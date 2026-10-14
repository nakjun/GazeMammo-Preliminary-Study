"""End-to-end ResNet50 + MIL fine-tuning on leakage-safe predicted patches."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data import ROOT,PatchDataset
from src.models import JointPatchClassifier
from src.metrics import classification_metrics,cluster_bootstrap,threshold_at_specificity
from pipeline import validate_prediction_provenance

def run(a):
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    device=torch.device(a.device);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    source=Path(a.features);cfg=json.loads((source/'config.json').read_text());frame=pd.read_csv(source/'manifest.csv')
    validate_prediction_provenance(argparse.Namespace(**cfg),frame)
    train=frame[frame.split=='train'].reset_index(drop=True);val=frame[frame.split=='val'].reset_index(drop=True)
    if set(train.patient_key)&set(val.patient_key):raise ValueError('Patient overlap')
    warm=torch.load(Path(a.init_classifier)/'best.pt',map_location='cpu',weights_only=False)
    old_manifest=pd.read_csv(Path(warm['args']['features'])/'manifest.csv')
    if set(old_manifest[old_manifest.split=='train'].patient_key)-set(train.patient_key):raise ValueError('Classifier initializer has seen outside training patients')
    model=JointPatchClassifier(k=cfg['k']).to(device);model.mil.load_state_dict(warm['state'])
    reference=next(model.visual.parameters()).detach().clone()
    opt=torch.optim.AdamW([{'params':model.visual.parameters(),'lr':a.backbone_lr},{'params':model.mil.parameters(),'lr':a.lr}],weight_decay=1e-4)
    loaders=[]
    for split,data in enumerate([train,val]):
        dataset=PatchDataset(data,cfg['method'],cfg['k'],cfg['crop_size'],cfg['seed'],cfg['predictions'],cfg['min_distance'],cfg.get('keep_predicted_centers',False))
        cached=[b for b in DataLoader(dataset,batch_size=None,num_workers=0)]
        loaders.append(DataLoader(cached,batch_size=a.batch_size,shuffle=split==0))
    def evaluate():
        model.eval();probabilities=[];attention=[]
        with torch.inference_mode():
            for b in loaders[1]:
                logit,w=model(b['patches'].to(device),b['mask'].to(device));probabilities.extend(logit.float().sigmoid().cpu().tolist());attention.extend(w.cpu().tolist())
        return probabilities,attention
    def checkpoint(epoch):return dict(state=model.state_dict(),end_to_end=True,k=cfg['k'],aggregation='mil',args=dict(vars(a),features=str(source.resolve())),epoch=epoch,training_patient_keys=sorted(train.patient_key.unique()))
    p,_=evaluate();initial=classification_metrics(val.label,p);best=initial['auroc'];best_epoch=0
    torch.save(checkpoint(0),out/'best.pt');history=[dict(epoch=0,validation=initial)]
    stale=0;print(json.dumps(history[0]),flush=True)
    for epoch in range(a.epochs):
        model.train();losses=[];opt.zero_grad()
        for step,b in enumerate(loaders[0]):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                logits,_=model(b['patches'].to(device),b['mask'].to(device));loss=nn.functional.binary_cross_entropy_with_logits(logits,b['label'].to(device))
            group_size=min(a.accumulation,len(loaders[0])-(step//a.accumulation)*a.accumulation)
            (loss/group_size).backward();losses.append(float(loss.detach()))
            if (step+1)%a.accumulation==0 or step+1==len(loaders[0]):nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();opt.zero_grad()
        p,_=evaluate();metrics=classification_metrics(val.label,p)
        change=float((next(model.visual.parameters()).detach()-reference).abs().mean())
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation=metrics,backbone_weight_change=change);history.append(record);print(json.dumps(record),flush=True)
        if metrics['auroc']>best:
            best=metrics['auroc'];best_epoch=epoch+1;stale=0;torch.save(checkpoint(best_epoch),out/'best.pt')
        else:stale+=1
        if stale>=a.patience:break
    torch.save(checkpoint(history[-1]['epoch']),out/'last.pt');best_state=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(best_state['state'])
    p,attention=evaluate();threshold=threshold_at_specificity(val.label,p,.9)
    result=val.copy();result['probability']=p;result['attention']=[json.dumps(x) for x in attention];result.to_csv(out/'validation_predictions.csv',index=False)
    for name,obj in [('history',history),('config',vars(a)),('threshold',dict(value=threshold,target_specificity=.9,source='validation')),
                     ('validation_metrics',dict(metrics=classification_metrics(val.label,p,threshold),auroc_ci=cluster_bootstrap(result),best_epoch=best_epoch,trainable_parameters=sum(p.numel() for p in model.parameters()),final_backbone_weight_change=history[-1].get('backbone_weight_change',0)))]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--init-classifier',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--accumulation',type=int,default=4);p.add_argument('--backbone-lr',type=float,default=1e-5);p.add_argument('--lr',type=float,default=1e-4)
    run(p.parse_args())
