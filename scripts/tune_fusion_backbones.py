"""Patient-disjoint inner tuning, followed by a fixed-epoch full-train refit.

No outer-validation labels or metrics are used for search selection.
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import gc
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data import ROOT,PatchDataset
from src.spatial import JointGlobalLocal
from src.metrics import classification_metrics,cluster_bootstrap,threshold_at_specificity

BASE=ROOT/'runs/fusion_hpo_seed42'
NAMES=['densenet161','efficientnet_b5','convnext_base','swin_b','vit_b_16']
HEADS=['global_logits','local_logits','fusion_logits']

def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def prepare():
    BASE.mkdir(parents=True,exist_ok=True)
    source=ROOT/'runs/spatial_v2_seed42/grid_control/features'
    f=pd.read_csv(source/'manifest.csv');cfg=json.loads((source/'config.json').read_text())
    from pipeline import validate_prediction_provenance
    validate_prediction_provenance(argparse.Namespace(**cfg),f)
    assert set(f.split)=={'train','val'} and len(f)==469
    payload=torch.load(ROOT/'runs/spatial_v2_cache/inputs.pt',map_location='cpu',weights_only=False)
    items=[]
    for split in ['train','val']:
        rows=f[f.split==split].reset_index(drop=True)
        ds=PatchDataset(rows,cfg['method'],cfg['k'],cfg['crop_size'],cfg['seed'],cfg.get('predictions'),cfg['min_distance'],True)
        for i in range(len(ds)):
            b=ds[i];b['image']=payload['items'][int(b['image_id'])]['image'];items.append(b)
    frame=pd.concat([f[f.split==s] for s in ['train','val']],ignore_index=True)
    assert [int(b['image_id']) for b in items]==frame.image_id.tolist()
    torch.save(items,BASE/'pixels.pt');frame.to_csv(BASE/'manifest.csv',index=False)
    inner_train=frame[(frame.split=='train')&(frame.crossfit_fold!=0)]
    inner_val=frame[(frame.split=='train')&(frame.crossfit_fold==0)]
    outer=frame[frame.split=='val']
    assert not set(inner_train.patient_key)&set(inner_val.patient_key)
    assert not set(frame[frame.split=='train'].patient_key)&set(outer.patient_key)
    save(BASE/'protocol.json',dict(seed=42,inner_train_images=len(inner_train),inner_val_images=len(inner_val),outer_val_images=len(outer),
        inner_split='existing patient crossfit_fold0',backbones=NAMES,selector='fixed image-only grid16',crop_size=224,whole_hw=[512,256],
        screen_backbone_lrs=[1e-6,3e-6,1e-5],head_lr=1e-4,screen_epochs=6,refine_max_epochs=12,refine_patience=5,
        warmup_epochs=2,selection='inner fusion AUROC only',refit='all376 training images; fixed inner-selected epoch; no outer early stopping',test_used=False))
    print('Cached',len(items),'images',flush=True)

def learning_rates(cfg,epoch,horizon=12):
    if epoch<cfg['warmup']:return 0.,cfg['head_lr']
    scale=1.
    if cfg['cosine']:
        progress=(epoch-cfg['warmup'])/max(1,horizon-cfg['warmup']-1)
        scale=.1+.9*.5*(1+math.cos(math.pi*min(1,progress)))
    return cfg['backbone_lr']*scale,cfg['head_lr']*scale

def augmented(image,patches,generator):
    flips=torch.rand(len(image),generator=generator)<.5
    if flips.any():
        image=image.clone();patches=patches.clone();flips=flips.to(image.device)
        image[flips]=image[flips].flip(-1);patches[flips]=patches[flips].flip(-1)
    return image,patches

def fit(name,cfg,train,validation,train_frame,val_frame,out,device,epochs,refit=False):
    out.mkdir(parents=True,exist_ok=True);save(out/'config.json',dict(backbone=name,**cfg,epochs=epochs,refit=refit))
    torch.manual_seed(42);np.random.seed(42)
    for _ in range(2):torch.empty((),dtype=torch.int64).random_()
    model=JointGlobalLocal(backbone_name=name,head_dropout=cfg['dropout']).to(device)
    optimizer=torch.optim.AdamW([{'params':model.visual.parameters(),'lr':cfg['backbone_lr']},
        {'params':model.head.parameters(),'lr':cfg['head_lr']}],weight_decay=cfg['weight_decay'])
    loader=DataLoader(train,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(42))
    val_loader=DataLoader(validation,batch_size=2)
    augmentation_rng=torch.Generator().manual_seed(4242)
    def evaluate(which):
        model.eval();pred={k:[] for k in HEADS}
        with torch.inference_mode():
            for b in which:
                values=model(b['image'].to(device),b['patches'].to(device),b['mask'].to(device))
                for key in HEADS:pred[key].extend(values[key].sigmoid().cpu().tolist())
        return pred
    best=-1;stale=0;history=[];started=time.monotonic()
    for epoch in range(epochs):
        model.train();warm=epoch<cfg['warmup']
        for p in model.visual.parameters():p.requires_grad=not warm
        if warm:model.visual.eval()
        backbone_lr,head_lr=learning_rates(cfg,epoch)
        optimizer.param_groups[0]['lr']=backbone_lr;optimizer.param_groups[1]['lr']=head_lr
        optimizer.zero_grad(set_to_none=True);losses=[]
        for step,b in enumerate(loader):
            image=b['image'].to(device);patches=b['patches'].to(device)
            if cfg['augmentation']:image,patches=augmented(image,patches,augmentation_rng)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(image,patches,b['mask'].to(device))
                loss=sum(nn.functional.binary_cross_entropy_with_logits(output[k],b['label'].to(device)) for k in HEADS)/3
            group_size=min(4,len(loader)-(step//4)*4);(loss/group_size).backward();losses.append(float(loss.detach()))
            if (step+1)%4==0 or step+1==len(loader):
                nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();optimizer.zero_grad(set_to_none=True)
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),backbone_lr=backbone_lr,head_lr=head_lr)
        if not refit:
            pred=evaluate(val_loader);record['inner']=classification_metrics(val_frame.label,pred['fusion_logits'])
            score=record['inner']['auroc']
            if not warm and score>best:
                best=score;stale=0
                torch.save(dict(kind='spatial_v2_fusion',state=model.state_dict(),model_config=model.model_config,epoch=epoch+1,
                    config=cfg,training_patient_keys=sorted(train_frame.patient_key.unique()),selection='inner AUROC'),out/'best.pt')
            elif not warm:stale+=1
        history.append(record);save(out/'history.json',history)
        print(name,out.name,json.dumps(record),flush=True)
        if not refit and epochs>6 and epoch+1>=6 and stale>=5:break
    if refit:
        torch.save(dict(kind='spatial_v2_fusion',state=model.state_dict(),model_config=model.model_config,epoch=epochs,config=cfg,
            training_patient_keys=sorted(train_frame.patient_key.unique()),selection='fixed epoch chosen on inner split; no outer early stopping'),out/'best.pt')
    else:
        ckpt=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(ckpt['state'])
    ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    predictions=evaluate(val_loader);results={}
    for key,values in predictions.items():
        f=val_frame.reset_index(drop=True).copy();f['probability']=values;f.to_csv(out/f'{key}_predictions.csv',index=False)
        results[key]=dict(metrics=classification_metrics(f.label,values,threshold_at_specificity(f.label,values)),auroc_ci=cluster_bootstrap(f)) if refit else dict(metrics=classification_metrics(f.label,values))
    train_pred=evaluate(DataLoader(train,batch_size=2))['fusion_logits']
    save(out/'metrics.json',dict(best_epoch=ckpt['epoch'],selected_head='fusion_logits',heads=results,
        train_metrics=classification_metrics(train_frame.label,train_pred),seconds=time.monotonic()-started,
        selection='inner' if not refit else 'fixed inner-selected epoch',test_used=False))
    result=(results['fusion_logits']['metrics']['auroc'],ckpt['epoch'])
    del model,optimizer,ckpt,output,loss;gc.collect();torch.cuda.empty_cache()
    return result

def worker(device,names):
    torch.set_num_threads(4);device=torch.device(device)
    frame=pd.read_csv(BASE/'manifest.csv');items=torch.load(BASE/'pixels.pt',map_location='cpu',weights_only=False)
    def subset(mask):
        indices=frame.index[mask].tolist();return [items[i] for i in indices],frame.loc[indices].reset_index(drop=True)
    train,tf=subset((frame.split=='train')&(frame.crossfit_fold!=0))
    inner,vf=subset((frame.split=='train')&(frame.crossfit_fold==0))
    full,ff=subset(frame.split=='train');outer,of=subset(frame.split=='val')
    assert not set(tf.patient_key)&set(vf.patient_key)
    for name in names:
        candidates=[]
        for lr in [1e-6,3e-6,1e-5]:
            cfg=dict(backbone_lr=lr,head_lr=1e-4,warmup=2,weight_decay=1e-4,dropout=.1,augmentation=False,cosine=False)
            trial=BASE/name/f'screen_lr{lr:g}'
            score,epoch=fit(name,cfg,train,inner,tf,vf,trial,device,6);candidates.append((score,cfg))
        chosen=max(candidates,key=lambda x:x[0])[1];refined=[]
        for regularized in [False,True]:
            cfg=dict(chosen)
            if regularized:cfg.update(weight_decay=1e-2,dropout=.3,augmentation=True,cosine=True)
            trial=BASE/name/('refine_regularized' if regularized else 'refine_minimal')
            score,epoch=fit(name,cfg,train,inner,tf,vf,trial,device,12);refined.append((score,epoch,cfg))
        score,epoch,cfg=max(refined,key=lambda x:x[0])
        save(BASE/name/'selected.json',dict(inner_auroc=score,epoch=epoch,config=cfg))
        fit(name,cfg,full,outer,ff,of,BASE/name/'refit',device,epoch,refit=True)
        print(name,'COMPLETE',flush=True)

def no_warm_control(device,names):
    """Test the original no-warmup option; select only on inner patients."""
    torch.set_num_threads(4);device=torch.device(device)
    frame=pd.read_csv(BASE/'manifest.csv');items=torch.load(BASE/'pixels.pt',map_location='cpu',weights_only=False)
    def subset(mask):
        ids=frame.index[mask].tolist();return [items[i] for i in ids],frame.loc[ids].reset_index(drop=True)
    train,tf=subset((frame.split=='train')&(frame.crossfit_fold!=0));inner,vf=subset((frame.split=='train')&(frame.crossfit_fold==0))
    full,ff=subset(frame.split=='train');outer,of=subset(frame.split=='val')
    for name in names:
        screens=[BASE/name/f'screen_lr{lr:g}' for lr in [1e-6,3e-6,1e-5]]
        started=time.monotonic()
        while not all((p/'metrics.json').exists() for p in screens):
            if time.monotonic()-started>7200:raise TimeoutError(name+' screening unavailable')
            time.sleep(5)
        best=max(screens,key=lambda p:json.loads((p/'metrics.json').read_text())['heads']['fusion_logits']['metrics']['auroc'])
        cfg=json.loads((best/'config.json').read_text());cfg={k:cfg[k] for k in ['backbone_lr','head_lr','warmup','weight_decay','dropout','augmentation','cosine']};cfg['warmup']=0
        score,epoch=fit(name,cfg,train,inner,tf,vf,BASE/name/'refine_no_warm',device,12)
        while not (BASE/name/'refit/metrics.json').exists():
            if time.monotonic()-started>7200:raise TimeoutError(name+' warm-up refit unavailable')
            time.sleep(5)
        previous=json.loads((BASE/name/'selected.json').read_text())
        if score>previous['inner_auroc']:
            selected=dict(inner_auroc=score,epoch=epoch,config=cfg,trial='refine_no_warm',refit_dir='refit_no_warm')
            fit(name,cfg,full,outer,ff,of,BASE/name/'refit_no_warm',device,epoch,refit=True)
        else:
            selected=dict(previous,trial='refine_regularized' if previous['config']['augmentation'] else 'refine_minimal',refit_dir='refit')
        save(BASE/name/'selected_final.json',selected);print(name,'NO_WARM_CONTROL_COMPLETE',flush=True)

def run():
    if list(BASE.glob('*/*/history.json')):raise FileExistsError('Existing trials; use --output with a fresh run directory')
    if not (BASE/'pixels.pt').exists():prepare()
    def launch(index):
        gpu=index%2;mode='worker' if index<2 else 'control';log_name=f'gpu{gpu}.log' if mode=='worker' else f'control_gpu{gpu}.log'
        with (BASE/log_name).open('w') as log:
            subprocess.run([sys.executable,__file__,mode,'--output',str(BASE),'--device',f'cuda:{gpu}','--names',*NAMES[gpu::2]],stdout=log,stderr=subprocess.STDOUT,check=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(launch,range(4)))
    subprocess.run([sys.executable,str(ROOT/'scripts/report_fusion_hpo.py'),'--output',str(BASE)],check=True)
    save(BASE/'completed.json',dict(status='complete',backbones=NAMES,no_warm_controls_complete=True,test_used=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker','control']);p.add_argument('--output',default=str(BASE));p.add_argument('--device',default='cuda:0');p.add_argument('--names',nargs='+',default=NAMES);a=p.parse_args();BASE=Path(a.output).resolve()
    if a.command=='prepare':prepare()
    elif a.command=='run':run()
    elif a.command=='worker':worker(a.device,a.names)
    else:no_warm_control(a.device,a.names)
