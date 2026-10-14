"""Train spatial gaze models, evaluate constrained policies, and run global/local fusion."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data import ROOT,load_gazes,reduce_gaze,whole_tensor,patch_tensor,seed_for
from src.spatial import SpatialTrajectory,gaussian_targets,JointGlobalLocal
from src.constraints import breast_mask,grid_mask,VisitConstraint,spatial_metrics
from src.metrics import dtw_distance,classification_metrics,threshold_at_specificity,cluster_bootstrap
from src.spatial_inference import predict_path

def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))
def setup(a):
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_num_threads(4)
    return torch.device(a.device)
def frames():return pd.read_csv(ROOT/'manifests/all.csv')
def subset(f,split,fold=None,labelled=True):
    f=f[f.split==split]
    if labelled:f=f[f.label>=0]
    if split=='train' and fold is not None:f=f[f.crossfit_fold!=fold]
    return f.sort_values('image_id').reset_index(drop=True)

def cache(a):
    setup(a);f=frames();f=f[f.split.isin(['train','val'])].sort_values('image_id')
    gazes=load_gazes();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);items={}
    for n,(_,r) in enumerate(f.iterrows()):
        with Image.open(ROOT/'data/image'/r.file_name) as source:image=source.convert('L')
        points=gazes[r.file_name];valid=np.isfinite(points).all(1)&(points[:,0]>=0)&(points[:,0]<image.width)&(points[:,1]>=0)&(points[:,1]<image.height)
        raw=torch.tensor(points[valid]/[image.width,image.height],dtype=torch.float32)
        density=gaussian_targets(raw,sigma_pixels=a.sigma).mean(0);density/=density.sum()
        segments=reduce_gaze(points,image.width,image.height,a.k,32);coords=torch.tensor(max(segments,key=len)/[image.width,image.height],dtype=torch.float32)
        t=torch.zeros(a.k,2);mask=torch.zeros(a.k,dtype=torch.bool);t[:len(coords)]=coords;mask[:len(coords)]=True
        items[int(r.image_id)]=dict(image=whole_tensor(image,size=(a.image_width,2*a.image_width)),coords=t,mask=mask,density=density,label=torch.tensor(float(r.label)),image_id=int(r.image_id))
        if n%50==0:print(f'cache {n}/{len(f)}',flush=True)
    torch.save(dict(items=items,k=a.k,grid=(24,48),sigma=a.sigma,image_size=(a.image_width,2*a.image_width)),out/'inputs.pt')
    f.to_csv(out/'manifest.csv',index=False);save(out/'config.json',vars(a))

def load_cache(a):return torch.load(Path(a.cache)/'inputs.pt',map_location='cpu',weights_only=False)
def samples(frame,payload):return [payload['items'][int(i)] for i in frame.image_id]

def foveal_cache(frame,payload,out):
    """Only previous GT crops; encoder never sees future fixations at an earlier step."""
    result=[]
    for n,(_,r) in enumerate(frame.iterrows()):
        b=dict(payload['items'][int(r.image_id)]);k=len(b['coords']);patches=torch.zeros(k,3,224,224);valid=torch.zeros(k,dtype=torch.bool)
        with Image.open(ROOT/'data/image'/r.file_name) as source:image=source.convert('L')
        for step in range(1,k):
            if b['mask'][step-1]:patches[step]=patch_tensor(image,b['coords'][step-1].numpy()*[image.width,image.height]);valid[step]=True
        b.update(previous_patches=patches,previous_valid=valid);result.append(b)
        if n%100==0:print(f'foveal crop cache {n}/{len(frame)}',flush=True)
    return result

def soft_loss(logits,targets,mask=None):
    loss=-(targets*logits.log_softmax(-1)).sum(-1)
    return loss[mask].mean() if mask is not None else loss.mean()

def train(a):
    device=setup(a);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);save(out/'config.json',vars(a))
    p=load_cache(a);f=frames();tr=subset(f,'train',a.holdout_fold,labelled=a.phase!='pretrain' and not a.all_gaze_train);va=subset(f,'val')
    assert not set(tr.patient_key)&set(va.patient_key)
    model=SpatialTrajectory(k=p['k'],fpn=a.fpn,foveal=a.foveal,backbone_name=a.backbone,dim=a.dim,motion_bias=a.motion_bias,input_size=p['image_size']).to(device)
    if a.init:
        c=torch.load(Path(a.init)/'best.pt',map_location='cpu',weights_only=False)
        if set(c['training_patient_keys'])-set(tr.patient_key):
            # Spatial pretraining may include unannotated patients from the same permitted train fold.
            allowed=set(subset(f,'train',a.holdout_fold,labelled=False).patient_key)
            if set(c['training_patient_keys'])-allowed:raise ValueError('Initializer patient leakage')
        if c.get('kind')!='spatial_v2':raise ValueError('Initializer must be a gaze model')
        if c['model_config']['fpn']!=a.fpn or c['model_config'].get('backbone_name','resnet50')!=a.backbone or c['model_config'].get('dim',128)!=a.dim:raise ValueError('Gaze initializer architecture mismatch')
        missing,unexpected=model.load_state_dict(c['state'],strict=False)
        if unexpected or set(missing)-{'motion.weight','motion.bias','motion_scale'}:raise ValueError('Unexpected gaze initializer keys')
    visual=list(model.visual.parameters());vid={id(x) for x in visual}
    optimizer=torch.optim.AdamW([{'params':visual,'lr':a.backbone_lr},{'params':[x for x in model.parameters() if id(x) not in vid],'lr':a.lr}],weight_decay=1e-4)
    data=[samples(x,p) for x in [tr,va]]
    if a.foveal and a.phase=='sequence':data=[foveal_cache(x,p,out) for x in [tr,va]]
    loaders=[DataLoader(d,batch_size=a.batch_size,shuffle=i==0) for i,d in enumerate(data)]
    def outputs(b):
        image=b['image'].to(device)
        if a.phase=='pretrain':return model.prior_logits(image)
        logits=model(image,b['coords'].to(device),b.get('previous_patches',None).to(device) if a.foveal else None,b.get('previous_valid',None).to(device) if a.foveal else None)
        return logits
    def loss(b):
        logits=outputs(b)
        if a.phase=='pretrain':return soft_loss(logits,b['density'].to(device))
        target=gaussian_targets(b['coords'].to(device),sigma_pixels=a.sigma)
        value=soft_loss(logits,target,b['mask'].to(device))
        if a.prior_weight:value=value+a.prior_weight*soft_loss(model.prior_logits(b['image'].to(device)),b['density'].to(device))
        return value
    def checkpoint(epoch):
        # Include pretraining patients in provenance as well as sequence-training patients.
        all_patients=set(tr.patient_key)
        if a.init:all_patients|=set(c['training_patient_keys'])
        return dict(kind='spatial_v2',state=model.state_dict(),model_config=model.model_config,training_patient_keys=sorted(all_patients),
                    selection_patient_keys=sorted(va.patient_key.unique()),epoch=epoch,args=vars(a))
    best=float('inf');stale=0;history=[];best_epoch=None
    reference_parameter=next(p for name,p in model.visual.named_parameters() if model.is_last_parameter(name));reference=reference_parameter.detach().clone()
    for epoch in range(a.epochs):
        model.set_backbone('frozen' if epoch<a.warmup_epochs else a.backbone_mode);model.train();train_losses=[]
        for b in loaders[0]:
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):value=loss(b)
            value.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();train_losses.append(float(value.detach()))
        model.eval();values=[];raw_values=[]
        with torch.inference_mode():
            for b in loaders[1]:
                logits=outputs(b)
                if a.phase=='pretrain':value=soft_loss(logits,b['density'].to(device))
                else:
                    valid=b['mask'].to(device);coords=b['coords'].to(device);gx,gy=model.grid
                    ids=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1)
                    nll=-logits.log_softmax(-1).gather(-1,ids[...,None]).squeeze(-1)
                    per_image=(nll*valid).sum(-1)/valid.sum(-1).clamp_min(1)
                    raw_values.extend(per_image.cpu().tolist())
                    value=soft_loss(logits,gaussian_targets(coords,sigma_pixels=a.sigma),valid)
                values.append((float(value),len(b['image'])))
        soft_score=sum(v*n for v,n in values)/sum(n for _,n in values)
        raw_score=float(np.mean(raw_values)) if raw_values else None
        score=raw_score if a.selection=='raw_nll' and a.phase=='sequence' else soft_score
        change=float((reference_parameter.detach()-reference).abs().mean())
        record=dict(epoch=epoch+1,train_loss=float(np.mean(train_losses)),validation_loss=soft_score,validation_raw_nll=raw_score,selection_score=score,backbone_mode=model.backbone_mode,backbone_last_weight_change=change);history.append(record);print(json.dumps(record),flush=True)
        if score<best:best=score;best_epoch=epoch+1;stale=0;torch.save(checkpoint(best_epoch),out/'best.pt')
        else:stale+=1
        if stale>=a.patience:break
    torch.save(checkpoint(history[-1]['epoch']),out/'last.pt');save(out/'history.json',history)
    save(out/'summary.json',dict(best_epoch=best_epoch,best_validation_loss=best,training_images=len(tr),validation_images=len(va),model_config=model.model_config))

@torch.no_grad()
def generate(a):
    device=setup(a);c=torch.load(Path(a.checkpoint)/'best.pt',map_location='cpu',weights_only=False)
    model=SpatialTrajectory(**c['model_config'],pretrained=False).to(device).eval();model.load_state_dict(c['state'])
    f=frames();f=f[(f.split==a.split)&(f.label>=0)].sort_values('image_id')
    if a.holdout_fold is not None:f=f[f.crossfit_fold==a.holdout_fold]
    if a.split=='test':raise ValueError('This development runner does not evaluate test')
    if set(f.patient_key)&set(c['training_patient_keys']):raise ValueError('Generation patient leakage')
    p=load_cache(a);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    coco=json.loads((ROOT/'manifests/instances_clipped.json').read_text());boxes={int(i):[] for i in f.image_id}
    for box in coco['annotations']:
        if box['image_id'] in boxes:boxes[box['image_id']].append(box['bbox'])
    prior_frame=subset(frames(),'train',c['args'].get('holdout_fold'),labelled=False)
    prior=torch.stack([p['items'][int(i)]['density'] for i in prior_frame.image_id]).mean(0).clamp_min(1e-8);prior/=prior.sum();prior=prior.to(device)
    predictions={};records=[]
    for n,(_,r) in enumerate(f.iterrows()):
        with Image.open(ROOT/'data/image'/r.file_name) as source:image=source.convert('L')
        result=predict_path(model,image,device,seed_for(a.seed,int(r.image_id)),a.strategy,a.prior_only,p['items'][int(r.image_id)]['image'],a.policy,a.radius,a.penalty)
        tissue=result['breast_mask'];memory=result['memory'];points=result['points'].tolist()
        predictions[str(r.image_id)]=points
        truth=p['items'][int(r.image_id)]['coords'][p['items'][int(r.image_id)]['mask']].numpy()
        target=p['items'][int(r.image_id)]['coords'][None].to(device);valid_target=p['items'][int(r.image_id)]['mask'].to(device)
        if a.prior_only:teacher=memory['prior'][:,None].expand(-1,model.k,-1)
        else:
            vectors=None
            if model.foveal:
                previous_crops=torch.stack([patch_tensor(image,xy.numpy()*[r.width,r.height]) for xy in target[0,:-1].cpu()]).to(device)
                encoded=model.visual.patch_vector(previous_crops);vectors=torch.zeros(1,model.k,model.patch.in_features,device=device);vectors[:,1:]=encoded
            teacher=model.teacher_logits(memory,target,vectors)
        gx,gy=model.grid;cells=(target[0,:,1]*gy).long().clamp(0,gy-1)*gx+(target[0,:,0]*gx).long().clamp(0,gx-1)
        logp=teacher[0].log_softmax(-1).gather(1,cells[:,None]).squeeze(1)[valid_target]
        logprior=prior[cells][valid_target].log()
        probabilities=teacher[0].softmax(-1);actual=probabilities.gather(1,cells[:,None])
        auc=((probabilities<actual).sum(-1)+.5*((probabilities==actual).sum(-1)-1))/(model.cells-1)
        nss=((actual.squeeze(1)-probabilities.mean(-1))/probabilities.std(-1,unbiased=False).clamp_min(1e-10))[valid_target]
        records.append(dict(image_id=int(r.image_id),patient_key=r.patient_key,raw_next_step_nll=float(-logp.mean()),information_gain_bits=float((logp-logprior).mean()/np.log(2)),
                            conditional_auc=float(auc[valid_target].mean()),conditional_nss=float(nss.mean()),normalized_dtw=dtw_distance(truth,np.array(points)),**spatial_metrics(points,tissue,boxes[int(r.image_id)])))
        if n%20==0:print(f'generate {n}/{len(f)}',flush=True)
    save(out/'predictions.json',predictions)
    save(out/'predictions.provenance.json',dict(method='trajectory',kind='spatial_v2',training_patient_keys=c['training_patient_keys'],generated_image_ids=f.image_id.tolist(),
                checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256((Path(a.checkpoint)/'best.pt').read_bytes()).hexdigest(),
                generation_policy=a.policy,generation_radius=a.radius,generation_penalty=a.penalty,generation_strategy=a.strategy,prior_only=a.prior_only,seed=a.seed))
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False)

def fusion(a):
    device=setup(a);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);save(out/'config.json',vars(a))
    cfg=json.loads((Path(a.features)/'config.json').read_text());frame=pd.read_csv(Path(a.features)/'manifest.csv')
    from src.data import PatchDataset
    from pipeline import validate_prediction_provenance
    validate_prediction_provenance(argparse.Namespace(**cfg),frame)
    data=[];payload=load_cache(a)
    for split in ['train','val']:
        rows=frame[frame.split==split].reset_index(drop=True)
        cached=[]
        if a.head_mode=='global':
            cached=[payload['items'][int(i)] for i in rows.image_id]
        else:
            dataset=PatchDataset(rows,cfg['method'],cfg['k'],cfg['crop_size'],cfg['seed'],cfg.get('predictions'),cfg['min_distance'],cfg.get('keep_predicted_centers',False))
            for b in DataLoader(dataset,batch_size=None):
                b['image']=payload['items'][int(b['image_id'])]['image'];cached.append(b)
        data.append((rows,cached))
    # Match the initial heads to the first fusion runs regardless of crop-cache iterator creation.
    torch.manual_seed(a.seed)
    for _ in range(2):torch.empty((),dtype=torch.int64).random_()
    import time
    started=time.monotonic()
    model=JointGlobalLocal(backbone_name=a.backbone).to(device)
    save(out/'model_info.json',dict(model_config=model.model_config,parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    optimizer=torch.optim.AdamW([{'params':model.visual.parameters(),'lr':1e-5},{'params':model.head.parameters(),'lr':1e-4}],weight_decay=1e-4)
    loaders=[DataLoader(x[1],batch_size=2,shuffle=i==0) for i,x in enumerate(data)]
    def outputs(b):
        if a.head_mode=='global':
            g=model.head.global_project(model.visual.patch_vector(b['image'].to(device)))
            return {'global_logits':model.head.global_head(g).squeeze(-1)}
        return model(b['image'].to(device),b['patches'].to(device),b['mask'].to(device))
    head_keys=['global_logits'] if a.head_mode=='global' else ['global_logits','local_logits','fusion_logits']
    selected_head='global_logits' if a.head_mode=='global' else 'fusion_logits'
    def evaluate():
        model.eval();pred={key:[] for key in head_keys}
        with torch.inference_mode():
            for b in loaders[1]:
                output=outputs(b)
                for key in pred:pred[key].extend(output[key].sigmoid().cpu().tolist())
        return pred
    best=-1;history=[];stale=0
    for epoch in range(a.epochs):
        model.train();optimizer.zero_grad();losses=[]
        for step,b in enumerate(loaders[0]):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                output=outputs(b)
                value=sum(nn.functional.binary_cross_entropy_with_logits(output[k],b['label'].to(device)) for k in head_keys)/len(head_keys)
            group_size=min(4,len(loaders[0])-(step//4)*4);(value/group_size).backward();losses.append(float(value.detach()))
            if (step+1)%4==0 or step+1==len(loaders[0]):nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();optimizer.zero_grad()
        predictions=evaluate();metrics={key:classification_metrics(data[1][0].label,value) for key,value in predictions.items()}
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation=metrics);history.append(record);print(json.dumps(record),flush=True)
        score=metrics[selected_head]['auroc']
        if score>best:
            best=score;stale=0;torch.save(dict(kind='spatial_v2_fusion',state=model.state_dict(),model_config=model.model_config,epoch=epoch+1,args=vars(a),features_cfg=cfg,training_patient_keys=sorted(data[0][0].patient_key.unique())),out/'best.pt')
        else:stale+=1
        if stale>=a.patience:break
    torch.save(dict(kind='spatial_v2_fusion',state=model.state_dict(),model_config=model.model_config,epoch=history[-1]['epoch'],args=vars(a),features_cfg=cfg),out/'last.pt')
    saved=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(saved['state']);predictions=evaluate();results={}
    for key,values in predictions.items():
        threshold=threshold_at_specificity(data[1][0].label,values);r=data[1][0].copy();r['probability']=values;r.to_csv(out/f'{key}_predictions.csv',index=False)
        results[key]=dict(metrics=classification_metrics(r.label,values,threshold),auroc_ci=cluster_bootstrap(r))
    save(out/'metrics.json',dict(best_epoch=saved['epoch'],selected_head=selected_head,heads=results));save(out/'history.json',history)
    save(out/'timing.json',dict(seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))

def parser():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ['cache','train','generate','fusion']:
        q=sub.add_parser(name);q.add_argument('--output',required=True);q.add_argument('--device',default='cuda:0');q.add_argument('--seed',type=int,default=42)
        q.add_argument('--cache',default=str(ROOT/'runs/spatial_v2_cache'));q.add_argument('--sigma',type=float,default=96);q.add_argument('--k',type=int,default=16)
        q.add_argument('--image-width',type=int,default=256)
        if name in ['train','fusion']:
            q.add_argument('--epochs',type=int,default=15);q.add_argument('--patience',type=int,default=4)
        if name=='train':
            q.add_argument('--phase',choices=['pretrain','sequence'],default='sequence');q.add_argument('--fpn',action='store_true');q.add_argument('--foveal',action='store_true')
            q.add_argument('--init');q.add_argument('--holdout-fold',type=int);q.add_argument('--batch-size',type=int,default=4)
            q.add_argument('--warmup-epochs',type=int,default=2);q.add_argument('--backbone-mode',choices=['frozen','last','all'],default='last')
            q.add_argument('--backbone-lr',type=float,default=1e-5);q.add_argument('--lr',type=float,default=1e-4)
            q.add_argument('--backbone',choices=['resnet50','densenet161','efficientnet_b5','convnext_base','swin_b','vit_b_16'],default='resnet50')
            q.add_argument('--dim',type=int,default=128);q.add_argument('--motion-bias',action='store_true');q.add_argument('--prior-weight',type=float,default=0)
            q.add_argument('--all-gaze-train',action='store_true');q.add_argument('--selection',choices=['soft_loss','raw_nll'],default='soft_loss')
        if name=='generate':
            q.add_argument('--checkpoint',required=True);q.add_argument('--split',choices=['train','val','test'],default='val');q.add_argument('--holdout-fold',type=int)
            q.add_argument('--strategy',choices=['argmax','sample'],default='sample');q.add_argument('--prior-only',action='store_true')
            q.add_argument('--policy',choices=['none','breast','soft','hard'],default='hard');q.add_argument('--radius',type=float,default=112);q.add_argument('--penalty',type=float,default=3)
        if name=='fusion':
            q.add_argument('--features',required=True);q.add_argument('--head-mode',choices=['fusion','global'],default='fusion')
            q.add_argument('--backbone',choices=['resnet50','densenet161','efficientnet_b5','convnext_base','swin_b','vit_b_16'],default='resnet50')
    return p
if __name__=='__main__':
    a=parser().parse_args();{'cache':cache,'train':train,'generate':generate,'fusion':fusion}[a.command](a)
