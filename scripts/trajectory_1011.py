"""Separate, reproducible 1011 experiment runner; never evaluates final test."""
import sys, json, time, hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.data import ROOT,load_gazes,whole_tensor,patch_tensor,seed_for
from src.spatial import gaussian_targets
from src.trajectory_1011 import Trajectory1011,segments,window_sample,objective,check_provenance,initialize,restore
from src.metrics import dtw_distance


def save(path,obj):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(obj,indent=2,allow_nan=False));temporary.replace(path)
def checkpoint_path(path):
    path=Path(path); return path/'best.pt' if path.is_dir() else path

def rows(split,limit=0,holdout_fold=None):
    f=pd.read_csv(ROOT/'manifests/all.csv'); f=f[f.split==split].sort_values('image_id')
    if split=='val':f=f[f.label>=0]
    if holdout_fold is not None:f=f[f.crossfit_fold!=holdout_fold] if split=='train' else f
    return f.head(limit) if limit else f

class Data:
    def __init__(self,frame,a,input_size):
        self.items=[]; self.a=a
        gazes=load_gazes()
        for _,r in frame.iterrows():
            with Image.open(ROOT/'data/image'/r.file_name) as src:
                image=src.convert('L'); size=(image.width,image.height)
                parts=segments(gazes[r.file_name],*size,a.min_distance)
                if not parts:raise ValueError(f'No gaze: {r.image_id}')
                valid=np.isfinite(gazes[r.file_name]).all(1)&(gazes[r.file_name][:,0]>=0)&(gazes[r.file_name][:,0]<size[0])&(gazes[r.file_name][:,1]>=0)&(gazes[r.file_name][:,1]<size[1])
                raw=torch.tensor(gazes[r.file_name][valid]/size,dtype=torch.float32)
                density=gaussian_targets(raw,sigma_pixels=a.sigma,sizes=size).mean(0)
                self.items.append(dict(row=r,image=whole_tensor(image,input_size),parts=[p/size for p in parts],size=size,density=density))
    def sample(self,item,epoch,mode,start=None,segment=None):
        return window_sample(item['parts'],int(item['row'].image_id),self.a.seed,epoch,mode,self.a.window,self.a.burn_in,self.a.mixed_ratio,start,segment)

def output(model,item,sample,device,memory=None,patch_features=None):
    coords=sample['coords'][None].to(device)
    if memory is None:memory=model.maps(item['image'][None].to(device))
    vectors=None
    if model.foveal:
        vectors=torch.zeros(1,len(coords[0]),model.patch.in_features,device=device)
        n=int(sample['valid'].sum())
        if patch_features is not None:
            if n>1:vectors[:,1:n]=patch_features[sample['segment']][sample['positions'][:n-1].to(device)]
        else:
            with Image.open(ROOT/'data/image'/item['row'].file_name) as image:
                if n>1:
                    crops=torch.stack([patch_tensor(image,p.numpy()*item['size']) for p in sample['coords'][:n-1]]).to(device)
                    vectors[:,1:n]=model.visual.patch_vector(crops)
    logits,gates=model.teacher_logits(memory,coords,vectors,return_diagnostics=True)
    return logits,gates,memory

@torch.no_grad()
def evaluation_features(model,item,device,chunk_size=16):
    """At most one image's dense memory and segment patch vectors live on device."""
    memory=model.maps(item['image'][None].to(device));features=None
    if model.foveal:
        features=[]
        with Image.open(ROOT/'data/image'/item['row'].file_name) as image:
            for segment in item['parts']:
                vectors=[];coords=torch.as_tensor(segment,dtype=torch.float32)
                for start in range(0,len(coords),chunk_size):
                    crops=torch.stack([patch_tensor(image,p.numpy()*item['size']) for p in coords[start:start+chunk_size]]).to(device)
                    vectors.append(model.visual.patch_vector(crops))
                features.append(torch.cat(vectors))
    return memory,features

@torch.no_grad()
def evaluate_data(model,data,device,prior,cache_features=True):
    model.eval(); records={key:[] for key in ('legacy16','contiguous','legacy16_steps','contiguous_steps')};summaries={}
    started=time.monotonic()
    for image_index,item in enumerate(data.items):
        memory,features=evaluation_features(model,item,device) if cache_features else (None,None)
        for mode in ('legacy16','contiguous'):
            windows=[data.sample(item,0,mode)] if mode=='legacy16' else [data.sample(item,0,mode,start=s,segment=j) for j,p in enumerate(item['parts']) for s in range(0,len(p),data.a.window)]
            window_scores=[]; window_igs=[]
            for sample in windows:
                logits,gates,_=output(model,item,sample,device,memory,features)
                _,nll=objective(logits,sample['coords'][None].to(device),sample['loss_mask'][None].to(device),grid=model.grid,sizes=item['size'])
                gx,gy=model.grid; xy=sample['coords']; ids=(xy[:,1]*gy).long().clamp(0,gy-1)*gx+(xy[:,0]*gx).long().clamp(0,gx-1)
                valid=sample['loss_mask']; values=nll[0].cpu()[valid]; ig=(-values-prior[ids[valid]].log())/np.log(2)
                window_scores.append(float(values.mean()));window_igs.append(float(ig.mean()))
                nll_cpu=nll[0].cpu();gates_cpu=gates[0].cpu()
                for t in torch.where(valid)[0].tolist():
                    records[mode+'_steps'].append(dict(image_id=int(item['row'].image_id),patient_key=item['row'].patient_key,segment=sample['segment'],start=sample['start'],sequence_step=t,segment_position=int(sample['positions'][t]),truncated_history=sample['truncated_history'],raw_nll=float(nll_cpu[t]),gate_global=float(gates_cpu[t,0]),gate_local=float(gates_cpu[t,1]),gate_return=float(gates_cpu[t,2])))
            records[mode].append(dict(image_id=int(item['row'].image_id),patient_key=item['row'].patient_key,raw_nll=float(np.mean(window_scores)),information_gain_bits=float(np.mean(window_igs)),windows=len(windows)))
        del memory,features
        if (image_index+1)%10==0 or image_index+1==len(data.items):
            print(json.dumps(dict(event='validation_progress',images=image_index+1,total_images=len(data.items),elapsed_seconds=time.monotonic()-started)),flush=True)
    for mode in ('legacy16','contiguous'):
        summaries[mode]=dict(raw_nll=float(np.mean([r['raw_nll'] for r in records[mode]])),information_gain_bits=float(np.mean([r['information_gain_bits'] for r in records[mode]])))
    return summaries,records


def run(a):
    if a.batch_size<1 or a.epochs<1 or a.steps<1 or a.log_every<1:raise ValueError('batch-size/epochs/steps must be positive')
    if a.command!='train' and not a.checkpoint:raise ValueError('--checkpoint is required')
    original_args=vars(a).copy()
    restored_cp=None
    if a.command!='train':
        restored_cp=torch.load(checkpoint_path(a.checkpoint),map_location='cpu',weights_only=False)
    for key,default in [('window',16),('burn_in',8),('min_distance',32),('sigma',64)]:
        if getattr(a,key) is None:setattr(a,key,restored_cp.get('args',{}).get(key,default) if restored_cp else default)
    torch.manual_seed(a.seed);torch.set_num_threads(a.threads);device=torch.device(a.device)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);save(out/'config.json',vars(a));started=time.monotonic()
    allowed=rows('train',holdout_fold=a.holdout_fold);allowed_val=rows('val')
    if a.command=='train':
        tr=rows('train',a.limit,a.holdout_fold);va=rows('val',a.val_limit)
        if a.selection_fold is not None:
            allowed=rows('train');allowed_val=allowed[allowed.crossfit_fold==a.selection_fold];allowed=allowed[allowed.crossfit_fold!=a.selection_fold]
            tr=allowed.head(a.limit) if a.limit else allowed;va=allowed_val.head(a.val_limit) if a.val_limit else allowed_val
        if set(tr.patient_key)&set(va.patient_key):raise ValueError('Split patient overlap')
        model=Trajectory1011(grid=(24,48),k=a.window,dim=a.dim,backbone_name=a.backbone,fpn=a.fpn,foveal=a.foveal,motion_bias=a.motion_bias,input_size=(a.image_width,2*a.image_width),spatial_history=a.spatial_history,behavior_mixture=a.behavior_mixture,history_length=a.history_length,pretrained=a.pretrained).to(device)
        init=None
        if a.init:
            init=torch.load(checkpoint_path(a.init),map_location='cpu',weights_only=False)
            check_provenance(init,allowed.patient_key,allowed_val.patient_key)
            save(out/'initializer.json',dict(path=str(checkpoint_path(a.init)),sha256=hashlib.sha256(checkpoint_path(a.init).read_bytes()).hexdigest(),**initialize(model,init)))
        train_data=Data(tr,a,model.input_size);val_data=Data(va,a,model.input_size)
        # Train-only full raw gaze density, uniform across images; independent of sequence windows.
        prior=torch.stack([i['density'] for i in train_data.items]).mean(0).clamp_min(1e-8);prior/=prior.sum()
        vp=list(model.visual.parameters());ids={id(p) for p in vp};optimizer=torch.optim.AdamW([dict(params=vp,lr=a.backbone_lr),dict(params=[p for p in model.parameters() if id(p) not in ids],lr=a.lr)],weight_decay=a.weight_decay)
        history=[];sampling=[];best=float('inf');updates=0
        for epoch in range(a.epochs):
            epoch_started=time.monotonic()
            model.set_backbone('frozen' if epoch<a.warmup_epochs else a.backbone_mode);model.train();losses=[]
            order=np.random.default_rng(seed_for(a.seed,epoch)).permutation(len(train_data.items));optimizer.zero_grad()
            for step,idx in enumerate(order):
                item=train_data.items[idx];sample=train_data.sample(item,epoch,a.mode)
                sampling.append(dict(epoch=epoch+1,image_id=int(item['row'].image_id),segment=sample['segment'],start=sample['start'],prefix=sample['prefix'],mode=sample['mode'],truncated_history=sample['truncated_history'],segment_positions=sample['positions'].tolist(),valid=sample['valid'].tolist(),loss_mask=sample['loss_mask'].tolist()))
                logits,_,memory=output(model,item,sample,device)
                loss,_=objective(logits,sample['coords'][None].to(device),sample['loss_mask'][None].to(device),a.nll_weight,a.sigma,item['size'],model.grid)
                if a.prior_weight:loss=loss+a.prior_weight*(-item['density'].to(device)*memory['prior'][0].log_softmax(-1)).sum()
                group_size=min(a.batch_size,len(order)-(step//a.batch_size)*a.batch_size)
                (loss/group_size).backward();losses.append(float(loss.detach()))
                if (step+1)%a.batch_size==0 or step+1==len(order):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();optimizer.zero_grad();updates+=1
                    if updates%a.log_every==0 or step+1==len(order):
                        print(json.dumps(dict(event='train_progress',epoch=epoch+1,images=step+1,total_images=len(order),gradient_updates=updates,mean_loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-epoch_started)),flush=True)
            summary,_=evaluate_data(model,val_data,device,prior);record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation=summary,epoch_seconds=time.monotonic()-epoch_started);history.append(record);print(json.dumps(record),flush=True)
            cp=dict(kind='trajectory_1011',state=model.state_dict(),model_config=model.model_config,args=vars(a),epoch=epoch+1,gradient_updates=updates,train_prior=prior,training_patient_keys=sorted(set(tr.patient_key)|set(init['training_patient_keys'] if init else [])),selection_patient_keys=sorted(set(va.patient_key)|set(init['selection_patient_keys'] if init else [])),initializer_provenance={k:init[k] for k in ('training_patient_keys','selection_patient_keys')} if init else None)
            score=summary[a.selection_mode]['raw_nll']
            if score<best:best=score;torch.save(cp,out/'best.pt')
            torch.save(cp,out/'last.pt')
            save(out/'history.json',history);save(out/'sampling.json',sampling)
        save(out/'history.json',history);save(out/'sampling.json',sampling)
    else:
        cp=restored_cp
        f=rows(a.split,a.limit,a.holdout_fold)
        if cp['args'].get('selection_fold') is not None:
            alltrain=rows('train');fold=cp['args']['selection_fold'];allowed=alltrain[alltrain.crossfit_fold!=fold];allowed_val=alltrain[alltrain.crossfit_fold==fold]
        if a.split=='train':
            if a.holdout_fold is None:raise ValueError('Train evaluation requires --holdout-fold')
            f=rows('train');f=f[f.crossfit_fold==a.holdout_fold];f=f.head(a.limit) if a.limit else f
        check_provenance(cp,allowed.patient_key,allowed_val.patient_key,f.patient_key,a.independent)
        model=restore(cp).to(device).eval();prior=cp['train_prior']
        if a.command=='evaluate' or a.gaze_diagnostics:data=Data(f,a,model.input_size)
        else:
            data=argparse.Namespace(items=[])
            for _,r in f.iterrows():
                with Image.open(ROOT/'data/image'/r.file_name) as image:data.items.append(dict(row=r,image=whole_tensor(image,model.input_size),size=(image.width,image.height)))
        save(out/'preprocessing.json',dict(effective={k:getattr(a,k) for k in ('window','burn_in','min_distance','sigma')},explicit_overrides={k:original_args[k] for k in ('window','burn_in','min_distance','sigma') if original_args[k] is not None}))
        save(out/'provenance.json',dict(checkpoint=str(checkpoint_path(a.checkpoint)),checkpoint_sha256=hashlib.sha256(checkpoint_path(a.checkpoint).read_bytes()).hexdigest(),training_patient_keys=cp['training_patient_keys'],selection_patient_keys=cp['selection_patient_keys'],independent_holdout=not bool(set(f.patient_key)&set(cp['selection_patient_keys'])),generated_image_ids=f.image_id.tolist()))
        if a.command=='evaluate':
            summary,records=evaluate_data(model,data,device,prior);save(out/'metrics.json',summary)
            for key,value in records.items():pd.DataFrame(value).to_csv(out/f'{key}.csv',index=False)
        else:
            predictions={};geometry=[]
            with torch.no_grad():
                for item in data.items:
                    memory=model.maps(item['image'][None].to(device));prefix=torch.zeros(1,0,2,device=device);hidden=None;gates=[]
                    generator=torch.Generator(device=device).manual_seed(seed_for(a.seed,int(item['row'].image_id)))
                    with Image.open(ROOT/'data/image'/item['row'].file_name) as image:
                        for step in range(a.steps):
                            vector=model.visual.patch_vector(patch_tensor(image,prefix[0,-1].cpu().numpy()*item['size'])[None].to(device)) if model.foveal and step else None
                            logp,hidden,diag=model.step_history(memory,prefix,hidden,vector)
                            if a.policy=='radius-hard' and step:
                                distance=(model.cell_xy[:,None]-prefix[0][None])*prefix.new_tensor(item['size']);bad=distance.square().sum(-1).sqrt().min(-1).values<a.radius
                                if bool(bad.all()):raise ValueError('radius-hard exhausted all cells; reduce radius')
                                logp=logp.masked_fill(bad[None],-torch.inf)
                            cell=int(logp.argmax(-1)) if a.strategy=='argmax' else int(torch.multinomial(logp.softmax(-1),1,generator=generator))
                            prefix=torch.cat([prefix,model.cell_xy[cell][None,None]],1);gates.append(diag['gates'][0].cpu().tolist())
                    p=prefix[0].cpu().numpy();predictions[str(item['row'].image_id)]=p.tolist();delta=np.diff(p,axis=0);distance=np.linalg.norm(delta,axis=1)
                    truth=data.sample(item,0,'legacy16') if a.gaze_diagnostics else None
                    dtw=dtw_distance(truth['coords'][truth['valid']].numpy(),p) if truth is not None else None
                    return_rate=float(np.mean([bool((np.linalg.norm((p[:j]-p[j])*item['size'],axis=1)<112).any()) for j in range(1,len(p))])) if len(p)>1 else 0.
                    geometry.append(dict(image_id=int(item['row'].image_id),normalized_dtw=dtw,step_distances_normalized=distance.tolist(),step_distances_pixels=np.linalg.norm(delta*item['size'],axis=1).tolist(),step_directions_radians_pixels=np.arctan2(delta[:,1]*item['size'][1],delta[:,0]*item['size'][0]).tolist(),return_rate_112px=return_rate,gates=gates))
            save(out/'predictions.json',predictions);save(out/'geometry.json',geometry)
    save(out/'timing.json',dict(seconds=time.monotonic()-started,peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))


def parser():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','evaluate','generate']);p.add_argument('--output',required=True)
    p.add_argument('--log-every',type=int,default=10);p.add_argument('--checkpoint');p.add_argument('--init');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mode',choices=['legacy16','contiguous','mixed'],default='mixed');p.add_argument('--mixed-ratio',type=float,default=.5);p.add_argument('--window',type=int);p.add_argument('--burn-in',type=int);p.add_argument('--min-distance',type=float)
    p.add_argument('--history-length',type=int,choices=[1,2,4],default=4);p.add_argument('--spatial-history',action='store_true');p.add_argument('--behavior-mixture',action='store_true');p.add_argument('--fpn',action='store_true');p.add_argument('--foveal',action='store_true');p.add_argument('--motion-bias',action='store_true');p.add_argument('--pretrained',action='store_true')
    p.add_argument('--backbone',default='resnet50');p.add_argument('--dim',type=int,default=128);p.add_argument('--image-width',type=int,default=256);p.add_argument('--sigma',type=float);p.add_argument('--nll-weight',type=float,default=.5);p.add_argument('--prior-weight',type=float,default=0)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--backbone-lr',type=float,default=1e-5);p.add_argument('--epochs',type=int,default=1);p.add_argument('--warmup-epochs',type=int,default=2);p.add_argument('--backbone-mode',choices=['frozen','last','all'],default='last')
    p.add_argument('--limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0);p.add_argument('--holdout-fold',type=int);p.add_argument('--selection-fold',type=int);p.add_argument('--batch-size',type=int,default=2);p.add_argument('--split',choices=['val','train'],default='val');p.add_argument('--independent',action='store_true');p.add_argument('--selection-mode',choices=['legacy16','contiguous'],default='legacy16')
    p.add_argument('--gaze-diagnostics',action='store_true');p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--steps',type=int,default=16);p.add_argument('--strategy',choices=['sample','argmax'],default='sample');p.add_argument('--policy',choices=['none','radius-hard'],default='none');p.add_argument('--radius',type=float,default=112)
    return p
if __name__=='__main__':run(parser().parse_args())
