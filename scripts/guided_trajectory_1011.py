"""Train weak-box guidance and predict from image+RNG ID only; preserve base weights."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.data import whole_tensor,patch_tensor,seed_for
from src.spatial import gaussian_targets
from src.trajectory_1011 import objective,check_provenance
from src.guided_trajectory_1011 import ABLATIONS,candidate_geometry,training_box_targets,balanced_proposal_loss,free_losses,initialize_from_base,restore_guided,rho_schedule,nonvisual_state
from trajectory_1011 import Data,rows
from train_attention_1011 import digest,save


def base_path(path):
    path=Path(path);return path/'best.pt' if path.is_dir() else path

def teacher(model,item,sample,image,geometry,device,include_raw=False):
    memory=model.maps(item['image'][None].to(device));coords=sample['coords'][None].to(device);vectors=None
    if model.foveal:
        n=int(sample['valid'].sum());vectors=torch.zeros(1,coords.shape[1],model.patch.in_features,device=device)
        if n>1:
            with torch.no_grad():vectors[:,1:n]=model.visual.patch_vector(torch.stack([patch_tensor(image,p.numpy()*image.size) for p in sample['coords'][:n-1]]).to(device))
    raw=model.teacher_logits(memory,coords,vectors) if include_raw else None
    return model.teacher_guided(memory,coords,geometry,vectors),memory,raw


def train_annotations(frame,path):
    allowlist={int(r.image_id):r.patient_key for _,r in frame.iterrows()};boxes={i:[] for i in allowlist}
    payload=json.loads(Path(path).read_text())
    for annotation in payload['annotations']:
        image_id=int(annotation['image_id'])
        if image_id in allowlist:boxes[image_id].append(annotation['bbox'])
    return allowlist,boxes

@torch.no_grad()
def validate(model,data,geometries,device,prior,visual_output=None):
    model.eval();records=[];free_records=[]
    for item in data.items:
        image_id=int(item['row'].image_id);geometry=geometries[image_id];sample=data.sample(item,0,'legacy16')
        with Image.open(ROOT/'data/image'/item['row'].file_name) as source:image=source.convert('L')
        logits,memory,raw=teacher(model,item,sample,image,geometry,device,include_raw=True);_,nll=objective(logits,sample['coords'][None].to(device),sample['loss_mask'][None].to(device),grid=model.grid,sizes=image.size)
        _,raw_nll=objective(raw,sample['coords'][None].to(device),sample['loss_mask'][None].to(device),grid=model.grid,sizes=image.size)
        records.append(dict(raw_raw_nll=float(raw_nll[0][sample['loss_mask'].to(device)].mean()),image_id=image_id,patient_key=item['row'].patient_key,guided_raw_nll=float(nll[0][sample['loss_mask'].to(device)].mean()),geometry_valid=geometry['geometry_valid']))
        rollout=model.predicted_rollout(memory,image,geometry,data.a.steps,seed_for(data.a.seed,image_id))
        terms=free_losses(rollout['logp'].exp(),geometry,rollout['relevance'],torch.empty(0,model.cells,device=device),rollout['indices'])
        ids=rollout['indices'].cpu();free_records.append(dict(image_id=image_id,geometry_valid=geometry['geometry_valid'],expected_geometry_cost=float(terms['geometry']),expected_repeat_cost=float(terms['repeat']),sampled_mean_foreground_fraction=float(geometry['foreground'][ids].mean()) if geometry['geometry_valid'] else None,sampled_mean_inbounds_fraction=float(geometry['inbounds'][ids].mean()),sampled_unique_locations=len(torch.unique(ids)),coordinates=rollout['points'].cpu().tolist(),box_metrics='Not evaluated; validation boxes are not read; these free metrics never select checkpoint'))
        if visual_output is not None and image_id in (64,73):
            directory=visual_output/f'case_{image_id}';directory.mkdir(parents=True,exist_ok=False)
            proposal=torch.nan_to_num(model.proposal(memory['keys']).sigmoid(),nan=0.)[0,:,0].cpu().numpy();plot_prediction(image,geometry,rollout,proposal,directory)
            save(directory/'metadata.json',dict(image_id=image_id,rho=model.rho,geometry_valid=geometry['geometry_valid'],observed_gaze_used_for_prediction=False,annotations_used=False,coordinates=rollout['points'].cpu().tolist()))
    return float(np.mean([r['guided_raw_nll'] for r in records])),records,free_records


def checkpoint(model,base,a,epoch,train_keys,selection_keys,prior):
    return dict(kind='guided_trajectory_1011',state=nonvisual_state(model),model_config=model.model_config,epoch=epoch,args=vars(a),base_checkpoint=str(base_path(a.base_checkpoint).resolve()),base_checkpoint_sha256=digest(base_path(a.base_checkpoint)),training_patient_keys=sorted(set(base['training_patient_keys'])|set(train_keys)),selection_patient_keys=sorted(set(base['selection_patient_keys'])|set(selection_keys)),train_prior=prior,annotation_supervision='Explicit train-only benign+malignant union-box occupancy and free-coverage surrogate; boxless unknown, not healthy-negative',guidance=dict(lesion=2.,background=-3.,edge=-.5,crop=224,rho=model.rho,proposal_detached=True),loss_weights=dict(gaze=1.,prior=a.prior_weight,proposal=.5,geometry=.5,visit=.5,repeat=.05),rollout_protocol='Detached sampled positions/crops, differentiable conditional q; no gradient through discrete sampling, not RL',proposal_valid=model.proposal_valid)


def selection_eligible(rho,force_rho=None):return force_rho is not None or rho==1.
def validate_schedule(epochs,force_rho):
    if force_rho is None and epochs<5:raise ValueError('Default rho ramp requires at least5 epochs for rho1 checkpoint selection; explicit force-rho or smoke are exceptions')

def train(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);save(output/'config.json',vars(a));started=time.monotonic();device=torch.device(a.device)
    allowed_train=rows('train');allowed_val=rows('val');tr=allowed_train.head(a.train_limit) if a.train_limit else allowed_train;va=allowed_val.head(a.val_limit) if a.val_limit else allowed_val
    base=torch.load(base_path(a.base_checkpoint),map_location='cpu',weights_only=False);check_provenance(base,allowed_train.patient_key,allowed_val.patient_key,va.patient_key)
    model=initialize_from_base(base,a.ablation,rho=0.).to(device);train_data=Data(tr,a,model.input_size);val_data=Data(va,a,model.input_size)
    allowlist,boxes=train_annotations(allowed_train,ROOT/'manifests/instances_clipped.json');targets={};geometries={}
    for item in train_data.items+val_data.items:
        row=item['row'];image_id=int(row.image_id)
        with Image.open(ROOT/'data/image'/row.file_name) as source:image=source.convert('L');geometries[image_id]=candidate_geometry(image,model.grid)
        if row.split=='train':targets[image_id]=training_box_targets(image_id,row.patient_key,boxes[image_id],item['size'],allowlist,model.grid)
    model.proposal_valid=any(t['known'] for t in targets.values()) and a.ablation in ('lesion','combined')
    prior=torch.stack([item['density'] for item in train_data.items]).mean(0).clamp_min(1e-8);prior/=prior.sum()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4);best=float('inf');best_epoch=None;history=[];updates=0
    save(output/'supervision_provenance.json',dict(base_sha256=digest(base_path(a.base_checkpoint)),training_image_ids=tr.image_id.tolist(),training_patient_keys=sorted(tr.patient_key.unique()),selection_image_ids=va.image_id.tolist(),annotation_source=str(ROOT/'manifests/instances_clipped.json'),annotation_sha256=digest(ROOT/'manifests/instances_clipped.json'),box_supervised_images=sum(t['known'] for t in targets.values()),unknown_box_images=sum(not t['known'] for t in targets.values()),mask_diagnostics={str(i):dict(valid=g['geometry_valid'],error=g['geometry_error']) for i,g in geometries.items()},validation_boxes_used=False,final_test_used=False))
    for epoch in range(1,a.epochs+1):
        model.rho=a.force_rho if a.force_rho is not None else rho_schedule(epoch);model.model_config['rho']=model.rho;model.train();loss_records=[]
        order=np.random.default_rng(seed_for(a.seed,epoch)).permutation(len(train_data.items))
        for index in order:
            item=train_data.items[index];image_id=int(item['row'].image_id);geometry=geometries[image_id];target=targets[image_id];sample=train_data.sample(item,epoch,a.mode);optimizer.zero_grad()
            with Image.open(ROOT/'data/image'/item['row'].file_name) as source:image=source.convert('L')
            logits,memory,_=teacher(model,item,sample,image,geometry,device)
            gaze,_=objective(logits,sample['coords'][None].to(device),sample['loss_mask'][None].to(device),.5,a.sigma,image.size,model.grid)
            prior_loss=-(item['density'].to(device)*memory['prior'][0].log_softmax(-1)).sum()
            proposal=balanced_proposal_loss(model.proposal(memory['keys']).squeeze(),target['occupancy'].to(device),target['known']) if a.ablation in ('lesion','combined') else gaze*0
            rollout=model.predicted_rollout(memory,image,geometry,a.steps,seed_for(a.seed,image_id+epoch*100000),rho=model.rho)
            terms=free_losses(rollout['logp'].exp(),geometry,rollout['relevance'],target['coverage'].to(device) if target['known'] and a.ablation in ('lesion','combined') else torch.empty(0,model.cells,device=device),rollout['indices'])
            if a.ablation in ('gaze_only','lesion'):terms['geometry']=gaze*0
            if a.ablation!='combined':terms['repeat']=gaze*0
            if a.ablation=='gaze_only':terms['visit']=gaze*0
            loss=gaze+a.prior_weight*prior_loss+.5*proposal+model.rho*(.5*terms['geometry']+.5*terms['visit']+.05*terms['repeat'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite guided loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();updates+=1
            record=dict(image_id=image_id,total=float(loss.detach()),gaze=float(gaze.detach()),prior=float(prior_loss.detach()),proposal=float(proposal.detach()),**{k:float(v.detach()) for k,v in terms.items()});loss_records.append(record);print(json.dumps(dict(event='train_image',epoch=epoch,gradient_updates=updates,**record)),flush=True)
        visual_output=output/'visual_checks'/f'epoch_{epoch:02d}' if epoch%a.visual_every==0 or epoch==a.epochs else None
        score,validation,free_validation=validate(model,val_data,geometries,device,prior,visual_output);save(output/f'free_validation_metrics_epoch{epoch:02d}.json',free_validation);eligible=selection_eligible(model.rho,a.force_rho)
        record=dict(selection_eligible=eligible,epoch=epoch,rho=model.rho,train_losses=loss_records,validation_raw_nll=float(np.mean([r['raw_raw_nll'] for r in validation])),validation_guided_nll=score,validation_images=validation);history.append(record);save(output/'history.json',history)
        if eligible and score<best:best=score;best_epoch=epoch;torch.save(checkpoint(model,base,a,epoch,tr.patient_key,va.patient_key,prior),output/'best.pt')
        torch.save(checkpoint(model,base,a,epoch,tr.patient_key,va.patient_key,prior),output/'last.pt')
        print(json.dumps(dict(event='epoch',epoch=epoch,rho=model.rho,validation_guided_nll=score)),flush=True)
    save(output/'summary.json',dict(gradient_updates=updates,seconds=time.monotonic()-started,best_validation_guided_nll=best,best_epoch=best_epoch,effective_selection_rho=a.force_rho if a.force_rho is not None else 1.,selection='Minimum guided legacy16 validation NLL only at rho1 after ramp, or explicit fixed force-rho. Warmup last checkpoints retained; free geometry/box metrics never select and may favor another optimum',full_optimization_claim=False,smoke=a.command=='smoke'))


def load_guided(path,base_override=None):
    path=base_path(path);checkpoint=torch.load(path,map_location='cpu',weights_only=False);base=base_path(base_override or checkpoint['base_checkpoint'])
    if digest(base)!=checkpoint['base_checkpoint_sha256']:raise ValueError('Base checkpoint SHA mismatch')
    model=restore_guided(checkpoint,torch.load(base,map_location='cpu',weights_only=False));model.proposal_valid=checkpoint.get('proposal_valid',True);return model.eval(),checkpoint


def plot_prediction(image,geometry,result,proposal,output):
    import os;os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl');import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    gx,gy=(24,48);fig,axes=plt.subplots(1,4,figsize=(15,7));points=result['points'].cpu().numpy()*image.size
    axes[0].imshow(image,cmap='gray',vmin=0,vmax=255);axes[0].plot(*points.T,color='cyan',linewidth=1);axes[0].scatter(*points.T,c=np.arange(len(points)),cmap='turbo',s=18);axes[0].set_title('Image-only guided rollout');axes[0].set(xlim=(0,image.width),ylim=(image.height,0));axes[0].axis('off')
    for axis,array,title in [(axes[1],proposal,'Learned proposal (not GT segmentation)'),(axes[2],result['base_logp'][0].exp().cpu().numpy(),'Raw distribution, first step'),(axes[3],result['logp'][0].exp().cpu().numpy(),'Guided distribution, first step')]:axis.imshow(np.asarray(array).reshape(gy,gx),cmap='magma');axis.set_title(title,fontsize=9);axis.axis('off')
    fig.tight_layout();fig.savefig(output/'trajectory.png',dpi=150);plt.close(fig)

@torch.no_grad()
def predict(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);model,checkpoint=load_guided(a.checkpoint,a.base_checkpoint_override);model=model.to(a.device)
    with Image.open(a.image) as source:image=source.convert('L')
    geometry=candidate_geometry(image,model.grid);memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device));result=model.predicted_rollout(memory,image,geometry,a.steps,seed_for(a.seed,a.image_id),a.strategy)
    proposal_logits=model.proposal(memory['keys']);proposal_ok=bool(model.proposal_valid and torch.isfinite(proposal_logits).all());proposal=torch.nan_to_num(proposal_logits.sigmoid(),nan=0.)[0,:,0].cpu().numpy();record=dict(image=str(Path(a.image).resolve()),image_id=a.image_id,checkpoint_epoch=checkpoint['epoch'],rho=model.rho,ablation=model.ablation,proposal_valid=proposal_ok,geometry_valid=geometry['geometry_valid'],geometry_error=geometry['geometry_error'],coordinates=result['points'].cpu().tolist(),proposal=proposal.tolist(),raw_first_step_distribution=result['base_logp'][0].exp().cpu().tolist(),guided_first_step_distribution=result['logp'][0].exp().cpu().tolist(),seed=a.seed,inputs='image pixels and supplied RNG image ID only; no observed gaze/annotations/labels',annotation_supervision=checkpoint['annotation_supervision'],rollout_protocol=checkpoint['rollout_protocol'])
    if a.posthoc_annotations:
        annotations=json.loads(Path(a.posthoc_annotations).read_text());boxes=[b['bbox'] for b in annotations['annotations'] if int(b['image_id'])==a.image_id]
        pixels=result['points'].cpu().numpy()*image.size;centers=np.round(pixels);lo=centers-112;hi=lo+224;coverage=[]
        for x,y,w,h in boxes:
            overlap=np.maximum(0,np.minimum(hi,[x+w,y+h])-np.maximum(lo,[x,y])).prod(-1)/(w*h);coverage.append(float(overlap.max()))
        save(output/'posthoc_box_metrics.json',dict(explicit_posthoc_only=True,used_for_prediction=False,image_id=a.image_id,lesion_boxes=boxes,best_patch_box_area_coverage=coverage,any_box_contact=any(c>0 for c in coverage),interpretation='Post-hoc crop/box geometry, not lesion segmentation or expected coverage surrogate'))
    save(output/'prediction.json',record);plot_prediction(image,geometry,result,proposal,output);print(json.dumps(dict(output=str(output),rho=model.rho,geometry_valid=geometry['geometry_valid'])),flush=True)


def parser():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','predict','smoke']);p.add_argument('--output',required=True);p.add_argument('--base-checkpoint',default=str(ROOT/'runs/full_1011_seed42/improved/best.pt'));p.add_argument('--base-checkpoint-override');p.add_argument('--checkpoint');p.add_argument('--image');p.add_argument('--image-id',type=int,default=64);p.add_argument('--posthoc-annotations',help='Optional explicitly requested annotations, read only after predictions for separate metrics');p.add_argument('--ablation',choices=ABLATIONS,default='combined');p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--visual-every',type=int,default=3);p.add_argument('--train-limit',type=int,default=0);p.add_argument('--val-limit',type=int,default=0);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--prior-weight',type=float,default=.25);p.add_argument('--force-rho',type=float);p.add_argument('--steps',type=int,default=16);p.add_argument('--mode',choices=['legacy16','contiguous','mixed'],default='mixed');p.add_argument('--window',type=int,default=16);p.add_argument('--burn-in',type=int,default=8);p.add_argument('--mixed-ratio',type=float,default=.5);p.add_argument('--min-distance',type=float,default=32);p.add_argument('--sigma',type=float,default=64);p.add_argument('--strategy',choices=['sample','argmax'],default='sample');return p
if __name__=='__main__':
    a=parser().parse_args()
    if a.epochs<1 or a.steps<1 or a.visual_every<1 or (a.force_rho is not None and not 0<=a.force_rho<=1):raise ValueError('epochs/steps positive, force-rho within[0,1]')
    torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    if a.command=='predict':
        if not a.checkpoint or not a.image:raise ValueError('Predict needs checkpoint and image')
        predict(a)
    else:
        if a.command=='train':validate_schedule(a.epochs,a.force_rho)
        if a.command=='smoke':a.epochs=1;a.train_limit=a.train_limit or 2;a.val_limit=a.val_limit or 2;a.force_rho=1.
        train(a)
