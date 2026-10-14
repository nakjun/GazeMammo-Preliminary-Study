"""Frozen clinical whole +control past; fresh256head and sameaugprefix controlteacher KL."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from src.gaze_mammo_clip_1011 import MammoGaze,restore_mammo_gaze
from src.mammo_clip_visual_1011 import SOURCE
from gaze_nll_adaptation_1011 import cache_records
from gaze_nll_adaptation_1011 import load_base
from src.gaze_nll_adaptation_1011 import initialize
from train_gaze_path_augmentation_1011 import load_pools
from src.gaze_path_augmentation_1011 import augmented_sample
from src.guided_trajectory_1011 import nonvisual_state
from src.proposal_utility_1011 import state_sha256
from train_attention_1011 import digest
from src.proposal_utility_1011 import file_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--source-pool',default='runs/dual_target_1011/train_gaze_pool_cache');p.add_argument('--pool-labels',default='runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt');p.add_argument('--medical-checkpoint',default='runs/dual_target_1011/mammo_clip_public/assets/b5-model-best-epoch-7.tar');p.add_argument('--whole-cache',default='runs/dual_target_1011/mammo_clip_public/whole_cache');p.add_argument('--output',default='runs/dual_target_1011/mammo_clip_public/gaze_seed42');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;model=MammoGaze(a.medical_checkpoint,a.base).to(a.device).eval();original,prov=cache_records(a);source=dict(medical_checkpoint=str(Path(a.medical_checkpoint).resolve()),medical_checkpoint_sha256=file_sha256(a.medical_checkpoint),past_checkpoint=str(Path(a.base).resolve()),past_checkpoint_sha256=file_sha256(a.base));sourcepools,_=load_pools(a.source_pool,original,digest(a.base),prov['visual_state_sha256'],a.cache,pool_labels=a.pool_labels);nativepath=Path(a.whole_cache);nativeheader=json.loads((nativepath/'provenance.json').read_text());whole_sha=state_sha256(model.visual.whole.state_dict());past_sha=state_sha256(model.visual.past.state_dict())
 if nativeheader['kind']!='mammo_clip_whole_cache_1011' or json.loads((nativepath/'status.json').read_text())['state']!='complete' or nativeheader['medical_checkpoint_sha256']!=source['medical_checkpoint_sha256'] or nativeheader['whole_state_sha256']!=whole_sha or nativeheader['source_cache_provenance_sha256']!=digest(Path(a.cache)/'provenance.json') or nativeheader['source_past_visual_sha256']!=past_sha:raise ValueError('Medicalwhole source mismatch')
 cpu_rng_before=torch.get_rng_state();cuda_rng_before=torch.cuda.get_rng_state(a.device) if str(a.device).startswith('cuda') else None
 with torch.random.fork_rng(devices=[]):
  base,teacher_checkpoint=load_base(a.base);teacher=initialize(base).to(a.device).eval()
 if not torch.equal(cpu_rng_before,torch.get_rng_state()) or (cuda_rng_before is not None and not torch.equal(cuda_rng_before,torch.cuda.get_rng_state(a.device))):raise ValueError('Teacher construction changed student RNG')
 teacher.requires_grad_(False);teacher_state_sha=state_sha256(teacher.state_dict());teacher_memory={}
 with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=False):
  for start in range(0,len(original),8):
   group=original[start:start+8];features=[torch.stack([r['features'][level] for r in group]).to(a.device) for level in range(3)];memory=teacher.maps_features(features)
   for i,r in enumerate(group):teacher_memory[int(r['image_id'])]={k:v[i].cpu() for k,v in memory.items()}
 reference={int(r['image_id']):r for r in original};ids=[int(meta['image_id']) for meta in nativeheader['records']]
 if len(set(ids))!=505 or set(ids)!=set(reference):raise ValueError('Native cohort mismatch')
 rows=[]
 for meta in nativeheader['records']:
  file=nativepath/f"{meta['image_id']}.pt"
  if digest(file)!=meta['sha256']:raise ValueError('Medicalwhole file changed')
  r=torch.load(file,weights_only=False);ref=reference[int(meta['image_id'])]
  if r['image_id']!=int(meta['image_id']) or r['patient_key']!=meta['patient_key'] or r['split']!=meta['split'] or r['patient_key']!=ref['patient_key'] or r['split']!=ref['split'] or r['image_sha256']!=ref['image_sha256'] or r['file_name']!=ref['file_name'] or r['image_size']!=ref['image_size'] or not torch.equal(r['coords'],ref['coords']) or not torch.equal(r['mask'],ref['mask']) or r['feature'].shape!=(2048,48,29) or r['feature'].dtype!=torch.float32 or not torch.isfinite(r['feature']).all():raise ValueError('Medicalwhole metadata/tensors mismatch')
  r['vectors']=ref['vectors'];rows.append(r)
 train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];pools=sourcepools
 if len(train)!=412 or len(val)!=93 or set(pools)!={r['image_id'] for r in train}:raise ValueError('Medicalgaze cohort/pool mismatch')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);config=dict(seed=42,epochs=15,batch_size=4,lr=1e-4,source=source,source_code_sha256={str(f):digest(f) for f in [ROOT/'src/gaze_mammo_clip_1011.py',ROOT/'src/mammo_clip_visual_1011.py',ROOT/'scripts/train_mammo_clip_gaze_1011.py',*SOURCE.glob('*.py')]},cache_sha256=digest(nativepath/'provenance.json'),source_cache=str(nativepath.resolve()),initial_head_state_sha256=state_sha256(model.head_state()),pool_cache_sha256=digest(Path(a.source_pool)/'provenance.json'),source_pool_cache=str(Path(a.source_pool).resolve()),medical_checkpoint=source['medical_checkpoint'],medical_checkpoint_sha256=source['medical_checkpoint_sha256'],past_checkpoint=source['past_checkpoint'],past_checkpoint_sha256=source['past_checkpoint_sha256'],inventory=model.inventory(),optimizer_proof_sha256=digest(ROOT/'runs/dual_target_1011/mammo_clip_public/gaze_head_optimizer_proof_ieee.json'),sampling='fixedendpoints +14interiorstratauniform; seed_for42,imageID+epoch100000; trainonly; canonicalval93fixed',loss='image-macrohardNLL +1forwardKL(controlteacher||MammoGaze); matchedsameaugmentedprefix; noGaussian/mirror/boxes; frozenUPMCwhole/originalcontrolpast; genuineimage-onlyBOSlearnedbyhead',teacher_base_sha256=digest(a.base),teacher_visual_cache_sha256=digest(Path(a.cache)/'provenance.json'),teacher_crop_pool_sha256=digest(Path(a.source_pool)/'provenance.json'),teacher_training_patient_keys=teacher_checkpoint['training_patient_keys'],teacher_selection_patient_keys=teacher_checkpoint['selection_patient_keys'],teacher_head_policy='scopedcudNNIEEE; frozenvisualcacheunchanged',numerical_policy=dict(matmul_allow_tf32=False,visual_cudnn_allow_tf32=True,head_cudnn_allow_tf32=False,teacher_cudnn_allow_tf32=False,cudnn_enabled=True));(out/'config.json').write_text(json.dumps(config,indent=2));optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);history=[];best=float('inf');started=time.monotonic();train_metrics={}
 def save(name,data):
  target=out/name;temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(target)
 def epoch(data,training):
  model.train(training);total=0.;kl_total=0.
  for start in range(0,len(data),4):
   group=data[start:start+4];f=torch.stack([r['feature'] for r in group]).to(a.device);coords=torch.stack([r['coords'] for r in group]).to(a.device);vectors=torch.stack([r['vectors'] for r in group]).to(a.device);mask=torch.stack([r['mask'] for r in group]).bool().to(a.device);target=(coords[:,:,1]*48).long().clamp(0,47)*24+(coords[:,:,0]*24).long().clamp(0,23)
   with torch.set_grad_enabled(training):
    raw=model.teacher_logits(model.maps_features([f]*3),coords,vectors)
    if not torch.isfinite(raw).all() or raw.logsumexp(-1).abs().max()>1e-5:raise ValueError('Medicalgaze qnormalization failure')
    losses=(-raw.gather(-1,target[...,None]).squeeze(-1)*mask).sum(1)/mask.sum(1)
    if training:
     memory={k:torch.stack([teacher_memory[r['image_id']][k] for r in group]).to(a.device) for k in ['keys','prior','global_context']};teacher_vectors=torch.stack([r['teacher_vectors'] for r in group]).to(a.device)
     with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=False):teacher_raw=teacher.teacher_logits(memory,coords,teacher_vectors)
     kl=(teacher_raw.exp()*(teacher_raw-raw)).sum(-1);kl_images=(kl*mask).sum(1)/mask.sum(1);kl_total+=float(kl_images.detach().sum());loss=losses.mean()+kl_images.mean()
     if not torch.isfinite(loss):raise ValueError('Medicalgaze loss nonfinite')
     optimizer.zero_grad();loss.backward()
     if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Medicalgaze gradient nonfinite')
     optimizer.step()
   total+=float(losses.detach().sum())
  if training:train_metrics['forward_kl']=kl_total/len(data)
  return total/len(data)
 for ep in range(16):
  if ep:
   order=torch.randperm(412).tolist();sampled=[];selections=[]
   for i in order:
    r=train[i];sample=augmented_sample(pools[r['image_id']],r['image_id'],ep,42);teacher_sample=sample
    if sample['pool_indices']!=teacher_sample['pool_indices'] or sample['raw_indices']!=teacher_sample['raw_indices'] or not torch.equal(sample['coords'],teacher_sample['coords']) or not torch.equal(sample['mask'],teacher_sample['mask']):raise ValueError('Student/teacher selectedpoint mismatch')
    sampled.append({**r,**{k:sample[k] for k in ['coords','vectors','mask']},'teacher_vectors':teacher_sample['vectors']});selections.append(dict(image_id=r['image_id'],pool_indices=sample['pool_indices'],raw_indices=sample['raw_indices']))
   
   (out/f'epoch_{ep:02d}_sampling.json').write_text(json.dumps(selections));tn=epoch(sampled,True)
  else:tn=epoch(train,False)
  vn=epoch(val,False)
  
  entry=dict(epoch=ep,train_nll=tn,validation_nll=vn,train_forward_kl=train_metrics.get('forward_kl'));history.append(entry);save('history.json',history);ck=dict(kind='mammo_clip_gaze_1011',state=model.head_state(),config=config,epoch=ep,whole_state_sha256=whole_sha,past_state_sha256=past_sha,training_patient_keys=sorted({r['patient_key'] for r in train}),selection_patient_keys=sorted({r['patient_key'] for r in val}))
  if vn<best:best=vn;torch.save(ck,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(entry,indent=2))
  torch.save(ck,out/'last.pt');save('state.json',dict(state='running',epoch=ep,best_nll=best,seconds=time.monotonic()-started));print(json.dumps(entry),flush=True)
 if state_sha256(teacher.state_dict())!=teacher_state_sha:raise ValueError('Frozen teacher state changed')
 if any(p.grad is not None or p.requires_grad for p in teacher.parameters()):raise ValueError('Frozen teacher received gradients')
 unchanged=state_sha256(model.visual.whole.state_dict())==whole_sha and state_sha256(model.visual.past.state_dict())==past_sha
 if not unchanged:raise ValueError('FrozenMammoWhole/Past changed')
 best_ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);restored=restore_mammo_gaze(best_ck,a.device)
 if state_sha256(restored.head_state())!=state_sha256(best_ck['state']):raise ValueError('Exact complete medicalhead restore failed')
 save('summary.json',dict(state='complete',epochs=15,best_nll=best,best_epoch=best_ck['epoch'],frozen_whole_and_past_unchanged=unchanged,frozen_teacher_unchanged=True,teacher_state_sha256=teacher_state_sha,complete_head_compact_restore_sha256=state_sha256(restored.head_state()),seconds=time.monotonic()-started));save('state.json',dict(state='complete',epoch=15,best_nll=best,best_epoch=best_ck['epoch']))
if __name__=='__main__':main()
