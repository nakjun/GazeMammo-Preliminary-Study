"""Single15 chronological clinical cross-view representation pilot."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np
from src.gaze_crossview_1011 import CrossViewGaze,contrastive_aux,restore_crossview,BLOCK_PREFIXES
from src.gaze_path_augmentation_1011 import augmented_sample
from src.gaze_nll_adaptation_1011 import initialize
from src.proposal_utility_1011 import file_sha256,state_sha256
from gaze_nll_adaptation_1011 import cache_records,load_base,batch
from train_gaze_path_augmentation_1011 import load_pools

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--pool-cache',default='runs/dual_target_1011/train_gaze_pool_cache');p.add_argument('--pool-labels',default='runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt');p.add_argument('--prefix-cache',default='runs/dual_target_1011/gaze_crossview_prefix_cache');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;original,prov=cache_records(a);pools,_=load_pools(a.pool_cache,original,file_sha256(a.base),prov['visual_state_sha256'],a.cache,pool_labels=a.pool_labels);base,_=load_base(a.base);model=CrossViewGaze(base).to(a.device);frozen=state_sha256(model.frozen_state());native=Path(a.prefix_cache);header=json.loads((native/'provenance.json').read_text())
 causal=ROOT/'runs/dual_target_1011/gaze_crossview_guard_v3/causal_past_proof.json'
 if header['base_checkpoint_sha256']!=file_sha256(a.base) or header['source_cache_provenance_sha256']!=file_sha256(Path(a.cache)/'provenance.json') or header['dtype']!='float32':raise ValueError('Prefixbase/canonicalprovenance/dtype guard')
 if json.loads((native/'state.json').read_text())['state']!='complete' or header['frozen_state_sha256']!=frozen or header['source_code_sha256']!=file_sha256(ROOT/'src/gaze_crossview_1011.py') or header['causal_past_proof_sha256']!=file_sha256(causal) or json.loads(causal.read_text())['state']!='passed':raise ValueError('Fullprefix/causal/source guards')
 reference={int(r['image_id']):r for r in original};prefixes={}
 if len(reference)!=505 or len(header['records'])!=505 or {r['image_id'] for r in header['records']}!=set(reference):raise ValueError('Canonical505prefixIDs')
 for meta in header['records']:
  path=native/f"{meta['image_id']}.pt"
  if file_sha256(path)!=meta['sha256']:raise ValueError('NativeprefixSHA')
  r=torch.load(path,map_location='cpu',weights_only=False);ref=reference[r['image_id']]
  if r['image_id']!=meta['image_id'] or r['patient_key']!=ref['patient_key'] or r['split']!=ref['split'] or r['image_sha256']!=ref['image_sha256'] or r['prefix5_23'].shape!=(512,32,16) or r['prefix5_23'].dtype!=torch.float32 or not torch.isfinite(r['prefix5_23']).all():raise ValueError('Nativeprefixidentity/shape')
  prefixes[r['image_id']]=r['prefix5_23']
 train=[r for r in original if r['split']=='train'];val=[r for r in original if r['split']=='val'];trainpatients={r['patient_key'] for r in train};valpatients={r['patient_key'] for r in val}
 if len(train)!=412 or len(val)!=93 or trainpatients&valpatients or set(pools)!={int(r['image_id']) for r in train}:raise ValueError('Protectedcohorts')
 with torch.random.fork_rng(devices=[]):teacher=initialize(base).to(a.device).eval()
 teacher.requires_grad_(False);teacher_before=state_sha256(teacher.state_dict());teacher_memories={}
 with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=False):
  for start in range(0,len(original),8):
   group=original[start:start+8];f=[torch.stack([r['features'][k] for r in group]).to(a.device) for k in range(3)];mem=teacher.maps_features(f)
   for j,r in enumerate(group):teacher_memories[int(r['image_id'])]={k:v[j].cpu() for k,v in mem.items()}
 out=Path(a.output);out.mkdir(exist_ok=False,parents=True);config=dict(seed=42,epochs=15,batch_size=8,whole_lr=1e-5,head_projector_lr=1e-4,weight_decay=1e-4,temperature=.2,aux_weight=.1,KL_weight=1.,source_checkpoint=str(Path(a.base).resolve()),source_checkpoint_sha256=file_sha256(a.base),prefix_cache=str(native.resolve()),prefix_cache_provenance_sha256=file_sha256(native/'provenance.json'),pool_cache_provenance_sha256=file_sha256(Path(a.pool_cache)/'provenance.json'),pool_labels_sha256=file_sha256(a.pool_labels),canonical_cache_provenance_sha256=file_sha256(Path(a.cache)/'provenance.json'),frozen_state_sha256=frozen,teacher_state_sha256=teacher_before,training_patient_keys=sorted(trainpatients),selection_patient_keys=sorted(valpatients),selected_whole_blocks=list(BLOCK_PREFIXES),inventory=dict(whole_blocks=6380544,gazehead=631314,projector=525312,total=7537170),loss='image-macrohardNLL+1forwardKL(originalcontrol||student)+.1image-macrocurrenttargetlocal/cropInfoNCE; tau.2',negative_policy='Sameimage originalpool up to16; exactnp.round224 fullrectzero intersection +continuouscenterdistance>=448; deterministicseed; skipzeroeligible',runtime='Noauxiliarytarget/negative/projector inputs; wholeCNN+frozenoriginalpastCNN; originalR14separate',selection='Minimumfixedval93canonicalhardNLL includingepoch0; exploratory',source_code_sha256={n:file_sha256(ROOT/n) for n in ['src/gaze_crossview_1011.py','scripts/train_gaze_crossview_1011.py','src/gaze_path_augmentation_1011.py']});(out/'config.json').write_text(json.dumps(config,indent=2));whole=[p for n,p in model.named_parameters() if n.startswith(BLOCK_PREFIXES)];head=[p for n,p in model.named_parameters() if not n.startswith('visual.')];optimizer=torch.optim.AdamW([dict(params=whole,lr=1e-5),dict(params=head,lr=1e-4)],weight_decay=1e-4);history=[];best=float('inf');started=time.monotonic()
 def save(name,value):(out/name).write_text(json.dumps(value,indent=2,allow_nan=False))
 def ckpt(epoch):return dict(kind='gaze_crossview_1011',state=model.compact_state(),frozen_state_sha256=frozen,config=config,epoch=epoch)
 def epoch(data,training,ep):
  model.train(training);nlls=[];kls=[];auxs=[];coverage=dict(anchors=0,eligible_anchors=0,skipped_anchors=0,negative_total=0)
  for start in range(0,len(data),8):
   group=data[start:start+8];f,c,mask,vectors=batch(group,a.device);prefix=torch.stack([prefixes[int(r['image_id'])] for r in group]).to(a.device)
   with torch.set_grad_enabled(training):
    mem,f5=model.maps_prefix(f[0],prefix,True);q=model.teacher_logits(mem,c,vectors);target=(c[...,1]*48).long().clamp(0,47)*24+(c[...,0]*24).long().clamp(0,23);scores=(-q.gather(-1,target[...,None]).squeeze(-1)*mask).sum(-1)/mask.sum(-1)
    if not torch.isfinite(q).all() or q.logsumexp(-1).abs().max()>1e-5:raise ValueError('Crossviewnormalization')
    if training:
     tm={k:torch.stack([teacher_memories[int(r['image_id'])][k] for r in group]).to(a.device) for k in ['keys','prior','global_context']}
     with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=False):tq=teacher.teacher_logits(tm,c,vectors)
     kli=((tq.exp()*(tq-q)).sum(-1)*mask).sum(-1)/mask.sum(-1);aux,stats=contrastive_aux(model.aux_queries(f5,c),group,pools,ep);loss=scores.mean()+kli.mean()+.1*aux
     if not torch.isfinite(loss):raise ValueError('Crossviewloss')
     optimizer.zero_grad();loss.backward()
     if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Crossviewgradient')
     optimizer.step();kls.extend(kli.detach().cpu().tolist());auxs.extend([float(aux.detach())]*len(group))
     for key in coverage:coverage[key]+=stats[key]
   nlls.extend(scores.detach().cpu().tolist())
  return float(np.mean(nlls)),float(np.mean(kls)) if kls else None,float(np.mean(auxs)) if auxs else None,coverage
 for ep in range(16):
  if ep:
   sampled=[];selections=[]
   for idx in torch.randperm(412).tolist():
    r=train[idx];identity=int(r['image_id']);sample=augmented_sample(pools[identity],identity,ep,42);sampled.append({**r,'image_id':identity,'coords':sample['coords'],'mask':sample['mask'],'vectors':sample['vectors'],'selected_pool_indices':sample['pool_indices']});selections.append(dict(image_id=identity,pool_indices=sample['pool_indices'],raw_indices=sample['raw_indices']))
   save(f'epoch_{ep:02d}_sampling.json',selections);tn,kl,aux,coverage=epoch(sampled,True,ep)
  else:tn,kl,aux,coverage=epoch(train,False,0)
  vn,_,_,_=epoch(val,False,ep)
  if ep==0 and abs(vn-5.802225369279102)>1e-4:raise ValueError('Initialstrictcontrolqidentity')
  entry=dict(epoch=ep,train_nll=tn,train_forwardKL=kl,train_aux_InfoNCE=aux,negative_coverage=coverage,validation_nll=vn);history.append(entry);save('history.json',history)
  if vn<best:best=vn;torch.save(ckpt(ep),out/'best.pt');save('best_validation.json',entry)
  if ep==0:torch.save(ckpt(0),out/'initial.pt')
  torch.save(ckpt(ep),out/'last.pt');save('state.json',dict(state='running',epoch=ep,best_nll=best,seconds=time.monotonic()-started));print(json.dumps(entry),flush=True)
 if state_sha256(model.frozen_state())!=frozen or state_sha256(teacher.state_dict())!=teacher_before or any(p.grad is not None for p in teacher.parameters()):raise ValueError('Frozenprefix/past/later/teacher changed')
 best_ck=torch.load(out/'best.pt',map_location='cpu',weights_only=False);restored=restore_crossview(best_ck,base);assert state_sha256(restored.compact_state())==state_sha256(best_ck['state']);save('summary.json',dict(state='complete',epochs=15,best_validation_nll=best,frozen_all_branches_unchanged=True,frozen_teacher_unchanged=True,exact_compact_restore=True,best_selected_state_SHA_verified=True,best_epoch=best_ck['epoch'],seconds=time.monotonic()-started));save('state.json',dict(state='complete',epoch=15,best_nll=best))
if __name__=='__main__':main()
