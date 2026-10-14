"""Matched DINO trial: only train chronological path sampling changes."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from gaze_dino_1011 import setup,group_data
from train_gaze_path_augmentation_1011 import load_pools
from src.gaze_path_augmentation_1011 import augmented_sample
from src.guided_trajectory_1011 import nonvisual_state
from src.proposal_utility_1011 import state_sha256
from train_attention_1011 import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--source-pool',default='runs/dual_target_1011/train_gaze_pool_cache');p.add_argument('--pool-labels',default='runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt');p.add_argument('--pool-cache',default='runs/dual_target_1011/dino/pool_cache');p.add_argument('--dino-cache',default='runs/dual_target_1011/dino/native_cache');p.add_argument('--output',default='runs/dual_target_1011/dino/gaze_augmented_seed42');p.add_argument('--device',default='cuda:0');a=p.parse_args();model,original,prov,source=setup(a);sourcepools,_=load_pools(a.source_pool,original,digest(a.base),prov['visual_state_sha256'],a.cache,pool_labels=a.pool_labels);nativepath=Path(a.dino_cache);nativeheader=json.loads((nativepath/'provenance.json').read_text());poolpath=Path(a.pool_cache);header=json.loads((poolpath/'provenance.json').read_text());status=json.loads((poolpath/'status.json').read_text());visualsha=state_sha256(model.visual.state_dict())
 if nativeheader['kind']!='dino_gaze_native_cache_1011' or nativeheader['smoke'] or json.loads((nativepath/'status.json').read_text())['state']!='complete' or nativeheader['source']!=source or nativeheader['visual_state_sha256']!=visualsha or nativeheader['canonical_cache_provenance_sha256']!=digest(Path(a.cache)/'provenance.json'):raise ValueError('Native frozen DINO dependency mismatch')
 expected=dict(source_pool_provenance_sha256=digest(Path(a.source_pool)/'provenance.json'),source_pool_labels_sha256=digest(a.pool_labels),source_dino_cache_provenance_sha256=digest(nativepath/'provenance.json'),canonical_source_provenance_sha256=digest(Path(a.cache)/'provenance.json'),sampler_source_sha256=digest(ROOT/'src/gaze_path_augmentation_1011.py'))
 if status['state']!='complete' or header['smoke'] or header['kind']!='dino_gaze_pool_cache_1011' or header['source']!=source or header['visual_state_sha256']!=visualsha or header['annotation_used'] is not False or any(header[k]!=v for k,v in expected.items()):raise ValueError('DINO pool source mismatch')
 reference={int(r['image_id']):r for r in original};ids=[int(meta['image_id']) for meta in nativeheader['records']]
 if len(set(ids))!=505 or set(ids)!=set(reference):raise ValueError('Native cohort mismatch')
 rows=[]
 for meta in nativeheader['records']:
  file=nativepath/f"{meta['image_id']}.pt"
  if digest(file)!=meta['sha256']:raise ValueError('Native file changed')
  r=torch.load(file,weights_only=False);ref=reference[int(r['image_id'])]
  if r['image_id']!=int(meta['image_id']) or r['patient_key']!=meta['patient_key'] or r['split']!=meta['split'] or r['image_sha256']!=ref['image_sha256'] or not torch.equal(r['coords'],ref['coords']) or not torch.equal(r['mask'],ref['mask']) or r['feature'].shape!=(384,64,32) or r['vectors'].shape!=(16,384) or r['feature'].dtype!=torch.float32 or r['vectors'].dtype!=torch.float32 or not torch.isfinite(r['feature']).all() or not torch.isfinite(r['vectors']).all() or not torch.equal(r['vectors'][0],torch.zeros(384)):raise ValueError('Native data mismatch')
  rows.append(r)
 train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val'];pools={};poolids=[int(m['image_id']) for m in header['records']]
 if len(set(poolids))!=412 or set(poolids)!={r['image_id'] for r in train}:raise ValueError('Train-only pool IDs mismatch')
 for meta in header['records']:
  file=poolpath/f"{meta['image_id']}.pt"
  if digest(file)!=meta['sha256']:raise ValueError('DINO pool file changed')
  pool=torch.load(file,weights_only=False);identity=int(meta['image_id']);ref=sourcepools[identity];row=next(r for r in train if r['image_id']==identity)
  if pool['image_id']!=identity or pool['patient_key']!=row['patient_key'] or pool['patient_key']!=meta['patient_key'] or pool['split']!='train' or meta['split']!='train' or pool['image_sha256']!=row['image_sha256'] or pool['source_pool_file_sha256']!=digest(Path(a.source_pool)/f'{identity}.pt') or not torch.equal(pool['coords'],ref['coords']) or not torch.equal(pool['raw_indices'],torch.as_tensor(ref['raw_indices'])) or pool['patch_vectors'].shape!=(len(pool['coords']),384) or pool['patch_vectors'].dtype!=torch.float32 or not torch.isfinite(pool['patch_vectors']).all():raise ValueError('DINO pool target/encoder mismatch')
  canonical=augmented_sample(pool,identity,0,augmented=False)
  if not torch.equal(canonical['coords'],row['coords']) or not torch.equal(canonical['vectors'],row['vectors']) or not torch.equal(canonical['mask'],row['mask']):raise ValueError('Canonical DINO reproduction failed')
  pools[identity]=pool
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);config=dict(seed=42,epochs=15,batch_size=8,lr=1e-4,source=source,source_code_sha256={str(Path(source['source'])/n):digest(Path(source['source'])/n) for n in ['vision_transformer.py','utils.py','LICENSE','README.md']},cache_sha256=digest(nativepath/'provenance.json'),source_cache=str(nativepath.resolve()),pool_cache_sha256=digest(poolpath/'provenance.json'),source_pool_cache=str(poolpath.resolve()),sampling='fixedendpoints +14interiorstratauniform; seed_for42,imageID+epoch100000; trainonly; canonicalval93fixed',loss='image-macrohardNLL only; matchedcanonicalDINOarchitecture/LR/batch/seed; noKL/mirror/boxes',numerical_policy=dict(matmul_allow_tf32=False,cudnn_allow_tf32=True));(out/'config.json').write_text(json.dumps(config,indent=2));optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);history=[];best=float('inf');started=time.monotonic()
 def epoch(data,training):
  model.train(training);total=0.
  for start in range(0,len(data),8):
   group=data[start:start+8];f,coords,vectors,mask,target=group_data(group,a.device)
   with torch.set_grad_enabled(training):
    raw=model.teacher_logits(model.maps_features([f]*3),coords,vectors);losses=(-raw.gather(-1,target[...,None]).squeeze(-1)*mask).sum(1)/mask.sum(1)
    if training:optimizer.zero_grad();losses.mean().backward();optimizer.step()
   total+=float(losses.detach().sum())
  return total/len(data)
 for ep in range(16):
  if ep:
   order=torch.randperm(412).tolist();sampled=[];selections=[]
   for i in order:
    r=train[i];sample=augmented_sample(pools[r['image_id']],r['image_id'],ep,42);sampled.append({**r,**{k:sample[k] for k in ['coords','vectors','mask']}});selections.append(dict(image_id=r['image_id'],pool_indices=sample['pool_indices'],raw_indices=sample['raw_indices']))
   (out/f'epoch_{ep:02d}_sampling.json').write_text(json.dumps(selections));tn=epoch(sampled,True)
  else:tn=epoch(train,False)
  vn=epoch(val,False)
  if ep==0 and abs(vn-6.904008065500567)>1e-4:raise ValueError('Matched seed42 initializer NLL changed')
  entry=dict(epoch=ep,train_nll=tn,validation_nll=vn);history.append(entry);(out/'history.json').write_text(json.dumps(history,indent=2));ck=dict(kind='dino_gaze_1011',state=nonvisual_state(model),config=config,epoch=ep,visual_state_sha256=visualsha,training_patient_keys=sorted({r['patient_key'] for r in train}),selection_patient_keys=sorted({r['patient_key'] for r in val}))
  if vn<best:best=vn;torch.save(ck,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(entry,indent=2))
  torch.save(ck,out/'last.pt');print(json.dumps(entry),flush=True)
 unchanged=state_sha256(model.visual.state_dict())==visualsha
 if not unchanged:raise ValueError('FrozenDINO changed')
 (out/'summary.json').write_text(json.dumps(dict(epochs=15,best_nll=best,frozen_visual_unchanged=unchanged,seconds=time.monotonic()-started),indent=2))
if __name__=='__main__':main()
