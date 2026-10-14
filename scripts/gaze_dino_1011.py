"""One frozen original-DINO representation pilot, exact canonical16 cohort."""
import argparse,json,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.data import whole_tensor,patch_tensor
from src.gaze_dino_1011 import DinoGaze
from gaze_nll_adaptation_1011 import cache_records
from src.guided_trajectory_1011 import nonvisual_state
from src.proposal_utility_1011 import state_sha256
from train_attention_1011 import digest

def setup(a):
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;source=json.loads((ROOT/'runs/dual_target_1011/dino/source_provenance.json').read_text())
 if digest(source['checkpoint'])!=source['checkpoint_sha256']:raise ValueError('DINO source changed')
 if subprocess.check_output(['git','-C',source['source'],'rev-parse','HEAD'],text=True).strip()!=source['commit'] or subprocess.run(['git','-C',source['source'],'diff','--quiet']).returncode:raise ValueError('Pinned DINO source changed')
 if digest(source['official_args_path'])!=source['official_args_sha256'] or digest(Path(source['source'])/'LICENSE')!=source['license_sha256']:raise ValueError('DINO metadata dependency changed')
 model=DinoGaze(source['source'],source['checkpoint']).to(a.device).eval();records,prov=cache_records(a);return model,records,prov,source

def group_data(rows,device):
 feature=torch.stack([r['feature'] for r in rows]).to(device);coords=torch.stack([r['coords'] for r in rows]).to(device);vectors=torch.stack([r['vectors'] for r in rows]).to(device);mask=torch.stack([r['mask'] for r in rows]).bool().to(device);target=(coords[:,:,1]*48).long().clamp(0,47)*24+(coords[:,:,0]*24).long().clamp(0,23);return feature,coords,vectors,mask,target

def cache(a):
 model,records,prov,source=setup(a);out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);selected=[r for r in records if int(r['image_id']) in [64,73]] if a.stage=='smoke' else records;proof=[];metadata=[];started=time.monotonic()
 with torch.no_grad():
  for index,r in enumerate(selected):
   identity=int(r['image_id']);path=ROOT/'data/image'/r['file_name']
   if digest(path)!=r['image_sha256']:raise ValueError('Image changed')
   with Image.open(path) as original:
    image=original.convert('L');whole=whole_tensor(image,model.input_size)[None].to(a.device);feature=model.visual(whole)[-1][0].cpu().contiguous();crops=torch.stack([patch_tensor(image,point.numpy()*image.size) for point in r['coords'][:-1]]);vectors=torch.zeros(16,384)
    for start in range(0,15,8):vectors[start+1:start+9]=model.visual.patch_vector(crops[start:start+8].to(a.device)).cpu()
    entry={k:r[k] for k in ['coords','mask','image_size','image_sha256','file_name','patient_key','split']};entry.update(image_id=identity,feature=feature,vectors=vectors)
    if identity in [64,73]:
     memory=model.maps_features([feature[None].to(a.device)]*3);coords=r['coords'][None].to(a.device);pv=vectors[None].to(a.device);raw=model.teacher_logits(memory,coords,pv);live=model.maps(whole);actual=model.teacher_logits(live,coords,pv);error=float((raw-actual).abs().max());hidden=None;steps=[]
     for t in range(16):q,hidden,_=model.step_history(memory,coords[:,:t],hidden,pv[:,t]);steps.append(q)
     step_error=float((raw-torch.stack(steps,1)).abs().max());changed=coords.clone();changed[:,8:]=1-changed[:,8:];future_error=float((raw[:,:9]-model.teacher_logits(memory,changed,pv)[:,:9]).abs().max());proof.append(dict(image_id=identity,cached_live_error=error,teacher_step_error=step_error,future_error=future_error,native_shape=list(feature.shape)))
     if max(error,step_error)>1e-4 or future_error>1e-6:raise ValueError('DINO causal/runtime proof failed')
    torch.save(entry,out/f'{identity}.pt');metadata.append(dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],sha256=digest(out/f'{identity}.pt')))
   if (index+1)%25==0:print(json.dumps(dict(completed=index+1,total=len(selected))),flush=True)
 header=dict(kind='dino_gaze_native_cache_1011',records=metadata,source=source,visual_state_sha256=state_sha256(model.visual.state_dict()),canonical_cache_provenance_sha256=digest(Path(a.cache)/'provenance.json'),canonical_provenance=prov,input_size=[256,512],grid=[24,48],crop_size=224,native_shape=[384,64,32],proof=proof,smoke=a.stage=='smoke',seconds=time.monotonic()-started)
 (out/'provenance.json').write_text(json.dumps(header,indent=2));(out/'status.json').write_text(json.dumps(dict(state='complete',images=len(selected)),indent=2));print(json.dumps(dict(seconds=header['seconds'],images=len(selected),proof=proof)),flush=True)

def train(a):
 model,original,prov,source=setup(a);path=Path(a.dino_cache);header=json.loads((path/'provenance.json').read_text());status=json.loads((path/'status.json').read_text())
 if status['state']!='complete' or header['smoke'] or len(header['records'])!=505:raise ValueError('Incomplete505 cache')
 if header['source']!=source or header['visual_state_sha256']!=state_sha256(model.visual.state_dict()) or header['canonical_cache_provenance_sha256']!=digest(Path(a.cache)/'provenance.json'):raise ValueError('DINO dependency changed')
 reference={int(r['image_id']):r for r in original};rows=[]
 ids=[int(meta['image_id']) for meta in header['records']]
 if len(set(ids))!=505 or set(ids)!=set(reference):raise ValueError('DINO cache IDset mismatch')
 for meta in header['records']:
  file=path/f"{meta['image_id']}.pt"
  if digest(file)!=meta['sha256']:raise ValueError('Native file changed')
  r=torch.load(file,weights_only=False)
  if r['image_id']!=int(meta['image_id']) or r['patient_key']!=meta['patient_key'] or r['split']!=meta['split']:raise ValueError('DINO metadata/file identity mismatch')
  ref=reference[r['image_id']]
  if r['feature'].dtype!=torch.float32 or r['vectors'].dtype!=torch.float32 or not torch.equal(r['vectors'][0],torch.zeros(384)):raise ValueError('DINO cached dtype/BOS mismatch')
  if r['feature'].shape!=(384,64,32) or r['vectors'].shape!=(16,384) or not torch.isfinite(r['feature']).all() or not torch.isfinite(r['vectors']).all():raise ValueError('Invalid DINO cached tensors')
  if r['image_sha256']!=ref['image_sha256'] or r['image_size']!=ref['image_size'] or r['file_name']!=ref['file_name']:raise ValueError('DINO image metadata mismatch')
  if r['patient_key']!=ref['patient_key'] or r['split']!=ref['split'] or not torch.equal(r['coords'],ref['coords']) or not torch.equal(r['mask'],ref['mask']):raise ValueError('Source path changed')
  rows.append(r)
 trainrows=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
 if len(trainrows)!=412 or len(val)!=93:raise ValueError('Wrongcohort')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);config=dict(seed=42,epochs=15,batch_size=8,lr=1e-4,loss='image-macro hardcellNLL only; canonical16; no boxes/augmentation',source=source,source_code_sha256={str(Path(source['source'])/name):digest(Path(source['source'])/name) for name in ['vision_transformer.py','utils.py','LICENSE','README.md']},cache_sha256=digest(path/'provenance.json'),source_cache=str(path.resolve()),source_canonical_cache=str(Path(a.cache).resolve()),dim=256,numerical_policy=dict(matmul_allow_tf32=False,cudnn_allow_tf32=True));(out/'config.json').write_text(json.dumps(config,indent=2));optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=1e-4);history=[];best=float('inf');visualsha=state_sha256(model.visual.state_dict());started=time.monotonic()
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
  tn=epoch([trainrows[i] for i in torch.randperm(412).tolist()],True) if ep else epoch(trainrows,False);vn=epoch(val,False);entry=dict(epoch=ep,train_nll=tn,validation_nll=vn);history.append(entry);(out/'history.json').write_text(json.dumps(history,indent=2));ck=dict(kind='dino_gaze_1011',state=nonvisual_state(model),config=config,epoch=ep,visual_state_sha256=visualsha,training_patient_keys=sorted({r['patient_key'] for r in trainrows}),selection_patient_keys=sorted({r['patient_key'] for r in val}))
  if vn<best:best=vn;torch.save(ck,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(entry,indent=2))
  torch.save(ck,out/'last.pt');print(json.dumps(entry),flush=True)
 unchanged=state_sha256(model.visual.state_dict())==visualsha
 if not unchanged:raise ValueError('FrozenDINO changed')
 (out/'summary.json').write_text(json.dumps(dict(epochs=15,best_nll=best,frozen_visual_unchanged=unchanged,seconds=time.monotonic()-started),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','cache','train']);p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--dino-cache',default='runs/dual_target_1011/dino/native_cache');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();train(a) if a.stage=='train' else cache(a)
