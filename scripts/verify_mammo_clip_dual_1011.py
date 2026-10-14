"""Same-config actual64/73 integration proof; no training or annotation-based rollout."""
import json,sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.mammo_clip_dual_output_1011 import build_mammo_clip_config,load_mammo_clip_dual
from src.dual_output_1011 import config_sha256
from src.proposal_utility_1011 import state_sha256,file_sha256
from src.data import patch_tensor
from mammo_clip_dual_output_1011 import teacher_vectors
from gaze_nll_adaptation_1011 import routing

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;out=ROOT/'runs/dual_target_1011/mammo_clip_unified_runtime';out.mkdir(parents=True,exist_ok=False);config=build_mammo_clip_config();payload=dict(config=config,sha256=config_sha256(config));(out/'config.json').write_text(json.dumps(payload,indent=2));model,guided,head=load_mammo_clip_dual(config,'cuda:0');before=state_sha256(model.state_dict());refs=[]
for dependency in config['reference_caches']:
 if file_sha256(dependency['raw_cache'])!=dependency['raw_cache_sha256']:raise ValueError('RawexpectedSHA changed')
 refs.append({int(r['image_id']):r for r in torch.load(dependency['raw_cache'],weights_only=False)['records']})
route={int(r['image_id']):r for r in routing()};proof=[];started=time.monotonic()
with torch.no_grad():
 for identity in [64,73]:
  reference=refs[0][identity];coords=reference['coords'][None].cuda();mask=reference['loss_mask'];imagepath=ROOT/'data/image'/route[identity]['file_name']
  with Image.open(imagepath) as source:
   image=source.convert('L');memory=model.maps(image);vectors=teacher_vectors(model,image,coords,mask,'cuda:0');q=model.teacher_logits(memory,coords,vectors);direct_vectors=torch.zeros(1,16,1024,device='cuda:0');crops=torch.stack([patch_tensor(image,c.numpy()*image.size) for c in reference['coords'][:-1]])
   for begin in range(0,15,8):direct_vectors[:,begin+1:begin+9]=model.medical.visual.patch_vector(crops[begin:begin+8].cuda())[None]
   direct_med=model.medical.teacher_logits(model.medical.maps(image),coords,direct_vectors);reused_med=model.medical.teacher_logits(memory['medical'],coords,vectors[...,:1024]);reuse=float((direct_med-reused_med).abs().max());pv=float((direct_vectors-vectors[...,:1024]).abs().max())
  base=torch.logaddexp(refs[0][identity]['raw_logp']+math.log(.75),refs[1][identity]['raw_logp']+math.log(.25));base-=base.logsumexp(-1,keepdim=True);base=torch.logaddexp(base+math.log(.75),refs[2][identity]['raw_logp']+math.log(.25));base-=base.logsumexp(-1,keepdim=True);base=torch.logaddexp(base+math.log(.75),refs[3][identity]['raw_logp']+math.log(.25));base-=base.logsumexp(-1,keepdim=True);cached=float((q[0].cpu()-base).abs().max());state=None;steps=[]
  for t in range(16):value,state,_=model.step_history(memory,coords[:,:t],state,vectors[:,t]);steps.append(value)
  steperror=float((q-torch.stack(steps,1)).abs().max());changed=coords.clone();changed[:,8:]=1-changed[:,8:];changedvectors=vectors.clone();changedvectors[:,9:]=torch.randn_like(changedvectors[:,9:]);future=model.teacher_logits(memory,changed,changedvectors);causal=float((q[:,:9]-future[:,:9]).abs().max());normal=float(q.logsumexp(-1).abs().max());target=reference['target'].cuda();proof.append(dict(image_id=identity,medical_fresh_vs_reused_logp_error=reuse,past_vector_error=pv,combined_cached_logp_error=cached,teacher_step_error=steperror,current_future_error=causal,normalization_error=normal,nll=float(-q[0].gather(1,target[:,None]).squeeze(1)[mask.cuda()].mean())))
  if max(reuse,cached,steperror)>1e-4 or pv>1e-5 or causal>1e-5 or normal>1e-5:raise ValueError('Medical combined actual proof failed')
if before!=state_sha256(model.state_dict()):raise ValueError('Frozen combined state changed')
# Reload factory has already verified strict selected head and all frozen dependencies.
report=dict(config_sha256=payload['sha256'],records=proof,state_unchanged=True,originalcontrol_past_reuse_sha256=state_sha256(model.medical.visual.past.state_dict()),compute=config['compute'],ROI_unchanged=config['roi']==config['parent_config']['config']['roi'],tolerance=1e-4,source_config_path=str(out/'config.json'),seconds=time.monotonic()-started);(out/'actual64_73_proof.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
