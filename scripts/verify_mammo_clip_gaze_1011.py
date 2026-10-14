"""Actual train-only head update, frozen encoders, causal/fullpixel compact restore proof."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.gaze_mammo_clip_1011 import MammoGaze,restore_mammo_gaze
from src.mammo_clip_visual_1011 import preprocess
from src.data import patch_tensor
from src.proposal_utility_1011 import state_sha256,file_sha256

torch.manual_seed(42);torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;medical=ROOT/'runs/dual_target_1011/mammo_clip_public/assets/b5-model-best-epoch-7.tar';past=ROOT/'runs/full_1011_seed42/control/best.pt';model=MammoGaze(medical,past).cuda().eval();whole_sha=state_sha256(model.visual.whole.state_dict());past_sha=state_sha256(model.visual.past.state_dict());head_before=state_sha256(model.head_state());rows={i:torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{i}.pt',weights_only=False) for i in [62,64,73]};features={};images={};proof=[];start=time.monotonic()
with torch.no_grad():
 for identity,row in rows.items():
  path=ROOT/'data/image'/row['file_name']
  if file_sha256(path)!=row['image_sha256']:raise ValueError('Source image changed')
  with Image.open(path) as source:image=source.convert('L');images[identity]=image.copy();tensor,metadata=preprocess(image);features[identity]=model.visual.whole(tensor[None].cuda()).cpu().contiguous()
for phase in ['initial','updated']:
 if phase=='updated':
  row=rows[62];coords=row['coords'][None].cuda();vectors=row['vectors'][None].cuda();f=features[62].cuda();model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4);raw=model.teacher_logits(model.maps_features([f]*3),coords,vectors);target=(coords[:,:,1]*48).long().clamp(0,47)*24+(coords[:,:,0]*24).long().clamp(0,23);optimizer.zero_grad();(-raw.gather(-1,target[...,None]).mean()).backward()
  if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()) or model.lateral[2].weight.grad is None or model.rnn.weight_ih_l0.grad is None:raise ValueError('Medicalhead finitegradient proof failed')
  optimizer.step();model.eval()
 for identity,row in rows.items():
  with torch.no_grad():
   coords=row['coords'][None].cuda();vectors=row['vectors'][None].cuda();f=features[identity].cuda();memory=model.maps_features([f]*3);cached=model.teacher_logits(memory,coords,vectors);crops=torch.stack([patch_tensor(images[identity],c.numpy()*images[identity].size) for c in row['coords'][:-1]]);actual_vectors=torch.zeros_like(vectors)
   for begin in range(0,15,8):actual_vectors[:,begin+1:begin+9]=model.visual.patch_vector(crops[begin:begin+8].cuda())[None]
   actual=model.teacher_logits(model.maps(images[identity]),coords,actual_vectors);error=float((actual-cached).abs().max());hidden=None;steps=[]
   for t in range(16):q,hidden,_=model.step_history(memory,coords[:,:t],hidden,vectors[:,t]);steps.append(q)
   step_error=float((cached-torch.stack(steps,1)).abs().max());modified=coords.clone();modified[:,8:]=1-modified[:,8:];pv=vectors.clone();pv[:,9:]=torch.randn_like(pv[:,9:]);future=model.teacher_logits(memory,modified,pv);future_error=float((cached[:,:9]-future[:,:9]).abs().max());norm=float(cached.logsumexp(-1).abs().max());proof.append(dict(phase=phase,image_id=identity,actual_cache_error=error,past_vector_error=float((vectors-actual_vectors).abs().max()),teacher_step_error=step_error,current_future_error=future_error,normalization_error=norm))
   if max(error,step_error)>1e-4 or future_error>1e-5 or norm>1e-5:raise ValueError('Medicalhead actual/causal/normalization proof failed')
if state_sha256(model.visual.whole.state_dict())!=whole_sha or state_sha256(model.visual.past.state_dict())!=past_sha or state_sha256(model.head_state())==head_before or any(p.grad is not None or p.requires_grad for p in model.visual.parameters()):raise ValueError('Frozen/head-update proof failed')
config=dict(medical_checkpoint=str(medical),medical_checkpoint_sha256=file_sha256(medical),past_checkpoint=str(past),past_checkpoint_sha256=file_sha256(past),source_code_sha256={str(ROOT/f):file_sha256(ROOT/f) for f in ['src/gaze_mammo_clip_1011.py','src/mammo_clip_visual_1011.py']});ck=dict(kind='mammo_clip_gaze_1011',state=model.head_state(),config=config,whole_state_sha256=whole_sha,past_state_sha256=past_sha);restored=restore_mammo_gaze(ck,'cuda:0')
with torch.no_grad():
 r=rows[64];coords=r['coords'][None].cuda();pv=r['vectors'][None].cuda();f=features[64].cuda();a=model.teacher_logits(model.maps_features([f]*3),coords,pv);b=restored.teacher_logits(restored.maps_features([f]*3),coords,pv);error=float((a-b).abs().max())
 if error>1e-6:raise ValueError('Medicalhead compact restore failed')
report=dict(records=proof,compact_restore_error=error,whole_frozen=True,past_frozen=True,head_updated=True,whole_sha256=whole_sha,past_sha256=past_sha,inventory=model.inventory(),seconds=time.monotonic()-start);out=ROOT/'runs/dual_target_1011/mammo_clip_public/gaze_head_optimizer_proof_ieee.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
