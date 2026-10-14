"""Teacher exact sampled prefix and actual64/73 IEEE-head reference proof."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from gaze_nll_adaptation_1011 import load_base
from src.gaze_nll_adaptation_1011 import initialize
from src.gaze_path_augmentation_1011 import augmented_sample
from src.data import whole_tensor,patch_tensor
from src.proposal_utility_1011 import state_sha256

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.manual_seed(42)
base,_=load_base('runs/full_1011_seed42/control/best.pt');teacher=initialize(base).cuda().eval();teacher.requires_grad_(False);before=state_sha256(teacher.state_dict());proof=[]
with torch.no_grad():
 for identity in [64,73]:
  row=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{identity}.pt',weights_only=False);features=[f[None].cuda() for f in row['features']];coords=row['coords'][None].cuda();vectors=row['vectors'][None].cuda()
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):memory=teacher.maps_features(features);cached=teacher.teacher_logits(memory,coords,vectors)
  with Image.open(ROOT/'data/image'/row['file_name']) as original:
   image=original.convert('L');actual_features=teacher.visual(whole_tensor(image,teacher.input_size)[None].cuda());crops=torch.stack([patch_tensor(image,c.numpy()*image.size) for c in row['coords'][:-1]]);actual_vectors=torch.zeros_like(vectors)
   for start in range(0,15,8):actual_vectors[:,start+1:start+9]=teacher.visual.patch_vector(crops[start:start+8].cuda())[None]
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):actual=teacher.teacher_logits(teacher.maps_features(actual_features),coords,actual_vectors)
  error=float((cached-actual).abs().max());proof.append(dict(image_id=identity,actual_teacher_logp_error=error))
  if error>1e-4:raise ValueError('Actual teacher reference parity failed')
 for identity in [62,65]:
  controlpool=torch.load(ROOT/f'runs/dual_target_1011/train_gaze_pool_cache/{identity}.pt',weights_only=False);dinopool=torch.load(ROOT/f'runs/dual_target_1011/dino/pool_cache/{identity}.pt',weights_only=False);cs=augmented_sample(controlpool,identity,1);ds=augmented_sample(dinopool,identity,1)
  if cs['pool_indices']!=ds['pool_indices'] or cs['raw_indices']!=ds['raw_indices'] or not torch.equal(cs['coords'],ds['coords']):raise ValueError('Sampled student/teacher mapping mismatch')
  row=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{identity}.pt',weights_only=False);coords=cs['coords'][None].cuda();vectors=cs['vectors'][None].cuda();features=[f[None].cuda() for f in row['features']]
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):
   memory=teacher.maps_features(features);raw=teacher.teacher_logits(memory,coords,vectors);changed=coords.clone();changed[:,8:]=.9;modified=vectors.clone();modified[:,9:]=torch.randn_like(modified[:,9:]);future=teacher.teacher_logits(memory,changed,modified)
  if not torch.allclose(raw.exp().sum(-1),torch.ones_like(raw[:,:,0]),atol=1e-5) or not torch.equal(raw[:,:9],future[:,:9]):raise ValueError('Teacher normalized causal proof failed')
  proof.append(dict(image_id=identity,sampled_mapping=True,future_error=0.,normalized=True,detached=not raw.requires_grad))
if before!=state_sha256(teacher.state_dict()) or any(p.grad is not None or p.requires_grad for p in teacher.parameters()):raise ValueError('Frozen teacher changed')
out=ROOT/'runs/dual_target_1011/dino/teacherkl_proof_v2.json';out.write_text(json.dumps(dict(records=proof,teacher_state_sha256=before,unchanged=True,policy='visualTF32True/headTF32False/matmulFalse'),indent=2));print(json.dumps(proof))
