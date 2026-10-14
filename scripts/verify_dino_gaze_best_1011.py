"""Actual updated DINO best checkpoint proof under the training backend."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.data import whole_tensor,patch_tensor
from src.gaze_dino_1011 import restore_dino

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
trial=ROOT/'runs/dual_target_1011/dino/gaze_seed42';checkpoint=torch.load(trial/'best.pt',weights_only=False);model=restore_dino(checkpoint,'cuda:0');proof=[]
with torch.no_grad():
 for identity in [64,73]:
  r=torch.load(ROOT/f'runs/dual_target_1011/dino/native_cache/{identity}.pt',weights_only=False);f=r['feature'][None].cuda();coords=r['coords'][None].cuda();pv=r['vectors'][None].cuda();memory=model.maps_features([f]*3);cached=model.teacher_logits(memory,coords,pv)
  with Image.open(ROOT/'data/image'/r['file_name']) as image:
   image=image.convert('L');whole=whole_tensor(image,model.input_size)[None].cuda();crops=torch.stack([patch_tensor(image,c.numpy()*image.size) for c in r['coords'][:-1]]);vectors=torch.zeros_like(pv)
   for start in range(0,15,8):vectors[:,start+1:start+9]=model.visual.patch_vector(crops[start:start+8].cuda())[None]
   live=model.teacher_logits(model.maps(whole),coords,vectors)
  hidden=None;steps=[]
  for t in range(16):q,hidden,_=model.step_history(memory,coords[:,:t],hidden,pv[:,t]);steps.append(q)
  error=float((cached-live).abs().max());step=float((cached-torch.stack(steps,1)).abs().max());vector=float((pv-vectors).abs().max());proof.append(dict(image_id=identity,actual_updated_logp_error=error,teacher_step_error=step,fresh224vector_error=vector))
  if max(error,step)>1e-4:raise ValueError('Updated actual proof failed')
(trial/'actual_best_restore_proof.json').write_text(json.dumps(dict(epoch=checkpoint['epoch'],policy=dict(matmulTF32=False,cudnnTF32=True),records=proof,tolerance=1e-4),indent=2));print(json.dumps(proof))
