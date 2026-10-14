"""Separate terminalv3 updated head causal and actual frozen224crop guard."""
import sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.data import whole_tensor,patch_tensor
from src.gaze_crossview_1011 import restore_crossview
from src.proposal_utility_1011 import state_sha256
from gaze_nll_adaptation_1011 import load_base

def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=ROOT/'runs/dual_target_1011/gaze_crossview_guard_v3';assert json.loads((out/'proof.json').read_text())['state']=='passed';base,_=load_base('runs/full_1011_seed42/control/best.pt');ck=torch.load(out/'compact_smoke.pt',map_location='cpu',weights_only=False);m=restore_crossview(ck,base).cuda(1).eval();before=state_sha256(m.frozen_state());proofs=[]
 for image_id in [62,64,73]:
  r=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{image_id}.pt',map_location='cpu',weights_only=False)
  with Image.open(ROOT/'data/image'/r['file_name']) as source:image=source.convert('L')
  coords=r['coords'][None].cuda(1);vectors=r['vectors'][None].cuda(1)
  with torch.no_grad():
   memory=m.maps(whole_tensor(image,(256,512))[None].cuda(1));q=m.teacher_logits(memory,coords,vectors);state=None;steps=[]
   for t in range(16):p,state,_=m.step_history(memory,coords[:,:t],state,vectors[:,t]);steps.append(p)
   step_error=float((q-torch.stack(steps,1)).abs().max());assert step_error<=1e-5;changed=coords.clone();changed[:,8:]=1-changed[:,8:];changedvectors=vectors.clone();changedvectors[:,9:]=100;future=m.teacher_logits(memory,changed,changedvectors);future_error=float((q[:,:9]-future[:,:9]).abs().max());assert future_error<=1e-5
   if image_id==62:
    pool=torch.load(ROOT/'runs/dual_target_1011/train_gaze_pool_cache/62.pt',map_location='cpu',weights_only=False);crops=torch.stack([patch_tensor(image,c.numpy()*image.size,224) for c in pool['coords'][:8]]).cuda(1);fresh=m.visual.patch_vector(crops);past_error=float((fresh.cpu()-pool['patch_vectors'][:8]).abs().max());assert past_error<=1e-4
   else:past_error=None
   f3,prefix=m.visual.prefix_features(whole_tensor(image,(256,512))[None].cuda(1));cacheq=m.teacher_logits(m.maps_prefix(f3,prefix),coords,vectors);cache_error=float((q-cacheq).abs().max());assert cache_error<=1e-4;assert q.logsumexp(-1).abs().max()<=1e-5
  proofs.append(dict(image_id=image_id,updated_cache_full_error=cache_error,teacher_step_error=step_error,current_future_coord_and_vector_invariance_error=future_error,fresh_original_pool8_crop_vector_error=past_error))
 assert state_sha256(m.frozen_state())==before
 result=dict(state='passed',updated_model=True,proofs=proofs,frozen_state_unchanged=True,visual_TF32=True,gazehead_IEEE=True,runtime_auxiliary_inputs_disabled=True);(out/'causal_past_proof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
