"""Actual64/73 source-exact prefix and originalpast reuse, childq separately."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.data import whole_tensor,patch_tensor
from src.crossview_dual_output_1011 import load_crossview_dual
from src.proposal_utility_1011 import state_sha256
@torch.no_grad()
def main():
 torch.set_num_threads(4);out=ROOT/'runs/dual_target_1011/crossview_unified_runtime';payload=json.loads((out/'config.json').read_text());model,_,_=load_crossview_dual(payload['config'],'cuda:1');before=state_sha256(model.cross.state_dict());proofs=[]
 for image_id in [64,73]:
  r=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{image_id}.pt',map_location='cpu',weights_only=False)
  with Image.open(ROOT/'data/image'/r['file_name']) as source:image=source.convert('L')
  memory=model.maps(image);whole=whole_tensor(image,(256,512))[None].cuda(1);f3,prefix=model.cross.visual.prefix_features(whole);full=model.cross.maps(whole);fe3=float((f3-memory['reuse_features']['feature3']).abs().max());pre=float((prefix-memory['reuse_features']['prefix5_23']).abs().max());assert fe3<=1e-5 and pre<=1e-5
  coords=r['coords'][None].cuda(1);cv=r['vectors'][None].cuda(1);reused=model.cross.teacher_logits(memory['cross'],coords,cv);fresh=model.cross.teacher_logits(full,coords,cv);qerror=float((reused-fresh).abs().max());assert qerror<=1e-4
  crops=torch.stack([patch_tensor(image,c.numpy()*image.size,224) for c in r['coords'][:8]]).cuda(1);parentvectors=model.visual.patch_vector(crops)[:,:1024];ownvectors=model.cross.visual.patch_vector(crops);verror=float((parentvectors-ownvectors).abs().max());assert verror<=1e-5
  proofs.append(dict(image_id=image_id,reused_feature3_error=fe3,reused_prefix5_23_error=pre,separate_cross_q_full_vs_reuse_error=qerror,reused_parent_control_past_vs_fresh_own_vector_error=verror))
 assert state_sha256(model.cross.state_dict())==before;result=dict(state='passed',combined_config_sha256=payload['sha256'],proofs=proofs,frozen_and_selected_state_unchanged=True,compute=payload['config']['compute'],no_auxiliary_runtime_input=True);(out/'reuse_proof64_73.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
