"""New25% frozen mammographic whole expert; originalcontrol past vectors reused."""
import json,math
import torch
from torch import nn
from .dual_output_1011 import ROOT,config_sha256
from .crossview_dual_output_1011 import load_crossview_dual
from .gaze_mammo_clip_1011 import restore_mammo_gaze
from .proposal_utility_1011 import state_sha256,file_sha256

class MammoMixture(nn.Module):
 def __init__(self,parent,medical):
  super().__init__();self.parent=parent;self.medical=medical;self.grid=parent.grid;self.cell_xy=parent.cell_xy;self.foveal=True;self.visual=parent.visual;self.vector_dims=parent.vector_dims
  if medical.patch.in_features!=self.vector_dims[0] or self.vector_dims[0]!=1024:raise ValueError('Originalcontrol medicalpast reuse dimension')
 @torch.no_grad()
 def maps(self,image):
  parent=self.parent.maps(image);medical=self.medical.maps(image)
  return dict(parent=parent,medical=medical,prior=parent['prior'],embedding=parent['embedding'])
 @torch.no_grad()
 def step_history(self,memory,prefix,hidden=None,patch_vector=None):
  ps,ms=(None,None) if hidden is None else hidden
  if patch_vector is not None and patch_vector.shape[-1]!=sum(self.vector_dims):raise ValueError('Parentpastvector dimension changed')
  p,ps,diagnostic=self.parent.step_history(memory['parent'],prefix,ps,patch_vector);past=patch_vector[...,:1024] if patch_vector is not None else None;m,ms,mdiagnostic=self.medical.step_history(memory['medical'],prefix,ms,past);m=m.log_softmax(-1);q=torch.logaddexp(p+math.log(.75),m+math.log(.25));q=q-q.logsumexp(-1,keepdim=True)
  if not torch.isfinite(q).all():raise ValueError('Medicalmixture finite failure')
  return q,(ps,ms),dict(parent=diagnostic,medical=mdiagnostic,probability_role='RAW.75currentCrossView+.25MammoUPMCwhole/controlpast; noR/geometrybias')
 def teacher_logits(self,memory,coords,patch_vectors=None):
  state=None;outputs=[]
  for t in range(coords.shape[1]):q,state,_=self.step_history(memory,coords[:,:t],state,patch_vectors[:,t] if patch_vectors is not None else None);outputs.append(q)
  return torch.stack(outputs,1)

def load_mammo_clip_dual(config,device='cpu'):
 if config['kind']!='mammo_clip_dual_output_1011':raise ValueError('Wrong medicalruntime kind')
 for file,sha in config['source_code_sha256'].items():
  if file_sha256(ROOT/file)!=sha:raise ValueError('Medicalruntime pinnedsource changed')
 payload=config['parent_config']
 if payload['sha256']!=config_sha256(payload['config']):raise ValueError('CurrentCrossView nestedconfig SHA mismatch')
 parent,guided,head=load_crossview_dual(payload['config'],device);dependency=config['probability']['parents'][-1]
 if file_sha256(dependency['path'])!=dependency['sha256']:raise ValueError('Medicalgaze best changed')
 ck=torch.load(dependency['path'],map_location='cpu',weights_only=False);medical=restore_mammo_gaze(ck,device);medical.requires_grad_(False)
 if state_sha256(medical.visual.past.state_dict())!=state_sha256(parent.cross.visual.past.state_dict()):raise ValueError('Exact originalcontrol medicalpast identity required')
 result=MammoMixture(parent,medical).to(device).eval();result.requires_grad_(False);return result,guided,head

def build_mammo_clip_config():
 parent=json.loads((ROOT/'runs/dual_target_1011/crossview_unified_runtime/config.json').read_text());trial=ROOT/'runs/dual_target_1011/mammo_clip_public/gaze_seed42';checkpoint=trial/'best.pt';raw=trial/'best_raw_cache.pt';proof=json.loads((trial/'independent_terminal_review.json').read_text());dependencies=list(parent['config']['probability']['parents']);dependencies.append(dict(role='MammoUPMCwhole_controlpast',path=str(checkpoint.resolve()),sha256=file_sha256(checkpoint)));source=['src/mammo_clip_dual_output_1011.py','scripts/mammo_clip_dual_output_1011.py','src/gaze_mammo_clip_1011.py','src/mammo_clip_visual_1011.py']
 return dict(kind='mammo_clip_dual_output_1011',parent_config=parent,source_code_sha256={f:file_sha256(ROOT/f) for f in source},reference_caches=parent['config']['reference_caches']+[dict(raw_cache=str(raw.resolve()),raw_cache_sha256=file_sha256(raw))],probability=dict(parents=dependencies,weights=[.75,.25],temperature=1,geometry_coefficient=0,recipe='RAW.75currentCrossView+.25MammoUPMCwhole/controlpast',numerics=parent['config']['probability']['numerics']),roi=parent['config']['roi'],compute=dict(independent_whole_image_encoders=6,preceding_crop_encoders=3,additional_crossview_whole_tail_evaluations=1,medical_whole_input=[912,1520],medical_encoder_pretrain='UPMConly image-text',original_control_past_vector_reused=True,no_new_medical_past_encoder_evaluation=True),claims=dict(validation='Exploratory fixedval93; NLL5.3 notmet',forced_path_nll_reported=False,independent_component_review_sha256=file_sha256(trial/'independent_terminal_review.json')))
