"""New25% cross-view expert; exact frozen control prefix/past-vector reuse."""
import json,math,sys
import torch
from torch import nn
from .dual_output_1011 import ROOT,config_sha256
from .dino_mdn_dual_output_1011 import load_dino_mdn_dual
from .gaze_crossview_1011 import restore_crossview
from .proposal_utility_1011 import file_sha256,state_sha256
class CrossViewMixture(nn.Module):
 def __init__(self,parent,cross):
  super().__init__();self.parent=parent;self.cross=cross;self.grid=parent.grid;self.cell_xy=parent.cell_xy;self.foveal=True;self.visual=parent.visual;self.vector_dims=parent.vector_dims
  if self.cross.patch.in_features!=self.vector_dims[0]:raise ValueError('Originalcontrol vectorreuse dimension')
 @torch.no_grad()
 def maps(self,image):
  control=self.parent.parent.parent.baseline.parents[0];captures={};handles=[control.visual.features[3].register_forward_hook(lambda m,a,o:captures.__setitem__('feature3',o.detach())),control.visual.features[5][23].register_forward_hook(lambda m,a,o:captures.__setitem__('prefix5_23',o.detach()))]
  try:memory=self.parent.maps(image)
  finally:
   for handle in handles:handle.remove()
  if set(captures)!={'feature3','prefix5_23'} or captures['prefix5_23'].shape[1:]!=(512,32,16):raise ValueError('Frozencontrolprefix capture')
  cross=self.cross.maps_prefix(captures['feature3'],captures['prefix5_23'])
  return dict(parent=memory,cross=cross,prior=memory['prior'],embedding=memory['embedding'],reuse_features=captures)
 @torch.no_grad()
 def step_history(self,memory,prefix,hidden=None,patch_vector=None):
  ps,cs=(None,None) if hidden is None else hidden
  if patch_vector is not None and patch_vector.shape[-1]!=sum(self.vector_dims):raise ValueError('Parentcropvector dimension')
  p,ps,pdiag=self.parent.step_history(memory['parent'],prefix,ps,patch_vector);cv=patch_vector[...,:self.cross.patch.in_features] if patch_vector is not None else None;c,cs,cdiag=self.cross.step_history(memory['cross'],prefix,cs,cv);c=c.log_softmax(-1);q=torch.logaddexp(p+math.log(.75),c+math.log(.25));q=q-q.logsumexp(-1,keepdim=True)
  if not torch.isfinite(q).all():raise ValueError('Crossviewmixture finite')
  return q,(ps,cs),dict(parent=pdiag,cross=cdiag,probability_role='RAW.75preservedDINO_MDN+.25crossview; noR/geometrybias')
 def teacher_logits(self,memory,coords,patch_vectors=None):
  state=None;outputs=[]
  for t in range(coords.shape[1]):q,state,_=self.step_history(memory,coords[:,:t],state,patch_vectors[:,t] if patch_vectors is not None else None);outputs.append(q)
  return torch.stack(outputs,1)

def prefix_parameters(state):return {n:v for n,v in state.items() if n.startswith(tuple(f'features.{i}.' for i in range(5))) or (n.startswith('features.5.') and int(n.split('.')[2])<24)}
def load_crossview_dual(config,device='cpu'):
 if config['kind']!='crossview_dual_output_1011':raise ValueError('Wrongcrossviewruntime')
 for path,h in config['source_code_sha256'].items():
  if file_sha256(ROOT/path)!=h:raise ValueError('Pinnednewruntime source')
 payload=config['parent_config']
 if config_sha256(payload['config'])!=payload['sha256']:raise ValueError('NestedparentSHA')
 parent,guided,head=load_dino_mdn_dual(payload['config'],device);dependency=config['probability']['parents'][-1]
 if file_sha256(dependency['path'])!=dependency['sha256']:raise ValueError('CrossviewbestSHA')
 ck=torch.load(dependency['path'],map_location='cpu',weights_only=False);sys.path.insert(0,str(ROOT/'scripts'));from gaze_nll_adaptation_1011 import load_base
 if file_sha256(ck['config']['source_checkpoint'])!=ck['config']['source_checkpoint_sha256']:raise ValueError('Crossview originalsourceSHA')
 base,_=load_base(ck['config']['source_checkpoint']);cross=restore_crossview(ck,base).to(device).eval();cross.requires_grad_(False);control=parent.parent.parent.baseline.parents[0]
 if state_sha256(cross.visual.past.state_dict())!=state_sha256(control.visual.state_dict()) or state_sha256(prefix_parameters(cross.visual.whole.state_dict()))!=state_sha256(prefix_parameters(control.visual.state_dict())):raise ValueError('Reusefrozenprefix/past identity proof')
 result=CrossViewMixture(parent,cross).to(device).eval();result.requires_grad_(False);return result,guided,head

def build_crossview_config():
 parent=json.loads((ROOT/'runs/dual_target_1011/dino_mdn_unified_runtime/config.json').read_text());trial=ROOT/'runs/dual_target_1011/gaze_crossview_seed42';d=trial/'best.pt';cache=trial/'fixed_export/best_raw_cache.pt';deps=list(parent['config']['probability']['parents']);deps.append(dict(role='crossview',path=str(d),sha256=file_sha256(d)));source=['src/crossview_dual_output_1011.py','scripts/crossview_dual_output_1011.py','src/gaze_crossview_1011.py']
 return dict(kind='crossview_dual_output_1011',parent_config=parent,source_code_sha256={p:file_sha256(ROOT/p) for p in source},reference_caches=parent['config']['reference_caches']+[dict(raw_cache=str(cache),raw_cache_sha256=file_sha256(cache))],probability=dict(parents=deps,weights=[.75,.25],temperature=1,geometry_coefficient=0,recipe='RAW.75preservedDINO_MDN+.25crossview',numerics=parent['config']['probability']['numerics']),roi=parent['config']['roi'],compute=dict(independent_whole_image_encoders=5,preceding_crop_encoders=3,additional_whole_tail_evaluations=1,whole_control_prefix_before_block24_reused=True,original_control_past_vector_reused=True,crossview_gaze_state_independent=True,original_ROI_encoder_independent=True),claims=dict(validation='Exploratory fixedval93; NLL5.3notmet',forced_path_nll_reported=False))
