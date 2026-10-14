"""Causal gaze with whole-only stride16 clinical cross-view auxiliary training."""
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .gaze_nll_adaptation_1011 import GazeNLLAdaptation1011
from .data import seed_for
from .proposal_utility_1011 import state_sha256
BLOCK_PREFIXES=tuple(f'visual.whole.features.5.{i}.' for i in [24,25,26])
class SplitCrossViewVisual(nn.Module):
 channels=[256,512,1024]
 def __init__(self,original):
  super().__init__();self.whole=copy.deepcopy(original);self.past=copy.deepcopy(original);self.requires_grad_(False)
  for i in [24,25,26]:self.whole.features[5][i].requires_grad_(True)
  self.eval()
 def prefix_features(self,image):
  with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):
   x=image;feature3=None
   for i in range(5):
    x=self.whole.features[i](x)
    if i==3:feature3=x
   for block in self.whole.features[5][:24]:x=block(x)
  return feature3,x
 def tail_features(self,prefix):
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):return self._tail_features(prefix)
 def _tail_features(self,prefix):
  x=prefix.contiguous(memory_format=torch.channels_last)
  for block in self.whole.features[5][24:]:x=block(x)
  feature5=x
  # Frozen parameters, live input gradients; no no_grad/detach here.
  feature7=self.whole.features[7](self.whole.features[6](x))
  return feature5,feature7
 def forward(self,image):
  f3,prefix=self.prefix_features(image);f5,f7=self.tail_features(prefix);return f3,f5,f7
 def patch_vector(self,image):
  with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):return self.past.patch_vector(image)

class CrossViewGaze(GazeNLLAdaptation1011):
 def __init__(self,base):
  config={k:v for k,v in base.model_config.items() if k not in ('ablation','rho')};super().__init__(**config,pretrained=False);self.load_state_dict(base.state_dict(),strict=True);self.visual=SplitCrossViewVisual(base.visual)
  with torch.random.fork_rng(devices=[]):self.aux_projector=nn.Linear(512,1024)
  self.visual.eval()
 def maps_features(self,features):
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):return super().maps_features(features)
 def teacher_logits(self,*args,**kwargs):
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):return super().teacher_logits(*args,**kwargs)
 def step_history(self,*args,**kwargs):
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):return super().step_history(*args,**kwargs)
 def maps_prefix(self,feature3,prefix,return_aux=False):
  f5,f7=self.visual.tail_features(prefix);memory=self.maps_features([feature3,f5,f7]);return (memory,f5) if return_aux else memory
 def aux_queries(self,feature5,coords):
  sampled=F.grid_sample(feature5,coords[:,None]*2-1,mode='bilinear',padding_mode='border',align_corners=False)[:, :,0].transpose(1,2)
  return F.normalize(self.aux_projector(sampled),dim=-1)
 def frozen_state(self):
  return {n:v for n,v in self.visual.state_dict().items() if not n.startswith(tuple(f'whole.features.5.{i}.' for i in [24,25,26]))}
 def compact_state(self):return {n:v for n,v in self.state_dict().items() if not n.startswith('visual.') or n.startswith(BLOCK_PREFIXES)}

def physical_rectangles(coords,image_size):
 # Exactly src.data.patch_tensor: numpy bankers rounding, full224 PIL rectangle.
 centers=np.asarray(coords,dtype=np.float64)*np.asarray(image_size,dtype=np.float64);rounded=np.round(centers).astype(np.int64);return np.concatenate([rounded-112,rounded+112],1),centers

def negative_indices(pool_coords,image_size,positive,epoch,image_id,step):
 rects,centers=physical_rectangles(pool_coords,image_size);r=rects[positive];inter=np.minimum(rects[:,2:],r[2:])-np.maximum(rects[:,:2],r[:2]);overlap=np.maximum(inter,0).prod(1);distance=np.linalg.norm(centers-centers[positive],axis=1);eligible=np.flatnonzero((overlap==0)&(distance>=448));eligible=eligible[eligible!=positive]
 if len(eligible)>16:eligible=np.sort(np.random.default_rng(seed_for(42,int(image_id)+epoch*100000+step*10000000)).choice(eligible,16,replace=False))
 return eligible.tolist()

def contrastive_aux(queries,group,pools,epoch):
 losses=[];anchors=0;eligible=0;negative_total=0
 for b,r in enumerate(group):
  pool=pools[r['image_id']];terms=[]
  for step,positive in enumerate(r['selected_pool_indices']):
   anchors+=1;neg=negative_indices(pool['coords'].cpu().numpy(),r['image_size'],positive,epoch,r['image_id'],step)
   if not neg:continue
   eligible+=1;negative_total+=len(neg);target=F.normalize(pool['patch_vectors'][[positive,*neg]].to(queries.device).detach(),dim=-1);scores=(target@queries[b,step])/.2;terms.append(-scores.log_softmax(0)[0])
  losses.append(torch.stack(terms).mean() if terms else queries[b].sum()*0)
 return torch.stack(losses).mean(),dict(anchors=anchors,eligible_anchors=eligible,skipped_anchors=anchors-eligible,negative_total=negative_total)

def restore_crossview(checkpoint,base):
 if checkpoint['kind']!='gaze_crossview_1011':raise ValueError('Wrongcrossviewcheckpoint')
 model=CrossViewGaze(base)
 if state_sha256(model.frozen_state())!=checkpoint['frozen_state_sha256'] or set(checkpoint['state'])!=set(model.compact_state()):raise ValueError('Crossviewfrozen/state guard')
 model.load_state_dict(checkpoint['state'],strict=False)
 if state_sha256(model.frozen_state())!=checkpoint['frozen_state_sha256']:raise ValueError('Frozenbranch changed')
 return model.eval()
