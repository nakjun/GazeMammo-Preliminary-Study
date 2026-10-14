"""TRAIN-only lesion proposal coverage and probabilistic duplicate auxiliary losses."""
import torch
from torch.nn import functional as F
from src.gaze_nll_adaptation_1011 import hard_nll

def objective(model,logits,coords,valid,boxes):
 p=logits.softmax(-1);imitation=hard_nll(logits,coords,valid);targets=[];have=[];xy=model.cell_xy
 for bs in boxes:
  target=torch.zeros(len(xy),dtype=torch.bool)
  for x0,y0,x1,y1 in bs:target|=(xy[:,0]>=max(0,x0-112/1500))&(xy[:,0]<=min(1,x1+112/1500))&(xy[:,1]>=max(0,y0-112/3000))&(xy[:,1]<=min(1,y1+112/3000))
  targets.append(target);have.append(bool(target.any()))
 targets=torch.stack(targets).to(p);have=torch.tensor(have,dtype=torch.bool)
 mass=(p[:,:4]*targets[:,None]).sum(-1).clamp(0,1-1e-6);coverage=1-(1-mass).prod(1);lesion=-coverage.clamp_min(1e-6).log();lesion=lesion[have].mean() if have.any() else imitation*0
 # 3x3 cells span <=89 physical pixels from a center; penalize expected near-duplicate proposals.
 blur=F.avg_pool2d(p.reshape(-1,1,48,24),3,stride=1,padding=1)*9;blur=blur.reshape_as(p);dup=p.new_zeros(());den=p.new_zeros(())
 for a in range(16):
  for b in range(a):
   mask=(valid[:,a]&valid[:,b]).to(p);dup+=((p[:,a]*blur[:,b]).sum(-1)*mask).sum();den+=mask.sum()
 duplicate=dup/den.clamp_min(1);loss=imitation+.2*lesion+.05*duplicate
 return loss,dict(imitation=imitation,coverage=lesion,duplicate=duplicate)
