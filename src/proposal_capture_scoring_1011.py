"""Image-only capture-aware proposal ranking at continuous action centers."""
import torch
from torch.nn import functional as F


def sample_capture_field(field,coords,grid=(24,48)):
    gx,gy=grid
    if field.numel()!=gx*gy or coords.ndim!=2 or coords.shape[-1]!=2 or not torch.isfinite(coords).all() or (coords<0).any() or (coords>1).any():raise ValueError('Invalid capture sampling field/coordinates')
    field=field.reshape(1,1,gy,gx);points=(2*coords.to(field)-1).reshape(1,1,-1,2)
    return F.grid_sample(field,points,mode='bilinear',padding_mode='border',align_corners=False).reshape(-1)


def geometric_capture_scores(original_relevance,capture_scores,alpha=.75,epsilon=1e-6,capture_valid=True):
    if not 0<=alpha<=1 or not 0<epsilon<1:raise ValueError('Invalid score mixture')
    r=original_relevance
    if not torch.isfinite(r).all() or (r<0).any() or (r>1).any():raise ValueError('Invalid original proposal relevance')
    c=capture_scores.to(r).reshape_as(r);valid=bool(capture_valid and torch.isfinite(c).all() and (c>=0).all() and (c<=1).all())
    if alpha==1 or not valid:return r.clone(),valid
    score=(alpha*r.clamp_min(epsilon).log()+(1-alpha)*c.clamp_min(epsilon).log()).exp().clamp(0,1)
    return score,valid
