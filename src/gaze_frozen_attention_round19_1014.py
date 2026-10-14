"""Separate dim8 local attention conditioned on immutable global context."""
import torch
from torch import nn
class FrozenAttentionResidual(nn.Module):
 def __init__(self,scale):
  super().__init__();self.scale=scale;self.project=nn.Sequential(nn.Linear(1024,8),nn.GELU(),nn.Dropout(.7));self.query=nn.Linear(32,8,bias=False);self.dropout=nn.Dropout(.5);self.output=nn.Linear(8,1);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
 def pool(self,ctx,v,mask):
  mask=mask.clone()
  if self.training:
   original=mask.clone();mask&=torch.rand(mask.shape)>.5;empty=~mask.any(1);mask[empty,original[empty].long().argmax(1)]=original[empty].any(1)
  z=self.project(v);score=(z*self.query(ctx)[:,None]).sum(-1)/(8**.5);weights=score.masked_fill(~mask,-1e4).softmax(1)*mask;weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12);return (z*weights[:,:,None]).sum(1)
 def forward(self,ctx,pctx,v,pv,mask,pmask,use):
  z=self.pool(ctx,v,mask);pz=self.pool(pctx,pv,pmask);z=torch.where(use[:,None],.5*(z+pz),z);return self.scale*self.output(self.dropout(z)).flatten()
