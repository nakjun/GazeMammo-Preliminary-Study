"""Bounded zero-initialized local correction to an immutable global MIL."""
import torch
from torch import nn
class LocalResidual(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Sequential(nn.Linear(1024,16),nn.GELU(),nn.Dropout(.5));self.net=nn.Sequential(nn.Linear(64,16),nn.GELU(),nn.Dropout(.5),nn.Linear(16,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def pool(self,features,mask):
  if self.training:
   orig=mask.clone();mask&=torch.rand(mask.shape)>.3;empty=~mask.any(1);mask[empty,orig[empty].long().argmax(1)]=orig[empty].any(1)
  z=self.project(features);mean=(z*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];maximum=z.masked_fill(~mask[...,None],-1e4).max(1).values;maximum=torch.where(mask.any(1)[:,None],maximum,torch.zeros_like(maximum));return torch.cat([mean,maximum],1)
 def forward(self,context,features,mask,pfeatures,pmask,use):
  z=self.pool(features,mask.clone());pz=self.pool(pfeatures,pmask.clone());z=torch.where(use[:,None],.5*(z+pz),z);return .5*torch.tanh(self.net(torch.cat([context,z],1)).flatten())
