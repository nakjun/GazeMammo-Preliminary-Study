"""Trainable global/local interaction, learned paired-view fusion, duplicate-aware."""
import math,torch
from torch import nn

class JointGazeFocus(nn.Module):
 def __init__(self,local=True,input_dim=1024,dim=64):
  super().__init__();self.local_enabled=local;self.global_encoder=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,dim));self.patch_encoder=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,dim),nn.GELU(),nn.Dropout(.3));self.query=nn.Linear(dim,dim,bias=False);self.key=nn.Linear(dim,dim,bias=False);self.coordinate_score=nn.Linear(2,1);self.interaction_score=nn.Linear(dim,1);self.classifier=nn.Sequential(nn.Linear(dim*4,64),nn.GELU(),nn.Dropout(.5),nn.Linear(64,1))
 def forward(self,global_views,patch_views,coordinates,patch_mask,partner_used):
  b=global_views.shape[0];g=self.global_encoder(global_views);own=g[:,0];other=torch.where(partner_used[:,None],g[:,1],own);context=.5*(own+other);difference=(own-other).abs();local=torch.zeros_like(context)
  if self.local_enabled:
   values=self.patch_encoder(patch_views);query=self.query(context)[:,None,None,:];score=(query*self.key(values)).sum(-1)/math.sqrt(values.shape[-1])+self.coordinate_score(coordinates).squeeze(-1)+self.interaction_score(values*context[:,None,None,:]).squeeze(-1)
   valid=patch_mask.clone();valid[:,1]&=partner_used[:,None];duplicates=((coordinates[:,:,:,None]-coordinates[:,:,None,:]).abs().amax(-1)<=1e-6)&valid[:,:,:,None]&valid[:,:,None,:];counts=duplicates.sum(-1).clamp_min(1);score=(score-counts.log()).flatten(1).masked_fill(~valid.flatten(1),-1e9);weights=score.softmax(1)*valid.flatten(1);weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-12);local=(values.flatten(1,2)*weights[...,None]).sum(1)
  return self.classifier(torch.cat([context,difference,local,context*local],1)).flatten()
