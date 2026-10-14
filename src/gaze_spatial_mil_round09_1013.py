"""Low-capacity location-preserving spatial MIL with gated local evidence."""
import torch
from torch import nn
class SpatialGazeMIL(nn.Module):
 def __init__(self,local=True):
  super().__init__();self.local=local;self.project=nn.Sequential(nn.Linear(1024,32),nn.GELU(),nn.Dropout(.4));self.position=nn.Linear(2,32,bias=False);self.attention=nn.Sequential(nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1));self.context=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(.5));self.global_head=nn.Linear(32,1);self.query=nn.Linear(32,32,bias=False);self.local_head=nn.Linear(32,1);self.gate=nn.Linear(64,1);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias);yy,xx=torch.meshgrid((torch.arange(32)+.5)/32,(torch.arange(16)+.5)/16,indexing='ij');self.register_buffer('xy',torch.stack([xx,yy],-1).reshape(512,2),persistent=False)
 def encode(self,g,l,coords,mask):
  tokens=self.project(g)+.1*self.position(self.xy*2-1)[None];weights=self.attention(tokens).squeeze(-1).softmax(1);context=self.context(torch.cat([(tokens*weights[...,None]).sum(1),tokens.mean(1)],1));local=context.new_zeros(context.shape)
  if self.local:
   if self.training:
    original=mask.clone();mask=mask&(torch.rand(mask.shape)>.3);empty=~mask.any(1);mask[empty,original[empty].long().argmax(1)]=original[empty].any(1)
   z=self.project(l)+.1*self.position(coords*2-1);score=(z*self.query(context)[:,None]).sum(-1)/(32**.5);w=score.masked_fill(~mask,-1e4).softmax(1)*mask;w=w/w.sum(1,keepdim=True).clamp_min(1e-8);local=(z*w[...,None]).sum(1)
  return context,local
 def forward(self,g,l,c,m,pg,pl,pc,pm,use):
  a,b=self.encode(g,l,c,m);pa,pb=self.encode(pg,pl,pc,pm);a=torch.where(use[:,None],.5*(a+pa),a);b=torch.where(use[:,None],.5*(b+pb),b);logit=self.global_head(a).flatten()
  if self.local:logit=logit+self.gate(torch.cat([a,b],1)).sigmoid().flatten()*self.local_head(b).flatten()
  return logit
