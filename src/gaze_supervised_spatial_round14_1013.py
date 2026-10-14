"""Low-capacity spatial MIL with training-only attention alignment and local mean/max evidence."""
import torch
from torch import nn
class GazeSupervisedMIL(nn.Module):
 def __init__(self,local=False):
  super().__init__();self.local=local;self.project=nn.Sequential(nn.Linear(1024,32),nn.GELU(),nn.Dropout(.4));self.position=nn.Linear(2,32,bias=False);self.attention=nn.Sequential(nn.Linear(32,16),nn.Tanh(),nn.Linear(16,1));self.context=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(.5));self.global_head=nn.Linear(32,1);self.local_project=nn.Sequential(nn.Linear(1024,32),nn.GELU(),nn.Dropout(.5));self.local_head=nn.Linear(64,1);self.gate=nn.Linear(96,1);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias);yy,xx=torch.meshgrid((torch.arange(32)+.5)/32,(torch.arange(16)+.5)/16,indexing='ij');self.register_buffer('xy',torch.stack([xx,yy],-1).reshape(512,2),persistent=False)
 def encode(self,g,l,mask):
  z=self.project(g)+.1*self.position(self.xy*2-1)[None];weights=self.attention(z).squeeze(-1).softmax(1);context=self.context(torch.cat([(z*weights[:,:,None]).sum(1),z.mean(1)],1));local=context.new_zeros(len(g),64)
  if self.local:
   if self.training:
    original=mask.clone();mask=mask&(torch.rand(mask.shape)>.3);empty=~mask.any(1);mask[empty,original[empty].long().argmax(1)]=original[empty].any(1)
   v=self.local_project(l);mean=(v*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];maximum=v.masked_fill(~mask[:,:,None],-1e9).max(1).values;maximum=torch.where(mask.any(1)[:,None],maximum,torch.zeros_like(maximum));local=torch.cat([mean,maximum],1)
  return context,local,weights
 def forward(self,g,l,m,pg,pl,pm,use):
  a,b,w=self.encode(g,l,m);pa,pb,pw=self.encode(pg,pl,pm);a=torch.where(use[:,None],.5*(a+pa),a);b=torch.where(use[:,None],.5*(b+pb),b);logit=self.global_head(a).flatten()
  if self.local:logit+=self.gate(torch.cat([a,b],1)).sigmoid().flatten()*self.local_head(b).flatten()
  return logit,w,pw
