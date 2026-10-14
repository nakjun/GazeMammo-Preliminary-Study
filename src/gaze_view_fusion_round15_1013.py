"""CC/MLO ordered low-capacity unbounded correction, with missing-view masks."""
import torch
from torch import nn
class ViewFusion(nn.Module):
 def __init__(self):
  super().__init__();self.project=nn.Sequential(nn.Linear(1024,16),nn.GELU(),nn.Dropout(.5));self.net=nn.Sequential(nn.Linear(194,16),nn.GELU(),nn.Dropout(.5),nn.Linear(16,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def pool(self,v,mask):
  mask=mask.clone()
  if self.training:
   original=mask.clone();mask&=torch.rand(mask.shape)>.3;empty=~mask.any(1);mask[empty,original[empty].long().argmax(1)]=original[empty].any(1)
  z=self.project(v);mean=(z*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];maximum=z.masked_fill(~mask[:,:,None],-1e9).max(1).values;maximum=torch.where(mask.any(1)[:,None],maximum,torch.zeros_like(maximum));return torch.cat([mean,maximum],1)
 def compose(self,a,b,la,lb,iscc,use):
  b=torch.where(use[:,None],b,a);lb=torch.where(use[:,None],lb,la);cc=torch.where(iscc[:,None],a,b);mlo=torch.where(iscc[:,None],b,a);lcc=torch.where(iscc[:,None],la,lb);lmlo=torch.where(iscc[:,None],lb,la);availability=torch.stack([iscc|use,~iscc|use],1).float();return torch.cat([cc,mlo,(cc-mlo).abs(),lcc,lmlo,(lcc-lmlo).abs(),availability],1)
 def forward(self,a,b,v,pv,mask,pmask,iscc,use):return self.net(self.compose(a,b,self.pool(v,mask),self.pool(pv,pmask),iscc,use)).flatten()
