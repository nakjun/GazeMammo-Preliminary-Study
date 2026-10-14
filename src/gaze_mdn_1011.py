"""Causal eight-component density with integrated fixed-grid cell mass."""
import math
import torch
from torch import nn
from torch.nn import functional as F

def normal_interval(lower,upper):
    # Upper-tail subtraction avoids cancellation when both bounds are positive.
    return torch.where(lower>0,torch.special.ndtr(-lower)-torch.special.ndtr(-upper),torch.special.ndtr(upper)-torch.special.ndtr(lower)).clamp_min(0)

class GazeMDN(nn.Module):
    def __init__(self):
        super().__init__();self.image=nn.Linear(1024,64);self.head=nn.Sequential(nn.Linear(77,128),nn.GELU(),nn.Linear(128,128),nn.GELU(),nn.Linear(128,40))
    def step_history(self,image,prefix,image_size):
        batch=image.shape[0];t=prefix.shape[1]
        if t>=16:raise ValueError('Step outside legacy16')
        history=image.new_zeros(batch,4,2);mask=image.new_zeros(batch,4)
        n=min(t,4)
        if n:
            observed=prefix[:,-n:].flip(1);valid=torch.isfinite(observed).all(-1);history[:,:n]=torch.nan_to_num(observed);mask[:,:n]=valid
        inputs=torch.cat([self.image(F.normalize(image,dim=-1)),history.flatten(1),mask,image.new_full((batch,1),t/15)],-1)
        params=self.head(inputs).reshape(batch,8,5);weights=params[:,:,0].double().softmax(-1);means=params[:,:,1:3].double().sigmoid();minimum=1/image_size.double();scales=minimum[:,None]+(1-minimum[:,None])*params[:,:,3:5].double().sigmoid()
        masses=[]
        for axis,bins in [(0,24),(1,48)]:
            edges=torch.linspace(0,1,bins+1,device=image.device,dtype=torch.float64);z=(edges[None,None]-means[:,:,axis,None])/scales[:,:,axis,None];masses.append(normal_interval(z[:,:,:-1],z[:,:,1:]))
        joint=masses[1][:,:,:,None]*masses[0][:,:,None,:];q=(weights[:,:,None,None]*joint).sum(1).flatten(1)+1e-12;q=q/q.sum(-1,keepdim=True)
        return q.log().float()
    def teacher_logits(self,image,coords,image_size):
        return torch.stack([self.step_history(image,coords[:,:t],image_size) for t in range(coords.shape[1])],1)
