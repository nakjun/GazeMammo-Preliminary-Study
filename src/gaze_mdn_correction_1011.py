"""Signed bounded causal MDN product correction of a frozen parent probability."""
import torch
from torch import nn
from src.gaze_mdn_1011 import GazeMDN

class GazeMDNCorrection(nn.Module):
    def __init__(self):
        super().__init__();self.mdn=GazeMDN();self.alpha=nn.Parameter(torch.zeros(16))
    def step_history(self,image,prefix,image_size,parent_logp):
        t=prefix.shape[1];raw=parent_logp+self.alpha[t].tanh()*self.mdn.step_history(image,prefix,image_size);return raw-raw.logsumexp(-1,keepdim=True)
    def teacher_logits(self,image,coords,image_size,parent_logp):
        return torch.stack([self.step_history(image,coords[:,:t],image_size,parent_logp[:,t]) for t in range(coords.shape[1])],1)
