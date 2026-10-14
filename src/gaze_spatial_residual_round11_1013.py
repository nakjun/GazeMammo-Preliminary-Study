"""Zero-init bounded correction from a gaze-selected global-context summary."""
import torch
from torch import nn
class SpatialResidual(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(64,16),nn.GELU(),nn.Dropout(.5),nn.Linear(16,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,context,summary,partner_summary,use):
  z=torch.where(use[:,None],.5*(summary+partner_summary),summary);return .5*torch.tanh(self.net(torch.cat([context,z],1)).flatten())
