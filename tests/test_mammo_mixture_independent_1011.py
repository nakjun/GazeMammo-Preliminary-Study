import torch
from torch import nn
from src.mammo_clip_dual_output_1011 import MammoMixture
class Parent(nn.Module):
    def __init__(self):super().__init__();self.grid=(2,1);self.cell_xy=torch.tensor([[.25,.5],[.75,.5]]);self.vector_dims=[1024,1024,384];self.visual=nn.Identity();self.seen=[]
    def step_history(self,memory,prefix,hidden,vector):self.seen.append((hidden,vector));return torch.tensor([[.8,.2]]).log(),11 if hidden is None else hidden+1,{}
class Medical(nn.Module):
    def __init__(self):super().__init__();self.patch=nn.Linear(1024,2);self.seen=[]
    def step_history(self,memory,prefix,hidden,vector):self.seen.append((hidden,vector));return torch.tensor([[.2,.8]]).log(),31 if hidden is None else hidden+1,{}

def test_own_medical_state_first1024_reuse_and_normalized_weights():
    parent=Parent();medical=Medical();model=MammoMixture(parent,medical);memory=dict(parent={},medical={});prefix=torch.zeros(1,1,2);vector=torch.arange(2432).float()[None];q,state,_=model.step_history(memory,prefix,None,vector);assert torch.allclose(q.exp(),torch.tensor([[.65,.35]]),atol=1e-7);assert state==(11,31);q,other,_=model.step_history(memory,prefix,state,vector);assert other==(12,32) and parent.seen[-1][0]==11 and medical.seen[-1][0]==31;assert parent.seen[-1][1] is vector and torch.equal(medical.seen[-1][1],vector[:,:1024])
