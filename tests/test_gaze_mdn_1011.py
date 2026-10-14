import torch
from src.gaze_mdn_1011 import GazeMDN,normal_interval

def test_mass_gradient():
 torch.manual_seed(42);m=GazeMDN();q=m.step_history(torch.randn(2,1024),torch.zeros(2,0,2),torch.tensor([[2000,4000],[1000,2000]]));assert torch.isfinite(q).all();assert torch.allclose(q.exp().sum(-1),torch.ones(2),atol=1e-6);(-q[:,500].mean()).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
def test_causality_teacher_step():
 m=GazeMDN().eval();image=torch.randn(1,1024);coords=torch.rand(1,16,2);size=torch.tensor([[2000,4000]]);changed=coords.clone();changed[:,8:]=.9
 a=m.teacher_logits(image,coords,size);b=m.teacher_logits(image,changed,size);assert torch.equal(a[:,:9],b[:,:9]);assert torch.equal(a[:,8],m.step_history(image,coords[:,:8],size))
def test_integral():
 a=torch.tensor([-100.,-2.,0.,2.],dtype=torch.float64);b=torch.tensor([-99.,0.,2.,100.],dtype=torch.float64);q=normal_interval(a,b);assert torch.isfinite(q).all() and (q>=0).all();assert abs(float(q[1])-.477249868)<1e-8;assert abs(float(q[2])-.477249868)<1e-8
