import torch
from src.gaze_mdn_correction_1011 import GazeMDNCorrection

def test_identity_and_gradient():
 m=GazeMDNCorrection();im=torch.randn(2,1024);coords=torch.rand(2,16,2);size=torch.tensor([[2000,4000],[1000,2000]]);parent=torch.randn(2,16,1152).log_softmax(-1);q=m.teacher_logits(im,coords,size,parent);assert (q-parent).abs().max()<1e-6;assert torch.allclose(q.exp().sum(-1),torch.ones(2,16),atol=1e-6);(-q[:,:,50].mean()).backward();assert torch.isfinite(m.alpha.grad).all() and m.alpha.grad.abs().sum()>0;assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.mdn.parameters())
def test_causal_step():
 m=GazeMDNCorrection();m.alpha.data.fill_(.2);im=torch.randn(1,1024);coords=torch.rand(1,16,2);size=torch.tensor([[2000,4000]]);parent=torch.randn(1,16,1152).log_softmax(-1);changed=coords.clone();changed[:,8:]=.9;a=m.teacher_logits(im,coords,size,parent);b=m.teacher_logits(im,changed,size,parent);assert torch.equal(a[:,:9],b[:,:9]);assert torch.equal(a[:,8],m.step_history(im,coords[:,:8],size,parent[:,8]))
