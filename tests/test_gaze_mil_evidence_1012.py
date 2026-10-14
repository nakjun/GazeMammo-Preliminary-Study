import torch
from scripts.train_gaze_mil_evidence_1012 import EvidenceMIL,fit_normalization

def test_mil_sparse_evidence_and_order_invariance():
    m=EvidenceMIL(dim=1,temperature=.5).eval()
    with torch.no_grad():
        m.global_head.weight.zero_();m.global_head.bias.zero_();m.patch_head.weight.fill_(1);m.patch_head.bias.zero_()
    g=torch.zeros(1,1);p=torch.zeros(1,16,1);p[0,0,0]=8
    assert float(m(g,p).detach())>float(p.mean())+5
    assert torch.allclose(m(g,p),m(g,p.flip(1)))

def test_mil_bag_bce_reaches_patch_and_global_heads():
    m=EvidenceMIL(dim=2);g=torch.ones(2,2);p=torch.rand(2,16,2)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(m(g,p),torch.tensor([0.,1.]));loss.backward()
    assert m.global_head.weight.grad.abs().sum()>0
    assert m.patch_head.weight.grad.abs().sum()>0

def test_mil_patch_normalization_ignores_validation():
    g=torch.rand(3,2);p=torch.rand(3,16,2);train=torch.tensor([0,1]);_,_,a=fit_normalization(g,p,train)
    g[2]=10000;p[2]=-10000;_,_,b=fit_normalization(g,p,train)
    assert all(torch.equal(a[k],b[k]) for k in a)

if __name__=='__main__':
    test_mil_sparse_evidence_and_order_invariance();test_mil_bag_bce_reaches_patch_and_global_heads();test_mil_patch_normalization_ignores_validation();print('3 meaningful MIL checks passed')
