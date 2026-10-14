import torch
from scripts.train_gaze_joint_summary_1012 import normalize,summarize

def test_normalization_does_not_fit_validation():
    x=torch.tensor([[0.,1.],[2.,3.],[100.,200.]])
    _,mean,std=normalize(x,torch.tensor([0,1]))
    altered=x.clone();altered[2]=torch.tensor([-1000.,5000.])
    _,other_mean,other_std=normalize(altered,torch.tensor([0,1]))
    assert torch.equal(mean,other_mean) and torch.equal(std,other_std)
    assert torch.equal(mean,torch.tensor([1.,2.]))

def test_summary_is_permutation_invariant_and_uses_all_patches():
    g=torch.tensor([[3.,4.]])
    patches=torch.arange(32,dtype=torch.float32).reshape(1,16,2)
    actual=summarize(g,patches)
    assert torch.equal(actual,summarize(g,patches.flip(1)))
    assert torch.equal(actual,torch.tensor([[3.,4.,15.,16.,30.,31.]]))
