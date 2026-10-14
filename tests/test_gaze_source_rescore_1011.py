import torch
from scripts.rescore_gaze_source_parents_1011 import strict

class CausalDummy:
    def step(self,memory,previous,older,patch_vector,hidden,index):
        value=previous.sum(-1)+(0 if patch_vector is None else patch_vector.sum(-1))
        return torch.stack([value,-value],-1),None

def test_strict_has_only_preceding_coordinates_and_crops():
    coords=torch.rand(1,16,2);vectors=torch.rand(1,16,3);vectors[:,0]=0;raw=strict(CausalDummy(),{},coords,vectors)
    changed=coords.clone();changed[:,8:]=0;newvectors=vectors.clone();newvectors[:,9:]=99
    assert torch.equal(raw[:,:9],strict(CausalDummy(),{},changed,newvectors)[:,:9])
    assert torch.equal(raw[:,0],torch.zeros_like(raw[:,0]))
