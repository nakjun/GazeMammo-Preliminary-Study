import unittest
import tempfile
from pathlib import Path
import torch
from src.proposal_utility_1011 import *


def test_small_boundary_box_crop_can_capture_without_cell_occupancy():
    t=utility_targets(1,'p',[[0,0,5,5]],(240,480),{1:'p'})
    assert t['utility'][0]==1
    assert (t['utility']>0).sum()>(t['occupancy']>0).sum()
    assert t['coverage'].shape==(1,1152)


def test_boxless_and_balanced_mse():
    t=utility_targets(1,'p',[],(240,480),{1:'p'})
    assert not t['known']
    p=torch.tensor([.5,.5],requires_grad=True)
    loss,stats=balanced_mse(p,torch.tensor([1.,0.]))
    assert float(loss.detach())==.25
    absent,_=balanced_mse(p,torch.ones(2));assert float(absent.detach())==.125
    skipped,_=balanced_mse(p,t['utility'][:2],False);skipped.backward();assert torch.equal(p.grad,torch.zeros(2))
    with unittest.TestCase().assertRaises(ValueError):utility_targets(2,'p',[],(240,480),{1:'p'})


def test_frozen_keys_and_initial_seed():
    model=freeze_model(torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.Dropout(.9)))
    with torch.no_grad():keys=model(torch.ones(2,3))
    seed_everything(42);a=ProposalUtilityHead(4);seed_everything(42);b=ProposalUtilityHead(4)
    assert state_sha256(a.state_dict())==state_sha256(b.state_dict())
    a(keys).sum().backward();assert all(p.grad is None for p in model.parameters())
    assert not model.training and a.weight.grad is not None


def test_cache_allowlist_and_hash():
    record=dict(image_id=1,patient_key='p',keys=torch.ones(1152,4),known=True,utility=torch.ones(1152),occupancy=torch.zeros(1152))
    cache=dict(kind='proposal_utility_features_1011',base_checkpoint_sha256='sha',annotation_sha256='ann',records=[record])
    assert len(validate_cache(cache,'sha',{1:'p'},'ann'))==1
    for sha,allow in [('wrong',{1:'p'}),('sha',{1:'other'}),('sha',{1:'p',2:'q'})]:
        with unittest.TestCase().assertRaises(ValueError):validate_cache(cache,sha,allow)


def test_restore_hash(tmp_path):
    seed_everything(42);head=ProposalUtilityHead(4);state=head.state_dict();path=tmp_path/'head.pt'
    checkpoint=dict(kind='proposal_utility_1011',base_checkpoint_sha256='sha',head_state=state,head_state_sha256=state_sha256(state),config=dict(dim=4))
    torch.save(checkpoint,path);restored,_=load_utility_head(path,'sha')
    keys=torch.ones(1,1152,4);assert torch.equal(predict_utility(head,{'keys':keys}),predict_utility(restored,keys))
    with unittest.TestCase().assertRaises(ValueError):load_utility_head(path,'wrong')
    checkpoint['head_state_sha256']='wrong';torch.save(checkpoint,path)
    with unittest.TestCase().assertRaises(ValueError):load_utility_head(path,'sha')


class UtilityTests(unittest.TestCase):
    def test_targets(self): test_small_boundary_box_crop_can_capture_without_cell_occupancy()
    def test_loss(self): test_boxless_and_balanced_mse()
    def test_freeze_seed(self): test_frozen_keys_and_initial_seed()
    def test_cache(self): test_cache_allowlist_and_hash()
    def test_restore(self):
        with tempfile.TemporaryDirectory() as directory: test_restore_hash(Path(directory))

if __name__=="__main__": unittest.main()
