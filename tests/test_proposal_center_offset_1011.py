import unittest,tempfile
from pathlib import Path
import numpy as np
import torch
from src.proposal_center_offset_1011 import *
from src.proposal_utility_1011 import seed_everything,state_sha256,freeze_model
try:
    from .test_proposal_budget_1011 import fixture,Model
except ImportError:
    from test_proposal_budget_1011 import fixture,Model


class OffsetTests(unittest.TestCase):
    def test_per_box_weights_and_exact_center_target(self):
        t=center_offset_targets(1,'p',[[0,0,10,10],[100,200,100,100]],(240,480),{1:'p'})
        self.assertAlmostEqual(float(t['weights'].sum()),1)
        for box in (0,1):self.assertAlmostEqual(float(t['weights'][t['assignment']==box].sum()),.5,places=6)
        self.assertEqual(t['positive_cells'],101);self.assertTrue(torch.allclose(t['targets'][0],torch.zeros(2),atol=1e-7));self.assertEqual(t['unassigned_box_indices'],[])
    def test_assignment_and_unassigned_diagnostic(self):
        t=center_offset_targets(1,'p',[[0,0,10,10],[0,0,10,10]],(240,480),{1:'p'})
        self.assertEqual(int(t['assignment'][0]),0);self.assertEqual(t['unassigned_box_indices'],[1])
    def test_clip_and_train_allowlist_boxless(self):
        t=center_offset_targets(1,'p',[[0,0,3000,3000]],(5000,5000),{1:'p'})
        self.assertGreater(t['clipped_component_fraction'],0);self.assertLessEqual(float(t['targets'].abs().max()),1)
        t=center_offset_targets(1,'p',[],(240,480),{1:'p'});self.assertFalse(t['known']);self.assertFalse(t['weights'].any())
        with self.assertRaises(ValueError):center_offset_targets(1,'other',[],(240,480),{1:'p'})
    def test_zero_init_seed_and_frozen_gradient(self):
        frozen=freeze_model(torch.nn.Linear(4,8));before=state_sha256(frozen.state_dict())
        with torch.no_grad():keys=frozen(torch.ones(1152,4))
        target=torch.full((1152,2),.2);weights=torch.ones(1152)/1152
        for arch in ('linear','spatial'):
            seed_everything(42);head=CenterOffsetHead(8,arch);initial=state_sha256(head.state_dict());seed_everything(42);other=CenterOffsetHead(8,arch);self.assertEqual(initial,state_sha256(other.state_dict()));self.assertFalse(head(keys).any())
            loss=offset_loss(head(keys),target,weights);loss.backward();self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in head.parameters()));self.assertTrue(all(p.grad is None for p in frozen.parameters()))
        self.assertEqual(before,state_sha256(frozen.state_dict()))
    def test_shift_bounds_zero_reproduction_and_rounding(self):
        anchors=torch.tensor([[.1,.2],[.9,.9]]);zero=torch.zeros_like(anchors)
        self.assertTrue(torch.equal(shifted_candidate_coords(anchors,zero,(1000,2000)),anchors))
        self.assertTrue(torch.equal(shifted_candidate_coords(anchors,torch.ones_like(anchors),(1000,2000),0),anchors))
        shifted=shifted_candidate_coords(anchors,torch.tensor([[-1.,-1.],[1.,1.]]),(1000,2000));self.assertTrue((shifted>=0).all());self.assertTrue((shifted<1).all())
        rect=offset_rectangles(shifted,(1000,2000));centers=np.round(shifted.numpy().astype(float)*[1000,2000]);self.assertTrue(np.array_equal(rect[:,:2].numpy(),centers-112));self.assertTrue(torch.equal(rect[:,2:]-rect[:,:2],torch.full((2,2),224.)))
    def test_strength_zero_exact_original_controls(self):
        from src.proposal_budget_1011 import budget_rollout
        from src.proposal_policy_1011 import rollout
        for budget in (0,3,8,12,16):
            m,mem,image,g=fixture()
            a=center_offset_rollout(m,mem,image,g,torch.ones_like(m.cell_xy),budget,0,16,42)
            b=budget_rollout(Model(),mem,image,g,budget,steps=16,seed=42)
            self.assertEqual(a['coordinates'],b['coordinates']);self.assertEqual(a['indices'],b['indices'])
        m,mem,image,g=fixture();a=center_offset_rollout(m,mem,image,g,torch.zeros_like(m.cell_xy),3,1,4,42);b=rollout(Model(),mem,image,g,'P5',4,42)
        self.assertEqual(a['coordinates'],b['coordinates']);self.assertEqual(a['indices'],b['indices'])
    def test_actual_continuous_history_and_rectangles(self):
        m,mem,image,g=fixture();offsets=torch.full_like(m.cell_xy,.1)
        r=center_offset_rollout(m,mem,image,g,offsets,2,1,4,42,'argmax')
        candidates=torch.tensor(r['candidate_coords']);points=torch.tensor(r['coordinates'])
        for t,index in enumerate(r['indices']):
            expected=candidates[index] if r['reservation_flags'][t] else m.cell_xy[index]
            self.assertTrue(torch.equal(points[t],expected))
            if t:self.assertTrue(torch.equal(m.histories[t][0][0,-1],points[t-1]))
        self.assertTrue(torch.equal(torch.tensor(r['actual_crop_rectangles']),offset_rectangles(points,image.size)))
    def test_relevance_p0_exact_and_perbox_totals(self):
        args=(1,'p',[[0,0,10,10],[100,200,100,100]],(240,480),{1:'p'})
        base=center_offset_targets(*args);r=torch.linspace(0,1,1152)
        same=center_offset_targets(*args,source_relevance=r,relevance_power=0)
        for key in ('targets','weights','assignment'):self.assertTrue(torch.equal(base[key],same[key]))
        for power in (1,2):
            t=center_offset_targets(*args,source_relevance=r,relevance_power=power)
            self.assertAlmostEqual(float(t['weights'].sum()),1,places=6)
            for box in (0,1):self.assertAlmostEqual(float(t['weights'][t['assignment']==box].sum()),.5,places=6)
    def test_invalid_relevance_finite_uniform_fallback(self):
        args=(1,'p',[[0,0,10,10],[100,200,100,100]],(240,480),{1:'p'});base=center_offset_targets(*args)
        for r in (torch.zeros(1152),torch.full((1152,),float('nan')),torch.full((1152,),-1.),torch.full((1152,),2.)):
            for power in (1,2):
                t=center_offset_targets(*args,source_relevance=r,relevance_power=power);self.assertTrue(torch.equal(t['weights'],base['weights']));self.assertTrue(torch.isfinite(t['weights']).all());self.assertEqual(t['relevance_fallback_box_indices'],[0,1])
    def test_original_proposal_and_cache_stay_frozen(self):
        from src.proposal_utility_1011 import validate_cache
        keys=torch.ones(1152,8);r=dict(image_id=1,patient_key='p',keys=keys,known=True,utility=torch.zeros(1152),occupancy=torch.zeros(1152));cache=dict(kind='proposal_utility_features_1011',base_checkpoint_sha256='base',annotation_sha256='annotation',records=[r]);validate_cache(cache,'base',{1:'p'},'annotation')
        original=freeze_model(torch.nn.Linear(8,1));before=state_sha256(original.state_dict())
        with torch.no_grad():scores=original(keys).sigmoid().squeeze(-1)
        t=center_offset_targets(1,'p',[[0,0,10,10]],(240,480),{1:'p'},source_relevance=scores,relevance_power=2);head=CenterOffsetHead(8);offset_loss(head(keys),t['targets'],t['weights']).backward()
        self.assertEqual(before,state_sha256(original.state_dict()));self.assertTrue(all(p.grad is None for p in original.parameters()));self.assertIsNone(keys.grad)
        with self.assertRaises(ValueError):validate_cache(cache,'wrong',{1:'p'},'annotation')
    def test_restore_hash(self):
        for arch in ('linear','spatial'):
            seed_everything(42);config=dict(dim=8,architecture=arch,grid=[24,48],max_offset_crops=1.);head=CenterOffsetHead(**config);state=head.state_dict();ck=dict(kind='proposal_center_offset_1011',base_checkpoint_sha256='base',head_state=state,head_state_sha256=state_sha256(state),config=config)
            with tempfile.TemporaryDirectory() as d:
                path=Path(d)/'last.pt';torch.save(ck,path);other,_=load_center_offset_head(path,'base');self.assertTrue(torch.equal(other(torch.ones(1,1152,8)),head(torch.ones(1,1152,8))))
                with self.assertRaises(ValueError):load_center_offset_head(path,'wrong')
                ck['head_state_sha256']='wrong';torch.save(ck,path)
                with self.assertRaises(ValueError):load_center_offset_head(path,'base')
    def test_masked_cells_and_boxless_gradient(self):
        pred=torch.zeros(3,2,requires_grad=True);target=torch.ones(3,2);weights=torch.tensor([1.,0,0]);offset_loss(pred,target,weights).backward();self.assertFalse(pred.grad[1:].any());pred.grad=None;offset_loss(pred,target,weights,False).backward();self.assertFalse(pred.grad.any())

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
