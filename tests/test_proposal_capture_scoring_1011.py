import unittest
import torch
from src.proposal_capture_scoring_1011 import *

class CaptureScoringTests(unittest.TestCase):
    def test_alpha1_exact_identity(self):
        r=torch.tensor([0.,1e-9,.3,1.]);score,_=geometric_capture_scores(r,torch.ones(4),1);self.assertTrue(torch.equal(score,r))
        score,valid=geometric_capture_scores(r,torch.full((4,),float('nan')),1);self.assertTrue(torch.equal(score,r));self.assertFalse(valid)
    def test_axis_mapping_and_shifted_sampling(self):
        gx,gy=24,48;x=torch.arange(gx)[None].expand(gy,-1);y=torch.arange(gy)[:,None].expand(-1,gx);field=(x+100*y).float();anchors=torch.tensor([[(3+.5)/gx,(7+.5)/gy],[(4+.5)/gx,(10+.5)/gy]]);sample=sample_capture_field(field,anchors);self.assertTrue(torch.allclose(sample,torch.tensor([703.,1004.]),atol=1e-4))
        shifted=anchors.clone();shifted[0,0]+=.5/gx;shifted[1,1]+=.5/gy;sample=sample_capture_field(field,shifted);self.assertTrue(torch.allclose(sample,torch.tensor([703.5,1054.]),atol=1e-4))
        borders=sample_capture_field(field,torch.tensor([[0.,0.],[1.,1.]]));self.assertTrue(torch.equal(borders,torch.tensor([0.,4723.])))
    def test_finite_bounded_recipe_and_nan_fallback(self):
        r=torch.tensor([0.,.25,1.]);c=torch.tensor([0.,1.,.25]);score,valid=geometric_capture_scores(r,c,.5);self.assertTrue(valid);self.assertTrue(torch.isfinite(score).all());self.assertTrue(((score>=0)&(score<=1)).all());self.assertTrue(torch.allclose(score,torch.tensor([1e-6,.5,.5])))
        score,valid=geometric_capture_scores(r,torch.tensor([0.,float('nan'),1.]),.75);self.assertFalse(valid);self.assertTrue(torch.equal(score,r))
        with self.assertRaises(ValueError):geometric_capture_scores(torch.tensor([float('nan')]),torch.ones(1))
    def test_alpha1_rollout_exact_original_control(self):
        try:from .test_proposal_budget_1011 import fixture,Model
        except ImportError:from test_proposal_budget_1011 import fixture,Model
        from src.proposal_center_offset_1011 import center_offset_rollout
        m,mem,image,g=fixture();offsets=torch.ones_like(m.cell_xy)*.1;r=m.proposal(mem['keys']).squeeze(-1).sigmoid();mixed,_=geometric_capture_scores(r,torch.rand_like(r),1)
        a=center_offset_rollout(m,mem,image,g,offsets,3,.5,4,42,ranking_relevance=mixed);b=center_offset_rollout(Model(),mem,image,g,offsets,3,.5,4,42)
        self.assertEqual(a['coordinates'],b['coordinates']);self.assertEqual(a['source_anchor_indices'],b['source_anchor_indices'])

if __name__=='__main__':unittest.main()
