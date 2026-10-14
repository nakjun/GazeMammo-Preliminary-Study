import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
import torch
from src.constraints import breast_mask,VisitConstraint,rollout
from src.models import TrajectoryModel

class ConstraintTests(unittest.TestCase):
    def test_mask_removes_disconnected_marker_and_fills_hole(self):
        a=np.zeros((100,100),dtype=np.uint8);a[20:80,20:80]=150;a[40:50,40:50]=0;a[2:5,90:95]=255
        mask=breast_mask(Image.fromarray(a));self.assertTrue(mask[45,45]);self.assertFalse(mask[3,92])
    def test_hard_mask_cannot_choose_outside(self):
        allowed=torch.tensor([[False,True,False,False]])
        state=VisitConstraint(allowed,torch.tensor([[100.,100.]]),(2,2),'breast')
        logits,valid=state.apply(torch.tensor([[100.,0.,90.,80.]]));self.assertEqual(int(logits.argmax()),1);self.assertTrue(valid.item())
    def test_distance_relaxation_never_revisits_or_leaves_breast(self):
        state=VisitConstraint(torch.tensor([[True,True,False,False]]),torch.tensor([[100.,100.]]),(2,2),'hard',radius=1000)
        chosen=[]
        for _ in range(3):
            logits,valid=state.apply(torch.zeros(1,4));idx=logits.argmax(1);state.update(idx,valid)
            if valid.item():chosen.append(int(idx))
        self.assertEqual(chosen,[0,1]);self.assertGreater(int(state.relaxations[0]),0)
    def test_minimum_distance_is_in_original_pixels(self):
        state=VisitConstraint(torch.ones(1,16,dtype=torch.bool),torch.tensor([[400.,400.]]),(4,4),'hard',radius=112)
        state.update(torch.tensor([5]),torch.tensor([True]));logits,_=state.apply(torch.zeros(1,16))
        self.assertFalse(torch.isfinite(logits[0,5]));self.assertFalse(torch.isfinite(logits[0,6]));self.assertTrue(torch.isfinite(logits[0,10]))
    def test_unconstrained_rollout_matches_original(self):
        torch.set_num_threads(2);m=TrajectoryModel(grid=(4,8),k=4,pretrained=False).eval();image=torch.randn(1,3,128,64)
        with torch.no_grad():memory=m.memory(image);a=m.generate(image)
        b=rollout(m,memory,torch.ones(1,32,dtype=torch.bool),torch.tensor([[1500.,3000.]]))
        self.assertTrue(torch.equal(a,b['coords']));self.assertTrue(b['valid'].all())
    def test_soft_penalty_allows_revisit(self):
        state=VisitConstraint(torch.ones(1,16,dtype=torch.bool),torch.tensor([[400.,400.]]),(4,4),'soft',radius=112)
        state.update(torch.tensor([5]),torch.tensor([True]));score=torch.zeros(1,16);score[0,5]=100
        logits,_=state.apply(score)
        self.assertTrue(torch.isfinite(logits[0,5]));self.assertLess(float(logits[0,5]),100);self.assertEqual(int(logits.argmax()),5)

if __name__=='__main__':unittest.main()
