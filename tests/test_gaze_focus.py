import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from src.spatial import SpatialTrajectory

class GazeFocusTests(unittest.TestCase):
    def test_motion_teacher_equals_autoregressive_steps(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        m=SpatialTrajectory(grid=(4,8),k=4,foveal=True,motion_bias=True,pretrained=False).eval()
        memory=dict(keys=torch.randn(2,32,128),prior=torch.randn(2,32),global_context=torch.randn(2,128))
        coords=torch.rand(2,4,2);patches=torch.randn(2,4,2048);previous=torch.zeros(2,2);older=previous;hidden=None;steps=[]
        with torch.no_grad():
            teacher=m.teacher_logits(memory,coords,patches)
            for t in range(4):
                output,hidden=m.step(memory,previous,older,patches[:,t],hidden,t);steps.append(output)
                older,previous=previous,coords[:,t]
        torch.testing.assert_close(teacher,torch.stack(steps,1),rtol=1e-5,atol=1e-5)
    def test_motion_never_uses_future_coordinates_or_visual_feedback(self):
        m=SpatialTrajectory(grid=(4,8),k=4,foveal=True,motion_bias=True,pretrained=False).eval()
        memory=dict(keys=torch.randn(1,32,128),prior=torch.randn(1,32),global_context=torch.randn(1,128))
        coords=torch.rand(1,4,2);patches=torch.randn(1,4,2048)
        with torch.no_grad():
            first=m.teacher_logits(memory,coords,patches)
            coords[:,2:]=torch.rand(1,2,2);patches[:,3:]=100
            second=m.teacher_logits(memory,coords,patches)
        torch.testing.assert_close(first[:,:3],second[:,:3])
    def test_motion_density_is_finite_differentiable_and_has_distinct_components(self):
        m=SpatialTrajectory(grid=(4,8),k=4,motion_bias=True,pretrained=False)
        q=torch.randn(2,3,128,requires_grad=True);previous=torch.rand(2,3,2)
        value=m.motion_logits(q,previous)
        self.assertEqual(tuple(value.shape),(2,3,32));self.assertTrue(torch.isfinite(value).all())
        value.square().mean().backward();self.assertTrue(torch.isfinite(m.motion.weight.grad).all())
        self.assertGreater(torch.nn.functional.softplus(m.motion_scale).item(),0)
        self.assertEqual(len(torch.unique(m.motion.bias.reshape(3,5)[:,1])),3)
