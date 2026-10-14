import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from PIL import Image
from src.spatial import SpatialTrajectory,gaussian_targets,GlobalLocalFusion

class SpatialTests(unittest.TestCase):
    def test_gaussian_is_normalized_and_spatially_local(self):
        p=torch.tensor([[.375,.375]]);q=gaussian_targets(p,(4,4),sigma_pixels=40,sizes=(400,400))
        self.assertAlmostEqual(float(q.sum()),1.,places=6)
        self.assertGreater(float(q[0,6]),float(q[0,15]))
        self.assertEqual(int(gaussian_targets(p,(4,4),0).argmax()),5)
    def test_teacher_never_uses_future_coordinates_or_patch_features(self):
        torch.set_num_threads(2);m=SpatialTrajectory(grid=(4,8),k=4,foveal=True,pretrained=False).eval()
        memory=dict(keys=torch.randn(1,32,128),prior=torch.randn(1,32),global_context=torch.randn(1,128))
        coords=torch.rand(1,4,2);patch=torch.rand(1,4,2048)
        with torch.no_grad():
            a=m.teacher_logits(memory,coords,patch)
            coords[:,2:]=torch.rand(1,2,2);patch[:,3:]=100
            b=m.teacher_logits(memory,coords,patch)
        self.assertTrue(torch.allclose(a[:,:3],b[:,:3]))
    def test_step_matches_teacher_logits(self):
        m=SpatialTrajectory(grid=(4,8),k=4,foveal=True,pretrained=False).eval()
        memory=dict(keys=torch.randn(1,32,128),prior=torch.randn(1,32),global_context=torch.randn(1,128))
        coords=torch.rand(1,4,2);patch=torch.rand(1,4,2048);previous=torch.zeros(1,2);older=previous;hidden=None;outputs=[]
        with torch.no_grad():
            a=m.teacher_logits(memory,coords,patch)
            for t in range(4):
                out,hidden=m.step(memory,previous,older,patch[:,t],hidden,t);outputs.append(out)
                older,previous=previous,coords[:,t]
        self.assertTrue(torch.allclose(a,torch.stack(outputs,1),atol=1e-5))
    def test_fusion_heads_and_padding(self):
        m=GlobalLocalFusion().eval();g=torch.randn(2,2048);p=torch.randn(2,4,2048);mask=torch.tensor([[True,True,False,False],[True,True,True,True]])
        with torch.no_grad():a=m(g,p,mask);p[0,2:]=99999;b=m(g,p,mask)
        self.assertTrue(torch.allclose(a['fusion_logits'],b['fusion_logits']))
        self.assertEqual(tuple(a['global_logits'].shape),(2,))
    def test_last_stage_finetuning_and_spatial_shape(self):
        m=SpatialTrajectory(grid=(4,8),k=4,fpn=True,pretrained=False);m.set_backbone('last');m.train()
        self.assertFalse(next(m.visual.stem.parameters()).requires_grad)
        self.assertTrue(next(m.visual.layer4.parameters()).requires_grad)
        out=m.maps(torch.randn(1,3,128,64));self.assertEqual(tuple(out['prior'].shape),(1,32))
    def test_image_only_policy_stays_inside_and_avoids_revisits(self):
        from unittest.mock import patch
        from src.spatial_inference import predict_path
        class FixedModel:
            grid=(4,8);cells=32;k=4;foveal=False
            def maps(self,x):return {'prior':torch.zeros(1,32)}
            def step(self,*args):return torch.zeros(1,32),None
        image=Image.fromarray(np.full((400,200),255,dtype=np.uint8))
        with patch('src.data.load_gazes',side_effect=AssertionError('Gaze during inference')):
            result=predict_path(FixedModel(),image,torch.device('cpu'),strategy='argmax')
        p=result['points'];self.assertEqual(len(np.unique(p,axis=0)),4)
        xy=p*[200,400];distance=np.linalg.norm(xy[:,None]-xy[None,:],axis=-1)
        self.assertGreaterEqual(float(distance[np.triu_indices(4,1)].min()),112)

if __name__=='__main__':unittest.main()
