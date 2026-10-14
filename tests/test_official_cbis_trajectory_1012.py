import tempfile
import unittest
from pathlib import Path
import torch
from src.official_cbis_trajectory_1012 import allowed_payloads,teacher_inputs,sha256,CONFIG

class TrajectoryProtocolTests(unittest.TestCase):
    def test_excluded_payload_is_not_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'gaze.csv';p.write_text('TEST.png;not valid python\nallowed.png;[([10,20], 2)]\n')
            values=allowed_payloads(p,{'allowed.png'})
            self.assertEqual(values['allowed.png'].tolist(),[[5.,10.]])
            self.assertEqual(len(sha256(p)),64)
    def test_teacher_vectors_are_strictly_previous(self):
        coords=torch.stack([torch.arange(80)/100,torch.arange(80)/100],1)
        vectors=torch.arange(80.)[:,None].repeat(1,1024)
        r=dict(image_id=123,parts=[coords],vectors=[vectors])
        for improved in (False,True):
            for epoch in range(6):
                c,m,v=teacher_inputs(r,epoch,improved)
                self.assertEqual(v[0].abs().sum().item(),0)
                valid=int(torch.nonzero(m)[-1])+1
                for t in range(1,valid):
                    self.assertAlmostEqual(v[t,0].item(),round(c[t-1,0].item()*100),places=5)
    def test_future_crops_do_not_affect_previous_input(self):
        r=dict(image_id=1,parts=[torch.tensor([[.1,.1],[.2,.2],[.3,.3]])],vectors=[torch.randn(3,1024)])
        before=teacher_inputs(r)[2].clone();r['vectors'][0][2].fill_(999)
        after=teacher_inputs(r)[2]
        self.assertTrue(torch.equal(before,after))
    def test_fixed_architecture(self):
        self.assertEqual(CONFIG['backbone_name'],'convnext_base')
        self.assertTrue(CONFIG['fpn'] and CONFIG['foveal'] and CONFIG['spatial_history'])
        self.assertEqual(CONFIG['history_length'],4)
if __name__=='__main__':unittest.main()
