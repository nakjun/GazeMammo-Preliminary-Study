import unittest
from pathlib import Path
import torch
from PIL import Image
from src.clinical_dg_probability_1011 import load_clinical_probability
from src.proposal_utility_1011 import state_sha256
from src.deepgaze_clinical_1011 import head_state

class ClinicalProbabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.path=Path('runs/dual_target_1011/deepgaze_clinical_cpu_smoke/last.pt')
        if not cls.path.exists():raise unittest.SkipTest('Updated clinical smoke checkpoint unavailable')
        cls.adapter,cls.metadata=load_clinical_probability(cls.path,'runs/dual_target_1011/unified_runtime/config_cudnn_ieee.json','cpu')
        cls.before=state_sha256(cls.adapter.model.state_dict())
        # Actual image64; no GT is an input to maps or step_history.
        import pandas as pd
        frame=pd.read_csv('manifests/all.csv');row=frame[frame.image_id==64].iloc[0]
        # Resolve file name from canonical validation cache without reading gaze payload.
        record=torch.load('runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/64.pt',map_location='cpu',weights_only=False)
        image=Image.open(Path('data/image')/record['file_name']).convert('L')
        cls.memory=cls.adapter.maps(image)
    def test_causal_teacher_step(self):
        coords=torch.tensor([[[.2,.3],[.6,.7],[.4,.8]]]);q=self.adapter.teacher_logits(self.memory,coords)
        step=torch.stack([self.adapter.step_history(self.memory,coords[:,:t])[0] for t in range(3)],1)
        torch.testing.assert_close(q,step,atol=0,rtol=0)
        altered=coords.clone();altered[:,1:]=1-altered[:,1:]
        torch.testing.assert_close(self.adapter.teacher_logits(self.memory,altered)[:,:2],q[:,:2],atol=0,rtol=0)
        torch.testing.assert_close(q.logsumexp(-1),torch.zeros(1,3),atol=1e-5,rtol=0)
    def test_updated_restore_frozen(self):
        from scripts.train_deepgaze_clinical_1011 import load_clinical
        restored,_=load_clinical(self.path,'cpu')
        self.assertEqual(state_sha256(head_state(restored)),state_sha256(head_state(self.adapter.model)))
        self.assertEqual(self.before,state_sha256(self.adapter.model.state_dict()))
        self.assertTrue(all(not p.requires_grad for p in self.adapter.parameters()))
    def test_input_guard(self):
        with self.assertRaises(ValueError):self.adapter.maps(torch.zeros(1,3,1024,512))
        with self.assertRaises(ValueError):self.adapter.step_history(self.memory,torch.full((1,1,2),float('nan')))
        with self.assertRaises(ValueError):self.adapter.step_history(self.memory,torch.zeros(1,1,2),patch_vector=torch.zeros(1,1))
