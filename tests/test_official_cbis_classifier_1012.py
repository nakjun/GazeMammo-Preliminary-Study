"""Protocol and matched frozen residual tests without imaging downloads."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from src.official_cbis_classifier_1012 import ARMS,SEEDS,restore,implementation_hashes,state_digest,forward,verify_freeze,extract_image,metrics,build_cache,trajectory_proof
from src.gaze_classifier_1011 import GazeDiagnosisHead
from src.gaze_attention_1011 import GazeAttention1011,head_state

class ClassifierTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model=GazeDiagnosisHead(input_dim=1024,dim=128,mode='global',dropout=.3)
        self.ck=dict(kind='official_cbis_classifier_1012',arm='global',seed=42,model_config=self.model.config,state=self.model.state_dict(),global_state_sha256=state_digest(self.model.state_dict()),source_provenance={'fresh':True},implementation_sha256=implementation_hashes())
    def test_old_checkpoint_rejected(self):
        ck=copy.deepcopy(self.ck);ck['kind']='gaze_patch_classifier_1012'
        with self.assertRaises(ValueError):restore(ck)
    def test_matched_residual_epoch_zero(self):
        global_model=restore(self.ck)
        model=GazeAttention1011(global_model,variant='mean_residual',input_dim=1024,dim=64,dropout=.2).eval()
        tensors={'global_features':torch.randn(3,1024),'gaze_improved_features':torch.randn(3,16,1024),'gaze_improved_coordinates':torch.rand(3,16,2),'gaze_improved_mask':torch.ones(3,16,dtype=torch.bool)}
        result=forward(model,tensors,torch.arange(3),'gaze_improved')
        self.assertTrue(torch.equal(result['logits'],result['global_logits']))
        self.assertTrue(all(not p.requires_grad for p in model.global_model.parameters()))
        patch=dict(self.ck,arm='gaze_improved',model_config=model.config,state=head_state(model))
        restore(patch,self.ck)
        other=copy.deepcopy(self.ck);other['seed']=43
        with self.assertRaises(ValueError):restore(patch,other)
    def test_source_edit_rejected(self):
        self.ck['implementation_sha256']={}
        with self.assertRaises(ValueError):restore(self.ck)
    def test_freeze_requires_all_heads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'freeze.json'
            body=dict(arms=list(ARMS),head_seeds=list(SEEDS),threshold=.5,tta=False,path_seed=42,dependencies=[],heads=[],trajectories=[])
            body['sha256']=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest();path.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError,'Incomplete'):verify_freeze(path)
            body['threshold']=.4;path.write_text(json.dumps(body))
            with self.assertRaisesRegex(ValueError,'digest'):verify_freeze(path)
    def test_test_cache_embargo_before_pixels(self):
        import pandas as pd
        frame=pd.DataFrame({'image_id':[1], 'patient_key':['P_1'], 'split':['train']})
        with patch('src.official_cbis_classifier_1012.population',return_value=(frame,frame)),patch('PIL.Image.open') as opened:
            with self.assertRaisesRegex(ValueError,'final freeze'):
                build_cache('/unused','/unused','/unused','/unused',split='test')
            opened.assert_not_called()
    def test_pixel_api_refuses_semantic_identity(self):
        with self.assertRaisesRegex(ValueError,'numeric'):
            extract_image(None,None,'Malignant_P_0001')
    def test_full_classifier_refuses_smoke_trajectory(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trajectory.pt';torch.save({'smoke':True},path)
            with self.assertRaisesRegex(ValueError,'Smoke trajectory'):
                trajectory_proof(path,pd.DataFrame())
    def test_metrics_fixed_threshold(self):
        score=metrics([0,0,1,1],[.1,.6,.4,.9])
        self.assertEqual(score['confusion_matrix'],[[1,1],[1,1]])
        self.assertEqual(score['threshold'],.5)
        self.assertEqual(score['accuracy'],.5)
        with self.assertRaises(ValueError):metrics([0,1],[np.nan,.5])

if __name__=='__main__':unittest.main()
