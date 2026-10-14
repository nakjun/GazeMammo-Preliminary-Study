import copy,importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
import torch
from predict_cxr_ensemble_1011 import validate_ensemble,probability_ensemble

class EnsembleTests(unittest.TestCase):
    def heads(self):
        source=dict(cache_sha256='c',global_checkpoint_sha256='g',gaze_checkpoint_sha256='z',final_test_used=False,gaze_provenance=dict(gaze_training_patient_keys=['a'],gaze_selection_patient_keys=['b'],seed=42,steps=16,strategy='sample',policy='none',patch_size=224,final_test_used=False))
        return [dict(kind='gaze_cxr_1011',model_config=dict(variant='dual_cross_drop'),source_provenance=copy.deepcopy(source),training_seed=seed,training_patient_keys=['a'],selection_patient_keys=['b']) for seed in (42,43,44)]
    def test_probability_mean_is_not_logit_mean(self):
        logits=[torch.tensor([0.]),torch.tensor([0.]),torch.tensor([8.])]
        result=probability_ensemble(logits);torch.testing.assert_close(result,sum(x.sigmoid() for x in logits)/3)
        self.assertGreater(float(abs(result-torch.stack(logits).mean(0).sigmoid())),.2)
    def test_incompatible_sources_and_patients_rejected(self):
        heads=self.heads();validate_ensemble(heads)
        for key in ('cache_sha256','global_checkpoint_sha256','gaze_checkpoint_sha256'):
            changed=copy.deepcopy(heads);changed[1]['source_provenance'][key]='wrong'
            with self.assertRaises(ValueError):validate_ensemble(changed)
        changed=copy.deepcopy(heads);changed[2]['training_patient_keys']=['other']
        with self.assertRaises(ValueError):validate_ensemble(changed)
    def test_duplicate_seeds_or_architecture_rejected(self):
        heads=self.heads();heads[1]['training_seed']=42
        with self.assertRaises(ValueError):validate_ensemble(heads)
        heads=self.heads();heads[1]['model_config']['variant']='dual_mean'
        with self.assertRaises(ValueError):validate_ensemble(heads)

if __name__=='__main__':unittest.main()
