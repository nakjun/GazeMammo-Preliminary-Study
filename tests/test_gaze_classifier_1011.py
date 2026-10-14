import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.gaze_classifier_1011 import GazeDiagnosisHead, image_features, validate_population


class ClassifierTests(unittest.TestCase):
    def test_sequence_gradient_and_checkpoint_roundtrip(self):
        torch.manual_seed(42)
        model = GazeDiagnosisHead(input_dim=8, dim=16, dropout=0)
        g, p, xy = torch.randn(3, 8), torch.randn(3, 16, 8), torch.rand(3, 16, 2)
        logits, attention = model(g, p, xy)
        self.assertEqual(tuple(logits.shape), (3,))
        torch.testing.assert_close(attention.sum(1), torch.ones(3))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.tensor([0., 1., 0.])).backward()
        for module in (model.global_project, model.local_project, model.sequence, model.attention):
            self.assertTrue(any(x.grad is not None and x.grad.abs().sum() > 0 for x in module.parameters()))
        restored = GazeDiagnosisHead(**model.config)
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored(g, p, xy)[0], logits)
        changed = model(g, p.flip(1), xy.flip(1))[0]
        self.assertFalse(torch.allclose(changed, logits))

    def test_global_head_does_not_use_local_inputs(self):
        model = GazeDiagnosisHead(input_dim=8, dim=16, mode='global').eval()
        g = torch.randn(2, 8)
        torch.testing.assert_close(model(g)[0], model(g, torch.randn(2, 16, 8), torch.rand(2, 16, 2))[0])

    def test_patient_provenance_and_unknown_label_exclusion(self):
        frame = pd.DataFrame(dict(image_id=[1, 2, 3, 4, 5, 6],
                                  patient_key=['a', 'b', 'c', 'd', 'e', 'f'],
                                  split=['train', 'train', 'val', 'val', 'train', 'test'],
                                  label=[0, 1, 0, 1, -1, 1]))
        cp = dict(training_patient_keys=['a', 'b', 'e'], selection_patient_keys=['c', 'd'])
        self.assertEqual(set(validate_population(frame, cp).image_id), {1, 2, 3, 4})
        with self.assertRaises(ValueError):
            validate_population(frame, dict(cp, training_patient_keys=['a', 'c']))
        with self.assertRaises(ValueError):
            validate_population(frame, dict(cp, selection_patient_keys=['f']))
        with self.assertRaises(ValueError):
            validate_population(frame.assign(patient_key=['a', 'b', 'a', 'd', 'e', 'f']), cp)

    def test_image_only_feature_extraction(self):
        class Visual(torch.nn.Module):
            def patch_vector(self, x):
                return x.mean((-1, -2))
        class FakeGaze(torch.nn.Module):
            input_size = (8, 16)
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
                self.visual = Visual()
                self.register_buffer('cell_xy', torch.tensor([[.25, .25], [.75, .75]]))
            def maps(self, image):
                return {}
            def step_history(self, memory, prefix, hidden, vector):
                return torch.tensor([[100., 0.]]), None, {}
        with patch('src.data.load_gazes', side_effect=AssertionError('GT must not be read')):
            features = image_features(FakeGaze(), Image.new('L', (1500, 3000), 100), 64, steps=4)
        self.assertEqual(features['gaze_features'].shape, (4, 3))
        self.assertEqual(features['grid_features'].shape, (4, 3))
        np.testing.assert_allclose(features['gaze_coordinates'], .25)
        self.assertTrue(all(torch.isfinite(x).all() for x in features.values()))


if __name__ == '__main__':
    unittest.main()
