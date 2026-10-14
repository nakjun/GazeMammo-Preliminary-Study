import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
from tune_fusion_backbones import learning_rates,augmented

class FusionHpoTests(unittest.TestCase):
    def test_cosine_warmup_and_refit_prefix(self):
        cfg=dict(warmup=2,backbone_lr=1e-5,head_lr=1e-4,cosine=True)
        self.assertEqual(learning_rates(cfg,0),(0.,1e-4))
        self.assertEqual(learning_rates(cfg,1),(0.,1e-4))
        self.assertEqual(learning_rates(cfg,2),(1e-5,1e-4))
        rates=[learning_rates(cfg,e)[0] for e in range(2,12)]
        self.assertTrue(all(a>=b for a,b in zip(rates,rates[1:])))
        self.assertAlmostEqual(rates[-1],1e-6)
        # Short refits must retain the inner trial's original horizon.
        self.assertEqual(learning_rates(cfg,4),learning_rates(cfg,4,horizon=12))

    def test_flip_preserves_global_patch_correspondence_and_inputs(self):
        images=torch.arange(24.).reshape(2,1,3,4)
        patches=images[:,None].repeat(1,3,1,1,1)
        old=images.clone();a,b=augmented(images,patches,torch.Generator().manual_seed(1))
        torch.testing.assert_close(images,old)
        for j in range(3):torch.testing.assert_close(a,b[:,j])
        self.assertTrue(any(torch.equal(a[i],old[i].flip(-1)) for i in range(2)))
