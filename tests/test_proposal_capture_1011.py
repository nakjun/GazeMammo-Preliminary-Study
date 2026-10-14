import unittest,tempfile
from pathlib import Path
import torch
from src.proposal_capture_1011 import *
from src.proposal_utility_1011 import seed_everything,state_sha256,freeze_model,balanced_mse


class CaptureTests(unittest.TestCase):
    def test_absolute_threshold_and_unattainable(self):
        t=capture_targets(1,'p',[[0,0,5,5]],(240,480),{1:'p'})
        self.assertTrue(t['known']);self.assertEqual(t['capture50'][0],1);self.assertEqual(t['absolute_coverage'][0,0],1)
        t=capture_targets(1,'p',[[0,0,2000,2000]],(2400,2400),{1:'p'})
        self.assertTrue(t['known']);self.assertFalse(t['attainable_capture50']);self.assertFalse(t['capture50'].any());self.assertLess(float(t['absolute_utility'].max()),.5)
    def test_boxless_and_allowlist(self):
        t=capture_targets(1,'p',[],(240,480),{1:'p'});self.assertFalse(t['known']);self.assertEqual(t['absolute_coverage'].shape,(0,1152))
        with self.assertRaises(ValueError):capture_targets(1,'other',[],(240,480),{1:'p'})
    def test_shapes_gradient_frozen_keys_and_seed(self):
        model=freeze_model(torch.nn.Linear(4,8));before=state_sha256(model.state_dict())
        with torch.no_grad():keys=model(torch.ones(2,1152,4))
        for architecture in ('linear','spatial'):
            seed_everything(42);head=CaptureHead(8,architecture);initial=state_sha256(head.state_dict());seed_everything(42);other=CaptureHead(8,architecture);self.assertEqual(initial,state_sha256(other.state_dict()))
            scores=head(keys);self.assertEqual(scores.shape,(2,1152));target=torch.zeros(1152);target[:10]=1
            loss,_=balanced_capture_bce(head.logits(keys)[0],target);loss.backward();self.assertTrue(any(p.grad is not None for p in head.parameters()));self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertEqual(before,state_sha256(model.state_dict()))
    def test_absent_positive_retained_and_boxless_skipped(self):
        z=torch.zeros(1152,requires_grad=True);loss,stats=balanced_capture_bce(z,torch.zeros(1152));self.assertIsNone(stats['positive']);self.assertAlmostEqual(float(loss.detach()),.5*torch.log(torch.tensor(2.)).item(),places=6);loss.backward();self.assertTrue((z.grad>0).all())
        z.grad=None;loss,_=balanced_capture_bce(z,torch.zeros(1152),False);loss.backward();self.assertFalse(z.grad.any())
    def test_absolute_mse_gradient(self):
        t=capture_targets(1,"p",[[0,0,2000,2000]],(2400,2400),{1:"p"})
        self.assertTrue((t["absolute_utility"]>0).any());self.assertFalse(t["capture50"].any())
        for arch in ("linear","spatial"):
            head=CaptureHead(8,arch);loss,_=balanced_mse(head(torch.ones(1152,8)),t["absolute_utility"]);loss.backward();self.assertTrue(any(p.grad is not None for p in head.parameters()))
    def test_cache_guard(self):
        r=dict(image_id=1,patient_key='p',keys=torch.zeros(1152,8),known=True,utility=torch.zeros(1152),occupancy=torch.zeros(1152));cache=dict(kind='proposal_utility_features_1011',base_checkpoint_sha256='base',annotation_sha256='annotation',records=[r]);validate_cache(cache,'base',{1:'p'},'annotation')
        with self.assertRaises(ValueError):validate_cache(cache,'wrong',{1:'p'},'annotation')
        with self.assertRaises(ValueError):validate_cache(cache,'base',{1:'other'},'annotation')
    def test_checkpoint_restore_hash(self):
        for arch in ('linear','spatial'):
            seed_everything(42);head=CaptureHead(8,arch);state=head.state_dict();checkpoint=dict(kind='proposal_capture_1011',base_checkpoint_sha256='base',head_state=state,head_state_sha256=state_sha256(state),config=dict(dim=8,architecture=arch,grid=[24,48]))
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'last.pt';torch.save(checkpoint,path);other,_=load_capture_head(path,'base');keys=torch.zeros(1,1152,8);self.assertTrue(torch.equal(predict_capture(other,{'keys':keys}),head(keys)))
                with self.assertRaises(ValueError):load_capture_head(path,'wrong')
                checkpoint['head_state_sha256']='wrong';torch.save(checkpoint,path)
                with self.assertRaises(ValueError):load_capture_head(path,'base')

if __name__=='__main__':torch.set_num_threads(4);unittest.main()
