import unittest
import torch
from src.fusion_backbones import interpolate_positions,FusionEncoder

class FusionBackboneTests(unittest.TestCase):
    def test_rectangular_positions_preserve_cls_and_gradients(self):
        positions=torch.randn(1,197,16,requires_grad=True)
        out=interpolate_positions(positions,32,16)
        self.assertEqual(tuple(out.shape),(1,513,16))
        torch.testing.assert_close(out[:,:1],positions[:,:1])
        out.square().sum().backward()
        self.assertTrue(torch.isfinite(positions.grad).all())
        self.assertGreater(positions.grad[:,1:].abs().sum().item(),0)
        self.assertIs(interpolate_positions(positions,14,14),positions)

    def test_vit_square_matches_native_encoder(self):
        torch.set_num_threads(4);torch.manual_seed(42)
        model=FusionEncoder('vit_b_16',pretrained=False).eval()
        image=torch.randn(1,3,224,224)
        with torch.inference_mode():
            tokens=model.conv_proj(image).flatten(2).transpose(1,2)
            tokens=torch.cat([model.class_token,tokens],1)
            expected=model.encoder(tokens)[:,0]
            actual=model.patch_vector(image)
        torch.testing.assert_close(actual,expected)
