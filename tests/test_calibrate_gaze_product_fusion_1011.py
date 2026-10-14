import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('product_fusion',Path(__file__).parents[1]/'scripts/calibrate_gaze_product_fusion_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_normalized_identity_controls_and_static_R_shape():
    torch.manual_seed(42);control=torch.randn(2,16,1152).log_softmax(-1);improved=torch.randn(2,16,1152).log_softmax(-1);r=torch.rand(2,1152)
    assert torch.allclose(m.product_logp(control,improved,r,1,1,0),control,atol=2e-6)
    assert torch.allclose(m.product_logp(control,improved,r,0,1,0),improved,atol=2e-6)
    q=m.product_logp(control,improved,r,.75,.75,4)
    assert torch.isfinite(q).all() and torch.allclose(q.exp().sum(-1),torch.ones(2,16),atol=1e-6)


def test_target_oracle_is_not_a_normalized_distribution():
    first=torch.tensor([.9,.1]);second=torch.tensor([.1,.9]);assert float(torch.maximum(first,second).sum())>1
