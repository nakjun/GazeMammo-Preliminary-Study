import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('history_audit',Path(__file__).parents[1]/'scripts/audit_gaze_history_gain_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_bos_arm_independently_resets_prefix_hidden_and_crop():
    class Model:
        def __init__(self):self.calls=0
        def step_history(self,memory,prefix,hidden=None,patch_vector=None):
            assert prefix.shape==(2,0,2) and hidden is None and patch_vector is None
            self.calls+=1;return memory['prior'].log_softmax(-1),torch.ones(1,2,3),{}
    model=Model();memory={'prior':torch.randn(2,1152)};q=m.bos_repeated(model,memory,2,16)
    assert model.calls==16 and torch.equal(q,q[:,:1].expand_as(q)) and torch.allclose(q.exp().sum(-1),torch.ones(2,16),atol=1e-6)
