import unittest,tempfile
from pathlib import Path
import torch
from torch import nn
from src.mdn_dual_output_1011 import MDNProductProbability
from src.gaze_mdn_correction_1011 import GazeMDNCorrection
class Parent(nn.Module):
    grid=(24,48);foveal=True;vector_dims=[2,3]
    def __init__(self):
        super().__init__();self.register_buffer('cell_xy',torch.rand(1152,2));self.visual=nn.Identity()
    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        state=0 if hidden is None else hidden;q=torch.arange(1152.).sin()[None]*(1+prefix.sum()+state/100)
        if patch_vector is not None:q=q+torch.arange(1152.).cos()[None]*patch_vector.sum()/100
        return q.log_softmax(-1),state+1,{}
class MDNDualTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4);torch.manual_seed(42);self.model=MDNProductProbability(Parent(),GazeMDNCorrection());self.memory=dict(parent={},embedding=torch.rand(1,1024),image_size=torch.tensor([[1500.,3000.]]),prior=torch.zeros(1,1152))
    def test_initial_identity(self):
        p=torch.rand(1,2,2);v=torch.rand(1,5);q,_,_=self.model.step_history(self.memory,p,patch_vector=v);base,_,_=self.model.parent.step_history({},p,patch_vector=v);torch.testing.assert_close(q,base,atol=1e-6,rtol=0)
    def test_causal_teacher_step(self):
        self.model.correction.alpha.data.fill_(.3);c=torch.rand(1,6,2);v=torch.rand(1,6,5);q=self.model.teacher_logits(self.memory,c,v);alt=c.clone();alt[:,3:]=1-alt[:,3:];q2=self.model.teacher_logits(self.memory,alt,v);self.assertTrue(torch.equal(q[:,:4],q2[:,:4]));state=None;steps=[]
        for t in range(6):p,state,_=self.model.step_history(self.memory,c[:,:t],state,v[:,t]);steps.append(p)
        self.assertTrue(torch.equal(q,torch.stack(steps,1)));torch.testing.assert_close(q.logsumexp(-1),torch.zeros(1,6),atol=1e-6,rtol=0)
    def test_restore(self):
        self.model.correction.alpha.data.fill_(.4);c=torch.rand(1,4,2);q=self.model.teacher_logits(self.memory,c)
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'state.pt';torch.save(self.model.state_dict(),path);m=MDNProductProbability(Parent(),GazeMDNCorrection());m.load_state_dict(torch.load(path,weights_only=True),strict=True);torch.testing.assert_close(q,m.teacher_logits(self.memory,c),atol=0,rtol=0)
