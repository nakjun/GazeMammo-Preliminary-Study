import unittest
import torch
from torch import nn
from tests.test_dual_output_1011 import Parent
from src.dual_output_1011 import ProbabilityComponent
from src.clinical_dual_output_1011 import ClinicalMixture

class Clinic(nn.Module):
    def __init__(self,baseline):super().__init__();self.baseline=baseline
    def step_history(self,memory,prefix):
        if prefix.shape[1]==0:return self.baseline.step_history(memory['bos'],prefix)
        q=torch.tensor([[.1,.2,.3,.4]]).log();return q,None,{}

class ClinicalMixTests(unittest.TestCase):
    def test_BOS_and_mixture(self):
        base=ProbabilityComponent([Parent(2,0),Parent(3,2)],[.75,.25]);mix=ClinicalMixture(base,Clinic(base));mem=base.maps(torch.zeros(1,3,10,10));memory=dict(baseline=mem,clinical=dict(bos=mem),prior=mem['prior']);empty=torch.empty(1,0,2)
        q,state,_=mix.step_history(memory,empty);expected,_,_=base.step_history(mem,empty);torch.testing.assert_close(q,expected,atol=1e-6,rtol=0)
        prefix=torch.tensor([[[.2,.3]]]);v=torch.arange(5.)[None];q,state,_=mix.step_history(memory,prefix,state,v);b,_,_=base.step_history(mem,prefix,(torch.ones(1,1),torch.full((1,1),3.)),v)
        torch.testing.assert_close(q.exp(),.75*b.exp()+.25*torch.tensor([[.1,.2,.3,.4]]),atol=1e-6,rtol=0)
    def test_causal_teacher_step(self):
        base=ProbabilityComponent([Parent(2,0),Parent(3,2)],[.75,.25]);mix=ClinicalMixture(base,Clinic(base));mem=base.maps(torch.zeros(1,3,10,10));memory=dict(baseline=mem,clinical=dict(bos=mem),prior=mem['prior']);coords=torch.rand(1,6,2);vectors=torch.rand(1,6,5);q=mix.teacher_logits(memory,coords,vectors);alt=coords.clone();alt[:,3:]=1-alt[:,3:];q2=mix.teacher_logits(memory,alt,vectors);self.assertTrue(torch.equal(q[:,:4],q2[:,:4]));state=None;steps=[]
        for t in range(6):p,state,_=mix.step_history(memory,coords[:,:t],state,vectors[:,t]);steps.append(p)
        self.assertTrue(torch.equal(q,torch.stack(steps,1)))
