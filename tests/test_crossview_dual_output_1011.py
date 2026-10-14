import unittest
import torch
from tests.test_dual_output_1011 import Parent
from src.dual_output_1011 import ProbabilityComponent
from src.crossview_dual_output_1011 import CrossViewMixture
class ReuseMixTests(unittest.TestCase):
 def setup_model(self):
  parent=ProbabilityComponent([Parent(2,0),Parent(3,2)],[.75,.25]);cross=Parent(2,7);m=CrossViewMixture(parent,cross);p=parent.maps(torch.zeros(1,3,10,10));c=cross.maps(torch.zeros(1,3,10,10));return m,dict(parent=p,cross=c,prior=p['prior'])
 def test_state_and_same_control_vector(self):
  m,mem=self.setup_model();prefix=torch.empty(1,0,2);v=torch.arange(5.)[None];q,state,_=m.step_history(mem,prefix,patch_vector=v);p,_,_=m.parent.step_history(mem['parent'],prefix,patch_vector=v);c,_,_=m.cross.step_history(mem['cross'],prefix,patch_vector=v[:,:2]);torch.testing.assert_close(q.exp(),.75*p.exp()+.25*c.exp(),atol=1e-6,rtol=0);self.assertEqual(float(state[0][0]),1);self.assertEqual(float(state[0][1]),3);self.assertEqual(float(state[1]),8)
 def test_causal_teacher_step(self):
  m,mem=self.setup_model();coords=torch.rand(1,6,2);v=torch.rand(1,6,5);q=m.teacher_logits(mem,coords,v);changed=coords.clone();changed[:,3:]=1-changed[:,3:];changedv=v.clone();changedv[:,4:]=100;q2=m.teacher_logits(mem,changed,changedv);self.assertTrue(torch.equal(q[:,:4],q2[:,:4]));state=None;steps=[]
  for t in range(6):p,state,_=m.step_history(mem,coords[:,:t],state,v[:,t]);steps.append(p)
  self.assertTrue(torch.equal(q,torch.stack(steps,1)));torch.testing.assert_close(q.logsumexp(-1),torch.zeros(1,6),atol=1e-6,rtol=0)
