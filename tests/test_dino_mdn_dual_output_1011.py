import unittest,tempfile
from pathlib import Path
import torch
from tests.test_dual_output_1011 import Parent
from src.dual_output_1011 import ProbabilityComponent
from src.dino_mdn_dual_output_1011 import DinoMDNProbability
class DinoMixTests(unittest.TestCase):
 def setup_model(self):
  parent=ProbabilityComponent([Parent(2,0),Parent(3,2)],[.75,.25]);dino=Parent(4,7);model=DinoMDNProbability(parent,dino);p=parent.maps(torch.zeros(1,3,10,10));d=dino.maps(torch.zeros(1,3,10,10));return model,dict(parent=p,dino=d,prior=p['prior'])
 def test_independent_states_and_vector_slices(self):
  m,mem=self.setup_model();prefix=torch.zeros(1,0,2);v=torch.arange(9.)[None];q,state,_=m.step_history(mem,prefix,patch_vector=v);p,ps,_=m.parent.step_history(mem['parent'],prefix,patch_vector=v[:,:5]);d,ds,_=m.dino.step_history(mem['dino'],prefix,patch_vector=v[:,5:]);torch.testing.assert_close(q.exp(),.75*p.exp()+.25*d.exp(),atol=1e-6,rtol=0);self.assertEqual(float(state[0][0]),1);self.assertEqual(float(state[0][1]),3);self.assertEqual(float(state[1]),8)
  with self.assertRaises(ValueError):m.step_history(mem,prefix,patch_vector=torch.zeros(1,8))
 def test_causal_teacher_step_normalization(self):
  m,mem=self.setup_model();c=torch.rand(1,6,2);v=torch.rand(1,6,9);q=m.teacher_logits(mem,c,v);alt=c.clone();alt[:,3:]=1-alt[:,3:];altv=v.clone();altv[:,4:]=100;q2=m.teacher_logits(mem,alt,altv);self.assertTrue(torch.equal(q[:,:4],q2[:,:4]));state=None;steps=[]
  for t in range(6):p,state,_=m.step_history(mem,c[:,:t],state,v[:,t]);steps.append(p)
  self.assertTrue(torch.equal(q,torch.stack(steps,1)));torch.testing.assert_close(q.logsumexp(-1),torch.zeros(1,6),atol=1e-6,rtol=0)
 def test_restore(self):
  m,mem=self.setup_model();c=torch.rand(1,4,2);v=torch.rand(1,4,9);q=m.teacher_logits(mem,c,v)
  with tempfile.TemporaryDirectory() as d:
   path=Path(d)/'state.pt';torch.save(m.state_dict(),path);other,mem2=self.setup_model();other.load_state_dict(torch.load(path,weights_only=True),strict=True);torch.testing.assert_close(q,other.teacher_logits(mem2,c,v),atol=0,rtol=0)
