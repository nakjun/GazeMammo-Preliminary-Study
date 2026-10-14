import inspect
import unittest
from unittest.mock import patch
import torch
from PIL import Image
from src.proposal_budget_1011 import budget_rollout,reservation_steps
from src.proposal_policy_1011 import rollout
from src.guided_trajectory_1011 import candidate_geometry


class Model:
    foveal=True;rho=1.;ablation='combined';proposal_valid=True
    cell_xy=torch.tensor([[.1,.1],[.4,.1],[.7,.1],[.9,.9]])
    def __init__(self):self.histories=[];self.visual=self
    def proposal(self,keys):return torch.tensor([[[4.],[3.],[2.],[1.]]])
    def step_history(self,memory,prefix,hidden,vector):
        self.histories.append((prefix.clone(),hidden,vector));return torch.tensor([[.1,.2,.3,.4]]).log_softmax(-1),(hidden or 0)+1,{}
    def patch_vector(self,crop):return crop.mean((2,3))


def fixture(size=224):
    m=Model();memory=dict(prior=torch.zeros(1,4),keys=torch.zeros(1,4,2));image=Image.new('L',(1000,1000))
    geometry=dict(geometry_valid=True,background=torch.zeros(4),edge=torch.zeros(4),crop_size=size,rectangles=torch.tensor([[0.,0,100,100],[300,0,400,100],[600,0,700,100],[900,900,1000,1000]]))
    return m,memory,image,geometry


class BudgetTests(unittest.TestCase):
    def test_schedule(self):
        self.assertEqual(reservation_steps(3,16,'interleaved'),[0,5,10]);self.assertEqual(reservation_steps(3,16,'first'),[0,1,2]);self.assertEqual(reservation_steps(0,16,'interleaved'),[])
        for k in (1,2,3,4,6,8):self.assertEqual(len(set(reservation_steps(k,16,'interleaved'))),k)
        with self.assertRaises(ValueError):reservation_steps(17)
    def test_actual_history_and_crop(self):
        m,mem,image,g=fixture(320);calls=[]
        def crop(image,center,crop_size):calls.append((center,crop_size));return torch.zeros(3,224,224)
        with patch('src.proposal_budget_1011.patch_tensor',side_effect=crop):r=budget_rollout(m,mem,image,g,2,'interleaved',320,4,strategy='argmax')
        self.assertEqual(r['reserved_steps'],[0,2]);self.assertEqual(r['indices'],[0,1,1,1]);self.assertEqual(len(calls),4)
        self.assertTrue(all(size==320 for _,size in calls))
        for t,(prefix,hidden,vector) in enumerate(m.histories):
            self.assertEqual(prefix.shape[1],t)
            if t:self.assertTrue(torch.equal(prefix[0,-1],m.cell_xy[r['indices'][t-1]]));self.assertEqual(hidden,t);self.assertIsNotNone(vector)
        self.assertEqual(len(r['coordinates']),4)
    def test_zero_budget_exact_baseline(self):
        m,mem,image,g=fixture()
        a=budget_rollout(m,mem,image,g,0,steps=4,seed=42)
        b=rollout(Model(),mem,image,g,'P0',4,42)
        self.assertEqual(a['indices'],b['indices']);self.assertEqual(a['coordinates'],b['coordinates'])
    def test_three_first_reproduces_existing_p5(self):
        m,mem,image,g=fixture()
        a=budget_rollout(m,mem,image,g,3,steps=4,seed=42)
        b=rollout(Model(),mem,image,g,"P5",4,42)
        self.assertEqual(a["indices"],b["indices"])
        self.assertEqual(a["coordinates"],b["coordinates"])
    def test_candidate_override(self):
        m,mem,image,g=fixture()
        a=budget_rollout(m,mem,image,g,2,steps=4,strategy="argmax",candidate_indices=[3,2,1])
        self.assertEqual(a["indices"][:2],[3,2])
        for indices in ([0,0],[4],[-1]):
            with self.assertRaises(ValueError):budget_rollout(m,mem,image,g,2,steps=4,candidate_indices=indices)
    def test_original_relevance_external_avoids_adapted_proposal(self):
        m,mem,image,g=fixture()
        def reject(_):raise AssertionError("Must use frozen original proposal instead of adapted keys")
        m.proposal=reject
        r=budget_rollout(m,mem,image,g,2,steps=4,original_relevance=torch.tensor([.1,.2,.8,.9]))
        self.assertEqual(r["reserved_candidate_indices"],[3,2])
    def test_geometry_matches_physical_crop(self):
        image=Image.new('L',(240,480));g=candidate_geometry(image,(24,48),320,mask=torch.ones(480,240).numpy())
        self.assertEqual(float(g['rectangles'][0,2]-g['rectangles'][0,0]),320)
        m,mem,image,g=fixture(224)
        with self.assertRaises(ValueError):budget_rollout(m,mem,image,g,crop_size=320)
    def test_no_supervision_inputs(self):
        params=set(inspect.signature(budget_rollout).parameters)
        self.assertFalse(params & {'boxes','labels','gaze','annotation','targets'})

if __name__=='__main__':unittest.main()
