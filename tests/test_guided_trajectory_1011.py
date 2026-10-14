import sys,importlib.util,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from src.guided_trajectory_1011 import candidate_geometry,training_box_targets,balanced_proposal_loss,visit_loss,free_losses,GuidedTrajectory1011,nonvisual_state,initialize_from_base,restore_guided,rho_schedule
from src.trajectory_1011 import Trajectory1011,objective

class GuidedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,**kwargs):return GuidedTrajectory1011(dim=8,grid=(2,2),pretrained=False,spatial_history=True,behavior_mixture=True,**kwargs).eval()
    def memory(self):return dict(keys=torch.randn(1,4,8,requires_grad=True),prior=torch.randn(1,4),global_context=torch.randn(1,8))
    def geometry(self):return candidate_geometry(Image.new('L',(100,200),128),(2,2),mask=np.ones((200,100),bool))
    def test_geometry_padding_edges_and_empty_fallback(self):
        image=Image.new('L',(100,200));g=candidate_geometry(image,(2,2));self.assertFalse(g['geometry_valid']);self.assertEqual(float(g['background'].sum()),0);self.assertTrue(torch.isfinite(g['edge']).all())
        mask=np.ones((200,100),bool);g=candidate_geometry(image,(2,2),mask=mask);self.assertTrue(g['geometry_valid']);self.assertTrue((g['foreground']<1).all());self.assertTrue((g['edge']>0).all());self.assertTrue((g['distance']>0).all());torch.testing.assert_close(g['foreground'],g['inbounds'])
    def test_union_tiny_box_coverage_and_train_allowlist(self):
        allow={1:'a'};t=training_box_targets(1,'a',[[10,10,1,1],[10,10,1,1]],(100,200),allow,(2,2))
        self.assertTrue(t['known']);self.assertGreater(float(t['occupancy'].sum()),0);self.assertAlmostEqual(float(t['occupancy'].sum()),1/5000,places=8);torch.testing.assert_close(t['coverage'].max(-1).values,torch.ones(2))
        with self.assertRaises(ValueError):training_box_targets(2,'val',[],(100,200),allow,(2,2))
        unknown=training_box_targets(1,'a',[],(100,200),allow,(2,2));self.assertFalse(unknown['known']);logits=torch.randn(4,requires_grad=True);loss=balanced_proposal_loss(logits,unknown['occupancy'],False);self.assertEqual(float(loss.detach()),0)
        balanced=balanced_proposal_loss(torch.zeros(4),torch.tensor([.01,0.,0.,0.]));self.assertAlmostEqual(float(balanced),float(np.log(2)),places=6)
    def test_geometry_and_visit_gradient_direction(self):
        logits=torch.zeros(1,2,requires_grad=True);q=logits.softmax(-1);g=dict(background=torch.tensor([0.,1.]),edge=torch.zeros(2),rectangles=torch.tensor([[0.,0.,224.,224.],[224.,0.,448.,224.]]),geometry_valid=True);loss=free_losses(q,g,torch.zeros(2),torch.empty(0,2),torch.tensor([0]))['geometry'];loss.backward();self.assertGreater(float(logits.grad[0,1]),0)
        logits=torch.zeros(16,2,requires_grad=True);loss=visit_loss(logits.softmax(-1),torch.tensor([[1.,0.]]));loss.backward();self.assertLess(float(logits.grad[0,0]),0)
        logits=torch.full((1000,2),-5.,requires_grad=True);long=visit_loss(logits.softmax(-1),torch.tensor([[.00001,0.]]));long.backward();self.assertTrue(torch.isfinite(long));self.assertTrue(torch.isfinite(logits.grad).all())
    def test_visit_empty_and_all_padding_skip(self):
        logits=torch.randn(3,4,requires_grad=True);q=logits.softmax(-1);coverage=torch.tensor([[1.,0.,0.,0.]])
        loss=visit_loss(q,coverage,torch.zeros(3,dtype=torch.bool));self.assertEqual(float(loss.detach()),0);loss.backward();self.assertEqual(float(logits.grad.abs().sum()),0)
        empty=torch.empty(0,4,requires_grad=True);self.assertEqual(float(visit_loss(empty,coverage).detach()),0)
    def test_schedule_selection_and_short_ramp_rejection(self):
        import guided_trajectory_1011 as runner
        self.assertFalse(runner.selection_eligible(0));self.assertFalse(runner.selection_eligible(2/3));self.assertTrue(runner.selection_eligible(1));self.assertTrue(runner.selection_eligible(0,0))
        with self.assertRaises(ValueError):runner.validate_schedule(3,None)
        runner.validate_schedule(5,None);runner.validate_schedule(1,1)

    def test_masks_repetition_and_loss_finite(self):
        g=self.geometry();logits=torch.randn(5,4,requires_grad=True);q=logits.softmax(-1);mask=torch.tensor([True,True,False,False,False]);terms=free_losses(q,g,torch.zeros(4),torch.tensor([[1.,0.,0.,0.]]),torch.tensor([0,0,1,2,3]),mask);loss=sum(terms.values());loss.backward();self.assertEqual(float(logits.grad[2:].abs().sum()),0);self.assertGreater(float(terms['repeat'].detach()),0);self.assertTrue(torch.isfinite(loss))
    def test_causal_teacher_shared_step_and_proposal_detach(self):
        m=self.model(foveal=True);mem=self.memory();coords=torch.rand(1,5,2);patches=torch.randn(1,5,m.patch.in_features);g=self.geometry();teacher=m.teacher_guided(mem,coords,g,patches);changed=coords.clone();changed[:,2:]=torch.rand_like(changed[:,2:]);newpatch=patches.clone();newpatch[:,3:]=torch.randn_like(newpatch[:,3:]);torch.testing.assert_close(teacher[:,:3],m.teacher_guided(mem,changed,g,newpatch)[:,:3])
        hidden=None
        for t in range(5):value,hidden,_=m.guide_step(mem,coords[:,:t],g,hidden,patches[:,t]);torch.testing.assert_close(value,teacher[:,t])
        loss,_=objective(teacher,coords,torch.ones(1,5,dtype=torch.bool),grid=(2,2));loss.backward();self.assertIsNone(m.proposal.weight.grad);self.assertIsNone(m.proposal.bias.grad);self.assertGreater(float(mem['keys'].grad.abs().sum()),0)
        m.zero_grad();prop=balanced_proposal_loss(m.proposal(mem['keys']).squeeze(),torch.tensor([1.,0.,0.,0.]));prop.backward();self.assertGreater(float(m.proposal.weight.grad.abs().sum()),0)
    def test_soft_edge_and_invalid_proposal_fallback(self):
        m=self.model();mem=self.memory();g=self.geometry();prefix=torch.empty(1,0,2);q,_,diag=m.guide_step(mem,prefix,g);self.assertTrue(torch.isfinite(q).all());self.assertTrue((q.exp()>0).all())
        with torch.no_grad():m.proposal.weight.fill_(float('nan'))
        q,_,diag=m.guide_step(mem,prefix,g);self.assertTrue(torch.isfinite(q).all());self.assertEqual(float(diag['relevance'].sum()),0)
    def test_nonvisual_roundtrip_and_frozen_backbone(self):
        base=Trajectory1011(dim=8,grid=(2,2),pretrained=False);cp=dict(state=base.state_dict(),model_config=base.model_config);m=initialize_from_base(cp,rho=.6);state=nonvisual_state(m);self.assertFalse(any(k.startswith('visual.') for k in state));checkpoint=dict(kind='guided_trajectory_1011',model_config=m.model_config,state=state);restored=restore_guided(checkpoint,cp).eval();mem=self.memory();g=self.geometry();coords=torch.rand(1,3,2);torch.testing.assert_close(m.eval().teacher_guided(mem,coords,g),restored.teacher_guided(mem,coords,g));m.set_backbone('all');self.assertTrue(all(not p.requires_grad for p in m.visual.parameters()));m.train();self.assertFalse(m.visual.training);self.assertEqual([rho_schedule(i) for i in range(1,6)],[0.,0.,1/3,2/3,1.])
        bad=dict(checkpoint);bad['state']=dict(state);bad['state'].pop('proposal.weight')
        with self.assertRaises(ValueError):restore_guided(bad,cp)
    def test_predict_does_not_read_gaze_or_annotations(self):
        import guided_trajectory_1011 as runner
        m=self.model();image=Image.new('L',(100,200),128);mem=self.memory();g=self.geometry()
        with patch('src.data.load_gazes',side_effect=AssertionError('Observed gaze read')),patch.object(runner,'train_annotations',side_effect=AssertionError('Annotation read')):
            result=m.predicted_rollout(mem,image,g,steps=3,seed=42)
        with patch.object(runner,'train_annotations',return_value={'val_box':[0,0,100,200]}):
            changed=m.predicted_rollout(mem,image,g,steps=3,seed=42)
        torch.testing.assert_close(result['points'],changed['points']);torch.testing.assert_close(result['logp'],changed['logp'])
        self.assertEqual(result['points'].shape,(3,2));self.assertTrue(torch.isfinite(result['logp']).all());self.assertTrue(result['logp'].requires_grad)
        result['logp'].exp().square().sum().backward();self.assertIsNone(m.proposal.weight.grad)

if __name__=='__main__':unittest.main()
