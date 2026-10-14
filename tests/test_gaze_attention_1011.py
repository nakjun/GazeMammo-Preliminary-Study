import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.gaze_classifier_1011 import GazeDiagnosisHead,image_features
from src.gaze_attention_1011 import GazeAttention1011,attention_loss,head_state,restore_head,validate_sources

class AttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,variant='contextual_residual'):
        baseline=GazeDiagnosisHead(input_dim=8,dim=16,mode='global',dropout=.5)
        return GazeAttention1011(baseline,variant=variant,input_dim=8,dim=8).eval()
    def inputs(self):return torch.randn(3,8),torch.randn(3,5,8),torch.rand(3,5,2)
    def test_initial_global_freeze(self):
        m=self.model();g,x,c=self.inputs();baseline=m.global_model(g)[0];m.train();out=m(g,x,c)
        self.assertTrue(torch.equal(out['logits'],baseline));self.assertTrue(torch.equal(out['residual'],torch.zeros(3)))
        self.assertFalse(m.global_model.training)
        self.assertTrue(all(not module.training for module in m.global_model.modules()));self.assertTrue(all(not p.requires_grad for p in m.global_model.parameters()))
        torch.testing.assert_close(out['alpha'],torch.full((3,),.1))
    def test_loss_gradient_auxiliary_and_invalid(self):
        m=self.model();m.train();g,x,c=self.inputs();mask=torch.ones(3,5,dtype=torch.bool);mask[0]=False
        out=m(g,x,c,mask);loss,parts=attention_loss(out,torch.tensor([0.,1.,0.]));loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.value[1].weight.grad.abs().sum()),0)
        self.assertGreater(float(m.score.weight.grad.abs().sum()),0);self.assertTrue(all(p.grad is None for p in m.global_model.parameters()))
        empty=m(g,x,c,torch.zeros_like(mask));empty_loss,empty_parts=attention_loss(empty,torch.tensor([0.,1.,0.]));self.assertEqual(float(empty_parts['auxiliary'].detach()),0)
        self.assertTrue(torch.equal(empty['logits'],empty['global_logits']));self.assertEqual(float(empty['attention'].sum().detach()),0);self.assertEqual(float(empty['alpha'].sum().detach()),0);self.assertEqual(float(empty['local_logits'].sum().detach()),0)
        self.assertTrue(torch.isfinite(empty_loss))
    def test_permutation_duplicates_and_mask(self):
        for variant in ('mean_residual','gated_residual','contextual_residual','grid_contextual_residual'):
            m=self.model(variant);torch.nn.init.normal_(m.delta.weight);torch.nn.init.normal_(m.delta.bias);g,x,c=self.inputs();mask=torch.ones(3,5,dtype=torch.bool);mask[:,4]=False
            original=m(g,x,c,mask);perm=torch.tensor([3,1,4,0,2]);changed=m(g,x[:,perm],c[:,perm],mask[:,perm]);torch.testing.assert_close(original['logits'],changed['logits'])
            self.assertEqual(float(original['attention'][:,4].sum().detach()),0)
            copied=m(g,torch.cat([x,x[:,:1]],1),torch.cat([c,c[:,:1]],1),torch.cat([mask,mask[:,:1]],1))
            torch.testing.assert_close(original['logits'],copied['logits']);torch.testing.assert_close(original['attention'][:,0],copied['attention'][:,0]+copied['attention'][:,-1])
            torch.testing.assert_close(copied['attention'].sum(-1),torch.ones(3))
    def test_signed_and_roundtrip(self):
        m=self.model();torch.nn.init.normal_(m.delta.weight);torch.nn.init.normal_(m.delta.bias);g,x,c=self.inputs();out=m(g,x,c)
        torch.testing.assert_close(out['signed_contribution'].sum(-1)+out['residual_bias'],out['residual'])
        cp=dict(kind='gaze_attention_1011',state=head_state(m),model_config=m.config);global_cp=dict(model_config=m.global_model.config,state=m.global_model.state_dict())
        self.assertFalse(any(key.startswith('global_model') for key in cp['state']))
        restored=restore_head(cp,global_cp).eval();torch.testing.assert_close(restored(g,x,c)['logits'],out['logits'])
    def test_sources_and_mismatch(self):
        frame=pd.DataFrame([dict(image_id=i,patient_key=str(i),split='train' if i<2 else 'val',label=i%2,file_name=f'{i}.png') for i in range(4)])
        provenance=dict(gaze_training_patient_keys=['0','1'],gaze_selection_patient_keys=['2','3'],gaze_checkpoint_sha256='a'*64,final_test_used=False,seed=42,steps=16,strategy='sample',policy='none',patch_size=224)
        cp=dict(training_patient_keys=['0','1'],selection_patient_keys=['2','3'],classifier_training_patient_keys=['0','1'],gaze_provenance=provenance)
        tensors=dict(global_features=torch.randn(4,1024),gaze_features=torch.randn(4,16,1024),grid_features=torch.randn(4,16,1024),gaze_coordinates=torch.rand(4,16,2),grid_coordinates=torch.rand(4,16,2));payload=dict(rows=frame.to_dict('records'),provenance=provenance,tensors=tensors)
        validate_sources(payload,cp,frame)
        bad=frame.copy();bad.loc[0,'patient_key']='test_patient'
        with self.assertRaises(ValueError):validate_sources(payload,cp,bad)
        tensors['gaze_coordinates'][0,0,0]=1.1
        with self.assertRaises(ValueError):validate_sources(payload,cp,frame)
    def test_image_only_features(self):
        class FakeGaze(torch.nn.Module):
            input_size=(16,32)
            def __init__(self):
                super().__init__();self.reference=torch.nn.Parameter(torch.zeros(1));self.cell_xy=torch.tensor([[.2,.2],[.8,.8]]);self.visual=self
            def eval(self):return self
            def maps(self,image):return {}
            def patch_vector(self,image):return torch.ones(len(image),8)
            def step_history(self,memory,prefix,hidden,vector):return torch.tensor([[1.,0.]]),None,{}
        gaze=FakeGaze()
        with patch('src.data.load_gazes',side_effect=AssertionError('Gaze read')):features=image_features(gaze,Image.new('L',(1500,3000),128),64,steps=5,include_grid=False)
        m=self.model();result=m(features['global_features'][None],features['gaze_features'][None],features['gaze_coordinates'][None]);self.assertTrue(torch.isfinite(result['logits']).all())

if __name__=='__main__':unittest.main()
