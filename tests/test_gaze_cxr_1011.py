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
from src.gaze_cxr_1011 import GazeCXR1011,VARIANTS,inbounds_fraction,attention_loss,head_state,restore_head,validate_sources

class CXRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,variant='dual_cross_drop_quality'):
        return GazeCXR1011(GazeDiagnosisHead(input_dim=8,dim=16,mode='global',dropout=.5),variant=variant,input_dim=8,dim=8,dropout=0).eval()
    def inputs(self):return [torch.randn(3,8),torch.randn(3,5,8),torch.rand(3,5,2),torch.randn(3,6,8),torch.rand(3,6,2)]
    def test_initial_global_and_frozen_gradient(self):
        m=self.model();args=self.inputs();baseline=m.global_model(args[0])[0];m.train();out=m(*args)
        torch.testing.assert_close(out['logits'],baseline,rtol=0,atol=0);self.assertFalse(m.global_model.training);self.assertTrue(all(not p.requires_grad for p in m.global_model.parameters()))
        loss,_=attention_loss(out,torch.tensor([0.,1.,0.]));loss.backward();self.assertTrue(torch.isfinite(loss));self.assertGreater(float(m.value[1].weight.grad.abs().sum()),0);self.assertTrue(all(p.grad is None for p in m.global_model.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
    def test_cross_attention_auxiliary_gradients(self):
        m=self.model('dual_cross');m.train();args=self.inputs();out=m(*args);loss,_=attention_loss(out,torch.tensor([0.,1.,0.]));loss.backward()
        for module in (m.query,m.key,m.cross_score,m.global_score,m.score,m.source_gate):
            self.assertTrue(torch.isfinite(module.weight.grad).all());self.assertGreater(float(module.weight.grad.abs().sum()),0)

    def test_missing_masks_nan_and_exact_fallback(self):
        for variant in VARIANTS:
            m=self.model(variant);torch.nn.init.normal_(m.delta.weight);args=self.inputs();gmask=torch.zeros(3,5,dtype=torch.bool);args[1][:]=float('nan');args[2][:]=float('nan')
            out=m(*args,gaze_mask=gmask);self.assertTrue(torch.isfinite(out['logits']).all());self.assertEqual(float(out['beta'][:,0].sum().detach()),0);self.assertEqual(float(out['grid_context'].abs().sum().detach()),0)
            empty=m(*args,gaze_mask=gmask,force_grid_absent=True);torch.testing.assert_close(empty['logits'],empty['global_logits'],rtol=0,atol=0);self.assertEqual(float(empty['beta'].sum().detach()),0);self.assertEqual(float(empty['alpha'].sum().detach()),0)
            loss,parts=attention_loss(empty,torch.tensor([0.,1.,0.]));self.assertEqual(float(parts['auxiliary'].detach()),0);self.assertTrue(torch.isfinite(loss))
    def test_independent_permutations_and_both_duplicate_sources(self):
        for variant in VARIANTS:
            m=self.model(variant);torch.nn.init.normal_(m.delta.weight);torch.nn.init.normal_(m.delta.bias);args=self.inputs();baseline=m(*args)
            changed=[args[0],args[1][:,[4,0,3,1,2]],args[2][:,[4,0,3,1,2]],args[3][:,[5,2,0,4,1,3]],args[4][:,[5,2,0,4,1,3]]];torch.testing.assert_close(m(*changed)['logits'],baseline['logits'])
            for feature_index,coordinate_index,source in [(1,2,'gaze'),(3,4,'grid')]:
                duplicated=list(args);duplicated[feature_index]=torch.cat([args[feature_index],args[feature_index][:,:1]],1);duplicated[coordinate_index]=torch.cat([args[coordinate_index],args[coordinate_index][:,:1]],1);out=m(*duplicated)
                torch.testing.assert_close(out['logits'],baseline['logits']);torch.testing.assert_close(out['beta'],baseline['beta']);torch.testing.assert_close(out[source+'_attention'][:,0]+out[source+'_attention'][:,-1],baseline[source+'_attention'][:,0])
                opposite='grid' if source=='gaze' else 'gaze';torch.testing.assert_close(out[opposite+'_context'],baseline[opposite+'_context'])
    def test_quality_geometry_and_prior(self):
        coords=np.array([[0,0],[.5,.5],[1.,.5],[-1.,.5]])
        np.testing.assert_allclose(inbounds_fraction(coords),[.25,1.,.5,0.]);torch.testing.assert_close(inbounds_fraction(torch.tensor(coords)),torch.tensor([.25,1.,.5,0.],dtype=torch.float64))
        for xy in coords:
            cx,cy=np.round(xy*[1500,3000]).astype(int);width=max(0,min(cx+112,1500)-max(cx-112,0));height=max(0,min(cy+112,3000)-max(cy-112,0));self.assertAlmostEqual(float(inbounds_fraction(xy)),width*height/224**2)
        m=self.model();args=self.inputs();args[1]=torch.ones(3,5,8);args[2]=torch.tensor([[[0.,0.],[.5,.5],[.6,.6],[.7,.7],[.8,.8]]]).expand(3,-1,-1);out=m(*args,uniform=True)
        ratio=out['gaze_attention'][:,0]/out['gaze_attention'][:,1];torch.testing.assert_close(ratio,torch.full((3,),.25))
        torch.nn.init.zeros_(m.query.weight);torch.nn.init.zeros_(m.key.weight)
        _,cross_weights=m.cross_context(torch.ones(1,1,8),torch.full((1,1,2),.5),torch.ones(1,1,dtype=torch.bool),torch.ones(1,2,8),torch.tensor([[[0.,0.],[.5,.5]]]),torch.ones(1,2,dtype=torch.bool),torch.ones(1,2),torch.tensor([[.25,1.]]))
        torch.testing.assert_close(cross_weights[0,0,0]/cross_weights[0,0,1],torch.tensor(.25))
        plain=self.model('dual_cross');plain.load_state_dict(m.state_dict());args=self.inputs();args[2]=.2+.6*args[2];args[4]=.2+.6*args[4]
        torch.testing.assert_close(m(*args)['logits'],plain(*args)['logits'])
    def test_quality_float32_boundary_matches_numpy_crop(self):
        coordinates=np.array([[(pixel+.5)/1500,.0005] for pixel in range(112)],dtype=np.float32)
        centers=np.round(coordinates*np.array([1500,3000])).astype(int);lo=centers-112;hi=lo+224
        intersection=np.maximum(0,np.minimum(hi,[1500,3000])-np.maximum(lo,[0,0]));expected=intersection.prod(-1)/224**2
        np.testing.assert_allclose(inbounds_fraction(torch.from_numpy(coordinates)).numpy(),expected,rtol=1e-7)
        np.testing.assert_allclose(inbounds_fraction(coordinates),expected,rtol=0,atol=0)

    def test_modality_dropout_reproducible_and_eval(self):
        m=self.model('dual_cross_drop');m.train();args=self.inputs();torch.manual_seed(42);a=m(*args);torch.manual_seed(42);b=m(*args);torch.testing.assert_close(a['logits'],b['logits']);self.assertTrue(torch.equal(a['gaze_mask'],b['gaze_mask']));self.assertTrue(a['grid_mask'].all())
        m.modality_dropout=1.;forced=m(*args);self.assertFalse(forced['gaze_mask'].any());m.eval();self.assertTrue(m(*args)['gaze_mask'].all());torch.testing.assert_close(m(*args)['logits'],m(*args)['logits'])
    def test_signed_checkpoint_roundtrip_and_mean_parameters(self):
        m=self.model();torch.nn.init.normal_(m.delta.weight);torch.nn.init.normal_(m.delta.bias);args=self.inputs();out=m(*args)
        torch.testing.assert_close(out['gaze_signed'].sum(-1)+out['grid_signed'].sum(-1)+out['residual_bias'],out['residual'])
        cp=dict(kind='gaze_cxr_1011',state=head_state(m),model_config=m.config);global_cp=dict(model_config=m.global_model.config,state=m.global_model.state_dict());restored=restore_head(cp,global_cp).eval();torch.testing.assert_close(restored(*args)['logits'],out['logits'])
        self.assertFalse(any(key.startswith('global_model.') for key in cp['state']));self.assertLess(sum(p.numel() for p in self.model('dual_mean').parameters()),sum(p.numel() for p in m.parameters()))
    def test_image_only_and_population_mismatch(self):
        class FakeGaze(torch.nn.Module):
            input_size=(16,32)
            def __init__(self):super().__init__();self.reference=torch.nn.Parameter(torch.zeros(1));self.cell_xy=torch.tensor([[.2,.2],[.8,.8]]);self.visual=self
            def eval(self):return self
            def maps(self,image):return {}
            def patch_vector(self,image):return torch.ones(len(image),8)
            def step_history(self,memory,prefix,hidden,vector):return torch.tensor([[1.,0.]]),None,{}
        with patch('src.data.load_gazes',side_effect=AssertionError('Actual gaze accessed')):features=image_features(FakeGaze(),Image.new('L',(1500,3000),128),64,steps=16,include_grid=True)
        args=[features[key][None] for key in ('global_features','gaze_features','gaze_coordinates','grid_features','grid_coordinates')];self.assertTrue(torch.isfinite(self.model()(*args)['logits']).all())
        frame=pd.DataFrame([dict(image_id=1,patient_key='a',split='train',label=0,file_name='x')]);payload=dict(rows=frame.to_dict('records'),provenance={})
        with self.assertRaises(ValueError):validate_sources(payload,{},frame.assign(patient_key='wrong'))

if __name__=='__main__':unittest.main()
