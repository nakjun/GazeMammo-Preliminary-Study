import sys
import unittest
import argparse
import tempfile
import json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.data import reduce_gaze,select_centers,patch_tensor
from src.metrics import threshold_at_specificity,cluster_bootstrap
from src.models import PatchClassifier,TrajectoryModel
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from pipeline import validate_prediction_provenance

class PipelineTests(unittest.TestCase):
    def test_recording_gap_does_not_create_transition(self):
        segments=reduce_gaze([[10,10],[15,15],[-1,0],[90,90],[95,95]],100,100,16,1)
        self.assertEqual([len(x) for x in segments],[2,2])
        self.assertTrue(np.array_equal(segments[1][0],[90,90]))
    def test_crop_keeps_center_and_pads_boundary(self):
        im=Image.fromarray(np.full((300,150),255,dtype=np.uint8))
        t=patch_tensor(im,[0,0]);self.assertEqual(tuple(t.shape),(3,224,224))
        self.assertLess(float(t[:,0,0].mean()),float(t[:,-1,-1].mean()))
    def test_selectors_are_deterministic_and_foreground_only(self):
        a=np.zeros((400,200),dtype=np.uint8);a[:,32:160]=100;im=Image.fromarray(a)
        for method in ['random','grid']:
            p=select_centers(im,method,16,42);q=select_centers(im,method,16,42)
            self.assertTrue(np.array_equal(p,q));self.assertTrue((p[:,0]>=32).all())
            self.assertEqual(len(p),16)
    def test_masked_patch_does_not_change_mil_output(self):
        torch.manual_seed(0);m=PatchClassifier().eval();x=torch.randn(1,3,2048);mask=torch.tensor([[True,True,False]])
        with torch.no_grad():
            a,w=m(x,mask);x[:,2]=100000;b,_=m(x,mask)
        self.assertTrue(torch.allclose(a,b));self.assertEqual(float(w[0,2]),0.)
    def test_decoder_is_causal(self):
        torch.manual_seed(0);m=TrajectoryModel(grid=(4,8),k=4,pretrained=False).eval()
        memory=torch.randn(1,8,128);a=torch.tensor([[32,1,2,3]]);b=torch.tensor([[32,1,20,21]])
        with torch.no_grad():x=m.decode(memory,a);y=m.decode(memory,b)
        self.assertTrue(torch.allclose(x[:,:2],y[:,:2],atol=1e-6))
    def test_backbone_channels_and_cached_forward(self):
        from src.models import BACKBONES
        torch.set_num_threads(2)
        for name,channels in BACKBONES.items():
            with self.subTest(backbone=name):
                model=TrajectoryModel(grid=(4,8),k=4,pretrained=False,backbone_name=name).eval()
                image=torch.randn(1,3,128,64);target=torch.tensor([[1,2,3,4]])
                with torch.no_grad():
                    visual=model.visual(image)
                    self.assertEqual(visual.shape[1],channels)
                    a=model(image,target);b=model.forward_features(visual,target)
                self.assertTrue(torch.allclose(a,b,atol=1e-6))
                self.assertFalse(any(p.requires_grad for p in model.visual.parameters()))
    def test_specificity_threshold_handles_ties(self):
        y=[0,0,0,0,1];p=[.5,.5,.5,.5,.7];t=threshold_at_specificity(y,p,.9)
        self.assertGreater(t,.5);self.assertGreaterEqual((np.array(p[:4])<t).mean(),.9)
    def test_sampling_is_reproducible_and_in_bounds(self):
        model=TrajectoryModel(grid=(4,8),k=4,pretrained=False).eval();image=torch.randn(1,3,128,64)
        torch.manual_seed(42);a=model.generate(image,strategy='sample')
        torch.manual_seed(42);b=model.generate(image,strategy='sample')
        self.assertTrue(torch.equal(a,b));self.assertEqual(tuple(a.shape),(1,4,2))
        self.assertTrue(bool(((a>0)&(a<1)).all()))
    def test_large_decoder_checkpoint_restores_architecture(self):
        from src.models import trajectory_from_checkpoint
        model=TrajectoryModel(grid=(4,8),k=4,dim=512,layers=6,heads=8,ff_multiplier=4,dropout=.2,pretrained=False).eval()
        c=dict(grid=(4,8),k=4,model_config=model.model_config,state=model.state_dict())
        restored=trajectory_from_checkpoint(c).eval()
        memory=torch.randn(1,8,512);inputs=torch.tensor([[32,1,2,3]])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model.decode(memory,inputs),restored.decode(memory,inputs)))
        self.assertEqual(len(restored.decoder.layers),6)
    def test_invalid_decoder_head_dimension_is_rejected(self):
        with self.assertRaises(ValueError):TrajectoryModel(dim=130,heads=8,pretrained=False)
    def test_legacy_checkpoint_without_model_config(self):
        from src.models import trajectory_from_checkpoint
        model=TrajectoryModel(grid=(4,8),k=4,pretrained=False)
        restored=trajectory_from_checkpoint(dict(grid=(4,8),k=4,state=model.state_dict()))
        self.assertEqual(restored.dim,128);self.assertEqual(len(restored.decoder.layers),2)
    def test_finetuning_updates_backbone_and_rejects_feature_cache(self):
        model=TrajectoryModel(grid=(4,8),k=4,pretrained=False,finetune_backbone=True).train()
        p=next(model.visual.parameters());before=p.detach().clone()
        optimizer=torch.optim.SGD(model.parameters(),lr=.001)
        logits=model(torch.randn(1,3,128,64),torch.tensor([[1,2,3,4]]));logits.square().mean().backward()
        self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.abs().sum()),0)
        optimizer.step();self.assertFalse(torch.equal(before,p))
        with self.assertRaises(ValueError):model.forward_features(torch.randn(1,2048,4,2),torch.tensor([[1,2,3,4]]))
    def test_joint_patch_encoder_receives_classification_gradient(self):
        from src.models import JointPatchClassifier
        model=JointPatchClassifier(k=2,pretrained=False).train();x=torch.randn(1,2,3,64,64);mask=torch.tensor([[True,False]])
        logits,w=model(x,mask);logits.sum().backward()
        self.assertGreater(float(next(model.visual.parameters()).grad.abs().sum()),0)
        self.assertEqual(float(w[0,1]),0.)
    def test_bootstrap_pairs_by_image_id(self):
        a=pd.DataFrame(dict(image_id=[1,2,3,4],patient_key=['a','a','b','c'],label=[0,1,0,1],probability=[.1,.9,.2,.8]))
        result=cluster_bootstrap(a,a.iloc[::-1],100)
        self.assertEqual(result['low'],0);self.assertEqual(result['high'],0)
    def test_patient_split_is_disjoint(self):
        p=Path(__file__).resolve().parents[1]/'manifests/all.csv'
        if not p.exists():self.skipTest('Run prepare first')
        f=pd.read_csv(p);self.assertTrue((f.groupby('patient_key').split.nunique()==1).all())
        self.assertTrue((f[f.source=='INbreast'].split=='quarantine').all())
    def test_prediction_provenance_rejects_training_patient(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'predictions.json';path.write_text('{}')
            meta={'method':'trajectory','generated_image_ids':[1],'training_patient_keys':['patient_a']}
            path.with_suffix('.provenance.json').write_text(json.dumps(meta))
            args=argparse.Namespace(method='trajectory',predictions=str(path))
            frame=pd.DataFrame(dict(image_id=[1],patient_key=['patient_a']))
            with self.assertRaisesRegex(ValueError,'trained on patient'):validate_prediction_provenance(args,frame)
            frame.patient_key='patient_b';validate_prediction_provenance(args,frame)
            meta.update(generation_policy='hard',generation_radius=112,generation_strategy='sample')
            path.with_suffix('.provenance.json').write_text(json.dumps(meta))
            args.generation_policy='hard';args.generation_radius=112;args.generation_strategy='sample'
            validate_prediction_provenance(args,frame)
            args.generation_policy='none'
            with self.assertRaisesRegex(ValueError,'generation_policy'):validate_prediction_provenance(args,frame)
    def test_image_only_inference_dataset_does_not_load_gaze(self):
        from unittest.mock import patch
        from src.data import ImageDataset
        p=Path(__file__).resolve().parents[1]/'manifests/classification.csv'
        if not p.exists():self.skipTest('Run prepare first')
        with patch('src.data.load_gazes',side_effect=AssertionError('Gaze access during inference')):
            item=ImageDataset(pd.read_csv(p).head(1))[0]
        self.assertEqual(tuple(item['image'].shape),(3,512,256))

if __name__=='__main__':unittest.main()
