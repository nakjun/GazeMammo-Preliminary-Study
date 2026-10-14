import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import torch
from src.trajectory_1011 import Trajectory1011,window_sample,segments,objective,restore,initialize,check_provenance

class TrajectoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def model(self,**kwargs):return Trajectory1011(dim=16,grid=(3,4),pretrained=False,**kwargs).eval()
    def memory(self):return dict(keys=torch.randn(2,12,16),prior=torch.randn(2,12),global_context=torch.randn(2,16))
    def test_windows(self):
        points=np.array([[0,0],[40,0],[np.nan,0],[80,0],[120,0],[160,0]],float)
        parts=segments(points,200,200);self.assertEqual([len(p) for p in parts],[2,3])
        s=window_sample(parts,1,length=2,burn_in=1,start=1,segment=1)
        self.assertEqual(s['coords'][:3].tolist(),parts[1].tolist());self.assertEqual(s['loss_mask'][:4].tolist(),[False,True,True,False])
        a=window_sample([np.arange(200).reshape(100,2)],1,epoch=2);b=window_sample([np.arange(200).reshape(100,2)],1,epoch=2)
        self.assertTrue(torch.equal(a['coords'],b['coords']))
        starts={window_sample([np.arange(200).reshape(100,2)],1,epoch=e)['start'] for e in range(10)};self.assertGreater(len(starts),1)
    def test_causal_probabilities_gradient(self):
        m=self.model(spatial_history=True,behavior_mixture=True);memory=self.memory();coords=torch.rand(2,5,2)
        teacher,gates=m.teacher_logits(memory,coords,return_diagnostics=True);changed=coords.clone();changed[:,2:]=torch.rand_like(changed[:,2:])
        self.assertTrue(torch.allclose(teacher[:,:3],m.teacher_logits(memory,changed)[:,:3]))
        hidden=None
        for i in range(5):
            value,hidden,diag=m.step_history(memory,coords[:,:i],hidden)
            self.assertTrue(torch.allclose(value,teacher[:,i]));self.assertTrue(torch.allclose(diag['branches'].exp().sum(-1),torch.ones_like(diag['branches'][:,:,0]),atol=1e-6))
        self.assertTrue(torch.allclose(teacher.exp().sum(-1),torch.ones(2,5),atol=1e-6));self.assertTrue((gates[:,0,1:]==0).all())
        loss,_=objective(teacher,coords,torch.ones(2,5,dtype=torch.bool),grid=m.grid);loss.backward()
        for name in ('behavior_gate.weight','history_head.0.weight','local_scale','return_scale','return_decay'):
            self.assertTrue(torch.isfinite(dict(m.named_parameters())[name].grad).all());self.assertGreater(float(dict(m.named_parameters())[name].grad.abs().sum()),0)
    def test_future_crop_causality(self):
        m=self.model(spatial_history=True,behavior_mixture=True,foveal=True,motion_bias=True)
        memory=self.memory();coords=torch.rand(2,5,2);patches=torch.randn(2,5,m.patch.in_features)
        teacher=m.teacher_logits(memory,coords,patches);changed=patches.clone();changed[:,3:]=torch.randn_like(changed[:,3:])
        self.assertTrue(torch.allclose(teacher[:,:3],m.teacher_logits(memory,coords,changed)[:,:3]))
        hidden=None
        for i in range(5):
            value,hidden,_=m.step_history(memory,coords[:,:i],hidden,patches[:,i])
            self.assertTrue(torch.allclose(value,teacher[:,i]))

    def test_mask(self):
        logits=torch.randn(1,4,12,requires_grad=True);coords=torch.rand(1,4,2);mask=torch.tensor([[False,True,True,False]])
        loss,_=objective(logits,coords,mask,grid=(3,4));loss.backward();self.assertEqual(float(logits.grad[:,[0,3]].abs().sum()),0)
    def test_restore_and_legacy(self):
        m=self.model(spatial_history=True,behavior_mixture=True);cp=dict(kind='trajectory_1011',model_config=m.model_config,state=m.state_dict());other=restore(cp)
        memory=self.memory();coords=torch.rand(2,3,2);self.assertTrue(torch.equal(m.teacher_logits(memory,coords),other.teacher_logits(memory,coords)))
        legacy=self.model();report=initialize(m,dict(kind='spatial_v2',model_config=legacy.model_config,state=legacy.state_dict()));self.assertTrue(report['missing'])
        bad=dict(kind='spatial_v2',model_config={**legacy.model_config,'fpn':True},state=legacy.state_dict())
        with self.assertRaises(ValueError):initialize(m,bad)
    def test_evaluation_cache_equivalence(self):
        import importlib.util,tempfile,argparse
        from unittest.mock import patch
        import pandas as pd
        from PIL import Image
        spec=importlib.util.spec_from_file_location('cache_runner',Path(__file__).resolve().parents[1]/'scripts/trajectory_1011.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data/image').mkdir(parents=True);Image.new('L',(64,128),128).save(root/'data/image/a.png')
            m=Trajectory1011(dim=16,input_size=(32,64),pretrained=False,spatial_history=True,behavior_mixture=True,foveal=True).eval()
            a=argparse.Namespace(seed=42,window=3,burn_in=2,mixed_ratio=.5)
            data=object.__new__(runner.Data);data.a=a
            data.items=[dict(row=pd.Series(dict(image_id=1,patient_key='b',file_name='a.png')),image=torch.randn(3,64,32),parts=[np.linspace(.1,.9,14)[:,None].repeat(2,axis=1),np.array([[.2,.3],[.3,.4]])],size=(64,128))]
            prior=torch.ones(1152)/1152
            with patch.object(runner,'ROOT',root):
                cached,cached_rows=runner.evaluate_data(m,data,torch.device('cpu'),prior)
                original,original_rows=runner.evaluate_data(m,data,torch.device('cpu'),prior,cache_features=False)
            for mode in ('legacy16','contiguous'):
                self.assertAlmostEqual(cached[mode]['raw_nll'],original[mode]['raw_nll'],places=5)
                for left,right in zip(cached_rows[mode+'_steps'],original_rows[mode+'_steps']):
                    self.assertAlmostEqual(left['raw_nll'],right['raw_nll'],places=5)
                    for key in ('gate_global','gate_local','gate_return'):self.assertAlmostEqual(left[key],right[key],places=6)

    def test_image_only_generation(self):
        import importlib.util,tempfile
        from unittest.mock import patch
        import pandas as pd
        from PIL import Image
        spec=importlib.util.spec_from_file_location('runner_1011',Path(__file__).resolve().parents[1]/'scripts/trajectory_1011.py')
        runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data/image').mkdir(parents=True);Image.new('L',(64,128),128).save(root/'data/image/a.png')
            m=Trajectory1011(dim=16,input_size=(32,64),pretrained=False,spatial_history=True,behavior_mixture=True)
            cp=dict(kind='trajectory_1011',state=m.state_dict(),model_config=m.model_config,args=dict(window=3,burn_in=1,min_distance=32,sigma=64),training_patient_keys=['a'],selection_patient_keys=['b'],train_prior=torch.ones(1152)/1152)
            torch.save(cp,root/'checkpoint.pt')
            def frame(split,*args,**kwargs):return pd.DataFrame([dict(patient_key='a' if split=='train' else 'b',image_id=1,file_name='a.png')])
            a=runner.parser().parse_args(['generate','--checkpoint',str(root/'checkpoint.pt'),'--output',str(root/'out'),'--limit','1','--steps','3'])
            with patch.object(runner,'ROOT',root),patch.object(runner,'rows',side_effect=frame),patch.object(runner,'load_gazes',side_effect=AssertionError('Gaze access')):runner.run(a)
            import json
            self.assertEqual(len(json.loads((root/'out/predictions.json').read_text())['1']),3)
            self.assertEqual(json.loads((root/'out/preprocessing.json').read_text())['effective']['window'],3)

    def test_provenance(self):
        cp=dict(training_patient_keys=['a'],selection_patient_keys=['b']);check_provenance(cp,['a'],['b'],['b'])
        with self.assertRaises(ValueError):check_provenance(cp,['a'],['b'],['b'],True)
        with self.assertRaises(ValueError):check_provenance(cp,['z'],['b'])

if __name__=='__main__':unittest.main()
