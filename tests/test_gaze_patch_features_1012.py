import inspect
import copy
import json
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.gaze_patch_features_1012 import extract_image_features,validate_development_population,_patient_proof,validate_feature_cache,DEFAULT_CONFIG,ARMS,SOURCE_SHA256
from src.proposal_center_offset_1011 import offset_rectangles

class Encoder(torch.nn.Module):
    def patch_vector(self,x): return x.mean((1,2,3))[:,None].expand(-1,1024)
class Probability(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(1));self.visual=torch.nn.Module();self.visual.encoders=torch.nn.ModuleList([Encoder()])
class FeatureTests(unittest.TestCase):
    def test_pixel_only_signature_and_random_determinism(self):
        self.assertEqual(list(inspect.signature(extract_image_features).parameters),['bundle','image','image_id','seed','arms'])
        image=Image.fromarray(np.full((256,256),120,dtype=np.uint8));bundle=(Probability(),None,None,{})
        a=extract_image_features(bundle,image,99,arms=('random','grid'));b=extract_image_features(bundle,image,99,arms=('random','grid'))
        for arm in ('random','grid'):
            self.assertTrue(torch.equal(a[arm]['coordinates'],b[arm]['coordinates']))
            self.assertEqual(a[arm]['features'].shape,(16,1024));self.assertTrue(a[arm]['mask'].all())
        self.assertTrue(torch.equal(a['global_features'],b['global_features']))
        with self.assertRaises(ValueError):extract_image_features(bundle,image,'MALIGNANT.png',arms=('random',))
    def test_raw_causal_history_has_no_roi_access(self):
        class Raw(Probability):
            def __init__(self):
                super().__init__();self.foveal=True;self.cell_xy=torch.tensor([[.25,.25],[.75,.75]]);self.seen=[]
                self.visual.patch_vector=self.visual.encoders[0].patch_vector
            def maps(self,image): return {'prior':torch.zeros(1,2)}
            def step_history(self,memory,prefix,state,vector):
                self.seen.append((prefix.clone(),None if vector is None else vector.clone()))
                return torch.tensor([[0.,-100.]]),None,{}
        raw=Raw();image=Image.fromarray(np.full((256,256),120,dtype=np.uint8))
        features=extract_image_features((raw,object(),object(),{}),image,9,arms=('gaze',))
        self.assertEqual([len(p[0]) for p,v in raw.seen],list(range(16)))
        self.assertIsNone(raw.seen[0][1]);self.assertTrue(all(v is not None for p,v in raw.seen[1:]))
        self.assertTrue(torch.equal(raw.seen[-1][0][0],features['gaze']['coordinates'][:15]))
    def test_proposal_does_not_call_probability(self):
        class Forbidden(Probability):
            def maps(self,*a): raise AssertionError('RAW maps invoked')
            def step_history(self,*a): raise AssertionError('RAW history invoked')
        class Guided:
            input_size=(256,512)
            cell_xy=torch.tensor([[i/16+.01,.5] for i in range(16)])
            def maps(self,x): return {'keys':torch.zeros(1,16,3)}
            def proposal(self,keys): return torch.zeros(1,16,1)
        image=Image.fromarray(np.full((256,256),120,dtype=np.uint8))
        with patch('src.gaze_patch_features_1012.predict_center_offsets',return_value=torch.zeros(1,16,2)),patch('src.gaze_patch_features_1012.proposal_budget_indices',return_value=list(range(16))):
            result=extract_image_features((Forbidden(),Guided(),None,{'config':{'roi':{'strength':.5,'nms':.15}}}),image,9,arms=('proposal',))
        self.assertFalse(result['selection_metadata']['proposal']['gaze_component_used'])
    def test_cache_rejects_mislabeled_row_and_source(self):
        manifest=pd.read_csv('manifests/all.csv');selected=manifest[(manifest.source=='CBIS-DDSM') & manifest.split.isin(['train','val']) & manifest.label.isin([0,1])]
        rows=pd.concat([selected[(selected.split==split)&(selected.label==label)].head(2) for split in ('train','val') for label in (0,1)]).to_dict('records')
        for r in rows:r.update(image_size=[1500,3000],image_path='unused',image_sha256='x')
        tensors={'global_features':torch.zeros(8,1024)}
        for arm in ARMS:
            tensors[arm+'_features']=torch.zeros(8,16,1024);tensors[arm+'_coordinates']=torch.zeros(8,16,2);tensors[arm+'_mask']=torch.ones(8,16,dtype=torch.bool);tensors[arm+'_rectangles']=torch.tensor([-112,-112,112,112]).expand(8,16,4)
        config=json.loads(DEFAULT_CONFIG.read_text())
        provenance=dict(source_config_sha256=SOURCE_SHA256,crop_size=224,feature_dim=1024,arms=list(ARMS),path_seed=42,source_config_path=str(DEFAULT_CONFIG),classification_encoder_sha256=config['config']['probability']['parents'][0]['visual_state_sha256'],manifest_path='manifests/all.csv',manifest_sha256='x',population='smoke8',dependencies=[],source_code_sha256={},source_config_file_sha256='x')
        cache=dict(kind='gaze_patch_classification_features_1012',rows=rows,tensors=tensors,provenance=provenance)
        with patch('src.gaze_patch_features_1012.file_sha256',return_value='x'):
            validate_feature_cache(cache)
            bad=copy.deepcopy(cache);bad['rows'][0]['label']=1
            with self.assertRaises(ValueError):validate_feature_cache(bad)
            bad=copy.deepcopy(cache);bad['provenance']['classification_encoder_sha256']='bad'
            with self.assertRaises(ValueError):validate_feature_cache(bad)
    def test_bankers_round_and_padding_rectangles(self):
        c=torch.tensor([[.125,.375]],dtype=torch.float64)
        self.assertEqual(offset_rectangles(c,(4,4)).tolist(),[[-112.,-110.,112.,114.]])
    def test_nested_patient_rejection(self):
        with self.assertRaises(ValueError):_patient_proof({'initializer_provenance':{'training_patient_keys':['TEST']}},{'T'},{'V'})
        _patient_proof({'selection_patient_keys':['V'],'training_patient_keys':['T']},{'T'},{'V'})
    def test_population_and_unknown_exclusion(self):
        rows=[];identity=0
        for split,label,n in [('train',0,187),('train',1,189),('val',0,46),('val',1,47)]:
            for i in range(n):identity+=1;rows.append(dict(image_id=identity,patient_key=split+str(i),split=split,label=label,source='CBIS-DDSM',file_name=str(identity)))
        rows.append(dict(image_id=999,patient_key='Q',split='quarantine',label=1,source='INbreast',file_name='x'))
        rows.append(dict(image_id=998,patient_key='U',split='train',label=-1,source='CBIS-DDSM',file_name='x'))
        self.assertEqual(len(validate_development_population(pd.DataFrame(rows),{})),469)
        rows[-1]['patient_key']='val0'
        with self.assertRaises(ValueError):validate_development_population(pd.DataFrame(rows),{})
if __name__=='__main__':unittest.main()
