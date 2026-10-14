import copy
import numpy as np
import pandas as pd
import unittest
import tempfile
from pathlib import Path
import torch
from src.gaze_classifier_1011 import GazeDiagnosisHead
from src.gaze_attention_1011 import GazeAttention1011
from src.gaze_patch_classifier_1012 import state_digest,restore,train_head,evaluate,metrics,paired_ci


def fixture():
    torch.manual_seed(4)
    frame=pd.DataFrame(dict(image_id=range(8),patient_key=['p'+str(i) for i in range(8)],split=['train']*4+['val']*4,label=[0,1,0,1]*2))
    tensors={'global_features':torch.randn(8,1024),'gaze_features':torch.randn(8,16,1024),'gaze_coordinates':torch.rand(8,16,2),'gaze_mask':torch.ones(8,16,dtype=torch.bool),'gaze_rectangles':torch.ones(8,16,4)}
    return frame,tensors


def test_mean_duplicate_permutation_and_fallback():
    torch.set_num_threads(1)
    model=GazeAttention1011(GazeDiagnosisHead(mode='global'),variant='mean_residual').eval()
    with torch.no_grad(): model.delta.weight.fill_(.02)
    g=torch.randn(2,1024);f=torch.randn(2,3,1024);c=torch.rand(2,3,2)
    a=model(g,f,c);order=torch.tensor([2,0,1]);b=model(g,f[:,order],c[:,order])
    assert torch.allclose(a['logits'],b['logits'],atol=1e-6)
    d=model(g,torch.cat([f,f[:,:1]],1),torch.cat([c,c[:,:1]],1))
    assert torch.allclose(a['logits'],d['logits'],atol=1e-6)
    empty=model(g,f,c,torch.zeros(2,3,dtype=torch.bool))
    assert torch.equal(empty['logits'],empty['global_logits'])
    assert not empty['attention'].any()


def test_mock_training_checkpoint_and_frozen_baseline(tmp_path):
    torch.set_num_threads(1);frame,tensors=fixture()
    global_ck,_,_=train_head(tensors,frame,'global',42,tmp_path/'global',{},epochs=2)
    original=state_digest(global_ck['state'])
    patch_ck,_,_=train_head(tensors,frame,'gaze',42,tmp_path/'gaze',{},epochs=2,global_checkpoint=global_ck)
    assert state_digest(global_ck['state'])==original
    model=restore(patch_ck,global_ck)
    assert all(not p.requires_grad for p in model.global_model.parameters())
    restored=restore(torch.load(tmp_path/'gaze'/'best.pt',weights_only=False),global_ck)
    indices=torch.arange(8)
    assert torch.equal(evaluate(model,tensors,indices,'gaze')['logits'],evaluate(restored,tensors,indices,'gaze')['logits'])
    fresh=GazeAttention1011(restore(global_ck),variant='mean_residual').eval()
    out=fresh(tensors['global_features'],tensors['gaze_features'],tensors['gaze_coordinates'])
    assert torch.equal(out['logits'],out['global_logits'])
    bad=copy.deepcopy(global_ck);bad['state']['head.1.bias']+=1
    with unittest.TestCase().assertRaisesRegex(ValueError,'hash'):restore(patch_ck,bad)
    bad=copy.deepcopy(patch_ck);bad['seed']=43
    with unittest.TestCase().assertRaisesRegex(ValueError,'provenance'):restore(bad,global_ck)


def test_metrics_and_patient_pairing():
    m=metrics([0,1,0,1],[.1,.8,.3,.6])
    assert m['accuracy']==1 and m['f1']==1 and m['confusion_matrix']==[[2,0],[0,2]]
    with unittest.TestCase().assertRaises(ValueError):metrics([0,0],[.1,.2])
    for invalid in ([float('nan'),.4], [-.1,.4], [.2,1.1]):
        with unittest.TestCase().assertRaises(ValueError): metrics([0,1],invalid)
    frame=pd.DataFrame(dict(image_id=[1,2,3,4],patient_key=['a','a','b','b'],label=[0,1,0,1],probability=[.1,.8,.3,.6]))
    ci=paired_ci(frame,frame.iloc[::-1].reset_index(drop=True),20)
    assert ci['delta']==0 and ci['low']==0 and ci['high']==0
    for field in ('patient_key','label','image_id'):
        bad=frame.copy();bad.loc[0,field]='wrong' if field=='patient_key' else 1 if field=='label' else 99
        with unittest.TestCase().assertRaises(ValueError): paired_ci(frame,bad,20)
    bad=frame.copy();bad.loc[0,'image_id']=2
    with unittest.TestCase().assertRaises(ValueError): paired_ci(frame,bad,20)

def test_malicious_cache_rejection():
    import json
    from src.gaze_patch_features_1012 import validate_feature_cache, SOURCE_SHA256, DEFAULT_CONFIG, ARMS
    config=json.loads(DEFAULT_CONFIG.read_text())
    provenance=dict(source_config_sha256=SOURCE_SHA256,crop_size=224,feature_dim=1024,arms=list(ARMS),path_seed=42,source_config_path=str(DEFAULT_CONFIG),classification_encoder_sha256=config['config']['probability']['parents'][0]['visual_state_sha256'])
    cache=dict(kind='gaze_patch_classification_features_1012',rows=[dict(image_id=1,split='test',label=1,patient_key='CBIS_x')],tensors={},provenance=provenance)
    with unittest.TestCase().assertRaisesRegex(ValueError,'population'): validate_feature_cache(cache)
    cache['rows'][0]['split']='train';cache['rows'][0]['label']=-1
    with unittest.TestCase().assertRaisesRegex(ValueError,'population'): validate_feature_cache(cache)
    cache['provenance']['classification_encoder_sha256']='0'*64
    with unittest.TestCase().assertRaisesRegex(ValueError,'encoder'): validate_feature_cache(cache)
    cache['provenance']['source_config_sha256']='0'*64
    with unittest.TestCase().assertRaisesRegex(ValueError,'topology'): validate_feature_cache(cache)

if __name__=='__main__':
    suite=unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(test_mean_duplicate_permutation_and_fallback))
    suite.addTest(unittest.FunctionTestCase(test_metrics_and_patient_pairing))
    suite.addTest(unittest.FunctionTestCase(test_malicious_cache_rejection))
    def lifecycle():
        with tempfile.TemporaryDirectory() as directory: test_mock_training_checkpoint_and_frozen_baseline(Path(directory))
    suite.addTest(unittest.FunctionTestCase(lifecycle))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
