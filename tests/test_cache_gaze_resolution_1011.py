import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('resolution_cache',Path(__file__).parents[1]/'scripts/cache_gaze_resolution_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_reused_crop_vectors_and_coordinates_are_exact():
    record=dict(features=[torch.ones(2,2,2)]*3,vectors=torch.randn(16,8),coords=torch.rand(16,2),mask=torch.ones(16,dtype=torch.bool))
    features=[torch.randn(2,4,4)]*3;changed=m.replace_whole_features(record,features,(512,1024),'encoder')
    assert changed['features'] is features and changed['whole_input_size']==[512,1024]
    for key in ('vectors','coords','mask'):assert torch.equal(changed[key],record[key])


def test_encoder_mismatch_and_size_alias_rejected():
    provenance=dict(visual_state_sha256='original',annotation_used=False,smoke=False)
    try:m.validate_source(provenance,'different',(512,1024))
    except ValueError as error:assert 'encoder' in str(error)
    else:raise AssertionError('Mismatched reused encoder accepted')
    try:m.validate_source(provenance,'original',(256,512))
    except ValueError as error:assert 'alias' in str(error)
    else:raise AssertionError('Aliased resolution accepted')


def test_native_feature_shapes_reject_semantic_grid_resize():
    features=[torch.zeros(channel,h,w) for channel,h,w in [(2,16,8),(3,8,4),(4,4,2)]]
    assert m.validate_native_features(features,(64,128),(2,3,4))==[(2,16,8),(3,8,4),(4,4,2)]
    features[0]=torch.zeros(2,48,24)
    try:m.validate_native_features(features,(64,128),(2,3,4))
    except ValueError as error:assert 'pyramid mismatch' in str(error)
    else:raise AssertionError('Shrunken semantic grid accepted')


def test_cached_live_map_paths_agree():
    class Model:
        def visual(self,x):return [torch.nn.functional.avg_pool2d(x,2),torch.nn.functional.avg_pool2d(x,4),torch.nn.functional.avg_pool2d(x,8)]
        def maps_features(self,features):
            fine=features[0];return dict(keys=fine.flatten(2).transpose(1,2),prior=fine.mean(1).flatten(1),global_context=fine.mean((2,3)))
        def maps(self,x):return self.maps_features(self.visual(x))
    model=Model();x=torch.randn(1,3,64,32);features=[feature[0].clone() for feature in model.visual(x)]
    assert all(value==0 for value in m.verify_live_cached_maps(model,x,features).values())
    features[0]+=1
    try:m.verify_live_cached_maps(model,x,features)
    except ValueError as error:assert 'mismatch' in str(error)
    else:raise AssertionError('Changed cached features accepted')
