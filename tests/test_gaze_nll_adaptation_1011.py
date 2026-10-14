import torch
from src.gaze_nll_adaptation_1011 import GazeNLLAdaptation1011,hard_nll,prior_targets


def test_hard_nll_image_macro_mask():
    raw=torch.zeros(2,3,4);raw[0,0,0]=2;coords=torch.zeros(2,3,2);mask=torch.tensor([[1,0,0],[1,1,1]],dtype=torch.bool)
    expected=(-raw[0,0].log_softmax(-1)[0]-raw[1,0].log_softmax(-1)[0])/2
    assert torch.allclose(hard_nll(raw,coords,mask,(2,2)),expected)
    assert torch.allclose(prior_targets(coords,mask,(2,2)).sum(-1),torch.ones(2))


def test_prefix_prediction_causal():
    torch.manual_seed(42);model=GazeNLLAdaptation1011(grid=(2,2),dim=8,pretrained=False,spatial_history=True,behavior_mixture=True)
    memory=dict(keys=torch.randn(1,4,8),prior=torch.randn(1,4),global_context=torch.randn(1,8));coords=torch.rand(1,4,2)
    changed=coords.clone();changed[:,2:]=1-changed[:,2:]
    assert torch.equal(model.teacher_logits(memory,coords)[:,:3],model.teacher_logits(memory,changed)[:,:3])


def test_cache_split_guards():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from gaze_nll_adaptation_1011 import routing,validate_records
    records=routing();assert validate_records(records)=={'train':412,'val':93}
    try:validate_records(records+[records[0]])
    except ValueError:pass
    else:raise AssertionError('Duplicate accepted')
    try:validate_records(records[:2])
    except ValueError:pass
    else:raise AssertionError('Incomplete full cache accepted')


def test_resolution_metadata_restores_without_constructor_kwargs():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from gaze_nll_adaptation_1011 import apply_cache_resolution
    from src.gaze_nll_adaptation_1011 import restore_adaptation
    model=GazeNLLAdaptation1011(grid=(2,2),dim=8,pretrained=False)
    apply_cache_resolution(model,dict(input_size=[512,1024],feature_memory_format='contiguous_nchw'))
    assert model.input_size==(512,1024)
    assert 'feature_memory_format' not in model.model_config
    checkpoint=dict(kind='gaze_nll_adaptation_1011',model_config=model.model_config,state={k:v for k,v in model.state_dict().items() if not k.startswith('visual.')},feature_memory_format='contiguous_nchw')
    restored=restore_adaptation(checkpoint,model)
    assert restored.input_size==(512,1024) and restored.feature_memory_format=='contiguous_nchw'
