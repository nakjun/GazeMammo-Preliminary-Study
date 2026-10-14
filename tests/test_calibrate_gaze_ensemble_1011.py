import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('ensemble',Path(__file__).parents[1]/'scripts/calibrate_gaze_ensemble_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def cache():
    coords=torch.tensor([[.5,.5],[.6,.6]]);target=(coords[:,1]*48).long()*24+(coords[:,0]*24).long()
    record=dict(image_id=1,patient_key='patient',split='val',coords=coords,valid=torch.ones(2,dtype=torch.bool),loss_mask=torch.ones(2,dtype=torch.bool),target=target,raw_logp=torch.randn(2,1152).log_softmax(-1))
    return {'records':[record]}


def test_alignment_targets_and_split_must_match():
    first=cache();second=cache();assert len(m.aligned_records([first,second]))==2
    second['records'][0]['target'][0]+=1
    try:m.aligned_records([first,second])
    except ValueError as error:assert 'target' in str(error)
    else:raise AssertionError('Misaligned target accepted')
    second=cache();second['records'][0]['split']='train'
    try:m.aligned_records([first,second])
    except ValueError as error:assert 'split' in str(error)
    else:raise AssertionError('Misaligned split accepted')


def test_probability_mixture_and_geometry_normalization():
    torch.manual_seed(42);first=torch.randn(2,3,1152).log_softmax(-1);second=torch.randn(2,3,1152).log_softmax(-1)
    logp=m.calibrated_logp(first,second,.25,1.)
    assert torch.allclose(logp.exp(),.25*first.exp()+.75*second.exp(),atol=1e-7)
    geometric=m.calibrated_logp(first,second,.5,.75,torch.rand(2,1152),torch.rand(2,1152),3.,.5)
    assert torch.isfinite(geometric).all() and torch.allclose(geometric.exp().sum(-1),torch.ones(2,3),atol=1e-6)


def test_duplicate_and_unaligned_ids_rejected():
    first=cache();first['records']*=2
    try:m.aligned_records([first])
    except ValueError as error:assert 'Duplicate' in str(error)
    else:raise AssertionError('Duplicate accepted')
