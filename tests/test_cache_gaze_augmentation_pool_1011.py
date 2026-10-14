import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('pool_cache',Path(__file__).parents[1]/'scripts/cache_gaze_augmentation_pool_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def source_and_pool():
    coords=torch.arange(40).reshape(20,2).float()/100
    canonical=torch.linspace(0,19,16).round().long();source=dict(coords=coords[canonical].clone(),mask=torch.ones(16,dtype=torch.bool),vectors=torch.randn(16,8))
    return source,coords,canonical


def test_canonical_coordinate_and_mask_guards():
    source,coords,canonical=source_and_pool();assert m.canonical_proof(coords,canonical,source)==16
    source['mask'][15]=False
    try:m.canonical_proof(coords,canonical,source)
    except ValueError:pass
    else:raise AssertionError('Changed canonical mask accepted')


def test_noncanonical_index_map_rejected():
    source,coords,canonical=source_and_pool();canonical[1]+=1
    try:m.canonical_proof(coords,canonical,source)
    except ValueError:pass
    else:raise AssertionError('Changed canonical point accepted')
