import importlib.util
from pathlib import Path
import numpy as np
from src.gaze_path_augmentation_1011 import selected_indices
spec=importlib.util.spec_from_file_location('pool_audit',Path(__file__).parents[1]/'scripts/audit_gaze_augmentation_pool_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_chosen_segment_tie_after_reduction_not_raw_length():
    first=np.column_stack([np.arange(20)*40,np.ones(20)*100]);second=np.column_stack([np.arange(30)*40,np.ones(30)*200]);points=np.concatenate([first,[[np.nan,0]],second]);pool,canonical,chosen,_=m.chosen_legacy_pool(points,(1500,3000))
    assert chosen==0 and len(pool)==20 and len(canonical)==16 and pool[-1]==19


def test_audit_and_trainer_share_exact_endpoint_sampler():
    indices=m.stratum_indices(191,42,62,3,True)
    assert np.array_equal(indices,selected_indices(191,62,3,42))
    assert indices[0]==0 and indices[-1]==190 and len(np.unique(indices))==16
    assert np.all(np.diff(indices)>0)
    assert not np.array_equal(indices,m.stratum_indices(191,42,62,4,True))
