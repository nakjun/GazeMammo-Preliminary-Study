import importlib.util
from pathlib import Path
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('zoom_audit',Path(__file__).parents[1]/'scripts/audit_gaze_zoom_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
from src.data import reduce_gaze


def test_invalid_break_and_min_distance_match_legacy():
    points=np.array([[0,0],[20,0],[40,0],[np.nan,0],[0,0],[40,0],[80,0]],float)
    indices,_=m.selected_indices(points,2000,3000)
    assert np.array_equal(indices,[4,5,6])
    assert np.array_equal(points[indices],max(reduce_gaze(points,2000,3000),key=len))


def test_legacy16_equal_reduced_length_chooses_first_segment():
    first=np.column_stack([np.arange(20)*40,np.ones(20)*100]);second=np.column_stack([np.arange(25)*40,np.ones(25)*200]);points=np.concatenate([first,[[np.nan,0]],second])
    indices,_=m.selected_indices(points,2000,3000)
    assert indices.max()<20 and len(indices)==16
    assert np.array_equal(points[indices],max(reduce_gaze(points,2000,3000),key=len))


def test_alignment_rejects_changed_cached_coordinate():
    points=np.array([[100,100],[150,150]],float);indices=np.array([0,1]);coords=torch.zeros(16,2);coords[:2]=torch.as_tensor(points/[1500,3000],dtype=torch.float32);mask=torch.zeros(16,dtype=torch.bool);mask[:2]=True;target=(coords[:,1]*48).long()*24+(coords[:,0]*24).long();record=dict(image_id=1,coords=coords,valid=mask,loss_mask=mask,target=target)
    assert m.alignment(points,indices,np.array([1500,3000]),record).shape==(2,2)
    record['coords'][0,0]+=.01
    try:m.alignment(points,indices,np.array([1500,3000]),record)
    except ValueError as error:assert 'alignment failed' in str(error)
    else:raise AssertionError('Changed prefix accepted')


def test_scale_change_tolerance_ignores_float_noise():
    assert m.scale_changes([.16,.160000000000003,.16])==0
    assert m.scale_changes([.16,.32,.16])==2
