import numpy as np
from src.proposal_region_centers_1011 import region_candidate_indices,snap_lower_tie


def test_bbox_half_grid_tie_snaps_lower():
    scores=np.zeros((8,8));scores[3,7]=.4;scores[3,6]=.15
    indices,regions=region_candidate_indices(scores.ravel(),(8,8),'bbox_center',.25,7,budget=1,return_regions=True)
    assert indices==[3*8+6] and regions[0]['bbox_center_xy']==[6.5,3.]
    assert snap_lower_tie(6.5)==6 and snap_lower_tie(6.5001)==7


def test_disconnected_component_not_used_by_centroid():
    scores=np.zeros((9,9));scores[4,4]=.8;scores[4,5]=.6;scores[4,1]=.7
    indices,regions=region_candidate_indices(scores.ravel(),(9,9),'centroid',.5,7,budget=1,return_regions=True)
    assert indices==[4*9+4]
    assert 4*9+1 not in regions[0]['component_indices']


def test_peak_preserves_stable_sort_and_crop_suppression():
    scores=[.8,.8,.5];rects=np.array([[0,0,224,224],[1,0,225,224],[300,0,524,224]])
    assert region_candidate_indices(scores,(3,1),'peak',rectangles=rects,budget=3)==[0,2]


def test_plateau_and_boundary_are_finite_unique():
    scores=np.ones((5,5))*.2
    indices,regions=region_candidate_indices(scores.ravel(),(5,5),'bbox_center',.25,7,return_regions=True)
    assert len(indices)==1 and len(set(indices))==len(indices) and all(0<=i<25 for i in indices)


def test_invalid_scores_rejected():
    try:region_candidate_indices([float('nan')],(1,1))
    except ValueError:pass
    else:raise AssertionError('Invalid scores accepted')


def test_fixed_relative_thresholds_produce_expected_component_change():
    scores=np.zeros((8,8));scores[3,7]=.4;scores[3,6]=.15
    low=region_candidate_indices(scores.ravel(),(8,8),'bbox_center',.25,7,budget=1)
    high=region_candidate_indices(scores.ravel(),(8,8),'bbox_center',.5,7,budget=1)
    assert low==[30] and high==[31]
