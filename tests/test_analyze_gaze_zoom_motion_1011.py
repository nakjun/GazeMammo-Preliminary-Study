import importlib.util
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('zoom_motion',Path(__file__).parents[1]/'scripts/analyze_gaze_zoom_motion_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_constant_raw_gt_scale_oracle_is_exact_descriptive_case():
    result=m.motion_oracle(np.array([500.,1000.]),np.array([500.,1000.]),1.,2.)
    assert result['oracle_constant_raw_residual_px']==0 and result['oracle_direction_cosine']==1
    assert result['world_long_raw_local'] and result['scale_change_over5pct']


def test_no_scale_change_matches_world_persistence():
    result=m.motion_oracle(np.array([10.,20.]),np.array([40.,60.]),1.,1.)
    assert result['oracle_constant_raw_residual_px']==result['restored_world_jump_px']==50.
    assert not result['oracle_better_than_world_persistence'] and np.isnan(result['oracle_direction_cosine'])
