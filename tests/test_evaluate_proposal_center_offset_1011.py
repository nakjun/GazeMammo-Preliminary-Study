import importlib.util,json,copy
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('offset_eval',Path(__file__).parents[1]/'scripts/evaluate_proposal_center_offset_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def trace():
    with (Path(__file__).parents[1]/'runs/dual_target_1011/center_offset_integration_smoke/traces.jsonl').open() as source:return json.loads(next(source))


def test_actual_smoke_trace_coordinate_proof():assert m.validate_trace(trace())[0].shape==(16,2)


def test_corrupted_mapping_and_annotation_flags_rejected():
    for key in ('coordinates','annotations_used'):
        changed=trace()
        if key=='coordinates':changed[key][0][0]+=.01
        else:changed[key]=True
        try:m.validate_trace(changed)
        except ValueError:pass
        else:raise AssertionError('Malformed protocol accepted')


def test_actual_rectangles_determine_foreground_and_inbounds():
    mask=np.ones((100,100),bool);rects=np.array([[-62,-62,162,162]],float)
    fg,inbounds=m.actual_geometry(rects,mask,(100,100),224)
    assert fg==inbounds==10000/(224**2)
