import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
spec=importlib.util.spec_from_file_location('proposal_metrics',Path(__file__).parents[1]/'scripts/evaluate_proposal_policy_1011.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_normalization_and_censored_first_hit():
    candidates=np.array([[0,0,224,224],[400,0,624,224]],float)
    result=m.lesion_metrics(candidates[1:],candidates,[0,1],[0,0,448,224])
    assert result['max_achievable_coverage']==.5
    assert result['normalized_coverage']<.5
    assert result['first_hit_step_05'] is None
    assert result['proposal_hit_path_miss_05_top1']==1


def test_spatial_distinct_suppression():
    rects=np.array([[0,0,224,224],[1,0,225,224],[300,0,524,224]])
    assert m.distinct_proposals([3,2,1],rects)==[0,2]


def test_patient_bootstrap_seed_averaging():
    frame=pd.DataFrame([dict(patient_key=p,image_id=i,policy=policy,seed=seed,score=value) for p,i in [('a',1),('a',2),('b',3)] for policy,value in [('original',0),('new',1)] for seed in [42,43,44]])
    result=m.paired_bootstrap(frame,'score','original','new',n=50)
    assert result['paired_images']==3 and result['patient_clusters']==2
    assert result['ci95']==[1,1]


def test_case73_saved_path_misses_recoverable_proposal():
    import json
    from PIL import Image
    trace=json.loads((m.ROOT/'runs/guided_1011_combined_seed42/best_predict_case73/prediction.json').read_text())
    boxes=json.loads((m.ROOT/'runs/guided_1011_combined_seed42/best_predict_case73/posthoc_box_metrics.json').read_text())['lesion_boxes']
    with Image.open(trace['image']) as image:geometry=m.candidate_geometry(image.convert('L'))
    rects=geometry['rectangles'].numpy();ranked=m.distinct_proposals(trace['proposal'],rects)
    result=m.lesion_metrics(m.rectangles(trace['coordinates'],geometry['image_size']),rects,ranked,boxes[0])
    assert result['actual_box_coverage']==0
    assert result['proposal_recall_05_top3']==1


def test_duplicate_traces_rejected(tmp_path):
    trace=dict(image_id=73,policy='P0',seed=42,coordinates=[])
    try:m.evaluate([trace,trace],tmp_path/'unused',bootstrap=1)
    except ValueError as exc:assert 'Duplicate trace' in str(exc)
    else:raise AssertionError('Duplicate accepted')


def test_new_schema_annotation_flags_rejected(tmp_path):
    trace=dict(schema_version='v1',image_id=73,policy='P0',seed=42,annotations_used=True,observed_gaze_used=False,labels_used=False)
    try:m.evaluate([trace],tmp_path/'unused',bootstrap=1)
    except ValueError as exc:assert 'annotation_used=false' in str(exc)
    else:raise AssertionError('Annotation flag accepted')


def test_argmax_baseline_suffix():
    assert m.baseline_for('P4_argmax','P0')=='P0_argmax'
    assert m.baseline_for('P4_argmax','P0_argmax')=='P0_argmax'
    assert m.baseline_for('P4','P0')=='P0'


def test_streaming_trace_compaction(tmp_path):
    import json
    path=tmp_path/'traces.jsonl'
    path.write_text(json.dumps(dict(image_id=73,coordinates=[[.5,.5]],steps=[dict(q=[1]*100)],geometry=dict(mask='huge')))+'\n')
    record=m.read_traces(path)[0]
    assert record['image_id']==73 and 'steps' not in record and 'geometry' not in record


def test_fixed_validation_routing_only_extra_validation():
    kept,routing=m.route_fixed_validation([{'image_id':1},{'image_id':2}],{1},{1,2})
    assert kept==[{'image_id':1}] and routing['excluded_image_ids']==[2]
    try:m.route_fixed_validation([{'image_id':3}],{1},{1,2})
    except ValueError as error:assert 'non-validation' in str(error)
    else:raise AssertionError('non-validation accepted')
