import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.test_metrics_1012 import overlap_fraction,clip_rectangles,coco_entries,detection_metrics,gaze_metrics,confidence,path_metrics

def test_physical_perfect_empty_and_clipping():
    rect=np.array([[0,0,224,224],[300,300,524,524]])
    assert np.array_equal(overlap_fraction(rect,[0,0,224,224]),[1.,0.]);assert np.array_equal(clip_rectangles([[-112,-112,112,112]],[1500,3000]),[[0,0,112,112]])

def test_coco_perfect_and_empty_known_gt():
    gt=[dict(id=1,image_id=1,category_id=2,bbox=[0,0,224,224])];images=[dict(id=1,width=1500,height=3000)];trace=dict(image_id=1,image_size=[1500,3000],actual_crop_rectangles=[[0,0,224,224]],relevance=[.9],source_anchor_indices=[0]);scores=detection_metrics(coco_entries([trace],gt,images));assert np.isclose(scores['proposal_AP'],1) and np.isclose(scores['proposal_AP50'],1) and scores['proposal_AR100']==scores['proposal_AR16']==1
    assert detection_metrics(coco_entries([],gt,images))['proposal_AP']==0

def test_confidence_order_changes_AP_not_geometry():
    gt=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,224,224])];images=[dict(id=1,width=1500,height=3000)];trace=dict(image_id=1,image_size=[1500,3000],actual_crop_rectangles=[[300,300,524,524],[0,0,224,224]],relevance=[.9,.1],source_anchor_indices=[0,1]);bad=detection_metrics(coco_entries([trace],gt,images))['proposal_AP'];trace['relevance']=[.1,.9];good=detection_metrics(coco_entries([trace],gt,images))['proposal_AP'];assert good>bad and np.isclose(good,1)

def test_uniform_gaze_metrics_and_perfect_path():
    record=dict(coords=np.full((16,2),.5),target=np.full(16,24*24+12),loss_mask=np.ones(16,bool),raw_logp=np.full((16,1152),-np.log(1152)));m=gaze_metrics(record,np.full(1152,1/1152));assert abs(m['conditional_AUC']-.5)<1e-12 and abs(m['conditional_NSS'])<1e-12 and abs(m['train_prior_IG_bits'])<1e-10
    truth=np.array([[.2,.2],[.3,.3],[.5,.3],[.6,.5]]);r=path_metrics(truth,truth,[1500,3000]);assert r['normalized_DTW']==0 and abs(r['ScanMatch_no_duration']-1)<1e-6 and abs(r['MultiMatch_position']-1)<1e-6

def test_manual_accumulation_matches_official_coco_tensor():
    import contextlib,io
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    images=[dict(id=1,width=1500,height=3000),dict(id=2,width=1500,height=3000)];gt=[dict(id=1,image_id=1,category_id=1,bbox=[0,0,224,224],area=224**2,iscrowd=0),dict(id=2,image_id=2,category_id=2,bbox=[500,500,224,224],area=224**2,iscrowd=0)];pred=[dict(image_id=1,category_id=1,bbox=[300,300,224,224],score=.95),dict(image_id=1,category_id=1,bbox=[0,0,224,224],score=.9),dict(image_id=2,category_id=1,bbox=[500,500,200,200],score=.8)]
    with contextlib.redirect_stdout(io.StringIO()):
        c=COCO();c.dataset=dict(info={},images=images,annotations=gt,categories=[dict(id=1,name='a'),dict(id=2,name='b')]);c.createIndex();d=c.loadRes(pred);ev=COCOeval(c,d,'bbox');ev.params.useCats=0;ev.params.maxDets=[1,10,100];ev.evaluate();ev.accumulate()
    entries={e['image_id']:e for e in ev.evalImgs if e is not None and np.array_equal(e['aRng'],ev.params.areaRng[0])};manual=detection_metrics(entries);precision=ev.eval['precision'][:,:,0,0,2];assert abs(manual['proposal_AP']-precision[precision>=0].mean())<1e-12;assert abs(manual['proposal_AP50']-precision[0].mean())<1e-12
    for k,index in [(1,0),(10,1),(100,2)]:assert abs(manual[f'proposal_AR{k}']-ev.eval['recall'][:,0,0,index].mean())<1e-12

def test_annotation_reader_only_decodes_allowed_objects():
    import tempfile,json
    from pathlib import Path
    from scripts.test_metrics_1012 import selected_annotations
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'a.json';path.write_text(json.dumps(dict(annotations=[dict(image_id=1,id=1,bbox=[0,0,1,1]),dict(image_id=999,id=2,bbox=['protected'])])));rows=selected_annotations(path,{1});assert len(rows)==1 and rows[0]['image_id']==1

if __name__=="__main__":
    import unittest
    suite=unittest.TestSuite(unittest.FunctionTestCase(value) for name,value in list(globals().items()) if name.startswith("test_") and callable(value))
    outcome=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if outcome.wasSuccessful() else 1)
