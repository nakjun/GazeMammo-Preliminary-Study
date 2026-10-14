"""Independent closing audit; consumes completed frozen results, never runs inference."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'runs/frozen_current_test_1012';out=base/'metrics'
m=json.loads((out/'metrics.json').read_text());f=pd.read_csv(out/'per_image.csv');l=pd.read_csv(out/'per_lesion.csv');s=pd.read_csv(out/'image_seed_averages.csv');manifest=pd.read_csv(ROOT/'manifests/all.csv');test=manifest[manifest.split=='test'];traj=pd.read_csv(ROOT/'manifests/trajectory.csv');annot=set(traj[traj.split=='test'].image_id)
assert len(f)==450 and len(s)==150 and len(l)==474
assert not f.duplicated(['image_id','seed']).any()
assert set(f.image_id)==set(test.image_id) and set(f.seed)=={42,43,44}
assert f.groupby('image_id').seed.nunique().eq(3).all()
assert f.loc[f.image_id.isin(annot),'known_annotation'].eq(1).all()
physical=['H']+[f'physical_recall{x}_step{k}' for x in [50,80] for k in [1,4,8,16]]
assert f.loc[~f.image_id.isin(annot),physical].isna().all().all()
assert set(l.image_id)==annot and l.groupby(['image_id','annotation_id']).seed.nunique().eq(3).all()
expected=f.groupby(['image_id','patient_key'],as_index=False).mean(numeric_only=True)
errors={}
for c,value in m['image_macro'].items():
 errors[c]=abs(float(expected[c].mean())-value);assert errors[c]<1e-12
for c,value in m['lesion_micro'].items():assert abs(float(l[c].mean())-value)<1e-12
for c,value in m['proposal_detection'].items():assert abs(float(np.mean([v[c] for v in m['proposal_detection_per_seed'].values()]))-value)<1e-12
assert m['proposal_detection']['proposal_AR16']==m['proposal_detection']['proposal_AR100']
for x in [50,80]:
 vals=f.loc[f.image_id.isin(annot),[f'physical_recall{x}_step{k}' for k in [1,4,8,16]]].to_numpy();assert (np.diff(vals,axis=1)>=0).all()
 d=m['first_hit'][str(x)];assert abs(l[f'first_hit_{x}'].notna().mean()-d['hit_fraction'])<1e-12
for p,h in json.loads((base/'metrics_implementation_audit.json').read_text())['source_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
for p,h in m['source_sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
result=dict(status='passed',gaze_images=150,valid_gaze_steps=2400,annotated_images=131,unknown_excluded=19,lesions=158,lesion_seed_rows=474,paths=450,seeds=[42,43,44],image_macro_recompute_max_error=max(errors.values()),all_metrics_nonmissing_counts={c:int(s[c].notna().sum()) for c in m['image_macro']},confidence_audit='independent_input_audit.json',source_and_metric_implementation_hashes_unchanged=True,AP_seed_average_not_pooled=True,AR100_equals_AR16=True,physical_metric='actual single224crop intersection/boxarea, not attainable-normalized recall',metrics_sha256=hashlib.sha256((out/'metrics.json').read_bytes()).hexdigest())
assert m['denominators']['valid_gaze_steps']==2400
(base/'independent_terminal_metrics_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
