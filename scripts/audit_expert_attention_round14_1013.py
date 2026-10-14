"""Independent metrics audit and expert-target patient allowlist proof."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from src.official_cbis_classifier_1012 import digest,metrics
root=Path('runs/gaze_focus_auroc88_1013');out=root/'round14_expert_attention_pilot42';m=json.load(open(out/'manifest.json'));assert m['state']=='complete';c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);v=f[f.split=='val'];assert len(v)==356 and v.patient_key.nunique()==183 and f.groupby('patient_key').split.nunique().max()==1;aud=[]
for r in m['results']:
 for mode in ['singleview','learned_paired_context_average','fixed_probability_average']:
  p=out/r['arm']/f'{mode}_predictions.csv';df=pd.read_csv(p);assert df.image_id.tolist()==v.image_id.tolist() and df.label.tolist()==v.label.tolist() and df.patient_key.tolist()==v.patient_key.tolist() and np.isfinite(df.probability).all();score=metrics(df.label,df.probability)
  for key in ['auroc','auprc','accuracy']:assert abs(score[key]-r[mode][key])<1e-12
  aud.append(dict(arm=r['arm'],mode=mode,metrics=score,sha256=digest(p)))
for p,sha in m['dependencies'].items():assert digest(p)==sha
from sklearn.model_selection import StratifiedGroupKFold
tr=np.flatnonzero(f.split=='train');fit,hold=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,f.label.values[tr],f.patient_key.values[tr]));fp=set(f.patient_key.iloc[tr[fit]]);hp=set(f.patient_key.iloc[tr[hold]]);vp=set(v.patient_key);counts={}
for tag,records in m['expert_target_access'].items():
 assert not {r['patient'] for r in records}&vp
 if tag.endswith('_inner'):assert {r['patient'] for r in records}<=fp and not {r['patient'] for r in records}&hp
 if tag.startswith('global_no_aux'):assert not records
 counts[tag]=len(records)
assert all(counts[tag]==420 for tag in counts if tag.endswith('_fullTRAIN') and not tag.startswith('global_no_aux'))
rows=[dict(arm=r['arm'],paired=r['learned_paired_context_average']['auroc'],single=r['singleview']['auroc'],epoch=r['selected_epoch'],train=r['train_paired_auc'],train_val_gap=r['train_paired_auc']-r['learned_paired_context_average']['auroc']) for r in m['results']];report=dict(state='complete',goal_met=False,pilot_seed42=True,results=m['results'],expert_target_access_counts=counts,valid_local_counts=m['valid_local_counts'],elapsed_minutes=m['elapsed_minutes'],attention_gradient_smoke=m['attention_gradient_smoke'],audit_passed=True,limitations=['Single classifier seed42 pilot; not three-seed mean.','Expert TRAIN420 fixation targets added only as attention supervision; inference never uses expert gaze.','Inner target construction opens only fit-patient TRAIN records; fullTRAIN target construction opens TRAIN420 only.','Auxiliary Gaussian sigma112 pixels, lambda.05 after three classification-only warmup epochs; no parameter sweep.','Predicted local K16 image-only OOF crops; nominal16 has fewer distinct pixel crops and random mask matches per-image valid count.']);(out/'final_report.json').write_text(json.dumps(report,indent=2));(out/'final_audit.json').write_text(json.dumps(dict(passed=True,csv=aud,expert_allowlist_passed=True,expert_target_counts=counts),indent=2));p=root/'progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(active_goal=.88,round='round14',state='complete_goal_unmet',output=str(out),table=rows),indent=2));tmp.replace(p);print(json.dumps(dict(table=rows,target_counts=counts,minutes=m['elapsed_minutes'])),flush=True)
