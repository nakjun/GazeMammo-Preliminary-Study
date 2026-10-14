"""Independent decision-score AUROC and PCA/SVM training-ID audit."""
import sys,json,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score
from src.official_cbis_classifier_1012 import digest
root=Path('runs/gaze_focus_auroc88_1013');out=root/'round16_rbf_svm_pilot42';m=json.load(open(out/'manifest.json'));assert m['state']=='complete';c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);tr=f[f.split=='train'];va=f[f.split=='val'];assert len(va)==356 and va.patient_key.nunique()==183 and f.groupby('patient_key').split.nunique().max()==1;aud=[];models={}
for r in m['results']:
 for mode in ['singleview','paired_feature_mean','fixed_decision_score_average']:
  p=out/r['arm']/f'{mode}_scores.csv';df=pd.read_csv(p);assert df.image_id.tolist()==va.image_id.tolist() and df.label.tolist()==va.label.tolist() and df.patient_key.tolist()==va.patient_key.tolist() and np.isfinite(df.decision_score).all();auc=float(roc_auc_score(df.label,df.decision_score));assert abs(auc-r[mode]['auroc'])<1e-12;aud.append(dict(arm=r['arm'],mode=mode,auroc=auc,sha256=digest(p)))
 p=out/r['arm']/'selected.pkl';ck=pickle.loads(p.read_bytes());assert ck['fit_image_ids']==tr.image_id.tolist() and ck['global_block']['fit_image_ids']==tr.image_id.tolist();assert ck['global_block']['pca'].n_components_==64 and ck['global_block']['pca'].whiten==False
 if ck['local_block'] is not None:assert ck['local_block']['fit_image_ids']==tr.image_id.tolist() and ck['local_block']['pca'].n_components_==64
 models[str(p)]=digest(p)
for p,sha in m['dependencies'].items():assert digest(p)==sha
rows=[dict(arm=r['arm'],paired=r['paired_feature_mean']['auroc'],single=r['singleview']['auroc'],train=r['train_paired_auroc'],train_val_gap=r['train_paired_auroc']-r['paired_feature_mean']['auroc'],config=r['selected_config']) for r in m['results']];report=dict(state='complete',goal_met=False,pilot_seed42=True,results=m['results'],VALID_effective_K=m['VALID_effective_K'],elapsed_minutes=m['elapsed_minutes'],audit_passed=True,limitations=['Single seed42 PCA/SVM pilot, not three-seed mean.','Uncalibrated decision scores; no probability/accuracy claim.','TRAIN-only per-block PCA64, internal patient selection of six fixed C/gamma settings, fullTRAIN refit then VALonce.','High-resolution wholeimage + freshOOF farthest-selected8 gaze and nonlinear RBF differ from old round04 lowresolution PCA logistic.','Spatial mean/max loses token location information relative to strong global spatialMIL reference.']);(out/'final_report.json').write_text(json.dumps(report,indent=2));(out/'final_audit.json').write_text(json.dumps(dict(passed=True,csv=aud,models=models,TRAINonly_fit_verified=True),indent=2));p=root/'progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(active_goal=.88,round='round16',state='complete_goal_unmet',output=str(out),table=rows),indent=2));tmp.replace(p);print(json.dumps(dict(table=rows,minutes=m['elapsed_minutes'])),flush=True)
