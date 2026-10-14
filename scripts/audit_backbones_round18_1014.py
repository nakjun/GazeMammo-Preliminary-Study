"""Independent same-protocol backbone comparison fullVAL metrics/source audit."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from src.official_cbis_classifier_1012 import digest,metrics
root=Path('runs/gaze_focus_auroc88_1013');out=root/'round18_backbone_comparison_pilot42';m=json.load(open(out/'manifest.json'));assert m['state']=='complete' and len(m['results'])==9;c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);v=f[f.split=='val'];tr=np.flatnonzero(f.split=='train');assert len(v)==356 and v.patient_key.nunique()==183 and f.groupby('patient_key').split.nunique().max()==1;fixed=torch.load(out/'fixed_coordinates.pt',weights_only=False,map_location='cpu');assert fixed['image_ids']==f.image_id.tolist() and fixed['valid_mask'].all();aud=[];models={};rows=[]
for r in m['results']:
 folder=out/r['backbone']/r['arm']
 for mode in ['singleview','paired_feature_mean','fixed_probability_average']:
  p=folder/f'{mode}_predictions.csv';df=pd.read_csv(p);assert df.image_id.tolist()==v.image_id.tolist() and df.label.tolist()==v.label.tolist() and df.patient_key.tolist()==v.patient_key.tolist() and np.isfinite(df.probability).all();score=metrics(df.label,df.probability)
  for key in ['auroc','auprc','accuracy']:assert abs(score[key]-r[mode][key])<1e-12
  aud.append(dict(backbone=r['backbone'],arm=r['arm'],mode=mode,metrics=score,sha256=digest(p)))
 p=folder/'selected.pt';ck=torch.load(p,weights_only=False,map_location='cpu');assert ck['epoch']==r['epoch'] and ck['weights_sha256']==m['weights'][r['backbone']]['sha256'];feature=torch.load(out/r['backbone']/'features.pt',weights_only=False,map_location='cpu');gs=feature['global_summary']
 if r['arm']=='global_only':ls=torch.zeros_like(gs)
 else:vv=feature['gaze_vectors'] if r['arm']=='gaze8' else feature['random_vectors'];ls=torch.cat([vv.mean(1),vv.max(1).values],1)
 x=torch.cat([gs,ls],1);assert torch.allclose(ck['mean'],x[tr].mean(0),rtol=1e-6,atol=1e-7) and torch.allclose(ck['std'],x[tr].std(0).clamp_min(.01),rtol=1e-6,atol=1e-7);models[str(p)]=digest(p);rows.append(dict(backbone=r['backbone'],arm=r['arm'],paired=r['paired_feature_mean']['auroc'],single=r['singleview']['auroc'],epoch=r['epoch'],train=r['train_paired_auroc'],train_val_gap=r['train_paired_auroc']-r['paired_feature_mean']['auroc']))
for p,sha in m['dependencies'].items():assert digest(p)==sha
for w in m['weights'].values():assert digest(w['path'])==w['sha256']
from sklearn.model_selection import StratifiedGroupKFold
fi,hi=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,f.label.values[tr],f.patient_key.values[tr]));assert not set(f.patient_key.iloc[tr[fi]])&set(f.patient_key.iloc[tr[hi]])
for r in m['results']:
 feature=torch.load(out/r['backbone']/'features.pt',weights_only=False,map_location='cpu');gs=feature['global_summary']
 if r['arm']=='global_only':ls=torch.zeros_like(gs)
 else:vv=feature['gaze_vectors'] if r['arm']=='gaze8' else feature['random_vectors'];ls=torch.cat([vv.mean(1),vv.max(1).values],1)
 x=torch.cat([gs,ls],1);ck=torch.load(out/r['backbone']/r['arm']/'inner_resume.pt',weights_only=False,map_location='cpu');assert torch.allclose(ck['mean'],x[tr[fi]].mean(0),rtol=1e-6,atol=1e-7) and torch.allclose(ck['std'],x[tr[fi]].std(0).clamp_min(.01),rtol=1e-6,atol=1e-7)
scores={(r['backbone'],r['arm']):r['paired_feature_mean']['auroc'] for r in m['results']};deltas={b:dict(gaze_minus_global=scores[b,'gaze8']-scores[b,'global_only'],gaze_minus_random=scores[b,'gaze8']-scores[b,'random8']) for b in m['protocol']['backbones']};preprocessing={p:digest(p) for p in ['src/data.py','src/gaze_backbones.py']};reviewpath=root/'backbone_result_review.json'
report=dict(within_backbone_deltas=deltas,review_criteria_path=str(reviewpath),review_criteria_sha256=digest(reviewpath),preprocessing_source_sha256=preprocessing,state='complete',goal_met=False,pilot_seed42=True,results=m['results'],benchmark=m['benchmark'],elapsed_minutes=m['elapsed_minutes'],audit_passed=True,limitations=['Same-protocol seed42 classifier pilots; not three-seed mean.','Trajectory generation stays original freshOOF ConvNeXt imitation, coordinates fixed across classifiers.','Backbone comparisons use standard ImageNet weights with different original training recipes, not isolated architecture causality.','Global FP16-map mean/max plus stock terminal normalization where present; local actualcrop mean vectors pooled mean/max.','Native feature dimensions differ but common block projection64/classifier16/dropout/.1wd/selection shared.','ConvNeXt caches reused with exact global parity and small CPU/GPU numeric crop drift; no prior model AUROC treated as same-protocol reference.','TRAINpatientinner epoch choice then freshfullTRAIN refit; VAL356once, TEST untouched.']);(out/'final_report.json').write_text(json.dumps(report,indent=2));(out/'final_audit.json').write_text(json.dumps(dict(passed=True,csv=aud,selected_model_sha256=models,TRAIN_only_normalizers_verified=True,fixed_coordinates_sha256=digest(out/'fixed_coordinates.pt')),indent=2));pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);p=root/'progress.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(dict(active_goal=.88,round='round18',state='complete_goal_unmet',output=str(out),table=rows,backbone_comparison_authorized=True),indent=2));tmp.replace(p);print(json.dumps(dict(table=rows,minutes=m['elapsed_minutes'])),flush=True)
