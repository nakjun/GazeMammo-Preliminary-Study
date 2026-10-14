"""Independent VAL CSV audit and TRAIN-only OOF trajectory coverage diagnostic."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from src.official_cbis_classifier_1012 import digest,metrics
out=Path('runs/gaze_focus_auroc88_1013/round12_gaze_extent_pilot42');base=out.parent/'round06_lesion_trajectory_pilot42';m=json.load(open(out/'manifest.json'));pred=json.load(open(out/'prediction_manifest_k16.json'));assert m['state']==pred['state']=='complete' and pred['completed_records']==2393
c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);vf=f[f.split=='val'];audit=[]
for r in m['results']:
 for mode in ['singleview','learned_paired_context_average','fixed_probability_average']:
  root=out if r['arm']!='global_only' else out.parent/'round09_spatial_mil_pilot42';p=root/r['arm']/f'{mode}_predictions.csv';df=pd.read_csv(p);assert df.image_id.tolist()==vf.image_id.tolist() and df.label.tolist()==vf.label.tolist() and df.patient_key.tolist()==vf.patient_key.tolist() and len(df)==356 and df.patient_key.nunique()==183 and np.isfinite(df.probability).all();score=metrics(df.label,df.probability)
  for key in ['auroc','auprc','accuracy']:assert abs(score[key]-r[mode][key])<1e-12
  audit.append(dict(arm=r['arm'],mode=mode,sha256=digest(p),metrics=score))
for p,sha in m['dependencies'].items():assert digest(p)==sha
for p,sha in pred['models'].items():assert digest(p)==sha
sel=torch.load(out/'diverse_selections.pt',weights_only=False,map_location='cpu');assert sel['image_ids']==f.image_id.tolist();boxes=json.load(open(base/'lesion_supervision_v2.json'))['boxes'];cases=[]
for k in [8,16]:
 s=sel['selection'][str(k)]
 for image_id,bs in boxes.items():
  i=sel['image_ids'].index(int(image_id));r=torch.load(out/'predicted_records_k16'/f'imitation_{image_id}.pt',weights_only=False,map_location='cpu');ck=torch.load(base/'trajectory_models'/f"imitation_{r['trajectory_model']}/selected.pt",weights_only=False,map_location='cpu');assert r['split']=='train' and r['patient_key'] not in ck['fit_patients'];assert digest(out/'predicted_records_k16'/f'imitation_{image_id}.pt')==pred['records'][f'imitation:{image_id}'];rect=r['rectangles'][s['indices'][i][s['mask'][i]]].numpy();hit=False;fraction=0.
  for box in bs:
   box=np.array(box)*[1500,3000,1500,3000];center=.5*(box[:2]+box[2:]);hit|=bool(((rect[:,:2]<=center)&(rect[:,2:]>=center)).all(1).any());area=np.maximum(np.minimum(rect[:,2:],box[2:])-np.maximum(rect[:,:2],box[:2]),0).prod(1);fraction=max(fraction,float(area.max()/np.prod(box[2:]-box[:2])))
  cases.append(dict(k=k,image_id=int(image_id),center_hit=hit,bbox_fraction=fraction,valid_count=len(rect)))
df=pd.DataFrame(cases);df.to_csv(out/'TRAIN_OOF_coverage.csv',index=False);coverage={str(k):dict(n=len(g),center_hit=float(g.center_hit.mean()),bbox_fraction_ge10percent=float((g.bbox_fraction>=.1).mean()),mean_bbox_fraction=float(g.bbox_fraction.mean()),mean_valid_count=float(g.valid_count.mean())) for k,g in df.groupby('k')};report=dict(state='complete',goal_met=False,pilot_seed=42,results=m['results'],effective_K=m['effective_K'],TRAIN_only_OOF_coverage=coverage,prediction_minutes=pred['elapsed_minutes'],classifier_minutes=m['elapsed_minutes'],audit_passed=True,limitations=['Single-seed pilot, not classifier three-seed mean.','Global MIL checkpoint fixed; residual selected using TRAIN patient holdout only.','Imitation trajectory supervision only; TRAIN lesion annotations used solely for acquisition diagnostic.','Nominal K differs from valid distinct crop count.']);(out/'final_report.json').write_text(json.dumps(report,indent=2));(out/'final_audit.json').write_text(json.dumps(dict(passed=True,csv=audit),indent=2));print(json.dumps(dict(results=[dict(arm=r['arm'],paired=r['learned_paired_context_average']['auroc']) for r in m['results']],coverage=coverage)),flush=True)
