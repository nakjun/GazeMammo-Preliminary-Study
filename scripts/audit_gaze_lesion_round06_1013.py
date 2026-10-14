"""Final independent metrics and TRAIN-only imageOOF lesion acquisition diagnostic."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from src.official_cbis_classifier_1012 import digest,metrics

out=Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42');m=json.load(open(out/'classifiers_k4/manifest.json'));pred=json.load(open(out/'prediction_manifest_k4.json'));traj=json.load(open(out/'trajectory_manifest.json'));audit=[]
for r in m['results']:
 for mode in ['singleview','learned_paired_feature_average','fixed_probability_average']:
  path=out/'classifiers_k4'/r['arm']/f'{mode}_predictions.csv';f=pd.read_csv(path);assert len(f)==356 and f.image_id.nunique()==356 and f.patient_key.nunique()==183 and np.isfinite(f.probability).all();score=metrics(f.label,f.probability)
  for key in ['auroc','auprc','accuracy']:assert abs(score[key]-r[mode][key])<1e-12
  audit.append(dict(arm=r['arm'],mode=mode,sha256=digest(path),auroc=score['auroc']))
for cp,sha in pred['models'].items():assert digest(cp)==sha
for p,sha in traj['dependencies'].items():assert digest(p)==sha
assert pred['completed_records']==4786 and pred['state']=='complete';boxes=json.load(open(out/'lesion_supervision_v2.json'))['boxes'];states={};records=[]
for arm in ['imitation','lesion_aux']:
 for name in ['fold0','fold1','fold2']:states[arm,name]=torch.load(out/'trajectory_models'/f'{arm}_{name}/selected.pt',weights_only=False,map_location='cpu')
 for image_id,bs in boxes.items():
  path=out/'predicted_records_k4'/f'{arm}_{image_id}.pt';r=torch.load(path,weights_only=False,map_location='cpu');assert r['split']=='train' and r['trajectory_model']!='fullTRAIN' and r['patient_key'] not in states[arm,r['trajectory_model']]['fit_patients'];assert digest(path)==pred['records'][f'{arm}:{image_id}'];rect=r['rectangles'].numpy();assert rect.shape==(4,4);centerhit=False;fraction=0.
  for box in bs:
   box=np.array(box)*[1500,3000,1500,3000];center=.5*(box[:2]+box[2:]);centerhit|=bool(((rect[:,:2]<=center)&(rect[:,2:]>=center)).all(1).any());areas=np.maximum(np.minimum(rect[:,2:],box[2:])-np.maximum(rect[:,:2],box[:2]),0).prod(1);fraction=max(fraction,float(areas.max()/np.prod(box[2:]-box[:2])))
  records.append(dict(arm=arm,image_id=int(image_id),patient_key=r['patient_key'],lesion_center_in_any_crop=centerhit,max_bbox_fraction=fraction,bbox_fraction_ge10percent=fraction>=.1,unique_centers=r['unique_centers']))
f=pd.DataFrame(records);f.to_csv(out/'TRAIN_OOF_acquisition_cases.csv',index=False);coverage={arm:dict(n=len(g),lesion_center_hit_rate=float(g.lesion_center_in_any_crop.mean()),bbox_fraction_ge10percent_rate=float(g.bbox_fraction_ge10percent.mean()),mean_max_bbox_fraction=float(g.max_bbox_fraction.mean()),mean_unique_centers=float(g.unique_centers.mean())) for arm,g in f.groupby('arm')};report=dict(state='complete',classifier_pilot_seed42=True,target=.88,goal_met=False,results=m['results'],valid_source_counts=m['valid_source_counts'],trajectory_training_minutes=traj['elapsed_minutes'],prediction_minutes=pred['elapsed_minutes'],classifier_minutes=m['elapsed_minutes'],TRAIN_only_OOF_acquisition=coverage,audit_passed=True,n_prediction_csv=12,limitations=['318 TRAIN lesion box supervision added; not pure gaze imitation.','OOF acquisition uses TRAIN boxes after image-only inference; no VAL lesion annotations used.','Coverage is bbox-based, not segmentation or verified lesion capture.','Single classifier seed42 pilot, not three-seed mean.','NominalK4 sources can have differing unique valid patch counts.','Teacher-forced inner coverage loss is not autonomous acquisition.']);(out/'final_report.json').write_text(json.dumps(report,indent=2));(out/'final_csv_and_checkpoint_audit.json').write_text(json.dumps(dict(passed=True,csv=audit,trajectory_checkpoints=8,TRAIN_OOF_diagnostic_images=318),indent=2));print(json.dumps(dict(results=[dict(arm=r['arm'],single=r['singleview']['auroc'],paired=r['learned_paired_feature_average']['auroc'],fixedavg=r['fixed_probability_average']['auroc'],train=r['train_paired_auroc']) for r in m['results']],coverage=coverage,classifier_minutes=m['elapsed_minutes'])),flush=True)
