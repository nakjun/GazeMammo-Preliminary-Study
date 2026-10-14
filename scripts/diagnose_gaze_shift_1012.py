"""TRAIN/VAL distribution and original-head subgroup audit; no model inputs added."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from src.official_cbis_classifier_1012 import restore,evaluate,metrics,digest

def main():
 torch.set_num_threads(4);out=Path('runs/gaze_shift_1012_round3');out.mkdir(exist_ok=False);path='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(path,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert set(f.split)=={'train','val'}
 f['lesion_type']=f.source_sn.str.split('-').str[0];f['view']=f.source_sn.str.rsplit('_',n=1).str[-1];f['side']=f.source_sn.str.extract('(LEFT|RIGHT)')[0]
 report=dict(cache_sha256=digest(path),test_used=False,distribution={},geometry={},performance={},limitations='Source-name subgroup metadata are diagnostics only, never model inputs. Feature norm is a proxy, not a measured foreground fraction.')
 for split in ['train','val']:
  z=f[f.split==split];patients=z.groupby('patient_key').label.max();report['distribution'][split]=dict(images=len(z),patients=len(patients),positive_image_fraction=float(z.label.mean()),positive_patient_any_fraction=float(patients.mean()),subgroups={col:{str(k):dict(n=len(g),positive_fraction=float(g.label.mean())) for k,g in z.groupby(col)} for col in ['lesion_type','view','side']})
 for arm in ['grid','random','gaze_baseline','gaze_improved']:
  xy=t[arm+'_coordinates'];norm=t[arm+'_features'].norm(dim=-1);nearest=torch.cdist(xy*torch.tensor([1500.,3000.]),xy*torch.tensor([1500.,3000.]));nearest.diagonal(dim1=1,dim2=2).fill_(float('inf'))
  report['geometry'][arm]={}
  for split in ['train','val']:
   idx=torch.tensor(np.flatnonzero(f.split==split));report['geometry'][arm][split]=dict(center_mean=xy[idx].mean((0,1)).tolist(),center_std=xy[idx].std((0,1)).tolist(),nearest_distance_px=float(nearest[idx].min(-1).values.mean()),patch_feature_norm_mean=float(norm[idx].mean()),patch_feature_norm_std=float(norm[idx].std()),duplicate_fraction=float(np.mean([1-len(torch.unique(x,dim=0))/16 for x in xy[idx]])))
  root=Path('runs/official_cbis_v2_experiment_1012/heads');gc=torch.load(root/'global_seed42/best.pt',map_location='cpu',weights_only=False);ck=torch.load(root/f'{arm}_seed42/best.pt',map_location='cpu',weights_only=False);model=restore(ck,gc);report['performance'][arm]={}
  for split in ['train','val']:
   ids=torch.tensor(np.flatnonzero(f.split==split));pred=evaluate(model,t,ids,arm)['logits'].sigmoid().numpy();z=f.iloc[ids.tolist()].copy();z['probability']=pred;groups={}
   for col in ['lesion_type','view','side']:
    groups[col]={str(k):metrics(g.label,g.probability) for k,g in z.groupby(col) if g.label.nunique()==2}
   report['performance'][arm][split]=dict(overall=metrics(z.label,z.probability),subgroups=groups);z.to_csv(out/f'{arm}_{split}_predictions.csv',index=False)
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({'distribution':report['distribution'],'performance':{a:{s:dict(auroc=x['overall']['auroc'],lesion_subgroup={k:v['auroc'] for k,v in x['subgroups']['lesion_type'].items()}) for s,x in q.items()} for a,q in report['performance'].items()}}))
if __name__=='__main__':main()
