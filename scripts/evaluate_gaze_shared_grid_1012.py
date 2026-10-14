"""Inference-only shared grid scorer transfer; fixed pooling, no fitting."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from scripts.train_gaze_multiview_1012 import pair_views
from src.official_cbis_classifier_1012 import digest,restore,metrics,paired_ci

def bags(t):
 b={a:(t[a+'_features'],t[a+'_coordinates'],t[a+'_mask']) for a in ['grid','gaze_baseline','gaze_improved','random']}
 for a in ['gaze_baseline','gaze_improved','random']:
  b['grid16_'+a+'16']=tuple(torch.cat((b['grid'][j],b[a][j]),1) for j in range(3))
 return b

@torch.inference_mode()
def infer(model,g,bag,ids):
 chunks=[model(g[x],bag[0][x],bag[1][x],bag[2][x]) for x in ids.split(32)]
 return {k:torch.cat([z[k].cpu() for z in chunks]) for k in ['logits','local_logits','residual','global_logits']}

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_shared_grid_1012_three_seed');out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert set(f.split)=={'train','val'} and (f.split=='val').sum()==356;partners,_,_=pair_views(f);va=torch.tensor(np.flatnonzero(f.split=='val'));tr=torch.tensor(np.flatnonzero(f.split=='train'));b=bags(t)
 m=dict(state='running',stage='three_seed_inference',test_used=False,seeds=[42,43,44],arms=list(b),source_sha256=digest(__file__),cache_sha256=digest(cp),dependencies={},model='Existing strong grid mean_residual head; identical weights shared across patch selections',pooling='Original duplicate-corrected uniform mean over unique coordinates; no learned fusion',multiview='Equal probability average of current and label-free same-breast opposite-view minimageid partner; fallback current',patch_budgets='16/32 singleview;32/64 pairedmultiview; missing16/32',results=[],calibration={})
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();preds={};heads=Path('runs/official_cbis_v2_experiment_1012/heads')
 for seed in m['seeds']:
  gp=heads/f'global_seed{seed}/best.pt';hp=heads/f'grid_seed{seed}/best.pt';m['dependencies'].update({str(p):digest(p) for p in [gp,hp]});model=restore(torch.load(hp,map_location='cpu',weights_only=False),torch.load(gp,map_location='cpu',weights_only=False));assert model.variant=='mean_residual'
  for arm,bag in b.items():
   result=infer(model,t['global_features'],bag,torch.arange(len(f)) if seed==42 else va);local=result['local_logits'];residual=result['residual'];prob=result['logits'].sigmoid().numpy();valprob=prob[va] if seed==42 else prob
   if seed==42:
    m['calibration'][arm]={s:{k:dict(mean=float(z[ids].mean()),std=float(z[ids].std()),q10=float(z[ids].quantile(.1)),q90=float(z[ids].quantile(.9))) for k,z in [('local_logit',local),('residual',residual)]} for s,ids in [('train',tr),('val',va)]}
   for mode in ['singleview','multiview']:
    p=valprob.copy()
    if mode=='multiview':
     pmap=dict(zip(f.iloc[va.tolist()].image_id,valprob));p=np.array([.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()])
    pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=p;pred.to_csv(out/f'{arm}_{mode}_seed{seed}_validation_predictions.csv',index=False);preds[(arm,mode,seed)]=pred;r=dict(arm=arm,mode=mode,seed=seed,metrics=metrics(pred.label,pred.probability));m['results'].append(r);save();print(json.dumps(dict(arm=arm,mode=mode,seed=seed,auroc=r['metrics']['auroc'])),flush=True)
 report=dict(results=m['results'],means={},paired={},original_matched_heads={})
 for arm in b:
  for mode in ['singleview','multiview']:
   vals=[r['metrics']['auroc'] for r in m['results'] if r['arm']==arm and r['mode']==mode];report['means'][arm+'_'+mode]=dict(mean_auroc=float(np.mean(vals)),sd_auroc=float(np.std(vals,ddof=1)),seed_aurocs=vals)
   for seed in m['seeds']:
    if arm!='grid':report['paired'][f'{arm}_{mode}_{seed}_minus_grid']=paired_ci(preds[(arm,mode,seed)],preds[('grid',mode,seed)],1000)
    if arm.startswith('grid16_gaze'):report['paired'][f'{arm}_{mode}_{seed}_minus_matchedrandom']=paired_ci(preds[(arm,mode,seed)],preds[('grid16_random16',mode,seed)],1000)
  if arm in ['grid','gaze_baseline','gaze_improved','random']:
   vals=[]
   for seed in m['seeds']:
    p=pd.read_csv(heads/f'{arm}_seed{seed}/validation_predictions.csv');vals.append(metrics(p.label,p.probability)['auroc'])
   report['original_matched_heads'][arm]=dict(mean_auroc=float(np.mean(vals)),seed_aurocs=vals)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(cp)==m['cache_sha256'];assert digest(__file__)==m['source_sha256'];assert all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';save();print(json.dumps(report['means']),flush=True)
if __name__=='__main__':main()
