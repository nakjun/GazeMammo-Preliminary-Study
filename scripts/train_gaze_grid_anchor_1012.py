"""Frozen strong-grid plus bounded extra16 residual, TRAIN-internal selection."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from src.official_cbis_classifier_1012 import restore,evaluate,metrics,digest,paired_ci

class BoundedResidual(torch.nn.Module):
 def __init__(self,dim):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(dim,32),torch.nn.GELU(),torch.nn.Dropout(.3),torch.nn.Linear(32,1));torch.nn.init.zeros_(self.net[-1].weight);torch.nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return .5*torch.tanh(self.net(x).flatten())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[42]);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False)
 cache=torch.load(a.cache,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors'];assert set(f.split)=={'train','val'} and len(f)==2393 and (f.split=='val').sum()==356
 tr=np.flatnonzero(f.split=='train');va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);pt=f.iloc[tr].groupby('patient_key').label.max();fitp,selp=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=912).split(pt.index,pt.values));fitids=torch.tensor(tr[f.iloc[tr].patient_key.isin(pt.index[fitp])]);selids=torch.tensor(tr[f.iloc[tr].patient_key.isin(pt.index[selp])]);assert not set(f.iloc[fitids].patient_key)&set(f.iloc[selids].patient_key)
 m=dict(state='running',stage='pilot42' if a.seeds==[42] else 'three_seed_confirmation',cache_sha256=digest(a.cache),source_sha256=digest(__file__),seeds=a.seeds,patch_budget=32,anchor_patch_budget=16,additional_patch_budget=16,test_used=False,arms=['gaze_baseline','gaze_improved','random'],selection='TRAIN-internal patient holdout BCE; VAL evaluated once per fitted candidate',limitation='Frozen anchor already trained on internal holdout patients; this is not unbiased OOF validation.',fit_patients=pt.index[fitp].tolist(),internal_selection_patients=pt.index[selp].tolist(),epochs=60,patience=12,residual_bound=.5,loss='bag BCE + .1 residual squared',lr=.0003,weight_decay=.01,results=[],dependencies={})
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();preds={}
 for seed in a.seeds:
  heads=Path('runs/official_cbis_v2_experiment_1012/heads');gp=heads/f'global_seed{seed}/best.pt';bp=heads/f'grid_seed{seed}/best.pt';m['dependencies'].update({str(gp):digest(gp),str(bp):digest(bp)});gc=torch.load(gp,map_location='cpu',weights_only=False);bc=torch.load(bp,map_location='cpu',weights_only=False);anchor=restore(bc,gc)
  logits=evaluate(anchor,t,torch.arange(len(f)),'grid')['logits'];baseline=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();baseline['probability']=logits[va].sigmoid().numpy();baseline.to_csv(out/f'grid16_seed{seed}_validation_predictions.csv',index=False);preds[('grid',seed)]=baseline
  for arm in m['arms']:
   source=t[arm+'_features'];x=torch.cat([t['global_features'],source.mean(1)-t['grid_features'].mean(1),source.max(1).values],1);mean=x[fitids].mean(0);std=x[fitids].std(0).clamp_min(.01);x=((x-mean)/std).clamp(-10,10)
   torch.manual_seed(seed);model=BoundedResidual(x.shape[1]);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);d=out/f'grid16_{arm}16_seed{seed}';d.mkdir();history=[];best=float('inf');stale=0
   for epoch in range(61):
    losses=[]
    if epoch:
     model.train();order=fitids[torch.randperm(len(fitids))]
     for ids in order.split(64):
      opt.zero_grad();res=model(x[ids]);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits[ids]+res,y[ids])+.1*res.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():res=model(x[selids]);sel=float(torch.nn.functional.binary_cross_entropy_with_logits(logits[selids]+res,y[selids])+.1*res.square().mean())
    history.append(dict(epoch=epoch,fit_loss=float(np.mean(losses)) if losses else None,internal_selection_loss=sel))
    if sel<best:best=sel;stale=0;state=copy.deepcopy(model.state_dict());be=epoch
    else:stale+=1
    if stale>=12:break
   model.load_state_dict(state);model.eval()
   with torch.inference_mode():res=model(x[va]);prob=(logits[va]+res).sigmoid().numpy()
   pred=baseline.copy();pred['probability']=prob;pred['anchor_logit']=logits[va].numpy();pred['residual_logit']=res.numpy();pred.to_csv(d/'validation_predictions.csv',index=False);preds[(arm,seed)]=pred;(d/'history.json').write_text(json.dumps(history));torch.save(dict(state=state,normalization=dict(mean=mean,std=std),seed=seed,arm=arm,epoch=be),d/'best.pt')
   r=dict(arm=arm,seed=seed,epoch=be,metrics=metrics(pred.label,pred.probability),mean_abs_residual=float(res.abs().mean()));m['results'].append(r);save();print(json.dumps(r),flush=True)
 report=dict(results=m['results'],paired={},anchor_metrics={str(s):metrics(preds[('grid',s)].label,preds[('grid',s)].probability) for s in a.seeds})
 for seed in a.seeds:
  for arm in m['arms']:
   report['paired'][f'{arm}_{seed}_minus_grid16']=paired_ci(preds[(arm,seed)],preds[('grid',seed)],1000)
   if arm!='random':report['paired'][f'{arm}_{seed}_minus_matchedrandom32']=paired_ci(preds[(arm,seed)],preds[('random',seed)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(a.cache)==m['cache_sha256'];assert digest(__file__)==m['source_sha256'];assert all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';save()
if __name__=='__main__':main()
