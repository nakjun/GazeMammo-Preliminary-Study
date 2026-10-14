"""Fresh patient-OOF bases + fixed regularized logit fusion; VAL once."""
import argparse,json,sys,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from src.official_cbis_classifier_1012 import digest,metrics,paired_ci
from scripts.train_gaze_multiview_1012 import pair_views

BASES=['global','grid224','grid448','gaze224','gaze448','random224']
FUSIONS={'gridcontext':['global','grid224','grid448'],'gridcontext_gaze224':['global','grid224','grid448','gaze224'],'gridcontext_random224':['global','grid224','grid448','random224'],'gridcontext_multiscalegaze':['global','grid224','grid448','gaze224','gaze448']}

def patient_folds(f,seed):
 tr=f[f.split=='train'];patient=tr.groupby('patient_key').label.max();result=[]
 for fit,hold in StratifiedKFold(n_splits=3,shuffle=True,random_state=seed).split(patient.index,patient.values):
  a=np.flatnonzero((f.split=='train')&f.patient_key.isin(patient.index[fit]));b=np.flatnonzero((f.split=='train')&f.patient_key.isin(patient.index[hold]));assert not set(f.iloc[a].patient_key)&set(f.iloc[b].patient_key);result.append((a,b))
 assert sorted(np.concatenate([b for _,b in result]).tolist())==np.flatnonzero(f.split=='train').tolist();return result

def fit_base(x,y,fit,predict,seed,epochs=20,dim=64):
 """Fixed epochs: y outside fit never read, and normalizer only fit rows."""
 torch.manual_seed(seed);fit=torch.as_tensor(fit);predict=torch.as_tensor(predict);mean=x[fit].mean(0);std=x[fit].std(0).clamp_min(.01);xf=((x[fit]-mean)/std).clamp(-10,10);xp=((x[predict]-mean)/std).clamp(-10,10);yf=y[fit]
 model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],dim),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(dim,1));opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1);history=[]
 for epoch in range(epochs):
  model.train();order=torch.randperm(len(fit));losses=[]
  for ids in order.split(64):
   opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(xf[ids]).flatten(),yf[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
  history.append(dict(epoch=epoch+1,fit_bce=float(np.mean(losses))))
 model.eval()
 with torch.inference_mode():p=model(xp).flatten().numpy()
 return p,dict(state=model.state_dict(),mean=mean,std=std,history=history,fit_indices=fit.tolist(),seed=seed,epochs=epochs)

def partner_logits(logits,indices,f,partners):
 """Same patient partners always in same OOF fold; no labels read."""
 mapping={int(idx):logits[i] for i,idx in enumerate(indices)};out=[]
 for idx in indices:
  partner=partners[int(idx)];assert all(j in mapping for j in partner)
  if partner:
   p=.5*(1/(1+np.exp(-np.clip(mapping[int(idx)],-20,20)))+1/(1+np.exp(-np.clip(mapping[partner[0]],-20,20))));p=np.clip(p,1e-6,1-1e-6);out.append(np.log(p/(1-p)))
  else:out.append(mapping[int(idx)])
 return np.stack(out)

def fit_meta(train_x,train_y,val_x):
 scaler=StandardScaler().fit(train_x);model=LogisticRegression(C=.1,max_iter=1000,solver='lbfgs',random_state=42).fit(scaler.transform(train_x),train_y);return model.predict_proba(scaler.transform(val_x))[:,1],dict(coefficients=model.coef_.tolist(),intercept=model.intercept_.tolist(),mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[42]);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';cache=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors'];assert len(f)==2393 and (f.split=='val').sum()==356 and set(f.split)=={'train','val'};tr=np.flatnonzero(f.split=='train');va=np.flatnonzero(f.split=='val');y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f)
 sources=['scripts/train_gaze_multiview_1012.py','src/official_cbis_classifier_1012.py'];manifest=dict(state='running',stage='feature_load',test_used=False,seeds=a.seeds,source_sha256=digest(__file__),cache_sha256=digest(cp),dependencies={p:digest(p) for p in sources},bases=BASES,fusions=FUSIONS,base_architecture='MLP64 GELU dropout.5 linear1, trainonly normalization floor.01 clamp10',epochs=20,lr=.001,weight_decay=.1,batch_size=64,meta='OOF-only StandardScaler + LogisticRegression C=.1; no VAL tuning',selection='No heldout/VAL epoch or coefficient selection, fixed20epochs',budgets={'gridcontext':32,'gridcontext_gaze224':48,'gridcontext_random224':48,'gridcontext_multiscalegaze':64},multiview_budgets='2xsingleview for paired; missingviewfallback1x',limitation='Fresh MLP bases differ from earlier strong mean_residual grid family; OOF is unbiased for these registered bases only. VAL reused exploratorily across earlier rounds.',nonmatched_control='multiscalegaze vs random224 differs additional448 scale; cannot attribute that contrast only to gaze',results=[],folds={})
 def save():(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 save();features448={a:[] for a in ['grid','gaze_baseline']};fp=Path('runs/gaze_multiscale_1012_round6/features');fm=json.loads((fp/'manifest.json').read_text());assert fm['state']=='complete';manifest['dependencies'][str(fp/'manifest.json')]=digest(fp/'manifest.json')
 for row in cache['rows']:
  path=fp/'records'/f"{row['image_id']}.pt";assert digest(path)==fm['records'][str(row['image_id'])]['record_sha256'];z=torch.load(path,map_location='cpu',weights_only=False)
  for arm in features448:features448[arm].append(z['features'][arm])
 features448={a:torch.stack(v) for a,v in features448.items()}
 def summary(p):return torch.cat([t['global_features'],p.mean(1),p.max(1).values],1)
 xs={'global':t['global_features'],'grid224':summary(t['grid_features']),'grid448':summary(features448['grid']),'gaze224':summary(t['gaze_baseline_features']),'gaze448':summary(features448['gaze_baseline']),'random224':summary(t['random_features'])};preds={};report=dict(results=[],base_metrics=[],paired={},meta_coefficients={})
 for seed in a.seeds:
  folds=patient_folds(f,seed);assert all(set(partners[int(i)])<=set(hold.tolist()) for _,hold in folds for i in hold);manifest['folds'][str(seed)]=[dict(fit_patients=sorted(f.iloc[fit].patient_key.unique()),heldout_patients=sorted(f.iloc[hold].patient_key.unique())) for fit,hold in folds];seedout=out/f'seed{seed}';seedout.mkdir();oof=np.full((len(f),len(BASES)),np.nan);vlog=np.empty((len(va),len(BASES)));manifest['stage']='oof_base_training';save()
  for col,name in enumerate(BASES):
   for fold,(fit,hold) in enumerate(folds):
    p,ck=fit_base(xs[name],y,fit,hold,seed+fold+100);oof[hold,col]=p;torch.save(ck,seedout/f'{name}_fold{fold}.pt');print(json.dumps(dict(seed=seed,base=name,fold=fold,state='complete')),flush=True)
   p,ck=fit_base(xs[name],y,tr,va,seed);vlog[:,col]=p;torch.save(ck,seedout/f'{name}_fulltrain.pt');report['base_metrics'].append(dict(seed=seed,base=name,train_oof=metrics(y[tr].numpy(),1/(1+np.exp(-np.clip(oof[tr,col],-20,20)))),val=metrics(y[va].numpy(),1/(1+np.exp(-np.clip(p,-20,20))))));assert np.isfinite(oof[tr,col]).all()
  np.savez(seedout/'base_logits.npz',train_ids=f.iloc[tr].image_id.values,val_ids=f.iloc[va].image_id.values,oof=oof[tr],validation=vlog)
  manifest['stage']='meta_train_and_val_once';save()
  for mode in ['singleview','multiview']:
   tx=oof[tr] if mode=='singleview' else partner_logits(oof[tr],tr,f,partners);vx=vlog if mode=='singleview' else partner_logits(vlog,va,f,partners)
   for arm,names in FUSIONS.items():
    cols=[BASES.index(n) for n in names];prob,coeff=fit_meta(tx[:,cols],y[tr].numpy(),vx[:,cols]);key=f'{arm}_{mode}_seed{seed}';report['meta_coefficients'][key]=coeff;pred=f.iloc[va][['image_id','patient_key','label']].copy();pred['probability']=prob;pred.to_csv(seedout/f'{arm}_{mode}_validation_predictions.csv',index=False);preds[(arm,mode,seed)]=pred;r=dict(seed=seed,arm=arm,mode=mode,metrics=metrics(pred.label,pred.probability));manifest['results'].append(r);report['results'].append(r);save();print(json.dumps(r),flush=True)
 for seed in a.seeds:
  for mode in ['singleview','multiview']:
   for arm in FUSIONS:
    if arm!='gridcontext':report['paired'][f'{arm}_{mode}_{seed}_minus_gridcontext']=paired_ci(preds[(arm,mode,seed)],preds[('gridcontext',mode,seed)],1000)
   report['paired'][f'gaze224_{mode}_{seed}_minus_matchedrandom224']=paired_ci(preds[('gridcontext_gaze224',mode,seed)],preds[('gridcontext_random224',mode,seed)],1000)
 report['means']={f'{arm}_{mode}':float(np.mean([r['metrics']['auroc'] for r in report['results'] if r['arm']==arm and r['mode']==mode])) for arm in FUSIONS for mode in ['singleview','multiview']};(out/'report.json').write_text(json.dumps(report,indent=2));assert digest(__file__)==manifest['source_sha256'] and digest(cp)==manifest['cache_sha256'] and all(digest(p)==h for p,h in manifest['dependencies'].items());manifest['state']='complete';manifest['stage']='report_complete';save()
if __name__=='__main__':main()
