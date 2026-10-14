"""Round13 locked quota14/2, maximumepoch60→120 only, exactpriorprefixguard."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.official_cbis_classifier_1012 import digest,restore,metrics,state_digest
from src.gaze_attention_1011 import attention_loss,head_state
from scripts.train_gaze_diagnosis_selector_1012 import build_candidates
from scripts.train_gaze_multiview_1012 import pair_views
from sklearn.metrics import roc_auc_score

class WeightedAUC:
 def __init__(self,y,p):self.y=np.asarray(y);_,self.group=np.unique(np.asarray(p),return_inverse=True)
 def __call__(self,w):
  pos=np.bincount(self.group,weights=w*self.y);neg=np.bincount(self.group,weights=w*(1-self.y),minlength=len(pos));pn=pos.sum()*neg.sum()
  return float(np.sum(pos*(np.cumsum(neg)-.5*neg))/pn) if pn else float('nan')

def mean_seed_bootstrap(frame,probabilities,repetitions=5000):
 """Patient cluster resampling, then mean of3 AUROCs/deltas; not ensemble."""
 patient,keys=pd.factorize(frame.patient_key);y=frame.label.to_numpy();rng=np.random.default_rng(42);calculators={key:[WeightedAUC(y,p) for p in ps] for key,ps in probabilities.items()};samples={key:[] for key in ['gaze_minus_grid','gaze_minus_random']}
 for _ in range(repetitions):
  counts=np.bincount(rng.integers(len(keys),size=len(keys)),minlength=len(keys));w=counts[patient];scores={a:float(np.mean([c(w) for c in cs])) for a,cs in calculators.items()}
  if not all(np.isfinite(v) for v in scores.values()):continue
  samples['gaze_minus_grid'].append(scores['gaze']-scores['grid']);samples['gaze_minus_random'].append(scores['gaze']-scores['random'])
 return {k:dict(low=float(np.quantile(v,.025)),high=float(np.quantile(v,.975)),valid_replicates=len(v)) for k,v in samples.items()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);pilot=Path('runs/gaze_coverage_quota_1012_round12_pilot42');control=Path('runs/gaze_selector_extension_1012_round11_pilot42');prior=Path('runs/gaze_coverage_quota_1012_round12_three_seed');previous=json.loads((prior/'manifest.json').read_text());assert previous['state']=='complete' and digest('scripts/confirm_gaze_quota_three_seed_1012.py')==previous['source_sha256'];pm=json.loads((pilot/'manifest.json').read_text());cm=json.loads((control/'manifest.json').read_text());assert pm['state']==cm['state']=='complete';assert digest('scripts/train_gaze_coverage_quota_1012.py')==pm['source_sha256'];cp='runs/official_cbis_v2_experiment_1012/development.pt';assert digest(cp)==pm['cache_sha256']==cm['cache_sha256'];c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert len(f)==2393 and set(f.split)=={'train','val'} and (f.split=='val').sum()==356;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f);valframe=f.iloc[va.tolist()][['image_id','patient_key','label']].reset_index(drop=True);deps={str(pilot/'manifest.json'):digest(pilot/'manifest.json'),str(control/'manifest.json'):digest(control/'manifest.json')};deps[str(prior/'manifest.json')]=digest(prior/'manifest.json');deps['scripts/confirm_gaze_quota_three_seed_1012.py']=digest('scripts/confirm_gaze_quota_three_seed_1012.py');deps.update(pm['dependencies']);deps.update({p:digest(p) for p in ['scripts/train_gaze_coverage_quota_1012.py','scripts/train_gaze_diagnosis_selector_1012.py','scripts/train_gaze_multiview_1012.py','src/official_cbis_classifier_1012.py','src/gaze_attention_1011.py']});assert all(digest(p)==h for p,h in deps.items());bags={'grid':(t['grid_features'],t['grid_coordinates'],t['grid_mask'])}
 for source in ['gaze_baseline','random']:
  arm='grid14_'+source+'2';p=pilot/f'{arm}_indices.pt';r=torch.load(p,map_location='cpu',weights_only=False);assert r['image_ids']==f.image_id.tolist() and r['source']==source and r['rule']=='fixed14/2';indices=r['indices'];assert indices.shape==(2393,16) and (indices[:,:14]<16).all() and (indices[:,14:]>=16).all();rows=torch.arange(len(f))[:,None];bags[arm]=tuple(x[rows,indices] for x in build_candidates(t,source));assert state_digest(dict(features=bags[arm][0],coordinates=bags[arm][1],mask=bags[arm][2]))==pm['bags_sha256'][arm];deps[str(p)]=digest(p)
 m=dict(state='running',stage='seed42_reuse_audit',source_sha256=digest(__file__),cache_sha256=digest(cp),dependencies=deps,seeds=[42,43,44],classifier_seeds_only=True,upstream_selector_seed=42,upstream_gaze_paths_seed=42,quota=[14,2],bags_fixed=True,epochs=120,patience=12,lr=.0003,weight_decay=.01,batch=32,test_used=False,oracle_used=False,patch_budget=16,candidate_budget=32,multiview='Extra image samebreastoppositeview fixedprobavg; paired acquisition64candidates/32selected crops; missingcurrentonly',results=[],reuse_audit={},prefix_reproduction={},trained_checkpoint_hashes={})
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();preds={}
 def read_verified(path):
  p=pd.read_csv(path);assert p[['image_id','patient_key','label']].reset_index(drop=True).equals(valframe);m['dependencies'][str(path)]=digest(path);return p
 for arm in bags:
  directory=control/'grid_seed42' if arm=='grid' else pilot/f'{arm}_seed42'
  for mode,name in [('singleview','validation_predictions.csv'),('multiview','multiview_validation_predictions.csv')]:
   pred=read_verified(directory/name);preds[(arm,mode,42)]=pred;score=metrics(pred.label,pred.probability);pred.to_csv(out/f'{arm}_{mode}_seed42_reused_predictions.csv',index=False);m.setdefault('seed42_initial_csv_audit',[]).append(dict(arm=arm,mode=mode,seed=42,metrics=score))
  m['reuse_audit'][arm]='IDs/patients/labelsorderexact;metricsindependentlyrecomputed;inputSHAfixed'
 m['stage']='fresh_all3seeds120_training';save();heads=Path('runs/official_cbis_v2_experiment_1012/heads')
 for seed in [42,43,44]:
  gp=heads/f'global_seed{seed}/best.pt';hp=heads/f'grid_seed{seed}/best.pt';m['dependencies'].update({str(p):digest(p) for p in [gp,hp]});gc=torch.load(gp,map_location='cpu',weights_only=False);hc=torch.load(hp,map_location='cpu',weights_only=False)
  for arm,bag in bags.items():
   torch.manual_seed(seed);model=restore(hc,gc);globalsha=state_digest(model.global_model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);d=out/f'{arm}_seed{seed}';d.mkdir();best=-1;stale=0;history=[];old_directory=(control/'grid_seed42' if arm=='grid' else pilot/f'{arm}_seed42') if seed==42 else prior/f'{arm}_seed{seed}';old_history=json.loads((old_directory/'history.json').read_text());old_ck=torch.load(old_directory/'best.pt',map_location='cpu',weights_only=False);old_predictions=pd.read_csv(old_directory/'validation_predictions.csv').set_index('image_id').loc[valframe.image_id].probability.to_numpy(dtype=np.float32);reproduction_key=f'{arm}_seed{seed}';m['dependencies'].update({str(old_directory/name):digest(old_directory/name) for name in ['history.json','best.pt','validation_predictions.csv']})
   def forward(ids):return model(t['global_features'][ids],bag[0][ids],bag[1][ids],bag[2][ids])
   for epoch in range(121):
    losses=[]
    if epoch:
     model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))]
     for ids in order.split(32):
      opt.zero_grad();loss=attention_loss(forward(ids),y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():prob=torch.cat([forward(ids)['logits'].sigmoid() for ids in va.split(32)]).numpy()
    score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,fitloss=float(np.mean(losses)) if losses else None,validation=score));assert state_digest(model.global_model.state_dict())==globalsha
    if epoch<len(old_history) and history[-1]!=old_history[epoch]:raise RuntimeError(f'Priorprefixexacthistorymismatch:{reproduction_key}:epoch{epoch}')
    if epoch==old_ck['epoch']:
     assert np.array_equal(prob,old_predictions) and state_digest(head_state(model))==state_digest(old_ck['state']);m['prefix_reproduction'].setdefault(reproduction_key,{})['priorbestprediction_and_headstate_exact']=True
    if epoch==len(old_history)-1:
     m['prefix_reproduction'].setdefault(reproduction_key,{}).update(history_prefix_exact=True,prior_last_epoch=epoch,history_rows_checked=len(old_history));save();print(json.dumps(dict(arm=arm,seed=seed,prior_prefix_reproduction='passed',last_prior_epoch=epoch)),flush=True)
    if score['auroc']>best:best=score['auroc'];stale=0;bp=prob.copy();be=epoch;torch.save(dict(state=copy.deepcopy(head_state(model)),epoch=epoch,arm=arm,seed=seed,frozen_global_sha256=globalsha),d/'best.pt')
    else:stale+=1
    if stale>=12:break
   assert m['prefix_reproduction'][reproduction_key]['history_prefix_exact'];m['trained_checkpoint_hashes'][str(d/'best.pt')]=digest(d/'best.pt')
   pred=valframe.copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);preds[(arm,'singleview',seed)]=pred;pmap=dict(zip(pred.image_id,bp));mv=pred.copy();mv['probability']=[.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()];mv.to_csv(d/'multiview_validation_predictions.csv',index=False);preds[(arm,'multiview',seed)]=mv;(d/'history.json').write_text(json.dumps(history))
   for mode,p in [('singleview',pred),('multiview',mv)]:r=dict(arm=arm,mode=mode,seed=seed,epoch=be,stopped_epoch=epoch,reused=False,metrics=metrics(p.label,p.probability));m['results'].append(r);save();print(json.dumps(r),flush=True)
 report=dict(results=m['results'],prefix_reproduction=m['prefix_reproduction'],trained_checkpoint_hashes=m['trained_checkpoint_hashes'],summary={},mean_head_paired_patient_CI={},ensemble_metrics={},independent_csv_audit={})
 # Re-read every stored CSV and verify independent computation versus recorded metrics.
 for (arm,mode,seed),pred in list(preds.items()):
  path=out/f'{arm}_seed{seed}'/('validation_predictions.csv' if mode=='singleview' else 'multiview_validation_predictions.csv');p=read_verified(path);score=metrics(p.label,p.probability);reference=next(r['metrics'] for r in m['results'] if r['arm']==arm and r['mode']==mode and r['seed']==seed);assert abs(score['auroc']-reference['auroc'])<1e-12;preds[(arm,mode,seed)]=p;report['independent_csv_audit'][f'{arm}_{mode}_{seed}']=dict(sha256=digest(path),auroc=score['auroc'],exact_identity=True)
 for mode in ['singleview','multiview']:
  probabilities={}
  for arm,key in [('grid','grid'),('grid14_gaze_baseline2','gaze'),('grid14_random2','random')]:
   ps=[preds[(arm,mode,s)].probability.to_numpy() for s in [42,43,44]];scores=[roc_auc_score(valframe.label,p) for p in ps];probabilities[key]=ps;report['summary'][arm+'_'+mode]=dict(mean_auroc=float(np.mean(scores)),sd_auroc=float(np.std(scores,ddof=1)),seed_aurocs=scores);ensemble=valframe.copy();ensemble['probability']=np.mean(ps,axis=0);ensemble.to_csv(out/f'{arm}_{mode}_ensemble_predictions.csv',index=False);report['ensemble_metrics'][arm+'_'+mode]=metrics(ensemble.label,ensemble.probability)
  ci=mean_seed_bootstrap(valframe,probabilities,5000);ci['gaze_minus_grid']['delta']=report['summary']['grid14_gaze_baseline2_'+mode]['mean_auroc']-report['summary']['grid_'+mode]['mean_auroc'];ci['gaze_minus_random']['delta']=report['summary']['grid14_gaze_baseline2_'+mode]['mean_auroc']-report['summary']['grid14_random2_'+mode]['mean_auroc'];report['mean_head_paired_patient_CI'][mode]=ci
 report['goal']=dict(primary='multiviewgazeusing3classifierseedmeanAUROC',threshold=.85,unrounded_mean_auroc=report['summary']['grid14_gaze_baseline2_multiview']['mean_auroc'],numeric_threshold_achieved=report['summary']['grid14_gaze_baseline2_multiview']['mean_auroc']>=.85,limitations='Classifierseeds only; upstreamselectors/gaze paths fixedseed42. Multiview extraimage protocol. Positive gaze contribution must be judged separately with matchedcontrols/CIs. Exploratory VAL reused.')
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(__file__)==m['source_sha256'] and digest(cp)==m['cache_sha256'] and all(digest(p)==h for p,h in m['dependencies'].items()) and all(digest(p)==h for p,h in m['trained_checkpoint_hashes'].items());m['state']='complete';m['stage']='report_complete';m['goal']=report['goal'];save();print(json.dumps({'summary':report['summary'],'goal':report['goal'],'mean_head_CI':report['mean_head_paired_patient_CI']}),flush=True)
if __name__=='__main__':main()
