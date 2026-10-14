"""Resumable fixedGrid14+K predictedgaze/random ablation; completedgoal untouched."""
import argparse,copy,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.grid14_patchcount_ablation_1013 import KS,pack_nested,slice_bag
from src.official_cbis_classifier_1012 import digest,restore,metrics,state_digest
from src.gaze_attention_1011 import attention_loss,head_state
from scripts.train_gaze_multiview_1012 import pair_views
from scripts.extend_quota_three_seed_epoch120_1012 import WeightedAUC

SEEDS=(42,43,44)
METRICS=('auroc','auprc','accuracy','sensitivity','specificity','balanced_accuracy','brier','f1')

def atomic_json(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def atomic_csv(path,frame):
 temp=path.with_suffix(path.suffix+'.tmp');frame.to_csv(temp,index=False);temp.replace(path)

def classifier_step(model,t,bag,ids,y,opt):
 opt.zero_grad();out=model(t['global_features'][ids],bag[0][ids],bag[1][ids],bag[2][ids]);loss=attention_loss(out,y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();return float(loss.detach())

@torch.inference_mode()
def predict(model,t,bag,ids):
 model.eval();return torch.cat([model(t['global_features'][part],bag[0][part],bag[1][part],bag[2][part])['logits'].sigmoid() for part in ids.split(32)]).numpy()

def bootstrap(frame,ps,repetitions=5000):
 codes,patients=pd.factorize(frame.patient_key);y=frame.label.to_numpy();cs={key:[WeightedAUC(y,p) for p in probs] for key,probs in ps.items()};samples={key:[] for key in ps if key!='baseline'};rng=np.random.default_rng(42)
 for _ in range(repetitions):
  count=np.bincount(rng.integers(len(patients),size=len(patients)),minlength=len(patients));w=count[codes];scores={key:float(np.mean([calc(w) for calc in calcs])) for key,calcs in cs.items()}
  if not all(np.isfinite(x) for x in scores.values()):continue
  for key in samples:samples[key].append(scores[key]-scores['baseline'])
 point={key:float(np.mean([calc(np.ones(len(y))) for calc in calcs])) for key,calcs in cs.items()}
 return {key:dict(delta=point[key]-point['baseline'],low=float(np.quantile(x,.025)),high=float(np.quantile(x,.975)),valid_replicates=len(x)) for key,x in samples.items()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--benchmark-only',action='store_true');a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=True);start=time.time();cp='runs/official_cbis_v2_experiment_1012/development.pt';origin=Path('runs/gaze_diagnosis_selector_1012_round10_pilot42');pilot=Path('runs/gaze_coverage_quota_1012_round12_pilot42');old=Path('runs/gaze_coverage_quota_1012_round13_epoch120_three_seed');om=json.loads((old/'manifest.json').read_text());pm=json.loads((pilot/'manifest.json').read_text());assert om['state']==pm['state']=='complete';assert digest(cp)==om['cache_sha256'];assert all(digest(p)==h for p,h in om['dependencies'].items());cache=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors'];assert len(f)==2393 and set(f.split)=={'train','val'} and (f.split=='val').sum()==356 and f.groupby('patient_key').split.nunique().max()==1;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));vf=f.iloc[va.tolist()][['image_id','patient_key','label']].reset_index(drop=True);assert vf.patient_key.nunique()==183;y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f)
 sources=['scripts/run_grid14_patchcount_ablation_1013.py','src/grid14_patchcount_ablation_1013.py','scripts/train_gaze_coverage_quota_1012.py','scripts/train_gaze_diagnosis_selector_1012.py','scripts/train_gaze_multiview_1012.py','src/gaze_attention_1011.py','src/official_cbis_classifier_1012.py','scripts/extend_quota_three_seed_epoch120_1012.py'];deps={p:digest(p) for p in sources};deps.update({str(old/'manifest.json'):digest(old/'manifest.json'),str(pilot/'manifest.json'):digest(pilot/'manifest.json')});packed={};selection={};audit={}
 for source in ['gaze_baseline','random']:
  path=origin/f'selector_{source}/selected_indices.pt';r=torch.load(path,map_location='cpu',weights_only=False);assert r['image_ids']==f.image_id.tolist() and r['candidate_source']==source and r['kind']=='patientOOFTRAIN_fullTRAINVAL_hardselection';assert digest(path)==pm['dependencies'][str(path)];deps[str(path)]=digest(path);packed[source],selection[source]=pack_nested(t,source,r);indices=selection[source]['indices'];reference=torch.load(pilot/f'grid14_{source}2_indices.pt',map_location='cpu',weights_only=False);assert torch.equal(indices[:,:16],reference['indices']);assert state_digest(dict(features=slice_bag(packed[source],2)[0],coordinates=slice_bag(packed[source],2)[1],mask=slice_bag(packed[source],2)[2]))==pm['bags_sha256'][f'grid14_{source}2'];deps[str(pilot/f'grid14_{source}2_indices.pt')]=digest(pilot/f'grid14_{source}2_indices.pt');audit[source]={}
  for k in KS:
   b=slice_bag(packed[source],k);audit[source][str(k)]={split:dict(n=int((f.split==split).sum()),valid_source_mean=float((b[2].sum(1)-14)[torch.tensor(np.flatnonzero(f.split==split))].float().mean()),padding_points=int((~b[2])[torch.tensor(np.flatnonzero(f.split==split))].sum()),zero_source_images=int(((b[2].sum(1)-14)==0)[torch.tensor(np.flatnonzero(f.split==split))].sum()),distance_relaxed_points=int(selection[source]['relaxed'][torch.tensor(np.flatnonzero(f.split==split)),:k].sum())) for split in ['train','val']}
  path=out/f'{source}_nested_selection.pt'
  if path.exists():existing=torch.load(path,map_location='cpu',weights_only=False);assert torch.equal(existing['indices'],indices) and torch.equal(existing['available'],selection[source]['available'])
  else:torch.save(dict(image_ids=f.image_id.tolist(),**selection[source]),path)
  deps[str(path)]=digest(path)
 assert torch.equal(packed['gaze_baseline'][0][:,:14],packed['random'][0][:,:14]) and torch.equal(selection['gaze_baseline']['indices'][:,:14],selection['random']['indices'][:,:14]);grid14=slice_bag(packed['gaze_baseline'],0);heads=Path('runs/official_cbis_v2_experiment_1012/heads')
 # Benchmark fresh copy on TRAIN only; no checkpoint/model selection and no old artifact mutation.
 if a.benchmark_only:
  seed=42;gc=torch.load(heads/'global_seed42/best.pt',map_location='cpu',weights_only=False);hc=torch.load(heads/'grid_seed42/best.pt',map_location='cpu',weights_only=False);measure=[]
  for k in [0,16]:
   torch.manual_seed(seed);model=restore(hc,gc);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);bag=grid14 if k==0 else slice_bag(packed['gaze_baseline'],k);model.train();classifier_step(model,t,bag,tr[:32],y,opt);tick=time.monotonic()
   for ids in tr[:256].split(32):classifier_step(model,t,bag,ids,y,opt)
   seconds=(time.monotonic()-tick)/8;measure.append(dict(k=k,step_seconds=seconds,full120epoch_head_minutes=seconds*64*120/60))
  estimate=27*float(np.mean([x['full120epoch_head_minutes'] for x in measure]));atomic_json(out/'benchmark.json',dict(state='complete',measurements=measure,upper_bound27fresh_heads_minutes=estimate,earlystop_expected=True,k2_exact_bag_reproduction=True,test_used=False));print(json.dumps(dict(benchmark=measure,upper_bound27fresh_heads_minutes=estimate)),flush=True);return
 configs=[('baseline',0)]+[(s,k) for s in ['gaze_baseline','random'] for k in KS if k];tasks=[dict(source=s,k=k,seed=seed,status='planned',best_epoch=None,latest_epoch=None) for s,k in configs for seed in SEEDS];mp=out/'manifest.json'
 if mp.exists():m=json.loads(mp.read_text());assert m['configuration']['ks']==list(KS) and m['dependencies']==deps;tasks=m['tasks']
 else:m=dict(state='running',stage='k2_prediction_reuse_audit',start_time=start,task_type='ordinaryablation;completednativegoaluntouched',configuration=dict(ks=list(KS),seeds=list(SEEDS),grid14_exactfixed=True,nested_source_prefix=True,max_budget='14+K;actualvalidreported',candidate_acquisition32=True,selection='originalabsOOFscore,96pxgreedysequence→relaxedremainingrank,maskpadonly,nogridfallback',epochs=120,patience=12,lr=.0003,weight_decay=.01,batch=32,head='originalmatchedgridmean_residualdim64dropout.2,frozenglobal',selection_metric='singleviewVALAUROC,tiesfirst,multiviewfixedprobavg',upstream_seed=42,test_used=False,oracle_used=False,globalbranch_immutable=True),dependencies=deps,cache_sha256=digest(cp),selection_audit=audit,tasks=tasks,checkpoints={},k2_reuse={},results=[],grid16_reference={})
 def tracking():
  m['tasks']=tasks;m['completed_heads']=sum(x['status'] in ['complete','reused'] for x in tasks);atomic_json(mp,m);atomic_csv(out/'results_tracking.csv',pd.DataFrame(tasks))
 tracking();preds={};epoch0={'gaze_baseline':False,'random':False}
 def paths(source,k,seed):return out/f'{source}_k{k}_seed{seed}'
 def write_metrics(task,p,mv,best_epoch,stopped_epoch,reused):
  task.update(status='reused' if reused else 'complete',best_epoch=best_epoch,latest_epoch=stopped_epoch)
  if task['source']=='baseline':task.update(valid_source_train_mean=0.,valid_source_val_mean=0.,nominal_patch_budget=14)
  else:task.update(valid_source_train_mean=audit[task['source']][str(task['k'])]['train']['valid_source_mean'],valid_source_val_mean=audit[task['source']][str(task['k'])]['val']['valid_source_mean'],nominal_patch_budget=14+task['k'])
  for mode,data in [('singleview',p),('multiview',mv)]:
   score=metrics(data.label,data.probability)
   for key in METRICS:task[f'{mode}_{key}']=score[key]
  task['best_singleview_auroc']=task['singleview_auroc'];tracking()
 for task in tasks:
  source,k,seed=task['source'],task['k'],task['seed'];d=paths(source,k,seed)
  if task['status'] in ['complete','reused']:
   for mode,name in [('singleview','validation_predictions.csv'),('multiview','multiview_validation_predictions.csv')]:preds[(source,k,mode,seed)]=pd.read_csv(d/name)
   continue
  gp=heads/f'global_seed{seed}/best.pt';hp=heads/f'grid_seed{seed}/best.pt';gc=torch.load(gp,map_location='cpu',weights_only=False);hc=torch.load(hp,map_location='cpu',weights_only=False);m.setdefault('initial_checkpoint_hashes',{}).update({str(gp):digest(gp),str(hp):digest(hp)});bag=grid14 if k==0 else slice_bag(packed[source],k);d.mkdir(exist_ok=True)
  if k==2:
   reference=old/f'grid14_{source}2_seed{seed}';ck=torch.load(reference/'best.pt',map_location='cpu',weights_only=False);model=restore(hc,gc);missing,unexpected=model.load_state_dict(ck['state'],strict=False);assert not unexpected and all(x.startswith('global_model.') for x in missing);p=pd.read_csv(reference/'validation_predictions.csv');mv=pd.read_csv(reference/'multiview_validation_predictions.csv');replay=predict(model,t,bag,va);assert np.array_equal(replay,p.probability.to_numpy(dtype=np.float32));assert p[['image_id','patient_key','label']].equals(vf);atomic_csv(d/'validation_predictions.csv',p);atomic_csv(d/'multiview_validation_predictions.csv',mv);m['k2_reuse'][f'{source}_{seed}']=dict(bag_exact=True,probabilities_float32_exact=True,reference_sha256={str(reference/name):digest(reference/name) for name in ['best.pt','history.json','validation_predictions.csv','multiview_validation_predictions.csv']});preds[(source,k,'singleview',seed)]=p;preds[(source,k,'multiview',seed)]=mv;hist=json.loads((reference/'history.json').read_text());write_metrics(task,p,mv,ck['epoch'],hist[-1]['epoch'],True);continue
  torch.manual_seed(seed);model=restore(hc,gc);globalsha=state_digest(model.global_model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);best=-1;stale=0;history=[];first_epoch=0;latest=d/'latest.pt'
  if latest.exists():
   checkpoint=torch.load(latest,map_location='cpu',weights_only=False);model.load_state_dict(checkpoint['state'],strict=False);opt.load_state_dict(checkpoint['optimizer']);torch.set_rng_state(checkpoint['rng']);history=checkpoint['history'];best=checkpoint['best'];stale=checkpoint['stale'];bp=checkpoint['best_probability'];be=checkpoint['best_epoch'];first_epoch=checkpoint['epoch']+1
  task['status']='running';m['stage']='classifier_training';tracking()
  epoch=first_epoch-1
  for epoch in ([] if latest.exists() and stale>=12 else range(first_epoch,121)):
   losses=[]
   if epoch:
    model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))]
    for ids in order.split(32):losses.append(classifier_step(model,t,bag,ids,y,opt))
   prob=predict(model,t,bag,va);score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,fitloss=float(np.mean(losses)) if losses else None,validation=score));assert state_digest(model.global_model.state_dict())==globalsha
   if score['auroc']>best:
    best=score['auroc'];stale=0;bp=prob.copy();be=epoch;temp=d/'best.tmp';torch.save(dict(state=copy.deepcopy(head_state(model)),epoch=epoch,source=source,k=k,seed=seed,frozen_global_sha256=globalsha),temp);temp.replace(d/'best.pt')
   else:stale+=1
   temp=d/'latest.tmp';torch.save(dict(state=head_state(model),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=epoch,history=history,best=best,stale=stale,best_probability=bp,best_epoch=be),temp);temp.replace(latest);task.update(latest_epoch=epoch,best_epoch=be,best_singleview_auroc=best,latest_singleview_auroc=score['auroc']);tracking()
   if stale>=12:break
  p=vf.copy();p['probability']=bp;mv=p.copy();pmap=dict(zip(p.image_id,bp));mv['probability']=[.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()];atomic_csv(d/'validation_predictions.csv',p);atomic_csv(d/'multiview_validation_predictions.csv',mv);atomic_json(d/'history.json',history);m['checkpoints'][str(d/'best.pt')]=digest(d/'best.pt');preds[(source,k,'singleview',seed)]=p;preds[(source,k,'multiview',seed)]=mv;write_metrics(task,p,mv,be,history[-1]['epoch'],False);print(json.dumps(dict(source=source,k=k,seed=seed,status='complete',best_epoch=be,single_auroc=task['singleview_auroc'],multi_auroc=task['multiview_auroc'],completed=m['completed_heads'],total=33)),flush=True)
 m['stage']='summary_and_bootstrap';tracking();summaries=[];probabilities={};ensemble={};independent={}
 for source,k in configs:
  for mode in ['singleview','multiview']:
   ps=[];aucs=[]
   for seed in SEEDS:
    path=paths(source,k,seed)/('validation_predictions.csv' if mode=='singleview' else 'multiview_validation_predictions.csv');p=pd.read_csv(path);assert p[['image_id','patient_key','label']].equals(vf) and len(p)==356 and np.isfinite(p.probability).all() and p.probability.between(0,1).all();score=metrics(p.label,p.probability);tracked=next(task for task in tasks if task['source']==source and task['k']==k and task['seed']==seed)
    for metric in METRICS:assert abs(score[metric]-tracked[f'{mode}_{metric}'])<(2e-7 if metric=='brier' else 1e-12)
    ps.append(p.probability.to_numpy());aucs.append(score['auroc']);independent[str(path)]=dict(sha256=digest(path),metrics={key:score[key] for key in METRICS},n=len(p),identity_exact=True,finite_probability=True,reported_metrics_recomputed=True)
   probabilities[(source,k,mode)]=ps;meanp=np.mean(ps,0);ensemble[f'{source}_k{k}_{mode}']=metrics(vf.label,meanp);ep=vf.copy();ep['probability']=meanp;atomic_csv(out/f'{source}_k{k}_{mode}_ensemble.csv',ep);summaries.append(dict(source=source,k=k,mode=mode,nominal_patch_budget=14+k,mean_auroc=float(np.mean(aucs)),sd_auroc=float(np.std(aucs,ddof=1)),seed42_auroc=aucs[0],seed43_auroc=aucs[1],seed44_auroc=aucs[2],ensemble_auroc=ensemble[f'{source}_k{k}_{mode}']['auroc']))
 for mode in ['singleview','multiview']:
  ps=[]
  for seed in SEEDS:
   path=old/f'grid_seed{seed}'/('validation_predictions.csv' if mode=='singleview' else 'multiview_validation_predictions.csv');p=pd.read_csv(path);assert p[['image_id','patient_key','label']].equals(vf);ps.append(p.probability.to_numpy());m.setdefault('reference_file_hashes',{})[str(path)]=digest(path)
  m['grid16_reference'][mode]=dict(mean_auroc=float(np.mean([metrics(vf.label,p)['auroc'] for p in ps])),sd_auroc=float(np.std([metrics(vf.label,p)['auroc'] for p in ps],ddof=1)));probabilities[('grid16',16,mode)]=ps
 summary=pd.DataFrame(summaries);atomic_csv(out/'summary.csv',summary);ci={}
 for mode in ['singleview','multiview']:
  p={'baseline':probabilities[('baseline',0,mode)]};p.update({f'{s}_k{k}':probabilities[(s,k,mode)] for s,k in configs if k});ci['vs_grid14_'+mode]=bootstrap(vf,p,5000)
  for k in KS[1:]:ci[f'gaze_vs_sameKrandom_{k}_{mode}']=bootstrap(vf,{'baseline':probabilities[('random',k,mode)],'gaze':probabilities[('gaze_baseline',k,mode)]},5000)['gaze']
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for ax,plot_mode in zip(axes,['singleview','multiview']):
  base=summary[(summary.source=='baseline')&(summary.mode==plot_mode)].iloc[0]
  for source,label,color in [('gaze_baseline','Predicted gaze','#2267a3'),('random','Matched random','#cf7b28')]:
   rows=summary[(summary.source==source)&(summary.mode==plot_mode)].sort_values('k');xs=[0]+rows.k.tolist();ys=[base.mean_auroc]+rows.mean_auroc.tolist();sd=[base.sd_auroc]+rows.sd_auroc.tolist();ax.errorbar(xs,ys,yerr=sd,marker='o',capsize=3,label=label,color=color)
  ax.axhline(base.mean_auroc,color='gray',linestyle=':',label='Grid14 baseline');ax.axhline(m['grid16_reference'][plot_mode]['mean_auroc'],color='#555555',linestyle='--',label='Grid16 reference');ax.set_xticks(KS);ax.set_xlabel('Additional source patches K (Grid14 fixed; budget 14 + K)');ax.set_ylabel('Validation mean classifier-seed AUROC');ax.set_title(plot_mode+' / fixed Grid14');ax.legend();ax.grid(alpha=.2)
 fig.text(.02,.01,'Error bars: SD across classifier seeds 42/43/44. Patient-sampling 95% CIs: paired_ci.json (5000 cluster resamples).',fontsize=8);fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'auroc_vs_k.png',dpi=180);plt.close(fig)
 for d in [m['dependencies'],m.get('initial_checkpoint_hashes',{}),m.get('reference_file_hashes',{}),m.get('checkpoints',{})]:assert all(digest(path)==sha for path,sha in d.items())
 for reuse in m['k2_reuse'].values():assert all(digest(path)==sha for path,sha in reuse['reference_sha256'].items())
 atomic_json(out/'paired_ci.json',ci);atomic_json(out/'report.json',dict(summary=summaries,grid16_reference=m['grid16_reference'],ensemble_metrics=ensemble,patient_CI=ci,selection_audit=audit,independent_csv_audit=independent,limitations='Fixedclassifierseeds/upstreamseed42;variable14+Kbudget/maskedavailablecount;predictedgazeonly,novalidationcoefficienttuning,noTEST. ReusedexploratoryVAL. SeedSD differs from patient-samplingCI. NominalsameK comparisons may have differentactualvalidsourcecounts duepixelduplication/masking;countsareexplicitandcannotbeclaimedstrictlyequalinformation.'))
 m.update(state='complete',stage='report_complete',finish_time=time.time());tracking();print(json.dumps(dict(state='complete',heads=33,fresh_heads=27,reused_k2_heads=6,elapsed_minutes=(time.time()-start)/60)),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  if '--output' in sys.argv:
   out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(exist_ok=True);atomic_json(out/'failure.json',dict(state='failed',error=repr(error)));path=out/'manifest.json'
   if path.exists():m=json.loads(path.read_text());m.update(state='failed',error=repr(error));atomic_json(path,m)
  raise
