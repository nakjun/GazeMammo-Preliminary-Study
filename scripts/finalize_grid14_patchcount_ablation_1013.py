"""Read-only model/CSV audit and recovery of final ablation plots/reports."""
import json,sys,os,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from scripts.run_grid14_patchcount_ablation_1013 import atomic_json,atomic_csv,bootstrap,METRICS
from src.official_cbis_classifier_1012 import digest,metrics
from src.grid14_patchcount_ablation_1013 import KS

def main():
 torch.set_num_threads(4);out=Path('runs/grid14_patchcount_ablation_1013');mp=out/'manifest.json';m=json.loads(mp.read_text());assert len(m['tasks'])==33 and all(t['status'] in ['complete','reused'] for t in m['tasks']);status=out/'postprocess_status.json';audit_source=digest(__file__);atomic_json(status,dict(state='running',stage='independent_csv_audit',source_sha256=audit_source,training_restart=False,original_failure=json.loads((out/'failure.json').read_text())))
 initial_hashes={str(p):digest(p) for p in sorted(out.glob('*_k*_seed*/*.csv'))};initial_hashes.update({str(p):digest(p) for p in sorted(out.glob('*_k*_seed*/best.pt'))});deps={**m['dependencies'],**m['initial_checkpoint_hashes'],**m.get('reference_file_hashes',{}),**m['checkpoints']};assert all(digest(p)==h for p,h in deps.items());assert len(m['checkpoints'])==27 and len(m['k2_reuse'])==6
 for entry in m['k2_reuse'].values():assert entry['bag_exact'] and entry['probabilities_float32_exact'] and all(digest(p)==h for p,h in entry['reference_sha256'].items())
 c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',map_location='cpu',weights_only=False,mmap=True);f=pd.DataFrame(c['rows']);vf=f[f.split=='val'][['image_id','patient_key','label']].reset_index(drop=True);assert len(vf)==356 and vf.patient_key.nunique()==183 and not set(f[f.split=='train'].patient_key)&set(vf.patient_key)
 a=torch.load(out/'gaze_baseline_nested_selection.pt',map_location='cpu',weights_only=False);b=torch.load(out/'random_nested_selection.pt',map_location='cpu',weights_only=False);assert torch.equal(a['indices'][:,:14],b['indices'][:,:14]) and a['available'][:,:14].all() and b['available'][:,:14].all();assert a['image_ids']==b['image_ids']==f.image_id.tolist()
 rows=[];ps={};csv_audit={};ensemble={};summary=[];configs=[('baseline',0)]+[(s,k) for s in ['gaze_baseline','random'] for k in KS if k]
 for source,k in configs:
  for mode,name in [('singleview','validation_predictions.csv'),('multiview','multiview_validation_predictions.csv')]:
   probabilities=[];scores=[]
   for seed in [42,43,44]:
    path=out/f'{source}_k{k}_seed{seed}'/name;p=pd.read_csv(path);assert p[['image_id','patient_key','label']].equals(vf) and np.isfinite(p.probability).all() and p.probability.between(0,1).all();score=metrics(p.label,p.probability);task=next(x for x in m['tasks'] if (x['source'],x['k'],x['seed'])==(source,k,seed))
    for metric in METRICS:assert abs(score[metric]-task[f'{mode}_{metric}'])<(2e-7 if metric=='brier' else 1e-12)
    probabilities.append(p.probability.to_numpy());scores.append(score);csv_audit[str(path)]=dict(sha256=digest(path),metrics=score,identity_exact=True,n=356,finite_probability=True)
   ps[(source,k,mode)]=probabilities;meanp=np.mean(probabilities,0);ensemble[f'{source}_k{k}_{mode}']=metrics(vf.label,meanp);record=dict(source=source,k=k,mode=mode,nominal_patch_budget=14+k,mean_auroc=float(np.mean([s['auroc'] for s in scores])),sd_auroc=float(np.std([s['auroc'] for s in scores],ddof=1)),seed42_auroc=scores[0]['auroc'],seed43_auroc=scores[1]['auroc'],seed44_auroc=scores[2]['auroc'],ensemble_auroc=ensemble[f'{source}_k{k}_{mode}']['auroc'])
   for metric in METRICS:record['mean_'+metric]=float(np.mean([s[metric] for s in scores]));record['sd_'+metric]=float(np.std([s[metric] for s in scores],ddof=1))
   record['valid_source_val_mean']=0. if k==0 else m['selection_audit'][source][str(k)]['val']['valid_source_mean'];record['valid_total_val_mean']=14+record['valid_source_val_mean'];summary.append(record)
 summary=pd.DataFrame(summary);atomic_csv(out/'summary.csv',summary);ci={};atomic_json(status,dict(state='running',stage='patient_bootstrap5000',source_sha256=audit_source,training_restart=False))
 for mode in ['singleview','multiview']:
  probabilities={'baseline':ps[('baseline',0,mode)]};probabilities.update({f'{s}_k{k}':ps[(s,k,mode)] for s,k in configs if k});ci['vs_grid14_'+mode]=bootstrap(vf,probabilities,5000)
  for k in KS[1:]:ci[f'gaze_vs_sameKrandom_{k}_{mode}']=bootstrap(vf,{'baseline':ps[('random',k,mode)],'gaze':ps[('gaze_baseline',k,mode)]},5000)['gaze']
 os.environ['MPLCONFIGDIR']='/tmp/grid14_ablation_mpl';import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(13,5))
 for ax,mode in zip(axes,['singleview','multiview']):
  baseline=summary[(summary['source']=='baseline')&(summary['mode']==mode)].iloc[0]
  for source,label,color in [('gaze_baseline','Predicted gaze','#2267a3'),('random','Matched random','#cf7b28')]:
   s=summary[(summary['source']==source)&(summary['mode']==mode)].sort_values('k');ax.errorbar([0]+s.k.tolist(),[baseline.mean_auroc]+s.mean_auroc.tolist(),yerr=[baseline.sd_auroc]+s.sd_auroc.tolist(),marker='o',capsize=3,label=label,color=color)
  ax.axhline(baseline.mean_auroc,color='gray',linestyle=':',label='Grid14 baseline');ax.axhline(m['grid16_reference'][mode]['mean_auroc'],color='#555555',linestyle='--',label='Grid16 reference');ax.set_xticks(KS);ax.set_xlabel('Additional source patches K (Grid14 fixed; budget 14 + K)');ax.set_ylabel('Validation mean classifier-seed AUROC');ax.set_title(mode+' / fixed Grid14');ax.legend();ax.grid(alpha=.2)
 fig.text(.015,.015,'Error bars: classifier-seed SD (42/43/44). Patient-sampling 95% CIs: paired_ci.json, 5000 cluster resamples. K16 valid counts may differ.',fontsize=8);fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'auroc_vs_k.png',dpi=180);plt.close(fig)
 assert all(digest(p)==h for p,h in initial_hashes.items()) and all(digest(p)==h for p,h in deps.items());audit=dict(state='passed',fresh_heads=27,reused_k2_heads=6,total_heads=33,csv_count=len(csv_audit),all_models_csv_unchanged_during_finalization=True,all_original_dependencies_unchanged=True,grid14_exactfixed_acrosssources=True,nested_prefix_preserved=True,maskpadding_no_gridfallback=True,val_images=356,val_patients=183,train_val_patient_disjoint=True,k2_6of6_original_predictions_exact=True,training_source_sha256=digest('scripts/run_grid14_patchcount_ablation_1013.py'),finalizer_source_sha256=audit_source,trained_checkpoint_sha256=m['checkpoints'],input_output_immutable_sha256=initial_hashes,test_used=False,oracle_used=False)
 atomic_json(out/'paired_ci.json',ci);atomic_json(out/'final_artifact_audit.json',audit);atomic_json(out/'report.json',dict(summary=summary.to_dict('records'),grid16_reference=m['grid16_reference'],ensemble_metrics=ensemble,patient_CI=ci,selection_audit=m['selection_audit'],independent_csv_audit=csv_audit,limitations='Ordinaryauthorizedablation;completedgoalnotreactivated. FixedGrid14,nominal14+Kbudget,sourceavailability/paddingreported. NominalsameK may have unequalactualvalidcounts,gaze16vsrandom16interpretationlimited. Upstreamseed42fixed,predictedgazeonly;classifierseedSD differs frompatientCI. ExploratoryfullVAL356,noTEST/expertsuppliedatdeployment.'))
 m.update(state='complete',stage='report_complete',finish_time=time.time(),postprocess_recovery=dict(reason='plotDataFrame.modeattributecollision',source_sha256=audit_source,training_restarted=False));atomic_json(mp,m);atomic_json(status,dict(state='complete',stage='report_complete',source_sha256=audit_source,training_restart=False));print(json.dumps(dict(state='complete',heads=33,csv_audit=len(csv_audit),plot=str(out/'auroc_vs_k.png'),multiview=summary[summary['mode']=='multiview'][['source','k','mean_auroc','sd_auroc','valid_source_val_mean']].to_dict('records'))),flush=True)
if __name__=='__main__':main()
