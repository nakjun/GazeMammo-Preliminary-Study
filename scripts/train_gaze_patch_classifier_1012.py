"""Train fresh matched controls and identical patch heads without TEST access."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from src.gaze_patch_classifier_1012 import ARMS,digest,train_head,metrics,paired_ci,implementation_hashes
from src.gaze_patch_features_1012 import validate_feature_cache

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--epochs',type=int,default=60);parser.add_argument('--seeds',type=int,nargs='+',default=[42,43,44])
    parser.add_argument('--k',type=int,choices=[4,8,16],default=16);parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.epochs<1 or args.epochs>60 or args.bootstrap<1: parser.error('epochs must be 1..60 and bootstrap positive')
    if args.smoke and (args.seeds!=[42] or args.epochs!=2): parser.error('Smoke requires --seeds 42 --epochs 2')
    if not args.smoke and args.seeds!=[42,43,44]: parser.error('Main protocol requires seeds 42 43 44')
    output=Path(args.output)
    if output.exists(): raise ValueError('Output must be new')
    cache=torch.load(args.cache,map_location='cpu',weights_only=False)
    frame=validate_feature_cache(cache)
    import pandas as pd
    if not isinstance(frame,pd.DataFrame): frame=pd.DataFrame(cache['rows'])
    expected='smoke8' if args.smoke else 'development469'
    if cache['provenance']['population']!=expected or len(frame)!=(8 if args.smoke else 469): raise ValueError('Explicit population protocol mismatch')
    torch.set_num_threads(args.threads);output.mkdir(parents=True)
    (output/'status.json').write_text(json.dumps(dict(state='running',smoke=args.smoke)))
    provenance=dict(cache['provenance'],cache_sha256=digest(args.cache))
    results={};seed_metrics=[]
    for seed in args.seeds:
        baseline,pred,score=train_head(cache['tensors'],frame,'global',seed,output/f'global_seed{seed}',provenance,args.epochs,args.k)
        results[('global',seed)]=pred;seed_metrics.append(dict(arm='global',seed=seed,metrics=score,best_epoch=baseline['epoch']))
        for arm in ARMS:
            checkpoint,pred,score=train_head(cache['tensors'],frame,arm,seed,output/f'{arm}_seed{seed}',provenance,args.epochs,args.k,baseline)
            results[(arm,seed)]=pred;seed_metrics.append(dict(arm=arm,seed=seed,metrics=score,best_epoch=checkpoint['epoch']))
    ensembles={};summary={}
    for arm in ('global',)+ARMS:
        members=[results[(arm,seed)] for seed in args.seeds];ensemble=members[0][['image_id','patient_key','label']].copy()
        ensemble['probability']=np.mean([p.probability.to_numpy() for p in members],axis=0)
        ensemble.to_csv(output/f'{arm}_ensemble_predictions.csv',index=False);ensembles[arm]=ensemble
        scores=[row['metrics'] for row in seed_metrics if row['arm']==arm]
        summary[arm]=dict(single_head_mean={key:float(np.mean([s[key] for s in scores])) for key in ('auroc','auprc','accuracy','balanced_accuracy','f1','sensitivity','specificity','brier')},single_head_sd={key:float(np.std([s[key] for s in scores],ddof=1)) if len(scores)>1 else 0. for key in ('auroc','auprc','accuracy','balanced_accuracy','f1','sensitivity','specificity','brier')},probability_ensemble=metrics(ensemble.label,ensemble.probability))
    pairs=[(arm,'global') for arm in ARMS]+[('gaze','grid'),('gaze','proposal')]
    paired={'ensemble':{a+' minus '+b:paired_ci(ensembles[a],ensembles[b],args.bootstrap) for a,b in pairs},'single_seed':{str(seed):{a+' minus '+b:paired_ci(results[(a,seed)],results[(b,seed)],args.bootstrap) for a,b in pairs} for seed in args.seeds}}
    report=dict(smoke=args.smoke,performance_evidence=not args.smoke,threshold=.5,k=args.k,seeds=args.seeds,seed_results=seed_metrics,summary=summary,paired_patient_bootstrap=paired,protocol='Frozen in-sample gaze transfer; reused exploratory validation; no TEST; selection CI does not remove validation selection bias',source_provenance=provenance,implementation_sha256=implementation_hashes())
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    lines=['# Gaze patch classifier development report', '', 'Smoke validation only; not performance evidence.' if args.smoke else 'Exploratory reused validation; frozen in-sample gaze transfer. No TEST used.', '', f'Patch budget K={args.k}; head seeds {args.seeds}; threshold 0.5.', '', '| Arm | Single-head AUROC mean ± SD | Probability ensemble AUROC |', '|---|---:|---:|']
    for arm, result in summary.items():
        lines.append(f"| {arm} | {result['single_head_mean']['auroc']:.4f} ± {result['single_head_sd']['auroc']:.4f} | {result['probability_ensemble']['auroc']:.4f} |")
    lines.extend(['', 'Paired patient bootstrap intervals and all classification metrics are in report.json. These intervals do not remove validation selection bias.', 'Attention and signed contributions are pooling explanations, not patch malignancy probabilities or lesion truth.'])
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    (output/'status.json').write_text(json.dumps(dict(state='complete',smoke=args.smoke,seeds=args.seeds,k=args.k)))
if __name__=='__main__': main()
