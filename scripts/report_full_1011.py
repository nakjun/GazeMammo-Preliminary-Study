"""Validate a completed paired suite and report matched image/patient comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METRICS=('raw_nll','information_gain_bits','normalized_dtw','return_rate_112px')


def read(path):return json.loads(Path(path).read_text())
def frame_ids(frame,expected,label):
    if 'image_id' not in frame or frame.image_id.duplicated().any():raise ValueError(f'{label}: duplicate/missing image IDs')
    if set(frame.image_id.astype(int))!=set(expected):raise ValueError(f'{label}: image ID set mismatch')
    return frame.sort_values('image_id').reset_index(drop=True)


def paired_bootstrap(frame,left,right,repeats=10000,seed=20261011):
    """Resample patients; retain all their paired images and image-weighted means."""
    values=frame[left].to_numpy(float)-frame[right].to_numpy(float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite paired values')
    groups=pd.DataFrame(dict(patient=frame.patient_key,delta=values)).groupby('patient').delta.agg(['sum','count'])
    if not len(groups):raise ValueError('Empty paired comparison')
    rng=np.random.default_rng(seed); samples=[];n=len(groups)
    for offset in range(0,repeats,1000):
        weights=rng.multinomial(n,np.full(n,1/n),size=min(1000,repeats-offset))
        samples.extend((weights@groups['sum'].to_numpy())/(weights@groups['count'].to_numpy()))
    low,high=np.quantile(samples,[.025,.975])
    return dict(delta=float(values.mean()),ci95=[float(low),float(high)],patients=n,images=len(frame),bootstrap_repeats=repeats,bootstrap_seed=seed,estimand='image mean difference; patient clusters resampled',difference='left minus right')


def load_model(suite,name,expected):
    run=suite/name;history=read(run/'history.json');config=read(run/'config.json')
    if len(history)!=15 or [r['epoch'] for r in history]!=list(range(1,16)):raise ValueError(f'{name}: expected complete 15 epochs')
    if config['seed']!=42 or config['epochs']!=15 or config.get('limit',0) or config.get('val_limit',0):raise ValueError(f'{name}: unexpected full-training configuration')
    for checkpoint in ('best.pt','last.pt'):
        if not (run/checkpoint).is_file():raise ValueError(f'{name}: missing {checkpoint}')
    best=min(history,key=lambda r:r['validation']['legacy16']['raw_nll'])
    eval_dir=suite/f'{name}_eval';provenance=read(eval_dir/'provenance.json')
    if provenance['independent_holdout']:raise ValueError('Unexpected independent holdout label for reused validation')
    if set(provenance['generated_image_ids'])!=set(expected):raise ValueError(f'{name}: evaluation provenance IDs')
    modes={}
    for mode in ('legacy16','contiguous'):
        frame=frame_ids(pd.read_csv(eval_dir/f'{mode}.csv'),expected,f'{name}/{mode}')
        for key in ('raw_nll','information_gain_bits'):
            if not np.isfinite(frame[key]).all():raise ValueError(f'{name}/{mode}: invalid {key}')
        modes[mode]=frame
    if not np.isclose(modes['legacy16'].raw_nll.mean(),best['validation']['legacy16']['raw_nll'],atol=2e-5):raise ValueError(f'{name}: saved evaluation differs from selected epoch')
    step_gates={}
    for mode in modes:
        steps=pd.read_csv(eval_dir/f'{mode}_steps.csv')
        if set(steps.image_id)!=set(expected):raise ValueError(f'{name}: gate step IDs')
        gates=steps[['gate_global','gate_local','gate_return']]
        if not np.isfinite(gates).all().all() or not np.allclose(gates.sum(axis=1),1,atol=1e-5):raise ValueError(f'{name}: invalid gates')
        step_gates[mode]=steps.groupby('image_id')[gates.columns.tolist()].mean().mean().to_dict()
    generations=[];generation_timings={}
    for strategy,seed in [('sample',42),('sample',43),('sample',44),('argmax',42)]:
        directory=suite/f'{name}_{strategy}_{seed}';cfg=read(directory/'config.json')
        if cfg['seed']!=seed or cfg['strategy']!=strategy or cfg['policy']!='none':raise ValueError(f'{name}: generation settings mismatch')
        predictions=read(directory/'predictions.json')
        if len(predictions)!=len(expected) or {int(k) for k in predictions}!=set(expected):raise ValueError(f'{name}: incomplete predictions')
        for image_id,points in predictions.items():
            points=np.asarray(points,float)
            if points.shape!=(16,2) or not np.isfinite(points).all() or (points<0).any() or (points>=1).any():raise ValueError(f'{name}: invalid prediction {image_id}')
        geometry=frame_ids(pd.DataFrame(read(directory/'geometry.json')),expected,f'{name}/{strategy}/{seed}')
        if not np.isfinite(geometry[['normalized_dtw','return_rate_112px']]).all().all():raise ValueError(f'{name}: missing geometry diagnostics')
        gp=read(directory/'provenance.json')
        if set(gp['generated_image_ids'])!=set(expected):raise ValueError(f'{name}: generation provenance IDs')
        geometry['strategy']=strategy;geometry['generation_seed']=seed;generations.append(geometry)
        generation_timings[f'{strategy}_{seed}']=read(directory/'timing.json')
    all_geometry=pd.concat(generations,ignore_index=True)
    sample=all_geometry[all_geometry.strategy=='sample'].groupby('image_id')[['normalized_dtw','return_rate_112px']].mean().reset_index()
    argmax=all_geometry[all_geometry.strategy=='argmax'][['image_id','normalized_dtw','return_rate_112px']]
    timing=read(run/'timing.json')
    details=dict(training_seed=42,best_epoch=best['epoch'],last_epoch=15,best_validation_raw_nll=best['validation']['legacy16']['raw_nll'],last_validation=history[-1]['validation'],training_seconds=timing['seconds'],peak_cuda_memory_bytes=timing['peak_cuda_memory_bytes'],evaluation_timing=read(eval_dir/'timing.json'),generation_timings=generation_timings,branch_gates_image_mean=step_gates,config=config,checkpoint_sha256=provenance['checkpoint_sha256'])
    return modes,sample,argmax,details


def align(left,right,expected,label):
    left=frame_ids(left,expected,label+'/left');right=frame_ids(right,expected,label+'/right')
    if not np.array_equal(left.image_id,right.image_id):raise ValueError('ID alignment failure')
    if 'patient_key' in right and not np.array_equal(left.patient_key,right.patient_key):raise ValueError('Patient key mismatch')
    return left.merge(right,on='image_id',suffixes=('_left','_right'),validate='one_to_one')


def report(suite,baseline,expected,patient_map,repeats=10000,seed=20261011):
    if repeats<1:raise ValueError('bootstrap repeats must be positive')
    suite=Path(suite);expected=set(expected)
    if not read(suite/'completion.json')['complete']:raise ValueError('Suite is not complete')
    status=read(suite/'status.json')
    if any(status[name]['state']!='complete' for name in ('improved','control')):raise ValueError('Workers are not complete')
    experiment=read(suite/'experiment.json')
    if experiment['test_used'] or experiment['training_images']!=412 or experiment['validation_images']!=93:raise ValueError('Suite population/test setting mismatch')
    models={name:load_model(suite,name,expected) for name in ('improved','control')};comparisons=[];ci={}
    matched=('init','backbone','dim','fpn','foveal','motion_bias','window','burn_in','image_width','sigma','prior_weight','nll_weight','batch_size','epochs','warmup_epochs','backbone_mode','backbone_lr','lr','weight_decay','seed','selection_mode')
    configs={name:models[name][3]['config'] for name in models}
    for key in matched:
        if configs['improved'].get(key)!=configs['control'].get(key):raise ValueError(f'Unmatched settings: {key}')
    if configs['improved'].get('mode')!='mixed' or configs['control'].get('mode')!='legacy16':raise ValueError('Unexpected comparison modes')
    for name,(modes,sample,argmax,details) in models.items():
        for mode,frame in modes.items():
            if frame.set_index('image_id').patient_key.to_dict()!=patient_map:raise ValueError(f'{name}: manifest patient mismatch')
    for mode in ('legacy16','contiguous'):
        joined=align(models['improved'][0][mode],models['control'][0][mode],expected,mode);joined['patient_key']=joined.patient_key_left
        for metric in ('raw_nll','information_gain_bits'):
            key=f'{mode}/{metric}/improved_minus_control';ci[key]=paired_bootstrap(joined,metric+'_left',metric+'_right',repeats,seed)
            comparisons.append(dict(comparison='improved vs control',target=mode,metric=metric,improved_mean=float(joined[metric+'_left'].mean()),reference_mean=float(joined[metric+'_right'].mean()),**ci[key]))
    for strategy,index in (('sample_3seed_image_mean',1),('argmax_42',2)):
        joined=align(models['improved'][index],models['control'][index],expected,strategy);joined['patient_key']=joined.image_id.map(patient_map)
        for metric in ('normalized_dtw','return_rate_112px'):
            key=f'{strategy}/{metric}/improved_minus_control';ci[key]=paired_bootstrap(joined,metric+'_left',metric+'_right',repeats,seed)
            comparisons.append(dict(comparison='improved vs control',target=strategy,metric=metric,improved_mean=float(joined[metric+'_left'].mean()),reference_mean=float(joined[metric+'_right'].mean()),**ci[key]))
    original=frame_ids(pd.read_csv(baseline),expected,'original baseline').rename(columns={'raw_next_step_nll':'original_raw_nll'})
    for name in models:
        joined=align(models[name][0]['legacy16'],original[['image_id','patient_key','original_raw_nll']],expected,name+'/original');joined['patient_key']=joined.patient_key_left
        key=f'legacy16/raw_nll/{name}_minus_original';ci[key]=paired_bootstrap(joined,'raw_nll','original_raw_nll',repeats,seed)
        comparisons.append(dict(comparison=f'{name} vs original ConvNeXt',target='legacy16',metric='raw_nll',improved_mean=float(joined.raw_nll.mean()),reference_mean=float(joined.original_raw_nll.mean()),**ci[key]))
    frame=pd.DataFrame(comparisons);details={name:models[name][3] for name in models}
    lines=['# Full 1011 exploratory validation comparison','', 'Training: seed42, full412 images,15epochs; both models continue the same ConvNeXt checkpoint. Generation samples: seeds42/43/44 averaged per image; these are not three independent training seeds. Validation:93 labelled images, patient paired bootstrap'+str(repeats)+' resamples. Final test and INbreast quarantine were not evaluated.','', 'Negative NLL/DTW differences and positive IG differences favor the candidate. Revisit rate differences describe behavior and do not establish quality. Confidence intervals condition on these trained/selected models; they do not remove checkpoint selection bias, repeated validation use, or training seed uncertainty.','', '| Comparison | Target | Metric | Candidate mean | Reference mean | Difference | Patient bootstrap95%CI |','|---|---|---|---:|---:|---:|---|']
    for row in comparisons:lines.append(f"| {row['comparison']} | {row['target']} | {row['metric']} | {row['improved_mean']:.6f} | {row['reference_mean']:.6f} | {row['delta']:+.6f} | [{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] |")
    lines+=['','![Learning curves; stars mark checkpoints selected by legacy16 validation NLL](learning_curves.png)','', 'The original ConvNeXt raw next-step NLL uses identical image IDs and legacy16 targets. Its hard-policy generated geometry is deliberately excluded from direct comparison with these unconstrained generations. Original IG is also excluded because train-prior definitions may differ.','']
    for name,detail in details.items():
        lines.append(f"{name}: best epoch {detail['best_epoch']}/15; last legacy16 NLL {detail['last_validation']['legacy16']['raw_nll']:.6f}; training time {detail['training_seconds']:.1f}s; peak CUDA allocation {detail['peak_cuda_memory_bytes']} bytes.")
        lines.append(f"Mean branch gates (uniform image means): {json.dumps(detail['branch_gates_image_mean'],sort_keys=True)}")
    lines+=['','Matched settings: raw NLL weight0.5, soft sigma64px, prior weight0.25, effective batch2,256×512 input,24×48 output, optimizer/backbone schedule and legacy16 validation checkpoint selection. Candidate combines mixed trajectories, recent4 nonlinear history, and behavior mixture; this joint experiment cannot identify the contribution of each component.','', 'Artifacts validated before reporting: complete15-epoch histories, best/last checkpoint existence, exact93 unique evaluation IDs and patient mappings, all four complete93-image generation sets per model,16 bounded coordinates per prediction, and finite NLL/IG/DTW/gates.']
    sources=[Path(baseline),suite/'experiment.json',suite/'completion.json']+[suite/name/'history.json' for name in models]
    source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    outputs={'report.md':'\n'.join(lines)+'\n','comparison.csv':frame.to_csv(index=False),'paired_ci.json':json.dumps(ci,indent=2),'report_details.json':json.dumps(dict(models=details,source_sha256=source_hashes,bootstrap_seed=seed),indent=2)}
    if any((suite/name).exists() for name in [*outputs,'learning_curves.png']):raise FileExistsError('Report artifacts already exist; preserve previous report')
    learning_curves(suite,float(original.original_raw_nll.mean()),details)
    for name,text in outputs.items():(suite/name).write_text(text)
    return frame,ci


def learning_curves(suite,original_nll,details):
    import os
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True)
    for axis,mode in zip(axes,('legacy16','contiguous')):
        for name,color in [('improved','#2878b5'),('control','#d97920')]:
            history=read(suite/name/'history.json')
            epochs=[record['epoch'] for record in history]
            nll=[record['validation'][mode]['raw_nll'] for record in history]
            axis.plot(epochs,nll,color=color,label=name,linewidth=1.8)
            selected=details[name]['best_epoch']
            axis.scatter([selected],[nll[selected-1]],color=color,marker='*',s=110,zorder=3)
            axis.annotate(f'epoch{selected}',(selected,nll[selected-1]),xytext=(-5 if selected>=12 else 4,6),ha='right' if selected>=12 else 'left',textcoords='offset points',fontsize=8,color=color)
        if mode=='legacy16':axis.axhline(original_nll,color='#555555',linestyle='--',linewidth=1.2,label=f'original ConvNeXt ({original_nll:.4f})')
        axis.set(title=f'{mode} validation NLL',xlabel='Epoch',ylabel='Raw NLL',xticks=[1,3,5,7,9,11,13,15],xlim=(.7,15.3))
        axis.grid(alpha=.2);axis.legend(fontsize=8)
    fig.suptitle('Training seed42 continuation; stars mark selected legacy16 checkpoints',fontsize=11)
    fig.tight_layout();fig.savefig(suite/'learning_curves.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('suite',type=Path);parser.add_argument('--baseline',type=Path,default=ROOT/'runs/gaze_focus_seed42/convnext_base_motion/validation_hard/metrics.csv');parser.add_argument('--bootstrap',type=int,default=10000);parser.add_argument('--seed',type=int,default=20261011);a=parser.parse_args()
    manifest=pd.read_csv(ROOT/'manifests/all.csv');manifest=manifest[(manifest.split=='val')&(manifest.label>=0)]
    if len(manifest)!=93 or manifest.image_id.duplicated().any():raise ValueError('Expected fixed93-image validation manifest')
    report(a.suite,a.baseline,manifest.image_id,manifest.set_index('image_id').patient_key.to_dict(),a.bootstrap,a.seed)

if __name__=='__main__':main()
