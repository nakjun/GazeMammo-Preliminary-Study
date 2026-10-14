"""Frozen-feature dual-source CXR adaptation; interventions never select models."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.gaze_cxr_1011 import GazeCXR1011,VARIANTS,restore_head,attention_loss,head_state,validate_sources
from src.gaze_classifier_1011 import GazeDiagnosisHead
from src.data import seed_for
from src.metrics import cluster_bootstrap
from train_attention_1011 import digest,save,metrics,paired_ci


def forward(model,tensors,indices,**options):
    args=[tensors['global_features'][indices],tensors['gaze_features'][indices],tensors['gaze_coordinates'][indices],tensors['grid_features'][indices],tensors['grid_coordinates'][indices]]
    for source in ('gaze','grid'):
        if source+'_mask' not in options and source+'_mask' in tensors:options[source+'_mask']=tensors[source+'_mask'][indices]
    return model(*args,**options)

@torch.inference_mode()
def evaluate(model,tensors,indices,**options):
    model.eval();outputs=[]
    for start,chunk in zip(range(0,len(indices),64),indices.split(64)):
        batch_options={key:value[start:start+len(chunk)] if torch.is_tensor(value) and value.ndim>0 and len(value)==len(indices) else value for key,value in options.items()}
        out=forward(model,tensors,chunk,**batch_options);outputs.append({key:value.cpu() for key,value in out.items() if not key.endswith('context')})
    return {key:torch.cat([out[key] for out in outputs]) for key in outputs[0]}


def prediction_frame(frame,indices,out):
    result=frame.iloc[indices].reset_index(drop=True)[['image_id','patient_key','file_name','label']].copy();result['probability']=out['logits'].sigmoid().numpy();return result

def check_alignment(left,right):
    a=left.sort_values('image_id').reset_index(drop=True);b=right.sort_values('image_id').reset_index(drop=True)
    for key in ('image_id','patient_key','label'):
        if a[key].tolist()!=b[key].tolist():raise ValueError(f'Paired frame mismatch {key}')
    return a,b

@torch.inference_mode()
def diagnose(model,tensors,frame,indices,directory,bootstrap):
    baseline=evaluate(model,tensors,indices);records=[];details={}
    for j,index in enumerate(indices.tolist()):
        row=frame.iloc[index];record=dict(image_id=int(row.image_id),split=row.split,alpha=float(baseline['alpha'][j]),residual=float(baseline['residual'][j]),gaze_branch_weight=float(baseline['beta'][j,0]),grid_branch_weight=float(baseline['beta'][j,1]))
        signed_total=float(baseline['gaze_signed'][j].sum()+baseline['grid_signed'][j].sum()+baseline['residual_bias'][j])
        if not np.isclose(signed_total,record['residual'],atol=3e-6):raise ValueError('CXR signed residual mismatch')
        detail=dict(**record,residual_bias=float(baseline['residual_bias'][j]))
        for source in ('gaze','grid'):
            a=baseline[source+'_attention'][j];q=baseline[source+'_quality'][j];entropy=float(-(a*a.clamp_min(1e-12).log()).sum())
            record[source+'_entropy']=entropy;record[source+'_effective_patches']=float(np.exp(entropy)) if bool(baseline[source+'_mask'][j].any()) else 0.
            record[source+'_padding_attention_mass']=float(a[q<1].sum());record[source+'_padding_effective_mass']=float((a[q<1]*baseline['beta'][j,0 if source=='gaze' else 1]).sum())
            detail[source]=dict(attention=a.tolist(),effective_attention=(a*baseline['beta'][j,0 if source=='gaze' else 1]).tolist(),signed_contribution=baseline[source+'_signed'][j].tolist(),quality=q.tolist(),coordinates=tensors[source+'_coordinates'][index].tolist())
        records.append(record);details[str(row.image_id)]=detail
    pd.DataFrame(records).to_csv(directory/'attention_diagnostics.csv',index=False);save(directory/'attention_diagnostics.json',details)
    val=indices[torch.tensor([frame.iloc[i].split=='val' for i in indices.tolist()])];base=evaluate(model,tensors,val);base_frame=prediction_frame(frame,val,base);direction=torch.where(base['logits']>=0,1.,-1.);rows=[];summaries={}
    interventions=[('gaze_missing',dict(force_gaze_absent=True)),('grid_missing',dict(force_grid_absent=True)),('uniform_attention',dict(uniform=True))]
    for source in ('gaze','grid'):
        for kind in ('top4','random4'):
            mask=base[source+'_mask'].clone()
            for j,index in enumerate(val.tolist()):
                available=torch.where(mask[j])[0].numpy();top=base[source+'_attention'][j].argsort(descending=True).numpy()
                chosen=top[np.isin(top,available)][:4] if kind=='top4' else np.random.default_rng(seed_for(42,int(frame.iloc[index].image_id))).permutation(available)[:4]
                mask[j,chosen]=False
            interventions.append((source+'_'+kind,{source+'_mask':mask}))
    for name,options in interventions:
        changed=evaluate(model,tensors,val,**options);changed_frame=prediction_frame(frame,val,changed)
        summaries[name]=dict(metrics=metrics(changed_frame.label,changed_frame.probability),paired_auroc_minus_original=paired_ci(changed_frame,base_frame,repetitions=bootstrap,seed=42))
        for j,index in enumerate(val.tolist()):rows.append(dict(image_id=int(frame.iloc[index].image_id),intervention=name,original_logit=float(base['logits'][j]),changed_logit=float(changed['logits'][j]),original_class_direction_logit_drop=float(direction[j]*(base['logits'][j]-changed['logits'][j]))))
    # Synthetic sensitivity test: features and coordinates are shuffled together, no labels move.
    shuffled=dict(tensors);order=torch.randperm(len(val),generator=torch.Generator().manual_seed(42))
    for key in ('gaze_features','gaze_coordinates','gaze_mask'):
        if key in tensors:
            shuffled[key]=tensors[key].clone();shuffled[key][val]=tensors[key][val[order]]
    changed=evaluate(model,shuffled,val);shuffled_frame=prediction_frame(frame,val,changed)
    for j,index in enumerate(val.tolist()):rows.append(dict(image_id=int(frame.iloc[index].image_id),intervention='shuffled_gaze_synthetic_sensitivity_only',original_logit=float(base['logits'][j]),changed_logit=float(changed['logits'][j]),original_class_direction_logit_drop=float(direction[j]*(base['logits'][j]-changed['logits'][j]))))
    summaries['shuffled_gaze_synthetic_sensitivity_only']=dict(metrics=metrics(shuffled_frame.label,shuffled_frame.probability),paired_auroc_minus_original=paired_ci(shuffled_frame,base_frame,repetitions=bootstrap,seed=42),permutation_image_ids=frame.iloc[val[order]].image_id.tolist(),used_for_selection=False)
    pd.DataFrame(rows).to_csv(directory/'intervention_logit_drops.csv',index=False);save(directory/'intervention_metrics.json',summaries)


def train_one(a,output,tensors,frame,global_checkpoint,provenance,variant,seed):
    torch.manual_seed(seed);directory=output/f'{variant}_seed{seed}';directory.mkdir();started=time.monotonic()
    global_model=GazeDiagnosisHead(**global_checkpoint['model_config']);global_model.load_state_dict(global_checkpoint['state']);model=GazeCXR1011(global_model,variant=variant)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    tr=torch.tensor(np.flatnonzero(frame.split.to_numpy()=='train'));va=torch.tensor(np.flatnonzero(frame.split.to_numpy()=='val'))
    if a.smoke:
        tr=torch.tensor(frame[frame.split=='train'].groupby('label').head(2).index.to_numpy());va=torch.tensor(frame[frame.split=='val'].groupby('label').head(2).index.to_numpy())
    labels=torch.tensor(frame.label.to_numpy(),dtype=torch.float32);best=-float('inf');best_epoch=0;stale=0;history=[]
    def checkpoint(epoch):return dict(kind='gaze_cxr_1011',state=head_state(model),model_config=model.config,epoch=epoch,training_seed=seed,source_provenance=provenance,training_patient_keys=global_checkpoint['training_patient_keys'],head_training_patient_keys=sorted(frame.iloc[tr].patient_key.unique()),selection_patient_keys=global_checkpoint['selection_patient_keys'],attention_untrained=epoch==0,threshold=.5)
    for epoch in range(a.epochs+1):
        losses=[]
        if epoch:
            model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))]
            for chunk in order.split(32):
                optimizer.zero_grad();loss,_=attention_loss(forward(model,tensors,chunk),labels[chunk]);loss.backward()
                if not torch.isfinite(loss):raise ValueError('Nonfinite CXR loss')
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()))
        out=evaluate(model,tensors,va);predictions=prediction_frame(frame,va,out);validation=metrics(predictions.label,predictions.probability);history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,validation=validation));save(directory/'history.json',history)
        if validation['auroc']>best:best=validation['auroc'];best_epoch=epoch;stale=0;torch.save(checkpoint(epoch),directory/'best.pt')
        elif epoch:stale+=1
        print(json.dumps(dict(variant=variant,seed=seed,epoch=epoch,auroc=validation['auroc'],best_epoch=best_epoch,stale=stale)),flush=True)
        if seed==42 and epoch in (1,5,10):
            from predict_cxr_attention_1011 import render_cached_case
            render_cached_case(model,tensors,frame,64,directory/f'case64_epoch{epoch:02d}',epoch,best_epoch)
        if stale>=12:break
    checkpoint_best=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);model=restore_head(checkpoint_best,global_checkpoint).eval();predictions=prediction_frame(frame,va,evaluate(model,tensors,va));predictions.to_csv(directory/'validation_predictions.csv',index=False)
    diagnose(model,tensors,frame,torch.cat([tr,va]),directory,a.bootstrap)
    result=dict(variant=variant,head_seed=seed,best_epoch=checkpoint_best['epoch'],epochs_completed=history[-1]['epoch'],metrics=metrics(predictions.label,predictions.probability),auroc_ci=cluster_bootstrap(predictions,repetitions=a.bootstrap,seed=seed),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),seconds=time.monotonic()-started,attention_untrained=checkpoint_best['epoch']==0)
    save(directory/'metrics.json',result);return result,predictions


def report(output,results,predictions,a):
    summaries={};ensemble={};rows=[];paired={};reference_dir=Path(a.global_checkpoint).parent.parent;expected=set(next(iter(predictions.values())).image_id)
    refs={'global':pd.read_csv(reference_dir/'global/validation_predictions.csv'),'old_mean_residual_ensemble':pd.read_csv(a.old_mean_reference)}
    refs={name:frame[frame.image_id.isin(expected)].sort_values('image_id').reset_index(drop=True) for name,frame in refs.items()}
    for variant in a.variants:
        frames=[predictions[(variant,seed)].sort_values('image_id').reset_index(drop=True) for seed in a.seeds]
        for frame in frames[1:]:check_alignment(frames[0],frame)
        ens=frames[0].copy();ens['probability']=np.mean([f.probability.to_numpy() for f in frames],axis=0);ens.to_csv(output/f'{variant}_ensemble_predictions.csv',index=False);ensemble[variant]=ens
        scores=[results[(variant,seed)] for seed in a.seeds]
        summaries[variant]=dict(single_head_statistics={key:dict(mean=float(np.mean([r['metrics'][key] for r in scores])),std=float(np.std([r['metrics'][key] for r in scores],ddof=1)) if len(scores)>1 else 0.) for key in ('auroc','auprc','brier','sensitivity','specificity')},ensemble_metrics=metrics(ens.label,ens.probability),ensemble_auroc_ci=cluster_bootstrap(ens,repetitions=a.bootstrap,seed=42))
        for seed,r,frame in zip(a.seeds,scores,frames):
            rows.append(dict(variant=variant,head_seed=seed,best_epoch=r['best_epoch'],trainable_parameters=r['trainable_parameters'],attention_untrained=r['attention_untrained'],**r['metrics']))
            for ref_name,ref in refs.items():left,right=check_alignment(frame,ref);paired[f'{variant}/seed{seed}/minus_{ref_name}']=paired_ci(left,right,repetitions=a.bootstrap,seed=42)
        for ref_name,ref in refs.items():left,right=check_alignment(ens,ref);paired[f'{variant}/ensemble/minus_{ref_name}']=paired_ci(left,right,repetitions=a.bootstrap,seed=42)
    comparisons=[('dual_cross','dual_mean'),('dual_cross_drop','dual_mean'),('dual_cross_drop_quality','dual_mean'),('dual_cross_drop','dual_cross'),('dual_cross_drop_quality','dual_cross_drop')]
    for left_name,right_name in comparisons:
        if left_name not in ensemble or right_name not in ensemble:continue
        for seed in a.seeds:left,right=check_alignment(predictions[(left_name,seed)],predictions[(right_name,seed)]);paired[f'{left_name}_minus_{right_name}/seed{seed}']=paired_ci(left,right,repetitions=a.bootstrap,seed=42)
        left,right=check_alignment(ensemble[left_name],ensemble[right_name]);paired[f'{left_name}_minus_{right_name}/ensemble']=paired_ci(left,right,repetitions=a.bootstrap,seed=42)
    pd.DataFrame(rows).to_csv(output/'seed_metrics.csv',index=False);save(output/'comparison.json',summaries);save(output/'paired_auroc_ci.json',paired)
    lines=['# CXR-inspired gaze+grid attention adaptation','','This is an adaptation of image/position/gaze relations and gaze-modality dropout principles, not GazeGNN or GII-ViT reproduction. Global/gaze encoder and generation seed42 are frozen; classifier head seeds '+','.join(map(str,a.seeds))+' are not independent full-pipeline seeds. No actual gaze, duration, ROI, or labels enter forward. Test/INbreast unused. Training transfer is in-sample, not OOF; repeated validation/checkpoint selection bias remains.','','| Variant | Single-head AUROC mean±std | Ensemble AUROC | Ensemble patient95%CI | Ensemble AUPRC | Brier | Sensitivity@0.5 | Specificity@0.5 |','|---|---:|---:|---|---:|---:|---:|---:|']
    for name,r in summaries.items():
        stats=r['single_head_statistics']['auroc'];m=r['ensemble_metrics'];ci=r['ensemble_auroc_ci'];lines.append(f"| {name} | {stats['mean']:.4f}±{stats['std']:.4f} | {m['auroc']:.4f} | [{ci['low']:.4f},{ci['high']:.4f}] | {m['auprc']:.4f} | {m['brier']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} |")
    lines+=['','Head seed metrics/CIs/parameter counts are in seed_metrics.csv and per-run metrics.json. Three-seed averaged probabilities form a separate ensemble. Mean/cross differ in positional/relation scoring and parameter counts because mean omits those modules; this is not a parameter-matched attention-only effect. Branch diagnostic importance is not gaze itself or malignancy probability. Signed source terms plus bias sum to the residual only in eval; this gate-fixed decomposition is not causal.','', '## Ensemble paired AUROC comparisons','']
    for key,value in paired.items():
        if '/ensemble' in key:lines.append(f"- {key}: {value['delta']:+.4f}, patient95%CI [{value['low']:.4f},{value['high']:.4f}]")
    for (variant,seed),r in results.items():
        if r['attention_untrained']:lines.append(f'{variant} seed{seed}: epoch0 global fallback selected; local attention untrained / residual zero.')
    lines+=['','Interventions (gaze/grid missing, source-specific top4/random4 removal, uniform attention, jointly shuffled gaze features/coordinates) are post-selection sensitivity checks. Shuffled gaze is synthetic sensitivity only, not a natural missingness or causal test; interventions never select or tune models. Quality is exact crop inbounds fraction only; it does not mask dark tissue or estimate lesion coverage. Training-only gaze bag dropout leaves global/grid unchanged.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def train(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);save(output/'config.json',vars(a));started=time.monotonic()
    payload=torch.load(a.features,map_location='cpu',weights_only=False);global_checkpoint=torch.load(a.global_checkpoint,map_location='cpu',weights_only=False);frame=validate_sources(payload,global_checkpoint,pd.read_csv(ROOT/'manifests/all.csv'))
    if digest(a.gaze_checkpoint)!=payload['provenance']['gaze_checkpoint_sha256']:raise ValueError('Source gaze hash mismatch')
    provenance=dict(cache_path=str(Path(a.features).resolve()),cache_sha256=digest(a.features),global_checkpoint=str(Path(a.global_checkpoint).resolve()),global_checkpoint_sha256=digest(a.global_checkpoint),gaze_checkpoint=str(Path(a.gaze_checkpoint).resolve()),gaze_checkpoint_sha256=digest(a.gaze_checkpoint),gaze_provenance=payload['provenance'],old_mean_reference=str(Path(a.old_mean_reference).resolve()),old_mean_reference_sha256=digest(a.old_mean_reference),final_test_used=False)
    save(output/'source_provenance.json',provenance);results={};predictions={}
    for variant in a.variants:
        for seed in a.seeds:
            save(output/'status.json',dict(complete=False,variant=variant,head_seed=seed));results[(variant,seed)],predictions[(variant,seed)]=train_one(a,output,payload['tensors'],frame,global_checkpoint,provenance,variant,seed)
    report(output,results,predictions,a);save(output/'status.json',dict(complete=True,seconds=time.monotonic()-started,smoke=a.smoke))


def parser():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--features',default=str(ROOT/'runs/classification_1011_seed42/features.pt'));p.add_argument('--global-checkpoint',default=str(ROOT/'runs/classification_1011_seed42/global/best.pt'));p.add_argument('--gaze-checkpoint',default=str(ROOT/'runs/full_1011_seed42/improved/best.pt'));p.add_argument('--old-mean-reference',default=str(ROOT/'runs/attention_1011_full/mean_residual_ensemble_predictions.csv'));p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS));p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--epochs',type=int,default=60);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--threads',type=int,default=4);p.add_argument('--smoke',action='store_true');return p
if __name__=='__main__':
    a=parser().parse_args()
    if not 1<=a.epochs<=60 or a.bootstrap<1:raise ValueError('Invalid epochs/bootstrap')
    torch.set_num_threads(a.threads);train(a)
