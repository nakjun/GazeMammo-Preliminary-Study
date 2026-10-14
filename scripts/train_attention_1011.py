"""Cached CPU attention-head training, interventions, and explicit head-seed reports."""
import argparse,hashlib,json,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.gaze_attention_1011 import GazeAttention1011,VARIANTS,attention_loss,head_state,restore_head,validate_sources
from src.gaze_classifier_1011 import GazeDiagnosisHead
from src.data import seed_for
from src.metrics import classification_metrics,cluster_bootstrap


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False));temporary.replace(path)
def metrics(labels,prob):
    result=classification_metrics(labels,prob,.5);result['brier']=float(np.mean((np.asarray(labels)-np.asarray(prob))**2));return result

def paired_ci(frame,reference,repetitions=1000,seed=42):
    result=cluster_bootstrap(frame,reference,repetitions=repetitions,seed=seed)
    result['delta']=metrics(frame.label,frame.probability)['auroc']-metrics(reference.label,reference.probability)['auroc']
    result['difference']='candidate minus reference AUROC'
    return result

def forward(model,tensors,indices,mask=None,uniform=False):
    source=model.source;coords=tensors[source+'_coordinates'][indices]
    if mask is None:mask=tensors.get(source+'_mask',torch.ones(tensors[source+'_coordinates'].shape[:2],dtype=torch.bool))[indices]
    return model(tensors['global_features'][indices],tensors[source+'_features'][indices],coords,mask,uniform)

@torch.inference_mode()
def evaluate(model,tensors,indices,batch=64):
    model.eval();outputs=[]
    for chunk in indices.split(batch):outputs.append({k:v.cpu() for k,v in forward(model,tensors,chunk).items() if k!='values'})
    return {key:torch.cat([o[key] for o in outputs]) for key in outputs[0]}

@torch.inference_mode()
def diagnose(model,tensors,frame,indices,directory):
    model.eval();out=evaluate(model,tensors,indices);records={};rows=[];deletions=[]
    for offset,index in enumerate(indices.tolist()):
        row=frame.iloc[index];weights=out['attention'][offset];valid=out['duplicate_counts'][offset]>0
        entropy=float(-(weights*weights.clamp_min(1e-12).log()).sum());signed=out['signed_contribution'][offset]
        bias=float(out['residual_bias'][offset]);residual=float(out['residual'][offset])
        if not np.isclose(float(signed.sum())+bias,residual,atol=2e-6):raise ValueError('Signed residual decomposition failure')
        record=dict(image_id=int(row.image_id),split=row.split,entropy=entropy,effective_patches=float(np.exp(entropy)),max_weight=float(weights.max()),top4_mass=float(weights.topk(min(4,len(weights))).values.sum()),alpha=float(out['alpha'][offset]),residual_logit=residual,residual_bias=bias)
        rows.append(record);records[str(row.image_id)]=dict(**record,attention=weights.tolist(),signed_contribution=signed.tolist(),coordinates=tensors[model.source+'_coordinates'][index].tolist())
    # Recompute attention and gate after each intervention; random masks are fixed by image ID.
    for split in ('train','val'):
        selected=indices[torch.tensor([frame.iloc[i].split==split for i in indices.tolist()])]
        if not len(selected):continue
        baseline=evaluate(model,tensors,selected);direction=torch.where(baseline['logits']>=0,1.,-1.)
        source_mask=tensors.get(model.source+'_mask',torch.ones(tensors[model.source+'_coordinates'].shape[:2],dtype=torch.bool))[selected]
        uniform=torch.cat([forward(model,tensors,chunk,uniform=True)['logits'].cpu() for chunk in selected.split(64)])
        for kind in ('top1','top4','random1','random4'):
            k=int(kind[-1]);mask=source_mask.clone()
            for j,index in enumerate(selected.tolist()):
                available=torch.where(mask[j])[0].numpy()
                chosen=baseline['attention'][j].argsort(descending=True)[:min(k,len(available))].numpy() if kind.startswith('top') else np.random.default_rng(seed_for(42,int(frame.iloc[index].image_id))).permutation(available)[:k]
                mask[j,chosen]=False
            changed=torch.cat([forward(model,tensors,chunk,mask[start:start+len(chunk)])['logits'].cpu() for start,chunk in zip(range(0,len(selected),64),selected.split(64))])
            for j,index in enumerate(selected.tolist()):deletions.append(dict(image_id=int(frame.iloc[index].image_id),split=split,intervention=kind,original_logit=float(baseline['logits'][j]),masked_logit=float(changed[j]),original_class_direction_logit_drop=float(direction[j]*(baseline['logits'][j]-changed[j])),uniform_absolute_probability_change=float(abs(baseline['logits'][j].sigmoid()-uniform[j].sigmoid()))))
    save(directory/'attention_diagnostics.json',records);pd.DataFrame(rows).to_csv(directory/'attention_diagnostics.csv',index=False);pd.DataFrame(deletions).to_csv(directory/'deletion_diagnostics.csv',index=False)


def train_one(a,output,tensors,frame,global_checkpoint,provenance,variant,seed):
    torch.manual_seed(seed);directory=output/f'{variant}_seed{seed}';directory.mkdir()
    baseline=GazeDiagnosisHead(**global_checkpoint['model_config']);baseline.load_state_dict(global_checkpoint['state'])
    model=GazeAttention1011(baseline,variant=variant,input_dim=tensors['global_features'].shape[-1]);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    tr=torch.tensor(np.flatnonzero(frame.split.to_numpy()=='train'));va=torch.tensor(np.flatnonzero(frame.split.to_numpy()=='val'))
    if a.smoke:
        tr=torch.tensor(frame[frame.split=='train'].groupby('label').head(2).index.to_numpy());va=torch.tensor(frame[frame.split=='val'].groupby('label').head(2).index.to_numpy())
    labels=torch.tensor(frame.label.to_numpy(),dtype=torch.float32);validation=frame.iloc[va].reset_index(drop=True)
    histories=[];started=time.monotonic();best=-float('inf');stale=0
    def checkpoint(epoch):
        return dict(kind='gaze_attention_1011',state=head_state(model),model_config=model.config,epoch=epoch,training_seed=seed,source_provenance=provenance,head_training_patient_keys=sorted(frame.iloc[tr].patient_key.unique()),training_patient_keys=global_checkpoint['training_patient_keys'],selection_patient_keys=global_checkpoint['selection_patient_keys'],threshold=.5,attention_untrained=epoch==0,protocol='Frozen in-sample gaze transfer; head seeds only, reused selection validation; no test')
    for epoch in range(a.epochs+1):
        losses=[]
        if epoch:
            model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))]
            for chunk in order.split(32):
                optimizer.zero_grad();value,parts=attention_loss(forward(model,tensors,chunk),labels[chunk]);value.backward()
                if not torch.isfinite(value):raise ValueError('Nonfinite attention loss')
                torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(value.detach()))
        result=evaluate(model,tensors,va);prob=result['logits'].sigmoid().numpy();score=metrics(validation.label,prob)
        record=dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,validation=score);histories.append(record);save(directory/'history.json',histories)
        if score['auroc']>best:
            best=score['auroc'];stale=0;torch.save(checkpoint(epoch),directory/'best.pt')
        elif epoch:stale+=1
        print(json.dumps(dict(variant=variant,seed=seed,epoch=epoch,auroc=score['auroc'],best_auroc=best,stale=stale)),flush=True)
        if stale>=12:break
    chosen=torch.load(directory/'best.pt',map_location='cpu',weights_only=False);model=restore_head(chosen,global_checkpoint).eval();result=evaluate(model,tensors,va);prob=result['logits'].sigmoid().numpy()
    predictions=validation[['image_id','patient_key','file_name','label']].copy();predictions['probability']=prob;predictions.to_csv(directory/'validation_predictions.csv',index=False)
    diagnose(model,tensors,frame,torch.cat([tr,va]),directory)
    summary=dict(variant=variant,training_seed=seed,best_epoch=chosen['epoch'],epochs_completed=histories[-1]['epoch'],attention_untrained=chosen['epoch']==0,metrics=metrics(predictions.label,prob),auroc_ci=cluster_bootstrap(predictions,repetitions=a.bootstrap,seed=seed),seconds=time.monotonic()-started,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    save(directory/'metrics.json',summary);return summary,predictions


def make_report(output,results,predictions,a):
    summaries={};rows=[];paired={};ensemble_frames={}
    reference_dir=Path(a.global_checkpoint).parent.parent
    references={name:pd.read_csv(reference_dir/name/'validation_predictions.csv') for name in ('global','gaze')}
    expected=set(next(iter(predictions.values())).image_id)
    references={name:frame[frame.image_id.isin(expected)].sort_values('image_id').reset_index(drop=True) for name,frame in references.items()}
    for variant in a.variants:
        selected=[results[(variant,seed)] for seed in a.seeds];frames=[predictions[(variant,seed)].sort_values('image_id').reset_index(drop=True) for seed in a.seeds]
        if any(frame.image_id.tolist()!=frames[0].image_id.tolist() or frame.patient_key.tolist()!=frames[0].patient_key.tolist() or frame.label.tolist()!=frames[0].label.tolist() for frame in frames):raise ValueError('Seed prediction alignment mismatch')
        ensemble=frames[0].copy();ensemble['probability']=np.mean([frame.probability.to_numpy() for frame in frames],axis=0);ensemble.to_csv(output/f'{variant}_ensemble_predictions.csv',index=False);ensemble_frames[variant]=ensemble
        seed_stats={key:dict(mean=float(np.mean([s['metrics'][key] for s in selected])),std=float(np.std([s['metrics'][key] for s in selected],ddof=1)) if len(selected)>1 else 0.) for key in selected[0]['metrics'] if isinstance(selected[0]['metrics'][key],(int,float)) and selected[0]['metrics'][key] is not None}
        summary=dict(single_head_seed_statistics=seed_stats,ensemble_metrics=metrics(ensemble.label,ensemble.probability),ensemble_auroc_ci=cluster_bootstrap(ensemble,repetitions=a.bootstrap,seed=42));summaries[variant]=summary
        for seed,r,frame in zip(a.seeds,selected,frames):
            rows.append(dict(variant=variant,training_seed=seed,best_epoch=r['best_epoch'],attention_untrained=r['attention_untrained'],**r['metrics']))
            for name,reference in references.items():
                if frame.image_id.tolist()!=reference.image_id.tolist() or frame.patient_key.tolist()!=reference.patient_key.tolist() or frame.label.tolist()!=reference.label.tolist():raise ValueError('Reference image/patient alignment mismatch')
                paired[f'{variant}/seed{seed}/minus_{name}']=paired_ci(frame,reference,repetitions=a.bootstrap,seed=42)
        for name,reference in references.items():paired[f'{variant}/ensemble/minus_{name}']=paired_ci(ensemble,reference,repetitions=a.bootstrap,seed=42)
    # Direct architecture comparisons remain distinct from references and ensembles.
    if 'contextual_residual' in ensemble_frames:
        for variant in ('mean_residual','gated_residual','grid_contextual_residual'):
            if variant not in ensemble_frames:continue
            for seed in a.seeds:paired[f'contextual_minus_{variant}/seed{seed}']=paired_ci(predictions[('contextual_residual',seed)].sort_values('image_id'),predictions[(variant,seed)].sort_values('image_id'),repetitions=a.bootstrap,seed=42)
            paired[f'contextual_minus_{variant}/ensemble']=paired_ci(ensemble_frames['contextual_residual'],ensemble_frames[variant],repetitions=a.bootstrap,seed=42)
    pd.DataFrame(rows).to_csv(output/'seed_metrics.csv',index=False);save(output/'comparison.json',summaries);save(output/'paired_auroc_ci.json',paired)
    lines=['# Gaze16 residual MIL attention comparison','','Frozen gaze training/generation seed42; classifier head training seeds '+','.join(map(str,a.seeds))+'. These are head seeds, not three complete pipeline retrainings. Global reference and cached patch/global features are fixed. Test/INbreast quarantine unused. Train transfer is in-sample, not OOF; validation selected the source gaze/global and these heads, so bootstrap does not remove selection bias.','','| Variant | Single-head AUROC mean±std | Ensemble AUROC | Ensemble patient95%CI | Ensemble AUPRC | Brier | Sensitivity@0.5 | Specificity@0.5 |','|---|---:|---:|---|---:|---:|---:|---:|']
    for variant,r in summaries.items():
        m=r['ensemble_metrics'];ci=r['ensemble_auroc_ci'];stats=r['single_head_seed_statistics']['auroc'];lines.append(f"| {variant} | {stats['mean']:.4f}±{stats['std']:.4f} | {m['auroc']:.4f} | [{ci['low']:.4f},{ci['high']:.4f}] | {m['auprc']:.4f} | {m['brier']:.4f} | {m['sensitivity']:.4f} | {m['specificity']:.4f} |")
    lines+=['','## Ensemble paired AUROC differences (separate from single-head seed summary)','']
    for key,value in paired.items():
        if '/ensemble' in key:lines.append(f"- {key}: {value['delta']:+.4f}, patient95%CI [{value['low']:.4f},{value['high']:.4f}]")
    lines+=['','Individual seed metrics/CIs and epoch0 selections: seed_metrics.csv and each run metrics.json. Ensembles average seed probabilities and are separate models; ensemble gains are not single-head gains.','']
    for name,reference in references.items():lines.append(f"Reference {name}: AUROC {metrics(reference.label,reference.probability)['auroc']:.4f} on identical images.")
    for (variant,seed),r in results.items():
        if r['best_epoch']==0:lines.append(f'{variant} seed{seed}: epoch0 selected; local attention is untrained and residual is zero (global fallback).')
    lines+=['','Attention is normalized importance, not malignancy probability. Signed contributions fix the learned gate and sum with the residual bias to the residual logit; this is not causal attribution. Top1/top4 and fixed random1/random4 deletion re-evaluate attention/gate. Original-class-direction logit drops and uniform-attention interventions are in each deletion_diagnostics.csv. Entropy and gate diagnostics are in each attention_diagnostics.csv/json; lower entropy alone is not success.','', 'The selected global baseline is eligible at epoch0 and is not a guarantee of independent generalization. No hyperparameters were retuned during this experiment.']
    (output/'report.md').write_text('\n'.join(lines)+'\n')


def train(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic();save(output/'config.json',vars(a))
    payload=torch.load(a.features,map_location='cpu',weights_only=False);global_checkpoint=torch.load(a.global_checkpoint,map_location='cpu',weights_only=False)
    manifest=pd.read_csv(ROOT/'manifests/all.csv');frame=validate_sources(payload,global_checkpoint,manifest)
    source=payload['provenance'];gaze_hash=digest(a.gaze_checkpoint)
    if gaze_hash!=source['gaze_checkpoint_sha256']:raise ValueError('Source gaze SHA mismatch')
    provenance=dict(cache_path=str(Path(a.features).resolve()),cache_sha256=digest(a.features),global_checkpoint=str(Path(a.global_checkpoint).resolve()),global_checkpoint_sha256=digest(a.global_checkpoint),gaze_checkpoint=str(Path(a.gaze_checkpoint).resolve()),gaze_checkpoint_sha256=gaze_hash,gaze_provenance=source,source_image_ids=frame.image_id.tolist(),classifier_training_seed_scope='head only',final_test_used=False)
    save(output/'source_provenance.json',provenance);results={};predictions={}
    for variant in a.variants:
        for seed in a.seeds:
            save(output/'status.json',dict(complete=False,variant=variant,head_seed=seed))
            results[(variant,seed)],predictions[(variant,seed)]=train_one(a,output,payload['tensors'],frame,global_checkpoint,provenance,variant,seed)
    make_report(output,results,predictions,a);save(output/'status.json',dict(complete=True,seconds=time.monotonic()-started,smoke=a.smoke))


def parser():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--features',default=str(ROOT/'runs/classification_1011_seed42/features.pt'));p.add_argument('--global-checkpoint',default=str(ROOT/'runs/classification_1011_seed42/global/best.pt'));p.add_argument('--gaze-checkpoint',default=str(ROOT/'runs/full_1011_seed42/improved/best.pt'));p.add_argument('--variants',nargs='+',choices=VARIANTS,default=list(VARIANTS));p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--epochs',type=int,default=60);p.add_argument('--bootstrap',type=int,default=1000);p.add_argument('--threads',type=int,default=4);p.add_argument('--smoke',action='store_true');return p
if __name__=='__main__':
    a=parser().parse_args()
    if not 1<=a.epochs<=60 or a.bootstrap<1:raise ValueError('epochs must be1..60 and bootstrap positive')
    torch.set_num_threads(a.threads);train(a)
