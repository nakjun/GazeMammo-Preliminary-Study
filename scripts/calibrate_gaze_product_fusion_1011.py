"""Limited normalized log-probability fusion; target-aware oracle is diagnostic only."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from calibrate_gaze_ensemble_1011 import aligned_records,per_image_nll,digest
from src.proposal_utility_1011 import state_sha256


def product_logp(control,improved,relevance,weight,temperature,gamma):
    if not 0<=weight<=1 or temperature<=0 or gamma<0:raise ValueError('Invalid fusion settings')
    return ((weight*control+(1-weight)*improved+gamma*relevance[:,None])/temperature).log_softmax(-1)


def original_relevance(a,reference):
    guided=torch.load(a.guided,map_location='cpu',weights_only=False);guided_sha=digest(a.guided);keys_cache=torch.load(a.train_keys,map_location='cpu',weights_only=False)
    if keys_cache['base_checkpoint_sha256']!=guided_sha or keys_cache.get('grid')!=[24,48]:raise ValueError('Original guided key source mismatch')
    train_ids={i for i,r in reference.items() if r['split']=='train'};result={};weight=guided['state']['proposal.weight'];bias=guided['state']['proposal.bias']
    if len(keys_cache['records'])!=412 or {int(r['image_id']) for r in keys_cache['records']}!=train_ids:raise ValueError('Train original-key cohort mismatch')
    for record in keys_cache['records']:
        image_id=int(record['image_id'])
        if record['patient_key']!=reference[image_id]['patient_key'] or record['keys'].shape!=(1152,weight.shape[1]):raise ValueError('Train original-key patient/shape mismatch')
        result[image_id]=torch.nn.functional.linear(record['keys'],weight,bias).squeeze(-1).sigmoid()
    with Path(a.val_traces).open() as source:
        for line in source:
            trace=json.loads(line);image_id=int(trace['image_id'])
            if image_id not in reference or reference[image_id]['split']!='val' or image_id in result:continue
            if trace['checkpoint_sha256']!=guided_sha or trace.get('grid')!=[24,48] or trace.get('proposal_source')!='original_occupancy' or trace.get('score_recipe') is not None or trace.get('annotations_used') is not False:raise ValueError('Validation original-R provenance mismatch')
            relevance=torch.as_tensor(trace.get('original_relevance',trace['relevance']),dtype=torch.float32)
            if not torch.equal(relevance,torch.as_tensor(trace['relevance'],dtype=torch.float32)):raise ValueError('Combined validation score is not originalR')
            result[image_id]=relevance
    if set(result)!=set(reference) or any(r.shape!=(1152,) or not torch.isfinite(r).all() or ((r<0)|(r>1)).any() for r in result.values()):raise ValueError('Incomplete/malformed originalR')
    return result,dict(guided_checkpoint_sha256=guided_sha,original_proposal_state_sha256=state_sha256({'weight':weight,'bias':bias}),train_original_keys_sha256=digest(a.train_keys),val_originalR_trace_sha256=digest(a.val_traces),guided_training_patient_keys=guided['training_patient_keys'],guided_selection_patient_keys=guided['selection_patient_keys'],protocol='TrainR from ORIGINAL guided cached image keys plus its original frozen Linear head; valR from SHA-checked original-occupancy image-only traces. No human-model keys, capture-combined scores or boxes enter fusion.')


def analyze(a):
    torch.set_num_threads(a.threads);started=time.monotonic();paths=[a.control,a.improved];caches=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];maps=aligned_records(caches);reference=maps[0]
    all_manifest=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','split']);val_manifest=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','split']);expected=set(all_manifest[all_manifest.split=='train'].image_id)|set(val_manifest[val_manifest.split=='val'].image_id)
    if set(reference)!=expected or len(expected)!=505:raise ValueError('Exact train412/val93 allowlist mismatch')
    relevance,r_proof=original_relevance(a,reference);rows=[];oracle_rows=[];recipe_image=[]
    for split in ('train','val'):
        ids=sorted(i for i,r in reference.items() if r['split']==split);records=[reference[i] for i in ids];control=torch.stack([r['raw_logp'] for r in records]);improved=torch.stack([maps[1][i]['raw_logp'] for i in ids]);target=torch.stack([r['target'] for r in records]);mask=torch.stack([r['loss_mask'] for r in records]).bool();r=torch.stack([relevance[i] for i in ids]);safe=torch.where(mask,target,torch.zeros_like(target));c=-control.gather(-1,safe[:,:,None]).squeeze(-1);v=-improved.gather(-1,safe[:,:,None]).squeeze(-1)
        for j,image_id in enumerate(ids):
            image_oracle=min(float(c[j][mask[j]].mean()),float(v[j][mask[j]].mean()))
            for t in torch.nonzero(mask[j],as_tuple=False).flatten().tolist():oracle_rows.append(dict(split=split,image_id=image_id,patient_key=records[j]['patient_key'],step=t,control_nll=float(c[j,t]),improved_nll=float(v[j,t]),oracle_best_target_nll=min(float(c[j,t]),float(v[j,t])),oracle_image_selected_mean_nll=image_oracle))
        for weight in (0.,.25,.5,.75,1.):
            for temperature in (.75,1.,1.25):
                for gamma in (0.,1.,2.,4.):
                    name=f'w{weight:g}_T{temperature:g}_R{gamma:g}';q=product_logp(control,improved,r,weight,temperature,gamma);scores=per_image_nll(q,target,mask);rows.append(dict(split=split,recipe=name,control_weight=weight,temperature=temperature,gamma_originalR=gamma,image_macro_nll=float(scores.double().mean()),images=len(ids)))
                    recipe_image.extend(dict(split=split,recipe=name,image_id=image_id,patient_key=records[j]['patient_key'],nll=float(scores[j])) for j,image_id in enumerate(ids))
    recipes=pd.DataFrame(rows);oracle=pd.DataFrame(oracle_rows);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);recipes.to_csv(output/'recipes.csv',index=False);pd.DataFrame(recipe_image).to_csv(output/'per_image_recipes.csv',index=False);oracle.to_csv(output/'per_step_oracle.csv',index=False);oracle.groupby(['split','image_id','patient_key']).mean(numeric_only=True).to_csv(output/'per_image_oracle.csv');best=recipes[recipes.split=='val'].sort_values('image_macro_nll').iloc[0].to_dict();oracle_summary={split:dict(images=g.image_id.nunique(),control_nll=float(g.groupby('image_id').control_nll.mean().mean()),improved_nll=float(g.groupby('image_id').improved_nll.mean().mean()),oracle_step_best_nll=float(g.groupby('image_id').oracle_best_target_nll.mean().mean()),oracle_image_best_nll=float(g.groupby('image_id').oracle_image_selected_mean_nll.first().mean())) for split,g in oracle.groupby('split')}
    result=dict(status='complete',formula='q=softmax((w logq_control+(1-w)logq_improved+gamma originalR)/T) over1152cells; image-onlyR repeated acrosssteps',oracle_warning='Best target-aware expert/image selection is unnormalized diagnostic ONLY; not deployable gating, not model performance, not goal achievement.',oracle=oracle_summary,best_exploratory_normalized=best,meets_nll_5_3=best['image_macro_nll']<=5.3,arithmetic_ensemble_reference=5.787842,sourceproof=dict(control_raw_sha256=digest(a.control),improved_raw_sha256=digest(a.improved),control_parent_sha256=caches[0]['base_checkpoint_sha256'],improved_parent_sha256=caches[1]['base_checkpoint_sha256'],originalR=r_proof),protocol='No training/CNN/newimagefeatures/boxes; exact505 IDs/patients/prefixes/targets/masks aligned. Choice on repeated val93 is exploratory; guided head trained412 and selected93 disclosed. If numerical5.3 threshold passes, actual-image runtime confirmation is still required.',seconds=time.monotonic()-started)
    (output/'metrics.json').write_text(json.dumps(result,indent=2));(output/'report.md').write_text('# Frozen probability product fusion\n\n'+result['protocol']+'\n\n'+result['oracle_warning']+'\n\nBest normalized val93 recipe: '+json.dumps(best)+'\n\nOracle diagnostics: '+json.dumps(oracle_summary)+'\n');print(json.dumps(dict(best=best,oracle=oracle_summary,seconds=result['seconds']),indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control',default='runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt');p.add_argument('--improved',default='runs/dual_target_1011/human_improved_raw_cache.pt');p.add_argument('--guided',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--train-keys',default='runs/proposal_visit_1011/train_feature_cache.pt');p.add_argument('--val-traces',default='runs/dual_target_1011/budget224_seed42/traces.jsonl');p.add_argument('--output',default='runs/dual_target_1011/product_fusion_audit');p.add_argument('--threads',type=int,default=4);analyze(p.parse_args())
