"""Fixed-target image-prior/BOS/full causal gaze audit, CPU only and no training."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gaze_nll_adaptation_1011 import cache_records,load_base,batch,apply_cache_resolution
from src.gaze_nll_adaptation_1011 import initialize
from train_attention_1011 import digest
from guided_trajectory_1011 import base_path


@torch.no_grad()
def bos_repeated(model,memory,batch_size,steps):
    """Every call is an independent BOS; no carried hidden, coordinates or past crop."""
    empty=memory['prior'].new_zeros(batch_size,0,2);outputs=[]
    for _ in range(steps):
        logp,_,_=model.step_history(memory,empty,hidden=None,patch_vector=None);outputs.append(logp)
    result=torch.stack(outputs,1)
    if not torch.equal(result,result[:,:1].expand_as(result)):raise ValueError('No-history distribution unexpectedly changes')
    return result


@torch.no_grad()
def audit(a):
    torch.set_num_threads(a.threads);records,provenance=cache_records(a);base,checkpoint=load_base(a.base);model=apply_cache_resolution(initialize(base),provenance).eval();original=torch.load(a.raw_cache,map_location='cpu',weights_only=False);reference={int(r['image_id']):r for r in original['records']}
    all_frame=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','split']);val_frame=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','split']);expected=set(all_frame[all_frame.split=='train'].image_id)|set(val_frame[val_frame.split=='val'].image_id)
    if len(records)!=505 or {int(r['image_id']) for r in records}!=expected or set(reference)!=expected:raise ValueError('Fixed train412/val93 allowlist mismatch')
    if original['base_checkpoint_sha256']!=digest(base_path(a.base)):raise ValueError('Frozen raw source parent hash mismatch')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);rows=[];alignment_errors=[];started=time.monotonic()
    for start in range(0,len(records),a.batch_size):
        group=records[start:start+a.batch_size];features,coords,mask,vectors=batch(group,'cpu');memory=model.maps_features(features);full=model.teacher_logits(memory,coords,vectors).log_softmax(-1);prior=memory['prior'].log_softmax(-1)[:,None].expand_as(full);bos=bos_repeated(model,memory,len(group),coords.shape[1]);gx,gy=model.grid;target=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1)
        kl_full_prior=(full.exp()*(full-prior)).sum(-1);kl_full_bos=(full.exp()*(full-bos)).sum(-1)
        for i,record in enumerate(group):
            image_id=int(record['image_id']);old=reference[image_id]
            if record['split']!=old['split'] or record['patient_key']!=old['patient_key'] or not torch.equal(coords[i],old['coords']) or not torch.equal(mask[i],old['loss_mask']) or not torch.equal(target[i][mask[i]],old['target'][mask[i]]):raise ValueError('Frozen targets/prefix alignment mismatch')
            error=float((full[i]-old['raw_logp']).abs().max());alignment_errors.append(error)
            if not torch.allclose(full[i],old['raw_logp'],rtol=1e-5,atol=2e-5):raise ValueError('Full causal recomputation mismatch')
            for t in range(coords.shape[1]):
                if not mask[i,t]:continue
                index=int(target[i,t]);prior_nll=float(-prior[i,t,index]);bos_nll=float(-bos[i,t,index]);full_nll=float(-full[i,t,index]);rows.append(dict(image_id=image_id,patient_key=record['patient_key'],split=record['split'],step=t,prior_nll=prior_nll,bos_repeated_nll=bos_nll,full_nll=full_nll,full_target_logp_gain_over_prior=prior_nll-full_nll,full_target_logp_gain_over_bos=bos_nll-full_nll,kl_full_to_prior=float(kl_full_prior[i,t]),kl_full_to_bos=float(kl_full_bos[i,t])))
    steps=pd.DataFrame(rows);names=['prior_nll','bos_repeated_nll','full_nll','full_target_logp_gain_over_prior','full_target_logp_gain_over_bos','kl_full_to_prior','kl_full_to_bos'];images=steps.groupby(['split','image_id','patient_key'])[names].mean().reset_index();per_step=steps.groupby(['split','step'])[names].mean().reset_index();summaries={split:dict(images=len(group),**group[names].mean().to_dict()) for split,group in images.groupby('split')}
    result=dict(protocol='No training. Same fixed legacy16 target cells/grid24x48/masks and frozen parent for all arms. Prior is image density logits normalized. BOS-repeated independently calls empty-prefix/no hidden/no crop at every timestep (time remains BOS0), not fictitious zero-coordinate history. Full sees observed past and its actual timestep; full-vs-BOS conflates timestep/motion dynamics with past-coordinate/crop history and is not a pure history contribution. All505 targets/prefixes match guarded raw cache; no test/quarantine images/labels/annotations read.',dynamic_scale=float(model.dynamic_scale),source_parent_sha256=digest(base_path(a.base)),source_cache_provenance_sha256=digest(Path(a.cache)/'provenance.json'),source_raw_cache_sha256=digest(a.raw_cache),source_visual_sha256=provenance['visual_state_sha256'],gaze_sha256=provenance['gaze_sha256'],manifest_sha256=provenance['manifest_sha256'],all_manifest_sha256=provenance['all_manifest_sha256'],training_image_ids=sorted(int(r['image_id']) for r in records if r['split']=='train'),validation_image_ids=sorted(int(r['image_id']) for r in records if r['split']=='val'),summaries=summaries,per_step=per_step.to_dict('records'),full_cached_max_logp_error=max(alignment_errors),seconds=time.monotonic()-started)
    steps.to_csv(output/'per_step.csv',index=False);images.to_csv(output/'per_image.csv',index=False);per_step.to_csv(output/'step_summary.csv',index=False);(output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));lines=['# Image prior and causal history audit','',result['protocol'],'',f"Learned dynamic_scale={result['dynamic_scale']:.6f}; full recomputation maximum cached-logp difference={result['full_cached_max_logp_error']:.3g}.",'','| Split | Prior NLL | Independent BOS repeated NLL | Full causal NLL | Full gain over prior | Full gain over BOS | KL full/prior |','|---|---:|---:|---:|---:|---:|---:|']
    for split,s in summaries.items():lines.append('| '+split+' | '+' | '.join(f'{s[k]:.6f}' for k in ['prior_nll','bos_repeated_nll','full_nll','full_target_logp_gain_over_prior','full_target_logp_gain_over_bos','kl_full_to_prior'])+' |')
    lines+=['','Step0–15 results are in step_summary.csv. Positive target-logp gain means full causal prediction helps on unchanged targets. KL reports distribution movement, not improvement. BOS-repeated retains learned image-conditioned first-step behavior and is distinct from the pure image prior. Full-vs-BOS includes timestep embeddings/motion terms as well as past coordinates/crops, so it is aggregate dynamic-branch gain, not an isolated causal-history contribution.'];(output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(dynamic_scale=result['dynamic_scale'],summaries=summaries,seconds=result['seconds']),indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--raw-cache',default='runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt');p.add_argument('--output',default='runs/dual_target_1011/history_gain_audit');p.add_argument('--threads',type=int,default=4);p.add_argument('--batch-size',type=int,default=4);audit(p.parse_args())
