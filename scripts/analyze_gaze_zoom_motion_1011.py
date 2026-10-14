"""Descriptive GT-scale oracle motion contrast and labels-only zoom targets; no training."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def motion_oracle(previous_raw,current_raw,previous_scale,current_scale):
    previous_world=np.asarray(previous_raw)/previous_scale;current_world=np.asarray(current_raw)/current_scale
    predicted=previous_world*previous_scale/current_scale;delta=current_world-previous_world;predicted_delta=predicted-previous_world
    raw_jump=float(np.linalg.norm(np.asarray(current_raw)-previous_raw));world_jump=float(np.linalg.norm(delta));residual=float(np.linalg.norm(current_world-predicted));predicted_jump=float(np.linalg.norm(predicted_delta));denominator=world_jump*predicted_jump
    return dict(raw_record_jump=raw_jump,restored_world_jump_px=world_jump,log_scale_ratio=float(np.log(current_scale/previous_scale)),scale_change_over5pct=bool(abs(np.log(current_scale/previous_scale))>np.log(1.05)),world_long_raw_local=bool(world_jump>224 and raw_jump<=112),oracle_constant_raw_residual_px=residual,oracle_direction_cosine=float(np.dot(delta,predicted_delta)/denominator) if denominator>1e-12 else np.nan,oracle_magnitude_error_px=abs(world_jump-predicted_jump),oracle_better_than_world_persistence=bool(residual<world_jump))


def analyze(a):
    root=Path(a.audit);steps=pd.read_csv(root/'per_step.csv');audit=json.loads((root/'metrics.json').read_text());cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    if hashlib.sha256(Path(a.cache).read_bytes()).hexdigest()!=audit['source_cache_sha256']:raise ValueError('Labels audit must use exact aligned source cache')
    records={int(r['image_id']):r for r in cache['records']}
    if set(records)!=set(steps.image_id):raise ValueError('Zoom labels/cache cohort mismatch')
    rows=[];labels=[];train_scales=steps[steps.split=='train'].recorded_scale.to_numpy();edges=np.quantile(train_scales,[.25,.5,.75])
    if len(np.unique(edges))!=3:raise ValueError('Cannot define four distinct train-quantile zoom bins')
    for image_id,group in steps.groupby('image_id'):
        group=group.sort_values('step');record=records[int(image_id)];n=len(group);mask=record['loss_mask'].bool()
        if int(mask.sum())!=n or not np.array_equal(group.step.to_numpy(),np.arange(1,n+1)):raise ValueError('Zoom step mask alignment mismatch')
        log_labels=torch.zeros(16);bin_labels=torch.full((16,),-100,dtype=torch.long);scales=group.recorded_scale.to_numpy();log_labels[:n]=torch.as_tensor(np.log(scales),dtype=torch.float32);bin_labels[:n]=torch.as_tensor(np.searchsorted(edges,scales,side='right'),dtype=torch.long)
        label_record=dict(record,zoom_log_scale_labels=log_labels,zoom_bin_labels=bin_labels,zoom_label_mask=mask.clone(),raw_indices=torch.as_tensor(group.raw_index.to_numpy(),dtype=torch.long),label_role='Supervised targets only; neither current nor past GT zoom may enter validation/generation model inputs')
        labels.append(label_record)
        for i in range(1,n):
            previous=group.iloc[i-1];current=group.iloc[i];values=motion_oracle(previous[['raw_record_x','raw_record_y']].to_numpy(dtype=float),current[['raw_record_x','raw_record_y']].to_numpy(dtype=float),previous.recorded_scale,current.recorded_scale)
            rows.append(dict(image_id=int(image_id),patient_key=record['patient_key'],split=record['split'],step=i+1,raw_parent_nll=float(current.raw_parent_nll),**values))
    frame=pd.DataFrame(rows);summary={}
    for split,group in frame.groupby('split'):
        changing=group[group.scale_change_over5pct];summary[split]=dict(transitions=len(group),scale_change_over5pct_count=len(changing),scale_change_over5pct_fraction=float(group.scale_change_over5pct.mean()),among_scale_change_world_long_raw_local_fraction=float(changing.world_long_raw_local.mean()),all_world_long_raw_local_fraction=float(group.world_long_raw_local.mean()),oracle_residual_median_px=float(changing.oracle_constant_raw_residual_px.median()),actual_world_jump_median_px=float(changing.restored_world_jump_px.median()),oracle_direction_cosine_mean=float(changing.oracle_direction_cosine.mean()),oracle_direction_cosine_median=float(changing.oracle_direction_cosine.median()),oracle_magnitude_error_median_px=float(changing.oracle_magnitude_error_px.median()),oracle_better_than_persistence_fraction=float(changing.oracle_better_than_world_persistence.mean()),scale_change_nll_transition_macro=float(changing.raw_parent_nll.mean()),no_scale_change_nll_transition_macro=float(group[~group.scale_change_over5pct].raw_parent_nll.mean()),all93_or412_nll_image_macro=audit['summary'][split]['raw_nll_image_macro'])
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);frame.to_csv(output/'per_transition.csv',index=False)
    primary='https://github.com/YanKong0408/MicEye-v2.0/blob/3f8774375aca45be5c4ba5af0a2649cbc61160ed/MainWindow.py'
    result=dict(protocol='Descriptive ORACLE contrast using previous/current GT scales only after exact selected16 reconstruction. No zoom head/model trained and no oracle scores count toward NLL5.3. High-scale subset NLL is not all93 achievement. Labels cache is for targets only; deployed zoom must be predicted causally from image/past coordinates/parent hidden, never GT zoom.',raw_coordinate_semantics=dict(primary_source=primary,source_commit='3f8774375aca45be5c4ba5af0a2649cbc61160ed',finding='MicEye-v2.0 getPointInImage subtracts image widget absolute origin from eye position; refresh records [widget-local scaled-image coordinate,imagescale]. These are raw-record/widget-local units, not invariant screen coordinates. Exact software version used by this dataset has not been established.',locations=['getPointInImage lines35–40','getImageGeometry lines243–251','refresh lines377–383']),summary=summary,zoom_bin_edges_train_only=edges.tolist(),target_parent_sha256=cache['base_checkpoint_sha256'],source_cache_sha256=hashlib.sha256(Path(a.cache).read_bytes()).hexdigest())
    torch.save(dict(kind='gaze_zoom_labels_only_1011',records=labels,zoom_bin_edges=edges.tolist(),provenance=result),output/'zoom_label_cache.pt');(output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));lines=['# GT-scale oracle motion audit','',result['protocol'],'',f"MicEye-v2.0 code records image-widget-local coordinates and image scale ([pinned primary code]({primary}#L35)). Screen-coordinate invariance is not assumed; exact dataset software revision remains unverified.",'','| Split | GT scale change >5% | World>224/raw≤112 among changes | Oracle direction cosine | Oracle residual median | World jump median |','|---|---:|---:|---:|---:|---:|']
    for split,s in summary.items():lines.append(f"| {split} | {s['scale_change_over5pct_fraction']:.1%} | {s['among_scale_change_world_long_raw_local_fraction']:.1%} | {s['oracle_direction_cosine_mean']:.3f} | {s['oracle_residual_median_px']:.1f}px | {s['actual_world_jump_median_px']:.1f}px |")
    lines+=['','The constant-raw oracle uses previousWorld×previousGTscale/currentGTscale. It is a descriptive transform contrast, not a deployable model or proof of a causal source of prediction error. Four-bin/current-log-scale labels use training quantiles only; labels are stored separately and must never become validation/generation inputs.']
    (output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',default='runs/dual_target_1011/zoom_audit_extended');p.add_argument('--cache',default='runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt');p.add_argument('--output',default='runs/dual_target_1011/zoom_motion_oracle_audit');analyze(p.parse_args())
