"""Audit full chosen legacy segment and deterministic stratum16 train augmentation; no CNN/training."""
import argparse,ast,json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from audit_gaze_zoom_1011 import alignment


def retained_segments(points,size,min_distance=32.):
    segments=[];current=[];last=None
    for index,point in enumerate(points):
        valid=np.isfinite(point).all() and (point>=0).all() and (point<np.asarray(size)).all()
        if not valid:
            if current:segments.append(np.asarray(current,dtype=int))
            current=[];last=None
        elif last is None or np.linalg.norm(point-last)>=min_distance:current.append(index);last=point
    if current:segments.append(np.asarray(current,dtype=int))
    return segments


def chosen_legacy_pool(points,size):
    segments=retained_segments(points,size)
    if not segments:raise ValueError('No valid retained segment')
    # Critical: longest AFTER each segment is reduced to at most16, ties first.
    chosen=max(range(len(segments)),key=lambda i:min(16,len(segments[i])));pool=segments[chosen]
    canonical=np.linspace(0,len(pool)-1,16).round().astype(int) if len(pool)>16 else np.arange(len(pool))
    return pool,canonical,chosen,len(segments)


def stratum_indices(length,seed,image_id,epoch,fixed_endpoints=True):
    from src.gaze_path_augmentation_1011 import selected_indices
    if not fixed_endpoints:raise ValueError('Approved sampler uses fixed endpoints plus14 interior strata')
    return selected_indices(length,image_id,epoch,seed,augmented=True)


def audit(a):
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);records={int(r['image_id']):r for r in cache['records']};frame=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','file_name','patient_key','split','width','height']);frame=frame[frame.image_id.isin(records)];allowed=frame.set_index('file_name')
    all_routing=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','split']);val_routing=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','split']);expected_ids=set(all_routing[all_routing.split=='train'].image_id)|set(val_routing[val_routing.split=='val'].image_id)
    if set(records)!=expected_ids:raise ValueError('Raw cache differs from independent fixed all-train412/trajectory-val93 allowlist')
    if len(frame)!=505 or frame.split.value_counts().to_dict()!={'train':412,'val':93}:raise ValueError('Requires guarded train412/val93 raw cache')
    rows=[];pool_records=[];seen=set()
    with (ROOT/'data/gaze_data/merged_gaze_data.csv').open() as source:
        for line in source:
            name,payload=line.split(';',1)
            if name not in allowed.index:continue
            if name in seen:raise ValueError('Duplicate allowed raw gaze')
            seen.add(name);row=allowed.loc[name];samples=ast.literal_eval(payload);scales=np.asarray([float(s[1]) for s in samples])
            with np.errstate(divide='ignore',invalid='ignore'):points=np.asarray([np.asarray(s[0],dtype=float)/float(s[1]) for s in samples])
            size=np.array([row.width,row.height]);pool,canonical,segment,count=chosen_legacy_pool(points,size);record=records[int(row.image_id)]
            if record['split']!=row.split or record['patient_key']!=row.patient_key:raise ValueError('Split/patient mismatch')
            alignment(points,pool[canonical],size,record)
            if not np.isfinite(scales[pool]).all() or (scales[pool]<=0).any():raise ValueError('Invalid pooled scale label')
            unique_points=len(np.unique(points[pool],axis=0));all_retained=sum(len(s) for s in retained_segments(points,size));draws=[stratum_indices(len(pool),42,int(row.image_id),epoch,True) for epoch in range(20)];visited=np.unique(np.concatenate(draws))
            rows.append(dict(image_id=int(row.image_id),patient_key=row.patient_key,split=row.split,raw_samples=len(samples),retained_all_segments=all_retained,valid_segments=count,chosen_segment=segment,chosen_pool_points=len(pool),unique_pool_locations=unique_points,canonical_points=len(canonical),reusable_cached_crop_vectors=max(0,len(canonical)-1),additional_crop_encodes=len(pool)-max(0,len(canonical)-1),pool_scale_unique=len(np.unique(np.round(scales[pool],8))),pool_scale_min=float(scales[pool].min()),pool_scale_max=float(scales[pool].max()),stratum20epochs_fixed_endpoints_unique_visited=len(visited),canonical_alignment_exact=True))
            if row.split=='train':pool_records.append(dict(image_id=int(row.image_id),patient_key=row.patient_key,split=row.split,pool_raw_indices=torch.as_tensor(pool),pool_coords=torch.as_tensor(points[pool]/size,dtype=torch.float32),pool_zoom_log_scale_labels=torch.as_tensor(np.log(scales[pool]),dtype=torch.float32),canonical_pool_indices=torch.as_tensor(canonical),canonical_coords=record['coords'],canonical_loss_mask=record['loss_mask'],zoom_labels_role='Targets only; not model inputs'))
    if seen!=set(allowed.index):raise ValueError('Missing allowed payload')
    table=pd.DataFrame(rows);summary={}
    for split,g in table.groupby('split'):
        n=g.chosen_pool_points;summary[split]=dict(images=len(g),pool_points=int(n.sum()),unique_location_sum=int(g.unique_pool_locations.sum()),canonical_targets=int(g.canonical_points.sum()),pool_to_canonical_ratio=float(n.sum()/g.canonical_points.sum()),pool_length_quantiles={str(q):float(n.quantile(q)) for q in (0,.25,.5,.75,1)},at_least={str(cut):dict(images=int((n>=cut).sum()),fraction=float((n>=cut).mean())) for cut in (32,64,128)},additional_train_crop_encodes=int(g.additional_crop_encodes.sum()) if split=='train' else 0,existing_crop_vectors_reusable=int(g.reusable_cached_crop_vectors.sum()),exact_cache_aligned_images=int(g.canonical_alignment_exact.sum()),stratum20epochs_fixed_endpoints_distinct_target_positions=int(g.stratum20epochs_fixed_endpoints_unique_visited.sum()))
    train=summary['train'];new_crops=train['additional_train_crop_encodes'];vectors_bytes=train['pool_points']*1024*4
    estimate=dict(training_images=412,all_filtered_crop_vectors=train['pool_points'],existing_canonical_previous_crops_reusable=train['existing_crop_vectors_reusable'],new_crop_cnn_inputs=new_crops,new_batch8_crop_calls_lower_bound=int(np.ceil(new_crops/8)),float32_1024dim_vector_storage_bytes=vectors_bytes,vector_storage_gb_decimal=vectors_bytes/1e9,source_whole_maps='Reference guarded existing low-resolution control cache; no whole-image CNN repeats, no validation crop repeats',runtime_sensitivity_minutes={f'{throughput}_crops_per_second':new_crops/throughput/60 for throughput in (100,250,500)},throughput_status='Scenario estimate only, not a measured benchmark; GPU throughput microbenchmark/root review required before full launch')
    result=dict(gaze_sha256=hashlib.sha256((ROOT/'data/gaze_data/merged_gaze_data.csv').read_bytes()).hexdigest(),manifest_sha256=hashlib.sha256((ROOT/'manifests/trajectory.csv').read_bytes()).hexdigest(),all_manifest_sha256=hashlib.sha256((ROOT/'manifests/all.csv').read_bytes()).hexdigest(),protocol='Audit only; original first-longest-after16-reduction segment selection, invalid breaks and32px retained spacing preserved. Train412 only proposed augmentation; validation93 exact canonical16 unchanged. Fixed first/last endpoints plus one sorted point in each of14 equal interior index strata per training epoch; exact shared src.gaze_path_augmentation_1011.selected_indices implementation. This retains full selected segment span/order, unlike contiguous windows. No new gaze/image/annotations or model training.',summary=summary,cache_estimate=estimate,source_raw_cache_sha256=hashlib.sha256(Path(a.cache).read_bytes()).hexdigest(),source_parent_sha256=cache['base_checkpoint_sha256'],sampler=dict(seed=42,interior_bins=14,fixed_endpoints=True,implementation='src.gaze_path_augmentation_1011.selected_indices',sampler_source_sha256=hashlib.sha256((ROOT/'src/gaze_path_augmentation_1011.py').read_bytes()).hexdigest(),epoch_dependent=True,fixed_endpoints_example=True,validation='Original exact canonical16 always'),caveat='All interior points are eligible; first/last endpoints remain fixed. Earlier16bins-then-endpoint-replacement audit is preserved separately and superseded by this exact shared sampler. Existing density-prior supervision may already use more gaze; this proposal specifically expands sequential next-step histories/targets.')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);table.to_csv(output/'per_image.csv',index=False);(output/'metrics.json').write_text(json.dumps(result,indent=2));torch.save(dict(kind='gaze_chosen_segment_pool_labels_1011',records=pool_records,provenance=result),output/'train_pool_labels.pt');lines=['# Chosen-segment augmentation feasibility','',result['protocol'],'','| Split | Pool points | Canonical targets | Median pool/image | ≥128points | Exact alignment |','|---|---:|---:|---:|---:|---:|']
    for split,s in summary.items():lines.append(f"| {split} | {s['pool_points']} | {s['canonical_targets']} | {s['pool_length_quantiles']['0.5']:.0f} | {s['at_least']['128']['fraction']:.1%} | {s['exact_cache_aligned_images']} |")
    lines+=['',f"Proposed train cache: {train['pool_points']:,} crop vectors, {new_crops:,} new CNN crops after reusing {train['existing_crop_vectors_reusable']:,} canonical previous-crop vectors; vector storage≈{vectors_bytes/1e9:.3f}GB. No whole-map re-extraction or validation changes.", '',estimate['throughput_status'],'',result['caveat']];(output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(summary=summary,cache_estimate=estimate),indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt');p.add_argument('--output',default='runs/dual_target_1011/augmentation_pool_audit');audit(p.parse_args())
