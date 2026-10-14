"""Audit discarded recorded gaze scale on fixed train412/val93; no model fitting."""
import argparse,ast,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def selected_indices(points,width,height,k=16,min_distance=32.):
    segments=[];current=[];last=None
    for index,point in enumerate(points):
        valid=np.isfinite(point).all() and 0<=point[0]<width and 0<=point[1]<height
        if not valid:
            if current:segments.append(current)
            current=[];last=None
        elif last is None or np.linalg.norm(point-last)>=min_distance:
            current.append(index);last=point
    if current:segments.append(current)
    if not segments:raise ValueError('No valid raw segment')
    reduced=[np.asarray(segment)[np.linspace(0,len(segment)-1,k).round().astype(int)] if len(segment)>k else np.asarray(segment) for segment in segments]
    chosen=max(range(len(reduced)),key=lambda i:len(reduced[i]))
    return reduced[chosen],dict(valid_segments=len(segments),selected_segment=chosen,selected_retained_segment_length=len(segments[chosen]),selected_steps=len(reduced[chosen]))


def scale_changes(values):return int(np.count_nonzero(~np.isclose(np.asarray(values)[1:],np.asarray(values)[:-1],rtol=1e-6,atol=1e-9)))


def alignment(points,indices,size,record):
    coords=torch.zeros_like(record['coords']);selected=torch.as_tensor(points[indices]/size,dtype=coords.dtype);coords[:len(selected)]=selected
    mask=torch.zeros_like(record['loss_mask'],dtype=torch.bool);mask[:len(selected)]=True
    if not torch.equal(coords,record['coords']) or not torch.equal(mask,record['loss_mask']) or not torch.equal(mask,record['valid']):raise ValueError(f'Raw-index reconstruction/cache alignment failed{record["image_id"]}')
    target=(coords[:,1]*48).long()*24+(coords[:,0]*24).long()
    if not torch.equal(target[mask],record['target'][mask]):raise ValueError('Target reconstruction mismatch')
    return selected


def audit(a):
    val_manifest=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','split']);val_ids=set(val_manifest[val_manifest.split=='val'].image_id)
    frame=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','file_name','patient_key','split','width','height']);frame=frame[(frame.split=='train')|((frame.split=='val')&frame.image_id.isin(val_ids))]
    if frame.split.value_counts().to_dict()!={'train':412,'val':93}:raise ValueError('Requires fixed train412/val93')
    allowed=frame.set_index('file_name');cache=torch.load(a.cache,map_location='cpu',weights_only=False);cached={int(r['image_id']):r for r in cache['records']}
    if set(cached)!=set(frame.image_id):raise ValueError('Raw cache cohort mismatch')
    images=[];steps=[];raw_scale_rows=[];seen=set()
    with (ROOT/'data/gaze_data/merged_gaze_data.csv').open() as source:
        for line in source:
            name,payload=line.split(';',1)
            if name not in allowed.index:continue # Other cohort payloads are never parsed.
            if name in seen:raise ValueError('Duplicate allowed gaze payload')
            seen.add(name);row=allowed.loc[name];samples=ast.literal_eval(payload);scales=np.asarray([float(sample[1]) for sample in samples]);points=[]
            for sample in samples:
                with np.errstate(divide='ignore',invalid='ignore'):point=np.asarray(sample[0],dtype=float)/float(sample[1])
                if point.shape!=(2,):raise ValueError('Malformed raw coordinate')
                points.append(point)
            points=np.asarray(points);indices,info=selected_indices(points,row.width,row.height);record=cached[int(row.image_id)]
            if record['split']!=row.split or record['patient_key']!=row.patient_key:raise ValueError('Split/patient cache mismatch')
            chosen=alignment(points,indices,np.asarray([row.width,row.height]),record);selected_scale=scales[indices]
            if not np.isfinite(selected_scale).all() or (selected_scale<=0).any():raise ValueError('Selected scale is not finite positive')
            valid=np.isfinite(points).all(1)&(points>=0).all(1)&(points<np.array([row.width,row.height])).all(1)&np.isfinite(scales)&(scales>0)
            valid_scale=scales[valid];span_scale=scales[int(indices[0]):int(indices[-1])+1];unique=lambda v:int(len(np.unique(np.round(v,8))))
            images.append(dict(image_id=int(row.image_id),patient_key=row.patient_key,split=row.split,raw_samples=len(samples),raw_valid_points=int(valid.sum()),raw_valid_unique_scales=unique(valid_scale),raw_scale_min=float(valid_scale.min()),raw_scale_max=float(valid_scale.max()),raw_scale_ratio=float(valid_scale.max()/valid_scale.min()),selected_unique_scales=unique(selected_scale),selected_scale_min=float(selected_scale.min()),selected_scale_max=float(selected_scale.max()),selected_scale_changes=scale_changes(selected_scale),raw_span_scale_changes=scale_changes(span_scale),raw_span_episodes=scale_changes(span_scale)+1,selected_raw_start=int(indices[0]),selected_raw_end=int(indices[-1]),**info))
            raw_scale_rows.extend(dict(split=row.split,scale=float(scale)) for scale in valid_scale)
            for step,index in enumerate(indices):
                target=int(record['target'][step]);nll=float(-record['raw_logp'][step,target]);past_scale=float(selected_scale[step-1]) if step else None;distance=float(np.linalg.norm((chosen[step]-chosen[step-1]).numpy()*[row.width,row.height])) if step else None
                steps.append(dict(image_id=int(row.image_id),patient_key=row.patient_key,split=row.split,step=step+1,raw_index=int(index),raw_record_x=float(samples[index][0][0]),raw_record_y=float(samples[index][0][1]),restored_x=float(points[index,0]),restored_y=float(points[index,1]),recorded_scale=float(scales[index]),past_recorded_scale=past_scale,scale_changed_from_past=bool(not np.isclose(scales[index],past_scale,rtol=1e-6,atol=1e-9)) if step else None,jump_pixels=distance,local_within112=int(distance<=112) if step else None,long_jump_over224=int(distance>224) if step else None,raw_parent_nll=nll))
    if seen!=set(allowed.index):raise ValueError('Missing allowed gaze payloads')
    image_frame=pd.DataFrame(images);step_frame=pd.DataFrame(steps);raw_frame=pd.DataFrame(raw_scale_rows);past=step_frame.dropna(subset=['past_recorded_scale']).copy();edges=np.unique(past[past.split=='train'].past_recorded_scale.quantile([.25,.5,.75]).to_numpy());past['past_scale_train_quantile_bin']=pd.cut(past.past_recorded_scale,[-np.inf,*edges,np.inf],duplicates='drop').astype(str)
    summary={}
    for split,group in image_frame.groupby('split'):
        selected=step_frame[step_frame.split==split];transitions=past[past.split==split];raw=raw_frame[raw_frame.split==split].scale
        summary[split]=dict(images=len(group),patients=group.patient_key.nunique(),aligned_cache_images=len(group),images_raw_scale_varies=int((group.raw_valid_unique_scales>1).sum()),images_selected16_scale_varies=int((group.selected_unique_scales>1).sum()),selected16_scale_varies_fraction=float((group.selected_unique_scales>1).mean()),images_selected16_spans_multiple_raw_zoom_episodes=int((group.raw_span_episodes>1).sum()),selected_adjacent_scale_change_fraction=float(transitions.scale_changed_from_past.mean()),raw_scale_quantiles={str(q):float(raw.quantile(q)) for q in (0,.25,.5,.75,1)},selected_scale_quantiles={str(q):float(selected.recorded_scale.quantile(q)) for q in (0,.25,.5,.75,1)},mean_raw_span_episodes=float(group.raw_span_episodes.mean()),max_raw_span_episodes=int(group.raw_span_episodes.max()),local_within112_rate=float(transitions.local_within112.mean()),long_jump_over224_rate=float(transitions.long_jump_over224.mean()),raw_nll_image_macro=float(selected.groupby('image_id').raw_parent_nll.mean().mean()))
    grouped=past.groupby(['split','past_scale_train_quantile_bin'],observed=True).agg(transitions=('raw_parent_nll','size'),images=('image_id','nunique'),local_within112_rate=('local_within112','mean'),long_jump_over224_rate=('long_jump_over224','mean'),nll_transition_macro=('raw_parent_nll','mean')).reset_index()
    for i,r in grouped.iterrows():
        subset=past[(past.split==r['split'])&(past.past_scale_train_quantile_bin==r['past_scale_train_quantile_bin'])];grouped.loc[i,'nll_image_macro_within_bin']=subset.groupby('image_id').raw_parent_nll.mean().mean()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);image_frame.to_csv(output/'per_image.csv',index=False);step_frame.to_csv(output/'per_step.csv',index=False);grouped.to_csv(output/'past_scale_groups.csv',index=False)
    result=dict(scale_run_definition='An episode is a contiguous run of near-equal recorded scalar, rtol1e-6/atol1e-9. UI zoom-animation samples can create several runs; these are not independently identified annotation events or clinical reading episodes.',protocol='Data audit only. Recorded scalar divided raw coordinates in legacy load_gazes; scale is an annotation-view magnification proxy, not clinical duration/natural-reading state. Only fixed train412/val93 payloads parsed; no test/quarantine or annotations loaded. All selected raw indices reconstructed with invalid breaks/min_distance32/legacy16 first-tie convention and exact cache coords/masks/target alignment. Current/future scale is not any model input; past-scale bins are descriptive and may be confounded by image/user/annotation stage.',source_cache=str(Path(a.cache).resolve()),source_cache_sha256=hashlib.sha256(Path(a.cache).read_bytes()).hexdigest(),source_parent_sha256=cache['base_checkpoint_sha256'],gaze_source_sha256=hashlib.sha256((ROOT/'data/gaze_data/merged_gaze_data.csv').read_bytes()).hexdigest(),summary=summary,past_scale_train_quantile_edges=edges.tolist(),past_scale_groups=grouped.to_dict('records'))
    (output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));lines=['# Recorded gaze scale audit','',result['protocol'],'','| Split | Images | Selected16 scale varies | Selected transitions changing scale | Multiple raw episodes spanned | Raw NLL |','|---|---:|---:|---:|---:|---:|']
    for split,g in summary.items():lines.append(f"| {split} | {g['images']} | {g['images_selected16_scale_varies']} ({g['selected16_scale_varies_fraction']:.1%}) | {g['selected_adjacent_scale_change_fraction']:.1%} | {g['images_selected16_spans_multiple_raw_zoom_episodes']} | {g['raw_nll_image_macro']:.6f} |")
    lines+=['',result['scale_run_definition'],'','Past-scale groups, jump rates and NLL are in past_scale_groups.csv. These are associations, not evidence that scale caused a model error. An image-only deployed policy cannot consume validation ground-truth zoom; any future latent zoom mechanism must predict its own causal state. No zoom model was developed in this audit.']
    (output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt');p.add_argument('--output',default='runs/dual_target_1011/zoom_audit');a=p.parse_args();audit(a)
