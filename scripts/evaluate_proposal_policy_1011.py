"""Posthoc val-only proposal/path diagnostics. Never called by generation."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.guided_trajectory_1011 import candidate_geometry
from src.data import ROOT,load_gazes,reduce_gaze
from src.metrics import dtw_distance
from src.constraints import breast_mask
from scipy import ndimage


def rectangles(points,size,crop=224):
    centers=np.round(np.asarray(points,dtype=float)*size)
    low=centers-crop//2
    return np.concatenate([low,low+crop],axis=1)


def iou(a,b):
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(overlap/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap))


def distinct_proposals(scores,rects,k=5,threshold=.3):
    selected=[]
    for index in np.argsort(-np.asarray(scores),kind='stable'):
        if all(iou(rects[index],rects[j])<=threshold for j in selected):selected.append(int(index))
        if len(selected)==k:break
    return selected


def coverage(rects,box):
    x,y,w,h=box
    if not np.isfinite(box).all() or w<=0 or h<=0:raise ValueError('Invalid lesion bbox')
    return np.maximum(0,np.minimum(rects[:,2:],[x+w,y+h])-np.maximum(rects[:,:2],[x,y])).prod(1)/(w*h)


def lesion_metrics(path_rects,candidate_rects,ranked,box):
    actual=coverage(path_rects,box);candidate=coverage(candidate_rects,box);maximum=float(candidate.max())
    normalized=actual/maximum if maximum>0 else np.full(len(actual),np.nan)
    out=dict(max_achievable_coverage=maximum,actual_box_coverage=float(actual.max()),any_overlap=int(actual.max()>0),normalized_coverage=float(np.nanmax(normalized)) if maximum>0 else None)
    for threshold in (.5,.8):
        tag=str(threshold).replace('.','');hits=np.flatnonzero(normalized>=threshold)
        out[f'first_hit_step_{tag}']=int(hits[0]+1) if len(hits) else None
        out[f'censored_{tag}']=int(not len(hits))
        for step in (1,4,8,16):
            out[f'recall_{tag}_step{step}']=int(np.any(normalized[:step]>=threshold)) if maximum>0 else None
            out[f'actual_recall_{tag}_step{step}']=int(np.any(actual[:step]>=threshold))
        for k in (1,3,5):
            hit=int(np.any(candidate[ranked[:k]]/maximum>=threshold)) if maximum>0 else None
            out[f'proposal_recall_{tag}_top{k}']=hit
            out[f'proposal_hit_path_miss_{tag}_top{k}']=int(hit and not len(hits)) if hit is not None else None
    return out


def paired_bootstrap(frame,metric,baseline,policy,n=2000,seed=1011):
    # Average generation seeds within image BEFORE pairing and patient resampling.
    means=frame.groupby(['patient_key','image_id','policy'])[metric].mean().unstack('policy')
    if baseline not in means or policy not in means:return None
    paired=means[[baseline,policy]].dropna();delta=paired[policy]-paired[baseline]
    clusters=[group.to_numpy() for _,group in delta.groupby(level='patient_key')]
    if not clusters:return None
    rng=np.random.default_rng(seed);values=[]
    for _ in range(n):values.append(float(np.concatenate([clusters[i] for i in rng.integers(len(clusters),size=len(clusters))]).mean()))
    return dict(delta=float(delta.mean()),ci95=np.quantile(values,[.025,.975]).tolist(),paired_images=len(paired),patient_clusters=len(clusters),bootstrap=n)


def read_traces(path):
    keep={'image_id','image_size','grid','crop_size','policy','seed','strategy','sampling_seed','coordinates','indices','relevance','proposal','schema_version','annotation_used','annotations_used','observed_gaze_used','gaze_used','labels_used','checkpoint','checkpoint_sha256','checkpoint_epoch','inputs'}
    def compact(trace):return {k:v for k,v in trace.items() if k in keep}
    path=Path(path)
    if path.suffix=='.jsonl':
        records=[]
        with path.open() as source:
            for line in source:
                if line.strip():records.append(compact(json.loads(line)))
        return records
    obj=json.loads(path.read_text())
    records=obj if isinstance(obj,list) else obj.get('records',obj.get('traces',[obj]))
    return [compact(trace) for trace in records]


def baseline_for(policy,baseline):
    return (baseline if baseline.endswith('_argmax') else baseline+'_argmax') if policy.endswith('_argmax') else baseline


def route_fixed_validation(traces,validation_ids,all_validation_ids):
    excluded=[];kept=[]
    for trace in traces:
        image_id=int(trace['image_id'])
        if image_id in validation_ids:kept.append(trace)
        elif image_id in all_validation_ids:excluded.append(image_id)
        else:raise ValueError(f'Refusing non-validation image {image_id}')
    return kept,dict(protocol='Fixed prior trajectory validation93 intersection; additional all.csv validation routing records excluded without performance selection',excluded_image_ids=sorted(set(excluded)),excluded_unique_images=len(set(excluded)),excluded_trace_count=len(excluded),input_trace_count=len(traces),analyzed_trace_count=len(kept))


def evaluate(traces,output,baseline='original',bootstrap=2000,expected_crop_size=224):
    manifest=pd.read_csv(ROOT/'manifests/trajectory.csv');val=manifest[manifest.split=='val'].set_index('image_id')
    all_manifest=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','split'])
    traces,routing=route_fixed_validation(traces,set(int(i) for i in val.index),set(int(i) for i in all_manifest[all_manifest.split=='val'].image_id))
    annotations=json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations'];boxes={int(i):[] for i in val.index}
    for annotation in annotations:
        if int(annotation['image_id']) in boxes:boxes[int(annotation['image_id'])].append(annotation['bbox'])
    gazes=load_gazes();images=[];lesions=[];cache={};seen=set();seed_sets={}
    for trace in traces:
        key=(int(trace['image_id']),trace.get('policy','original'),int(trace.get('seed',42)),trace.get('strategy','sample'))
        if key in seen:raise ValueError(f'Duplicate trace {key}')
        seen.add(key)
        seed_sets.setdefault((key[0],key[1]),set()).add((key[2],key[3]))
        if trace.get('schema_version') and trace.get('annotations_used',trace.get('annotation_used')) is not False or (trace.get('schema_version') and (trace.get('observed_gaze_used') is not False or trace.get('labels_used') is not False)):raise ValueError('New schema requires annotation_used=false')
    for trace in traces:
        image_id=int(trace['image_id'])
        if image_id not in val.index:raise ValueError(f'Refusing non-validation image {image_id}')
        row=val.loc[image_id];grid=tuple(trace.get('grid',(24,48)));crop=int(trace.get('crop_size',224))
        if grid!=(24,48) or crop!=expected_crop_size:raise ValueError(f'Protocol requires24x48grid and explicit{expected_crop_size}px crop')
        key=(image_id,grid,crop)
        if key not in cache:
            with Image.open(ROOT/'data/image'/row.file_name) as source:
                image=source.convert('L');geometry=candidate_geometry(image,grid,crop)
                mask=breast_mask(image) if geometry['geometry_valid'] else None
                boundary_map=ndimage.distance_transform_edt(np.pad(mask,1))[1:-1,1:-1] if mask is not None else None
                diagnostics=[]
                for box in boxes[image_id]:
                    cx,cy=np.round([box[0]+box[2]/2,box[1]+box[3]/2]).astype(int)
                    cx=int(np.clip(cx,0,image.width-1));cy=int(np.clip(cy,0,image.height-1))
                    distance=float(boundary_map[cy,cx]) if boundary_map is not None else None
                    group='mask_invalid' if distance is None else ('center_outside' if not mask[cy,cx] else ('boundary_le112px' if distance<=112 else 'interior_gt112px'))
                    diagnostics.append((distance,group))
                geometry['lesion_diagnostics']=diagnostics
                del mask,boundary_map
            cache[key]=geometry
        geometry=cache[key];size=np.array(geometry['image_size']);points=np.asarray(trace['coordinates'])
        if points.shape!=(16,2) or not np.isfinite(points).all() or np.any((points<0)|(points>=1)):raise ValueError('Expected finite normalized16step coordinates')
        rects=rectangles(points,size,crop);candidates=geometry['rectangles'].numpy()
        scores=trace.get('relevance',trace.get('proposal'))
        if scores is None or len(scores)!=len(candidates):raise ValueError('Missing candidate relevance')
        ranked=distinct_proposals(scores,candidates)
        indices=np.asarray(trace.get('indices',np.argmin(((points[:,None]-geometry['coords'].numpy()[None])**2).sum(2),axis=1)),dtype=int)
        if indices.shape!=(16,) or np.any((indices<0)|(indices>=len(candidates))):raise ValueError('Invalid grid indices')
        if not np.allclose(points,geometry['coords'].numpy()[indices],atol=1e-6):raise ValueError('Coordinate/index mismatch')
        previous_max=[max(iou(rects[i],rects[j]) for j in range(i)) for i in range(1,len(rects))]
        previous=[iou(rects[i],rects[j]) for i in range(1,len(rects)) for j in range(i)]
        policy=trace.get('policy','original')
        if trace.get('strategy')=='argmax' and not policy.endswith('_argmax'):policy+='_argmax'
        base=dict(image_id=image_id,patient_key=row.patient_key,policy=policy,strategy=trace.get('strategy','sample'),seed=trace.get('seed',42),image_label=row.image_label,known_annotation=int(bool(boxes[image_id])),lesion_count=len(boxes[image_id]))
        segments=reduce_gaze(gazes[row.file_name],*size,k=16,min_distance=32)
        item=dict(base,normalized_dtw=dtw_distance(max(segments,key=len)/size,points),mean_foreground=float(geometry['foreground'][indices].mean()) if geometry['geometry_valid'] else None,mean_inbounds=float(geometry['inbounds'][indices].mean()),mean_boundary_distance_px=float(geometry['distance'][indices].mean()) if geometry['geometry_valid'] else None,high_background_patch_fraction=float((geometry['foreground'][indices]<.5).float().mean()) if geometry['geometry_valid'] else None,unique_locations=len(np.unique(indices)),mean_max_previous_crop_iou=float(np.mean(previous_max)) if previous_max else 0.,mean_allpairs_crop_iou=float(np.mean(previous)) if previous else 0.,max_previous_crop_iou=max(previous,default=0.),geometry_valid=int(geometry['geometry_valid']))
        records=[]
        for j,box in enumerate(boxes[image_id]):
            distance,boundary_group=geometry['lesion_diagnostics'][j]
            record=dict(base,lesion_index=j,lesion_boundary_distance_px=distance,lesion_boundary_group=boundary_group,lesion_box_area_px=float(box[2]*box[3]),**lesion_metrics(rects,candidates,ranked,box))
            records.append(record);lesions.append(record)
        if records:
            for name in records[0]:
                if name not in base and name not in ('lesion_index','lesion_boundary_group'):
                    values=[r[name] for r in records if r[name] is not None]
                    item[name]=float(np.mean(values)) if values else None
        images.append(item)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frame=pd.DataFrame(images);lesion_frame=pd.DataFrame(lesions);frame.to_csv(output/'per_image.csv',index=False);lesion_frame.to_csv(output/'per_lesion.csv',index=False)
    numeric=[c for c in frame.select_dtypes(include='number') if c not in ('image_id','seed')]
    summaries={policy:{metric:float(group.groupby('image_id')[metric].mean().mean()) for metric in numeric if group[metric].notna().any()} for policy,group in frame.groupby('policy')}
    incomplete=[]
    for (image_id,policy),seeds in seed_sets.items():
        reference=seed_sets.get((image_id,baseline))
        if reference is not None and seeds!=reference:incomplete.append(dict(image_id=image_id,policy=policy,seeds=sorted(seeds),baseline_seeds=sorted(reference)))
    if incomplete:raise ValueError(f'Unmatched generation seed sets: {incomplete[:5]}')
    comparisons={policy:{metric:paired_bootstrap(frame,metric,baseline_for(policy,baseline),policy,bootstrap) for metric in numeric} for policy in frame.policy.unique() if policy not in (baseline,baseline+'_argmax')}
    subgroups={}
    for column in ['image_label']:
        subgroups[column]={str(label):group.groupby('policy')[numeric].mean().replace({np.nan:None}).to_dict() for label,group in frame.groupby(column)}
    metrics=dict(protocol='Exploratory val93 development; annotations posthoc only; boxless unknown; no test/INbreast evaluation; no duration or train prior metrics',definitions=dict(H=f'maximum single {expected_crop_size}px crop intersection area / lesion bbox area',normalized='H / maximum H over all grid candidate crops',image_recall='mean lesion recall per known annotated image, then mean images',gap='joint proposal-hit and path-miss / known reachable lesions; conditional denominator also reported',dtw='Repository normalized-coordinate Euclidean DTW / (truth length + prediction length), legacy16 observed reference only',bootstrap='Seeds averaged within image; paired image differences; patient clusters sampled with replacement'),denominators=dict(validation_manifest_images=len(val),evaluated_images=int(frame.image_id.nunique()),known_annotated_images=int(frame[frame.known_annotation==1].image_id.nunique()),known_lesions=int(lesion_frame[['image_id','lesion_index']].drop_duplicates().shape[0]) if len(lesion_frame) else 0),summaries=summaries,paired_patient_bootstrap=comparisons,posthoc_subgroups=subgroups)
    if len(lesion_frame):
        size_cut=224.**2;lesion_frame['lesion_size_group']=np.where(lesion_frame.lesion_box_area_px<=size_cut,'small_le_224square','large_gt_224square')
        lesion_frame.to_csv(output/'per_lesion.csv',index=False)
        metrics['lesion_size_fixed_area_px']=size_cut
        metrics['lesion_subgroups']={column:{str(label):group.groupby('policy').mean(numeric_only=True).replace({np.nan:None}).to_dict() for label,group in lesion_frame.groupby(column)} for column in ['lesion_boundary_group','lesion_size_group','image_label']}
        metrics['lesion_summaries']={p:g.groupby(['image_id','lesion_index']).mean(numeric_only=True).mean().to_dict() for p,g in lesion_frame.groupby('policy')}
        metrics['proposal_path_gaps']={}
        for p,g in lesion_frame.groupby('policy'):
            averaged=g.groupby(['image_id','lesion_index']).mean(numeric_only=True);metrics['proposal_path_gaps'][p]={}
            for t in ('05','08'):
                for k in (1,3,5):
                    hit=averaged[f'proposal_recall_{t}_top{k}'];miss=averaged[f'proposal_hit_path_miss_{t}_top{k}'];metrics['proposal_path_gaps'][p][f'{t}_top{k}']=dict(known_reachable_lesions=int(hit.notna().sum()),proposal_hit_equivalent=float(hit.sum()),miss_equivalent=float(miss.sum()),joint_gap=float(miss.mean()),conditional_path_miss=float(miss.sum()/hit.sum()) if hit.sum()>0 else None)
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    metrics['crop_protocol']=dict(crop_size_px=expected_crop_size,patch_budget=16,grid=[24,48],area_relative_to224=(expected_crop_size/224)**2,explicit_generation_trace_size_required=True)
    metrics['cohort_routing']=routing
    metrics['trace_completeness']=dict(trace_count=len(traces),policies={p:dict(images=int(g.image_id.nunique()),seeds=sorted(int(x) for x in g.seed.unique()),complete_validation=bool(g.image_id.nunique()==len(val))) for p,g in frame.groupby('policy')})
    metrics['engineering_screen']={}
    if baseline in summaries:
        for policy,summary in summaries.items():
            reference=baseline_for(policy,baseline)
            if reference not in summaries:continue
            ref=summaries[reference]
            delta=summary.get('recall_05_step16',0)-ref.get('recall_05_step16',0);fgdrop=ref.get('mean_foreground',0)-summary.get('mean_foreground',0);dtwdelta=summary.get('normalized_dtw',0)-ref.get('normalized_dtw',0)
            metrics['engineering_screen'][policy]=dict(baseline=reference,recall50_delta=delta,foreground_proxy_drop=fgdrop,dtw_delta=dtwdelta,passes=bool(delta>=.05 and fgdrop<=.02 and dtwdelta<=.005),exploratory_only=True)
    metrics=clean(metrics)
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    lines=['# Posthoc proposal-to-visit diagnostics','',metrics['protocol'], '',f"Evaluated {metrics['denominators']['evaluated_images']} images; {metrics['denominators']['known_annotated_images']} have explicit boxes ({metrics['denominators']['known_lesions']} lesions). Missing boxes remain unknown.",'',f'Coverage is maximum individual {expected_crop_size}px crop coverage, normalized by achievable grid coverage. Proposal top1/3/5 uses relevance ranking with crop IoU >0.3 suppression. Recall horizons are steps 1/4/8/16; first hits are censored when absent.','', 'Seeds are averaged per image before paired patient cluster bootstrap. Argmax must carry a separate policy name. DTW uses the existing legacy16 reference convention; no duration or train prior is scored. NLL belongs in separate teacher-forced reports.','','| Policy | Recall .5 at16 | Recall .8 at16 | Actual coverage | DTW |','|---|---:|---:|---:|---:|']
    for p,s in summaries.items():lines.append('| '+p+' | '+' | '.join(f'{s.get(k,float("nan")):.4f}' for k in ['recall_05_step16','recall_08_step16','actual_box_coverage','normalized_dtw'])+' |')
    lines+=['','Detailed denominators, proposal/path gaps, bootstrap comparisons and posthoc label/size/GT-boundary subgroups are in metrics.json. Diagnostics are not selection inputs.']
    lines+=['',f"Fixed cohort routing: excluded {routing['excluded_unique_images']} extra all.csv validation images ({routing['excluded_trace_count']} traces); analyzed prior trajectory validation93. Excluded IDs: {routing['excluded_image_ids']}."]
    lines+=['','Engineering screen against '+baseline+': recall50 gain >=5pp, foreground proxy drop <=2pp, DTW increase <=0.005. Exploratory engineering criteria; bootstrap uncertainty is reported in metrics.json.']
    for policy,screen in metrics['engineering_screen'].items():lines.append(f"{policy}: recall50 delta {screen['recall50_delta']:+.4f}; foreground proxy drop {screen['foreground_proxy_drop']:+.4f}; DTW delta {screen['dtw_delta']:+.4f}; pass={screen['passes']}")
    (output/'report.md').write_text('\n'.join(lines)+'\n');return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--traces',required=True);parser.add_argument('--output',required=True);parser.add_argument('--baseline',default='original');parser.add_argument('--crop-size',type=int,choices=(224,256,320,384,448),default=224);parser.add_argument('--bootstrap',type=int,default=2000);args=parser.parse_args();evaluate(read_traces(args.traces),args.output,args.baseline,args.bootstrap,args.crop_size)
