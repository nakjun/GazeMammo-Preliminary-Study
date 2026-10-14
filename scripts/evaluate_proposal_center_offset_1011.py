"""Explicit posthoc continuous-ROI protocol; canonical grid evaluator stays unchanged."""
import argparse,ast,json,sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_proposal_policy_1011 import rectangles,distinct_proposals,lesion_metrics,iou,paired_bootstrap
from src.constraints import breast_mask
from src.metrics import dtw_distance
from src.data import reduce_gaze


def validate_score_recipe(trace,candidate):
    recipe=trace.get('score_recipe')
    if recipe is None:return
    if recipe.get('kind')!='originalR_capture_geometric':raise ValueError('Unknown declared score recipe')
    expected_sampling=dict(mode='bilinear',align_corners=False,padding_mode='border',grid=[24,48])
    if recipe.get('sampling')!=expected_sampling or recipe.get('invalid_capture_fallback')!='originalR' or recipe.get('alpha1_exact_identity') is not True:raise ValueError('Capture sampling/identity protocol mismatch')
    field=torch.as_tensor(trace['capture_field'],dtype=torch.float32);original=torch.as_tensor(trace['original_relevance'],dtype=torch.float32)
    if field.numel()!=1152 or original.numel()!=1152 or not torch.isfinite(field).all() or not torch.isfinite(original).all() or ((field<0)|(field>1)).any() or ((original<0)|(original>1)).any():raise ValueError('Malformed capture/original field')
    alpha=float(recipe['alpha']);epsilon=float(recipe['epsilon'])
    if not 0<=alpha<=1 or not 0<epsilon<1 or not isinstance(recipe['capture_valid'],bool):raise ValueError('Malformed capture mixture settings')
    points=(2*torch.as_tensor(candidate,dtype=torch.float32)-1).reshape(1,1,-1,2)
    sampled=torch.nn.functional.grid_sample(field.reshape(1,1,48,24),points,mode='bilinear',padding_mode='border',align_corners=False).reshape(-1)
    if not torch.allclose(sampled,torch.as_tensor(trace['capture_scores'],dtype=torch.float32),rtol=1e-5,atol=2e-6):raise ValueError('Capture field sample mismatch')
    scores=original if alpha==1 or not recipe['capture_valid'] else (alpha*original.clamp_min(epsilon).log()+(1-alpha)*sampled.clamp_min(epsilon).log()).exp().clamp(0,1)
    if (alpha==1 or not recipe['capture_valid']) and not torch.equal(scores,torch.as_tensor(trace['relevance'],dtype=torch.float32)):raise ValueError('Exact originalR identity failed')
    if not torch.allclose(scores,torch.as_tensor(trace['relevance'],dtype=torch.float32),rtol=1e-5,atol=2e-6):raise ValueError('Declared capture ranking mismatch')
    from src.proposal_policy_1011 import proposal_budget_indices
    candidate_rects=torch.as_tensor(rectangles(candidate,trace['image_size'],224),dtype=torch.float32)
    ranked=proposal_budget_indices(torch.as_tensor(trace['relevance'],dtype=torch.float32),candidate_rects,int(trace['requested_roi_budget']),float(trace['max_crop_iou']))
    if ranked!=trace['reserved_candidate_indices']:raise ValueError('Capture reserved NMS order mismatch')

def validate_trace(trace,crop_size=224):
    if trace.get('coordinate_protocol')!='center_offset_continuous_roi_v1':raise ValueError('Wrong explicit continuous ROI protocol')
    if any(trace.get(key) is not False for key in ('annotations_used','observed_gaze_used','labels_used')):raise ValueError('Generation must exclude annotations/gaze/labels')
    if trace.get('grid')!=[24,48] or trace.get('crop_size')!=crop_size:raise ValueError('Grid/crop protocol mismatch')
    points=np.asarray(trace['coordinates'],dtype=float);candidate=np.asarray(trace['candidate_coords'],dtype=float);indices=np.asarray(trace['source_anchor_indices']);flags=np.asarray(trace['reservation_flags'],dtype=bool)
    if points.shape!=(16,2) or candidate.shape!=(1152,2) or not np.isfinite(points).all() or not np.isfinite(candidate).all() or np.any((points<0)|(points>=1)) or np.any((candidate<0)|(candidate>=1)):raise ValueError('Invalid normalized actual/candidate points')
    if indices.shape!=(16,) or flags.shape!=(16,) or not np.issubdtype(indices.dtype,np.integer) or np.any((indices<0)|(indices>=1152)):raise ValueError('Invalid anchors/reservation flags')
    if not np.array_equal(indices,np.asarray(trace['indices'])) or not np.array_equal(np.flatnonzero(flags),trace['reserved_steps']):raise ValueError('Conflicting aliases/reserved step list')
    ids=np.arange(1152);standard=np.stack([((ids%24)+.5)/24,((ids//24)+.5)/48],1).astype(np.float32)
    offsets=np.asarray(trace['offset_predicted'],dtype=np.float32);strength=float(trace['offset_strength'])
    if offsets.shape!=(1152,2) or not np.isfinite(offsets).all() or np.any(np.abs(offsets)>1.000001) or not 0<=strength<=1:raise ValueError('Malformed bounded offset table')
    predicted=standard if strength==0 else np.minimum(np.maximum(standard+strength*crop_size*offsets/np.asarray(trace['image_size'],dtype=np.float32),0),1-1/np.asarray(trace['image_size'],dtype=np.float32))
    if not np.allclose(candidate,predicted,rtol=0,atol=1e-7):raise ValueError('Candidate offset reconstruction mismatch')
    if len(trace['steps'])!=16 or not np.array_equal(flags,[s['reserved_proposal'] for s in trace['steps']]) or not np.array_equal(indices,[s['source_anchor_index'] for s in trace['steps']]):raise ValueError('Step reservation/anchor aliases mismatch')
    expected=np.where(flags[:,None],candidate[indices],standard[indices])
    if not np.allclose(points,expected,rtol=0,atol=1e-7):raise ValueError('Reserved/free coordinate mapping mismatch')
    rects=rectangles(points,trace['image_size'],crop_size)
    if not np.array_equal(rects,np.asarray(trace['actual_crop_rectangles'])):raise ValueError('Actual crop rounding mismatch')
    validate_score_recipe(trace,candidate)
    return points,candidate,indices,flags,rects


def actual_geometry(rects,mask,size,crop_size,integral=None):
    w,h=size;lo=np.maximum(rects[:,:2],[0,0]).astype(int);hi=np.minimum(rects[:,2:],[w,h]).astype(int);areas=np.maximum(0,hi-lo).prod(1)/crop_size**2
    if mask is None:return None,float(areas.mean())
    if integral is None:integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
    x0,y0=lo.T;x1,y1=hi.T;fg=(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])/crop_size**2
    return float(fg.mean()),float(areas.mean())


def evaluate(a):
    val=pd.read_csv(ROOT/'manifests/trajectory.csv');val=val[val.split=='val'].set_index('image_id');records=[];seen=set()
    with Path(a.traces).open() as source:
        for line in source:
            trace=json.loads(line);image_id=int(trace['image_id'])
            if image_id not in val.index:raise ValueError('Continuous offset evaluation refuses non-prior93 image')
            key=(image_id,trace['policy'],trace['seed'],trace.get('strategy','sample'))
            if key in seen:raise ValueError('Duplicate trace')
            seen.add(key);validate_trace(trace,a.crop_size);records.append(trace)
    annotations=json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations'];boxes={int(i):[] for i in val.index}
    for annotation in annotations:
        if int(annotation['image_id']) in boxes:boxes[int(annotation['image_id'])].append(annotation['bbox'])
    needed={val.loc[r['image_id']].file_name for r in records};gazes={}
    with (ROOT/'data/gaze_data/merged_gaze_data.csv').open() as source:
        for line in source:
            name,payload=line.split(';',1)
            if name in needed:gazes[name]=np.asarray([np.asarray(s[0],dtype=float)/float(s[1]) for s in ast.literal_eval(payload)])
    images=[];lesions=[]
    for image_id in sorted(set(r['image_id'] for r in records)):
        row=val.loc[image_id]
        with Image.open(ROOT/'data/image'/row.file_name) as source:image=source.convert('L')
        try:mask=breast_mask(image);mask=mask if mask.any() else None
        except Exception:mask=None
        integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1) if mask is not None else None
        truth=max(reduce_gaze(gazes[row.file_name],*image.size),key=len)/image.size
        for trace in [r for r in records if r['image_id']==image_id]:
            points,candidate,indices,flags,rects=validate_trace(trace,a.crop_size)
            if trace['image_size']!=list(image.size):raise ValueError('Image size mismatch')
            candidate_rects=rectangles(candidate,image.size,a.crop_size);ranked=distinct_proposals(trace['relevance'],candidate_rects)
            fg,inbounds=actual_geometry(rects,mask,image.size,a.crop_size,integral);previous=[iou(rects[i],rects[j]) for i in range(1,16) for j in range(i)];previous_max=[max(iou(rects[i],rects[j]) for j in range(i)) for i in range(1,16)]
            base=dict(image_id=int(image_id),patient_key=row.patient_key,policy=trace['policy'],seed=int(trace['seed']),known_annotation=int(bool(boxes[image_id])),lesion_count=len(boxes[image_id]),offset_strength=trace['offset_strength'])
            item=dict(base,mean_foreground=fg,mean_inbounds=inbounds,normalized_dtw=dtw_distance(truth,points),unique_locations=len(np.unique(np.round(points*np.array(image.size)),axis=0)),mean_max_previous_crop_iou=float(np.mean(previous_max)),mean_allpairs_crop_iou=float(np.mean(previous)),actual_roi_budget=int(flags.sum()))
            per_lesion=[]
            for lesion_index,box in enumerate(boxes[image_id]):
                grid_ids=np.arange(1152);source_coords=np.stack([((grid_ids%24)+.5)/24,((grid_ids//24)+.5)/48],1).astype(np.float32);source_rects=rectangles(source_coords,image.size,a.crop_size)
                shifted_metrics=lesion_metrics(rects,candidate_rects,ranked,box)
                action_union=np.concatenate([candidate_rects,source_rects]);union_metrics=lesion_metrics(rects,action_union,ranked,box)
                lesion=dict(base,lesion_index=lesion_index,shifted_proposal_oracle=shifted_metrics['max_achievable_coverage'],**union_metrics)
                for metric,value in shifted_metrics.items():
                    if metric.startswith('proposal_'):lesion['shifted_'+metric]=value
                lesions.append(lesion);per_lesion.append(lesion)
            if per_lesion:
                for name in per_lesion[0]:
                    if name in base or name=='lesion_index':continue
                    values=[r[name] for r in per_lesion if r[name] is not None];item[name]=float(np.mean(values)) if values else None
            images.append(item)
    frame=pd.DataFrame(images)
    if a.baseline not in set(frame.policy):raise ValueError('Baseline policy missing')
    baseline_seeds=frame[frame.policy==a.baseline].groupby('image_id').seed.apply(set)
    for (image_id,policy),group in frame.groupby(['image_id','policy']):
        if image_id not in baseline_seeds.index or set(group.seed)!=baseline_seeds.loc[image_id]:raise ValueError('Unmatched image/seed baseline comparison')
    lesion_frame=pd.DataFrame(lesions);metrics_columns=['actual_recall_05_step16','actual_recall_08_step16','actual_box_coverage','recall_05_step16','mean_foreground','mean_inbounds','normalized_dtw','unique_locations','mean_max_previous_crop_iou']
    summaries={policy:group.groupby('image_id')[metrics_columns].mean().mean().to_dict() for policy,group in frame.groupby('policy')};comparisons={policy:{metric:paired_bootstrap(frame,metric,a.baseline,policy,a.bootstrap) for metric in metrics_columns} for policy in frame.policy.unique() if policy!=a.baseline}
    result=dict(protocol='Explicit continuous image-only ROI recentering,224px physical crops,16patch budget. Reserved actions use shifted candidate table; free actions use original coarse-grid anchors. Annotation boxes/gaze are read posthoc only. Foreground/inbounds derive ACTUAL crop rectangles. Path normalized recall uses the union of shifted ROI candidates and original-grid free actions; shifted proposal oracle and its own normalized ranking recall are reported separately. Original grid upper bound is not a bound on this continuous protocol. Human teacher NLL is separate; exploratory prior val93, not independent test.',baseline=a.baseline,crop_size=a.crop_size,summaries=summaries,paired_patient_bootstrap=comparisons,denominators=dict(evaluated_images=int(frame.image_id.nunique()),known_images=int(frame[frame.known_annotation==1].image_id.nunique()),known_lesions=len(lesion_frame[['image_id','lesion_index']].drop_duplicates())),trace_completeness={policy:dict(images=int(group.image_id.nunique()),seeds=sorted(group.seed.unique().tolist()),paths=len(group),complete_prior93=group.image_id.nunique()==93) for policy,group in frame.groupby('policy')})
    def clean(value):
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,float) and not np.isfinite(value):return None
        return value
    result=clean(result);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);frame.to_csv(output/'per_image.csv',index=False);lesion_frame.to_csv(output/'per_lesion.csv',index=False);(output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False));lines=['# Continuous ROI offset posthoc','',result['protocol'],'','| Policy | Actual box50% recall | Actual coverage | Foreground proxy | DTW |','|---|---:|---:|---:|---:|']
    for policy,s in summaries.items():lines.append('| '+policy+' | '+' | '.join(f'{s[k]:.4f}' for k in ('actual_recall_05_step16','actual_box_coverage','mean_foreground','normalized_dtw'))+' |')
    (output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result['summaries'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--traces',required=True);p.add_argument('--output',required=True);p.add_argument('--baseline',required=True);p.add_argument('--crop-size',type=int,choices=(224,),default=224);p.add_argument('--bootstrap',type=int,default=2000);evaluate(p.parse_args())
