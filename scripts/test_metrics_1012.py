"""One frozen CBIS test posthoc: unknown annotations excluded, proposal AP class-agnostic."""
import argparse,json,re,sys,hashlib,importlib.util,contextlib,io
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_proposal_center_offset_1011 import validate_trace
from src.metrics import dtw_distance

def selected_annotations(path,allowed):
    # Read bytes/text but decode annotation objects ONLY after checking allowed image ID.
    text=Path(path).read_text();start=re.search(r'"annotations"\s*:\s*\[',text).end();depth=0;quoted=False;escape=False;begin=None;result=[]
    for i in range(start,len(text)):
        c=text[i]
        if quoted:
            if escape:escape=False
            elif c=='\\':escape=True
            elif c=='"':quoted=False
            continue
        if c=='"':quoted=True
        elif c=='{':
            if depth==0:begin=i
            depth+=1
        elif c=='}':
            depth-=1
            if depth==0:
                raw=text[begin:i+1];match=re.search(r'"image_id"\s*:\s*(\d+)',raw)
                if match and int(match.group(1)) in allowed:result.append(json.loads(raw))
        elif c==']' and depth==0:break
    return result

def overlap_fraction(rects,box):
    x,y,w,h=box;b=np.array([x,y,x+w,y+h]);inter=np.maximum(np.minimum(rects[:,2:],b[2:])-np.maximum(rects[:,:2],b[:2]),0).prod(1);return inter/(w*h)

def clip_rectangles(rects,size):
    r=np.asarray(rects,float).copy();r[:,[0,2]]=r[:,[0,2]].clip(0,size[0]);r[:,[1,3]]=r[:,[1,3]].clip(0,size[1]);return r

def coverage_rows(trace,annotations):
    rects=np.asarray(trace['actual_crop_rectangles'],float);rows=[]
    for j,annotation in enumerate(annotations):
        box=annotation['bbox'];h=overlap_fraction(rects,box);row=dict(image_id=int(trace['image_id']),seed=int(trace['seed']),lesion_index=j,annotation_id=annotation['id'],H=float(h.max()))
        for threshold in [.5,.8]:
            name='50' if threshold==.5 else '80';hit=np.flatnonzero(h>=threshold);row[f'first_hit_{name}']=int(hit[0]+1) if len(hit) else None;row[f'censored_{name}']=not len(hit)
            for k in [1,4,8,16]:row[f'physical_recall{name}_step{k}']=float(h[:k].max()>=threshold)
        rows.append(row)
    return rows

def confidence(trace):
    # Frozen image-only rule; no annotation-dependent scoring or thresholds.
    return np.asarray(trace['relevance'],float)[np.asarray(trace['source_anchor_indices'],int)]

def coco_entries(traces,annotations,images):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    with contextlib.redirect_stdout(io.StringIO()):
        gt=COCO();gt.dataset=dict(info={},images=images,annotations=[{**a,'area':a['bbox'][2]*a['bbox'][3],'iscrowd':int(a.get('iscrowd',0))} for a in annotations],categories=[dict(id=i,name=str(i)) for i in sorted({a['category_id'] for a in annotations})]);gt.createIndex();pred=[]
        for t in traces:
            clipped=clip_rectangles(t['actual_crop_rectangles'],t['image_size'])
            for rect,score in zip(clipped,confidence(t)):
                x,y,x2,y2=rect
                if x2>x and y2>y:pred.append(dict(image_id=t['image_id'],category_id=gt.dataset['categories'][0]['id'],bbox=[x,y,x2-x,y2-y],score=float(score)))
        if not pred:return {i['id']:dict(gtIgnore=np.array([bool(a.get('iscrowd',0)) for a in annotations if a['image_id']==i['id']]),dtScores=[],dtMatches=np.zeros((10,0)),dtIgnore=np.zeros((10,0),bool)) for i in images}
        dt=gt.loadRes(pred);ev=COCOeval(gt,dt,'bbox');ev.params.useCats=0;ev.params.imgIds=[i['id'] for i in images];ev.params.maxDets=[1,10,100];ev.evaluate()
        entries={e['image_id']:e for e in ev.evalImgs if e and np.array_equal(e['aRng'],ev.params.areaRng[0])}
    return entries

def detection_metrics(entries,ids=None):
    selected=[entries[i] for i in (ids if ids is not None else sorted(entries)) if entries[i] is not None];ngt=sum(int((np.asarray(e['gtIgnore'])==0).sum()) for e in selected);output={}
    for k in [1,10,16,100]:
        scores=np.concatenate([np.asarray(e['dtScores'][:k]) for e in selected]) if selected else np.array([]);order=np.argsort(-scores,kind='mergesort');tp=np.concatenate([(e['dtMatches'][:,:k]>0)&~e['dtIgnore'][:,:k] for e in selected],1)[:,order] if selected else np.zeros((10,0),bool);fp=np.concatenate([(e['dtMatches'][:,:k]==0)&~e['dtIgnore'][:,:k] for e in selected],1)[:,order] if selected else np.zeros((10,0),bool);aps=[];recalls=[]
        for t in range(10):
            hits=np.cumsum(tp[t]);errors=np.cumsum(fp[t]);recall=hits/ngt if ngt else np.zeros_like(hits,float);precision=hits/(hits+errors+np.spacing(1));precision=np.maximum.accumulate(precision[::-1])[::-1];positions=np.searchsorted(recall,np.linspace(0,1,101),'left');samples=np.zeros(101);valid=positions<len(precision);samples[valid]=precision[positions[valid]];aps.append(float(samples.mean()));recalls.append(float(recall[-1]) if len(recall) else 0.)
        output[f'proposal_AR{k}']=float(np.mean(recalls)) if ngt else None
        if k==100:output.update(proposal_AP=float(np.mean(aps)),proposal_AP50=aps[0],proposal_AP75=aps[5])
    return output

def gaze_metrics(record,prior):
    q=np.asarray(record['raw_logp'],float);coords=np.asarray(record['coords'],float);mask=np.asarray(record['loss_mask'],bool);target=np.asarray(record['target'],int);expected=np.minimum((coords[:,1]*48).astype(int),47)*24+np.minimum((coords[:,0]*24).astype(int),23)
    if q.shape!=(16,1152) or mask.shape!=(16,) or not mask.any() or not np.array_equal(target,expected) or not np.isfinite(q).all():raise ValueError('Malformed fixed teacher record')
    q=q-np.logaddexp.reduce(q,axis=-1)[:,None];p=np.exp(q);nss=[];auc=[]
    for t in np.flatnonzero(mask):
        values=p[t];value=values[target[t]];auc.append((np.sum(values<value)+.5*(np.sum(values==value)-1))/1151);nss.append((value-values.mean())/values.std() if values.std()>1e-12 else 0.)
    return dict(nll=float(-q[np.arange(16)[mask],target[mask]].mean()),LL_uniform_bits=float((np.log(1152)+q[np.arange(16)[mask],target[mask]].mean())/np.log(2)),conditional_AUC=float(np.mean(auc)),conditional_NSS=float(np.mean(nss)),train_prior_IG_bits=float(((q[np.arange(16)[mask],target[mask]]-np.log(prior[target[mask]]))/np.log(2)).mean()),valid_gaze_steps=int(mask.sum()))

def load_local(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def path_metrics(truth,pred,size):
    result=dict(normalized_DTW=float(dtw_distance(truth,pred)));scan=load_local('frozen_scanmatch_1012',ROOT/'literature/code/HAT/common/scanmatch.py');s=scan.ScanMatch(Xres=size[0],Yres=size[1],Xbin=24,Ybin=48,Threshold=3.5,TempBin=0);a=s.fixationToSequence(truth*np.array(size)).astype(np.int32);b=s.fixationToSequence(pred*np.array(size)).astype(np.int32);result['ScanMatch_no_duration']=float(s.match(a,b)[0]);mm=load_local('frozen_multimatch_1012',ROOT/'literature/code/HAT/common/multimatch.py')
    if len(truth)>=3:
        # NaN marks unknown duration. Only official spatial functions are called;
        # no simplification/grouping/duration score or fabricated time.
        a=mm.gen_scanpath_structure(np.c_[truth*np.array(size),np.full(len(truth),np.nan)]);b=mm.gen_scanpath_structure(np.c_[pred*np.array(size),np.full(len(pred),np.nan)]);matrix=mm.cal_vectordifferences(a,b);assignment=np.arange(matrix.size).reshape(matrix.shape);graph=mm.createdirectedgraph(matrix.shape,matrix,assignment);path,_=mm.dijkstra(graph,0,matrix.size-1);args=(a,b,path,assignment);raw=[np.median(fn(*args)) for fn in [mm.cal_vectordifferencealongpath,mm.cal_angulardifference,mm.cal_lengthdifference,mm.cal_positiondifference]];normalized=mm.normaliseresults(raw+[np.nan],sz=size)[:4]
        for name,value in zip(['vector','direction','length','position'],normalized):result['MultiMatch_'+name]=float(value)
    return result

def bootstrap_rows(frame,columns,repetitions=10000):
    groups=frame.groupby('patient_key');patients=list(groups.groups);counts=np.array([len(groups.get_group(p)) for p in patients]);draw=np.random.default_rng(42).multinomial(len(patients),np.full(len(patients),1/len(patients)),repetitions);out={}
    for col in columns:
        totals=np.array([groups.get_group(p)[col].sum() for p in patients]);validcounts=np.array([groups.get_group(p)[col].notna().sum() for p in patients]);values=(draw@totals)/(draw@validcounts);out[col]=dict(low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)))
    return out

def run(a):
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);allframe=pd.read_csv(ROOT/'manifests/all.csv');test=allframe[allframe.split=='test'].set_index('image_id');trajectory=pd.read_csv(ROOT/'manifests/trajectory.csv');annotated=set(trajectory[trajectory.split=='test'].image_id.astype(int));assert len(test)==150 and len(annotated)==131 and test.source.eq('CBIS-DDSM').all();assert not set(test.patient_key)&set(allframe[allframe.split!='test'].patient_key);annotations=selected_annotations(ROOT/'manifests/instances_clipped.json',annotated);boxes={i:[] for i in annotated}
    for ann in annotations:boxes[int(ann['image_id'])].append(ann)
    assert all(boxes.values());raw=torch.load(a.teacher,map_location='cpu',weights_only=False);teacher={int(r['image_id']):r for r in raw['records']};assert len(teacher)==len(raw['records']) and set(teacher)==set(test.index);prior_path=Path(a.train_raw);train_payload=torch.load(prior_path,map_location='cpu',weights_only=False);train_records=[r for r in train_payload['records'] if r['split']=='train'];allowed_train=allframe[allframe.split=='train'].set_index('image_id');assert len(train_records)==412 and {int(r['image_id']) for r in train_records}==set(allowed_train.index);counts=np.ones(1152,dtype=float)
    for r in train_records:
        assert r['patient_key']==allowed_train.loc[int(r['image_id'])].patient_key;np.add.at(counts,np.asarray(r['target'])[np.asarray(r['loss_mask'],bool)],1)
    prior=counts/counts.sum();(out/'train_prior.json').write_text(json.dumps(dict(recipe='Dirichlet1 marginalcanonicaltrain412hist fixedbeforeTESTmetrics',probabilities=prior.tolist(),source_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest()),indent=2));traces=[];seen=set();images=[];lesions=[]
    with Path(a.traces).open() as stream:
        for line in stream:
            t=json.loads(line);i=int(t['image_id']);key=(i,int(t['seed']));assert i in test.index and key not in seen;seen.add(key);validate_trace(t,224);assert t['evaluation_split']=='test';traces.append(t);row=dict(image_id=i,patient_key=test.loc[i].patient_key,seed=t['seed'],known_annotation=int(i in annotated))
            if i in teacher:
                r=teacher[i];assert r['patient_key']==row['patient_key'] and r['split']=='test';row.update(gaze_metrics(r,prior));truth=np.asarray(r['coords'])[np.asarray(r['loss_mask'],bool)];row.update(path_metrics(truth,np.asarray(t['coordinates']),t['image_size']))
            rows=coverage_rows(t,boxes.get(i,[]));lesions.extend([{**r,'patient_key':row['patient_key']} for r in rows])
            if rows:
                for col in ['H']+[f'physical_recall{x}_step{k}' for x in [50,80] for k in [1,4,8,16]]:row[col]=float(np.mean([r[col] for r in rows]))
            images.append(row)
    seeds=sorted({s for _,s in seen});assert seeds==[42,43,44] and seen=={(i,s) for i in test.index for s in seeds};frame=pd.DataFrame(images);lesion=pd.DataFrame(lesions);numeric=[c for c in frame.columns if c not in ['image_id','patient_key','seed']];macro=frame.groupby(['image_id','patient_key'],as_index=False)[numeric].mean();det={};entries={};cocoimages=[dict(id=int(i),width=int(test.loc[i].width),height=int(test.loc[i].height)) for i in sorted(annotated)]
    for seed in seeds:entries[seed]=coco_entries([t for t in traces if t['seed']==seed and t['image_id'] in annotated],annotations,cocoimages);det[seed]=detection_metrics(entries[seed])
    detection={k:float(np.mean([det[s][k] for s in seeds])) for k in det[seeds[0]]};columns=['nll','conditional_AUC','conditional_NSS','train_prior_IG_bits','normalized_DTW','ScanMatch_no_duration','MultiMatch_vector','MultiMatch_direction','MultiMatch_length','MultiMatch_position','LL_uniform_bits','H']+[f'physical_recall{x}_step{k}' for x in [50,80] for k in [1,4,8,16]];ci=bootstrap_rows(macro,columns);patients=sorted(set(test.loc[list(annotated)].patient_key));by={p:sorted(i for i in annotated if test.loc[i].patient_key==p) for p in patients};rng=np.random.default_rng(42);ap_samples={k:[] for k in detection}
    for _ in range(a.ap_bootstrap):
        ids=[i for p in rng.choice(patients,len(patients),replace=True) for i in by[p]];scores=[detection_metrics(entries[s],ids) for s in seeds]
        for key in detection:ap_samples[key].append(np.mean([r[key] for r in scores]))
    for key,values in ap_samples.items():ci[key]=dict(low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)))
    metrics=dict(protocol='ONE frozen CBIS-DDSM TEST; goal remains paused; no training/test fits/threshold selection/INbreast payload parsing',denominators=dict(test_images=150,gaze_images=len(teacher),annotated_images=131,unannotated_unknown_images=19,lesions=len(annotations),paths=len(traces),seeds=seeds,patient_clusters=int(test.patient_key.nunique()),valid_gaze_steps=sum(int(np.asarray(r['loss_mask']).sum()) for r in teacher.values())),image_macro={c:float(macro[c].mean()) for c in columns},lesion_micro={c:float(lesion[c].mean()) for c in ['H']+[f'physical_recall{x}_step{k}' for x in [50,80] for k in [1,4,8,16]]},proposal_detection=detection,proposal_detection_per_seed=det,patient_bootstrap_ci95=ci,definitions=dict(H='Max single actual224 fullcrop intersection/GTboxarea; annotation boxes unknown for19 excluded',detection='CLASS-AGNOSTIC proposalAP COCOIoU.50:.05:.95,101recallpoints,standardmaxDets[1,10,100] withONLY16actualdetections; AR1/10/100 andAR16=AR100; no category detector/SOTA claim',confidence='OriginalguidedR[source_anchor_index] for every reserved/free actualcrop; fixed beforetestGT',boxes='PhysicalH fullrect; IoU/AP crops clippedtoimagebounds, GTclippedoriginalannotations; zero-paddingoutsideimage excludedfromIoU area',gaze='ConditionalonepositivecellAUC/1151negativecells; NSSpopulationstd; IGbits vs fixed Dirichlet1 canonicaltrain412 marginal; image-macrovalidlegacy16; unit-square Euclidean DTW/(truthN+predN); not physical-diagonal DTW',paths='OfficialHATvendoredScanMatch24x48Threshold3.5TempBin0; MultiMatch4spatialdimensions,no grouping,no duration score, unknown durationsNaN',bootstrap=f'Patientclusterseed42; generationseeds averagedwithinimage first; 10000primary; {a.ap_bootstrap}AP/ARresamplescachedCOCOentries forcompute'),source_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [Path(a.traces),Path(a.teacher),prior_path,ROOT/'manifests/all.csv',ROOT/'manifests/trajectory.csv',ROOT/'manifests/instances_clipped.json']})
    metrics['first_hit']={str(threshold):dict(hit_fraction=float(lesion[f'first_hit_{threshold}'].notna().mean()),mean_step_among_hits=float(lesion[f'first_hit_{threshold}'].dropna().mean()),censored_fraction=float(lesion[f'first_hit_{threshold}'].isna().mean()),denominator='known lesions x three generation seeds; no-hit censored at16, never assigned artificial hit') for threshold in [50,80]}
    frame.to_csv(out/'per_image.csv',index=False);lesion.to_csv(out/'per_lesion.csv',index=False);macro.to_csv(out/'image_seed_averages.csv',index=False);(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    lines=['# Frozen CBIS-DDSM test evaluation','',metrics['protocol'],'','150 gaze images; 131 annotated images; 19 images with unknown annotation status excluded from lesion and detection scores. Each image has three 16-crop paths.','', '| Gaze metric (image macro) | Estimate | Patient 95% CI |','|---|---:|---:|']
    for key in columns[:11]:
        interval=ci[key];lines.append(f"| {key} | {metrics['image_macro'][key]:.6f} | [{interval['low']:.6f}, {interval['high']:.6f}] |")
    lines+=['','| Budget | Physical box50% recall (image macro) | Physical box80% recall (image macro) | Box50% recall (lesion micro) |','|---|---:|---:|---:|']
    for k in [1,4,8,16]:lines.append(f"| {k} | {metrics['image_macro'][f'physical_recall50_step{k}']:.6f} | {metrics['image_macro'][f'physical_recall80_step{k}']:.6f} | {metrics['lesion_micro'][f'physical_recall50_step{k}']:.6f} |")
    lines+=['','| Class-agnostic proposal metric | Estimate | Patient 95% CI |','|---|---:|---:|']
    for key,value in detection.items():lines.append(f"| {key} | {value:.6f} | [{ci[key]['low']:.6f}, {ci[key]['high']:.6f}] |")
    lines+=['','Physical recall measures maximum single-crop intersection divided by box area, not normalized attainable coverage. Proposal AP uses clipped actual crop boxes and frozen original-guided anchor relevance; it is not a malignancy classifier or a class-aware lesion detector. Standard COCO maxDets=100 has only 16 predictions, so AR100 equals AR16.','','NLL, conditional AUC and NSS use the same 24x48 target grid. Information gain uses a fixed Dirichlet-1 marginal from canonical TRAIN412. DTW uses repository unit-square Euclidean distance divided by the sum of path lengths, not physical image-diagonal distance. ScanMatch has no time bins; MultiMatch reports four spatial dimensions without grouping or duration.','','First-hit and censored fractions, all exact denominators, per-seed AP and source hashes are in metrics.json. Patient clusters are resampled with all their images; seeds are averaged first. Primary intervals use 10,000 draws; nonadditive AP/AR use 2,000 cached-matching draws.','','Primary metric implementations: [COCO evaluation](https://github.com/cocodataset/cocoapi), [ScanMatch](https://github.com/hsogo/gazeparser), [MultiMatch](https://github.com/adswa/multimatch_gaze).']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--traces',required=True);p.add_argument('--teacher',required=True);p.add_argument('--train-raw',default='runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt');p.add_argument('--output',required=True);p.add_argument('--ap-bootstrap',type=int,default=2000);run(p.parse_args())
