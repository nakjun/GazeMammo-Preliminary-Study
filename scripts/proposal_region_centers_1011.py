"""Image-only region-center generation and separately invoked posthoc candidate preview."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.proposal_region_centers_1011 import region_candidate_indices
from src.proposal_budget_1011 import budget_rollout
from src.data import whole_tensor,seed_for
from src.guided_trajectory_1011 import candidate_geometry
from proposal_policy_1011 import image_records
from guided_trajectory_1011 import load_guided,base_path
from train_attention_1011 import digest
from evaluate_proposal_policy_1011 import rectangles,coverage


def candidates(scores,size,mode,threshold,budget=16,max_iou=.3):
    ids=np.arange(1152);coords=np.stack([((ids%24)+.5)/24,((ids//24)+.5)/48],axis=1)
    return region_candidate_indices(scores,(24,48),mode,threshold,7,rectangles(coords,size,224),budget,max_iou,True)


def predict(a):
    torch.set_num_threads(a.threads);records=image_records(a);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);(output/'config.json').write_text(json.dumps(vars(a),indent=2));started=time.monotonic();completed=0
    def status(state,error=None):(output/'status.json').write_text(json.dumps(dict(state=state,images=len(records),completed_paths=completed,seconds=time.monotonic()-started,error=error)))
    status('running')
    try:
        model,checkpoint=load_guided(a.checkpoint,a.base_checkpoint_override);model.to(a.device).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        checkpoint_hash=digest(base_path(a.checkpoint))
        with (output/'traces.jsonl').open('w') as sink:
            for item in records:
                with Image.open(item['image']) as source:image=source.convert('L')
                geometry=candidate_geometry(image,model.grid,224);geometry['crop_size']=224
                with torch.no_grad():
                    memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device));scores=model.proposal(memory['keys']).squeeze(-1).sigmoid()[0].cpu().numpy()
                for mode in a.modes:
                    for threshold in a.thresholds:
                        if mode=='peak' and threshold!=a.thresholds[0]:continue
                        ranked,regions=candidates(scores,image.size,mode,threshold,16,a.max_iou)
                        for seed in a.seeds:
                            for budget in a.budgets:
                                name=f'RC_{mode}_t{threshold:g}_K{budget}_first_crop224'+('_argmax' if a.strategy=='argmax' else '')
                                result=budget_rollout(model,memory,image,geometry,budget,'first',224,16,seed_for(seed,item['image_id']),a.strategy,a.max_iou,candidate_indices=ranked)
                                record=dict(schema_version='proposal_region_centers_1011_v1',image_id=item['image_id'],image=str(Path(item['image']).resolve()),image_size=list(image.size),grid=[24,48],crop_size=224,policy=name,strategy=a.strategy,seed=seed,sampling_seed=seed_for(seed,item['image_id']),checkpoint=str(base_path(a.checkpoint).resolve()),checkpoint_sha256=checkpoint_hash,checkpoint_epoch=checkpoint['epoch'],rho=model.rho,ablation=model.ablation,inputs='Image pixels, original proposal R and supplied RNG only; snapped image-only region centers',annotations_used=False,annotation_used=False,observed_gaze_used=False,gaze_used=False,labels_used=False,proposal_source='original_occupancy',heuristic=True,output_role='ROI search path; human gaze likelihood evaluated separately',region_mode=mode,region_relative_threshold=threshold,region_window=7,snap_tie_rule='Nearest coarse-grid center; half-grid ties choose lower coordinate',ranked_regions=regions,max_crop_iou=a.max_iou,**result)
                                sink.write(json.dumps(record,allow_nan=False)+'\n');sink.flush();summary={k:record[k] for k in ('image_id','policy','seed','coordinates','indices','crop_size','reserved_candidate_indices','checkpoint_sha256')};(output/f'{name}_image{item["image_id"]}_seed{seed}.json').write_text(json.dumps(summary));completed+=1
                status('running');print(json.dumps(dict(image_id=item['image_id'],completed_paths=completed)),flush=True)
        status('complete')
    except Exception as error:status('failed',str(error));raise


def preview(a):
    # Read only original image-derived relevance; annotations are not loaded here.
    import csv
    with (ROOT/'manifests/trajectory.csv').open() as source:allowed={int(row['image_id']) for row in csv.DictReader(source) if row['split']=='val'}
    seen=set();records=[];checkpoint_hashes=set()
    with Path(a.traces).open() as source:
        for line in source:
            trace=json.loads(line);image_id=int(trace['image_id'])
            if image_id not in allowed or image_id in seen or trace.get('policy') not in ('P0','B0_first_crop224') or int(trace['seed'])!=42:continue
            if trace.get('annotations_used',trace.get('annotation_used')) is not False:raise ValueError('Original trace generation used annotations')
            seen.add(image_id);checkpoint_hashes.add(trace.get('checkpoint_sha256'))
            for mode in ('peak','centroid','bbox_center'):
                for threshold in ((.25,) if mode=='peak' else (.25,.5)):
                    indices,regions=candidates(trace['relevance'],trace['image_size'],mode,threshold,16,a.max_iou)
                    records.append(dict(image_id=image_id,image_size=trace['image_size'],mode=mode,threshold=threshold,indices=indices,regions=regions))
    if seen!=allowed:raise ValueError('Candidate preview requires all fixed93 image-only scores')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);(output/'candidate_regions.json').write_text(json.dumps(dict(source_traces=str(Path(a.traces).resolve()),source_checkpoint_sha256=sorted(checkpoint_hashes),protocol='Image-only candidate recentering; no observed gaze/boxes/labels loaded; fixed thresholds and window; not actual generated path',max_iou=a.max_iou,records=records),indent=2))


def preview_evaluate(a):
    obj=json.loads(Path(a.preview).read_text());annotations=json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations'];boxes={}
    for annotation in annotations:boxes.setdefault(int(annotation['image_id']),[]).append(annotation['bbox'])
    results={};cases={}
    for record in obj['records']:
        image_id=record['image_id'];ids=np.asarray(record['indices'],dtype=int);coords=np.stack([((ids%24)+.5)/24,((ids//24)+.5)/48],axis=1).astype(np.float32);rects=rectangles(coords,record['image_size'],224)
        if not boxes.get(image_id):continue
        for budget in (8,12,16):
            name=f"{record['mode']}_t{record['threshold']:g}_K{budget}";hits=[];actual=[]
            for box in boxes[image_id]:
                values=coverage(rects[:budget],box);best=float(values.max()) if len(values) else 0.;hits.append(best>=.5);actual.append(best)
            result=dict(image_id=image_id,known_lesions=len(hits),actual50=float(np.mean(hits)),actual_coverage=float(np.mean(actual)),candidates=min(budget,len(ids)))
            results.setdefault(name,[]).append(result)
            if image_id==73:cases[name]=dict(result,selected_indices=ids[:budget].tolist())
    summary=dict(protocol='Posthoc GT-only ranking preview over fixed93; coverage of reserved candidates only, not16-step generated trajectories. Extra nonreserved model samples are absent. No settings are chosen automatically.',max_iou=obj['max_iou'],image_macro={name:dict(actual50=float(np.mean([r['actual50'] for r in group])),actual_coverage=float(np.mean([r['actual_coverage'] for r in group])),known_images=len(group),known_lesions=sum(r['known_lesions'] for r in group),mean_candidates=float(np.mean([r['candidates'] for r in group]))) for name,group in results.items()},case73=cases)
    Path(a.output).write_text(json.dumps(summary,indent=2));print(json.dumps(summary['image_macro'],indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('predict');p.add_argument('--checkpoint',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--base-checkpoint-override');p.add_argument('--manifest',default='manifests/all.csv');p.add_argument('--split',default='val');p.add_argument('--image-ids',nargs='+',type=int);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=4);p.add_argument('--modes',nargs='+',choices=('peak','centroid','bbox_center'),default=['peak','centroid','bbox_center']);p.add_argument('--thresholds',nargs='+',type=float,choices=(.25,.5),default=[.25,.5]);p.add_argument('--budgets',nargs='+',type=int,choices=(8,12,16),default=[8,12,16]);p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--strategy',choices=('sample','argmax'),default='sample');p.add_argument('--max-iou',type=float,default=.3)
    p=sub.add_parser('preview');p.add_argument('--traces',required=True);p.add_argument('--output',required=True);p.add_argument('--max-iou',type=float,default=.3)
    p=sub.add_parser('preview-evaluate');p.add_argument('--preview',required=True);p.add_argument('--output',required=True)
    a=parser.parse_args();{'predict':predict,'preview':preview,'preview-evaluate':preview_evaluate}[a.command](a)
