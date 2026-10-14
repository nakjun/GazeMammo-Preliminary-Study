"""Generate physical-crop budget experiments; annotations are posthoc only."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.data import whole_tensor,seed_for
from src.guided_trajectory_1011 import candidate_geometry
from src.proposal_budget_1011 import budget_rollout
from proposal_policy_1011 import image_records
from guided_trajectory_1011 import load_guided,base_path
from train_attention_1011 import digest


def predict(a):
    torch.set_num_threads(a.threads);records=image_records(a);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);(output/'config.json').write_text(json.dumps(vars(a),indent=2));started=time.monotonic();completed=0
    def status(state,error=None):(output/'status.json').write_text(json.dumps(dict(state=state,images=len(records),completed_paths=completed,seconds=time.monotonic()-started,error=error)))
    status('running')
    try:
        model,checkpoint=load_guided(a.checkpoint,a.base_checkpoint_override);model.to(a.device).eval()
        for p in model.parameters():p.requires_grad_(False)
        checkpoint_hash=digest(base_path(a.checkpoint));capture=None;capture_checkpoint=None;capture_hash=None
        if a.capture_checkpoint:
            from src.proposal_capture_1011 import load_capture_head,predict_capture
            capture,capture_checkpoint=load_capture_head(a.capture_checkpoint,checkpoint_hash,a.device);capture_hash=digest(a.capture_checkpoint)
        with (output/'traces.jsonl').open('w') as sink:
            for item in records:
                with Image.open(item['image']) as source:image=source.convert('L')
                geometry=candidate_geometry(image,model.grid,a.crop_size);geometry['crop_size']=a.crop_size
                with torch.no_grad():
                    memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device))
                    capture_scores=predict_capture(capture,memory) if capture is not None else None
                static=dict(valid=geometry['geometry_valid'],error=geometry['geometry_error'])
                for seed in a.seeds:
                    for layout in a.layouts:
                        for budget in a.budgets:
                            if layout=='interleaved' and budget in (0,16) and 'first' in a.layouts:continue
                            source_label=capture_checkpoint['target']+'_'+capture_checkpoint['config']['architecture'] if capture is not None and budget>0 else 'original_occupancy'
                            name=f'B{budget}_{layout}_crop{a.crop_size}'+(('_'+source_label) if capture is not None and budget>0 else '')+('_argmax' if a.strategy=='argmax' else '')
                            result=budget_rollout(model,memory,image,geometry,budget,layout,a.crop_size,16,seed_for(seed,item['image_id']),a.strategy,a.max_iou,proposal_relevance=capture_scores if budget>0 else None)
                            record=dict(schema_version='proposal_budget_1011_v1',image_id=item['image_id'],image=str(Path(item['image']).resolve()),image_size=list(image.size),grid=list(model.grid),crop_size=a.crop_size,policy=name,policy_base='budget',strategy=a.strategy,seed=seed,sampling_seed=seed_for(seed,item['image_id']),checkpoint=str(base_path(a.checkpoint).resolve()),checkpoint_sha256=checkpoint_hash,checkpoint_epoch=checkpoint['epoch'],rho=model.rho,ablation=model.ablation,geometry=static,inputs='image pixels and RNG image ID only',annotation_used=False,annotations_used=False,gaze_used=False,observed_gaze_used=False,labels_used=False,proposal_source=source_label,capture_checkpoint_sha256=capture_hash if budget>0 else None,capture_base_checkpoint_sha256=capture_checkpoint['base_checkpoint_sha256'] if capture_checkpoint is not None and budget>0 else None,heuristic=budget>0,output_role='ROI search path; gaze likelihood is evaluated separately',max_crop_iou=a.max_iou,**result)
                            sink.write(json.dumps(record,allow_nan=False)+'\n');sink.flush()
                            summary={k:record[k] for k in ('image_id','policy','strategy','seed','coordinates','indices','checkpoint_sha256','crop_size','reserved_steps','reserved_candidate_indices')}
                            (output/f'{name}_image{item["image_id"]}_seed{seed}.json').write_text(json.dumps(summary))
                            completed+=1
                    status('running');print(json.dumps(dict(image_id=item['image_id'],seed=seed,completed_paths=completed)),flush=True)
        status('complete')
    except Exception as exc:status('failed',str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--base-checkpoint-override');p.add_argument('--capture-checkpoint');p.add_argument('--manifest',default='manifests/all.csv');p.add_argument('--split',default='val');p.add_argument('--image-ids',nargs='+',type=int);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=4);p.add_argument('--budgets',nargs='+',type=int,default=[1,2,3,4,6,8]);p.add_argument('--layouts',nargs='+',choices=['first','interleaved'],default=['first','interleaved']);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);p.add_argument('--strategy',choices=['sample','argmax'],default='sample');p.add_argument('--crop-size',type=int,choices=[224,320],default=224);p.add_argument('--max-iou',type=float,default=.3);predict(p.parse_args())
