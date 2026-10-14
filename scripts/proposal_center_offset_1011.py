"""Image-only continuous ROI-center refinement; source gaze grid stays24x48."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.data import whole_tensor,seed_for
from src.guided_trajectory_1011 import candidate_geometry
from src.proposal_center_offset_1011 import load_center_offset_head,predict_center_offsets,center_offset_rollout
from proposal_policy_1011 import image_records
from guided_trajectory_1011 import load_guided,base_path
from train_attention_1011 import digest


def predict(a):
    torch.set_num_threads(a.threads);records=image_records(a);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);(output/'config.json').write_text(json.dumps(vars(a),indent=2));started=time.monotonic();completed=0
    def status(state,error=None):(output/'status.json').write_text(json.dumps(dict(state=state,images=len(records),completed_paths=completed,seconds=time.monotonic()-started,error=error)))
    status('running')
    try:
        model,checkpoint=load_guided(a.checkpoint,a.base_checkpoint_override);model.to(a.device).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        base_hash=digest(base_path(a.checkpoint));head,head_checkpoint=load_center_offset_head(a.offset_checkpoint,base_hash,a.device);head_hash=digest(a.offset_checkpoint);architecture=head_checkpoint['config']['architecture'];relevance_power=head_checkpoint.get('relevance_power',0);architecture_label=architecture+(f'_p{relevance_power}' if relevance_power else '')
        with (output/'traces.jsonl').open('w') as sink:
            for item in records:
                with Image.open(item['image']) as source:image=source.convert('L')
                geometry=candidate_geometry(image,model.grid,224)
                with torch.no_grad():memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device));offsets=predict_center_offsets(head,memory)[0]
                for seed in a.seeds:
                    for nms in (a.nms_thresholds or [a.max_iou]):
                        for strength in a.strengths:
                            for budget in a.budgets:
                                if a.skip_half_nms03 and strength==.5 and nms==.3:continue
                                strength_name=str(strength).replace('.','p');name=f'O_{architecture_label}_s{strength_name}_K{budget}'+(('_nms'+str(nms).replace('.','p')) if a.policy_nms_label or a.nms_thresholds else '')+('_argmax' if a.strategy=='argmax' else '')
                                result=center_offset_rollout(model,memory,image,geometry,offsets,budget,strength,16,seed_for(seed,item['image_id']),a.strategy,nms)
                                trace=dict(schema_version='proposal_center_offset_1011_v1',coordinate_protocol='center_offset_continuous_roi_v1',image_id=item['image_id'],image=str(Path(item['image']).resolve()),image_size=list(image.size),grid=list(model.grid),crop_size=224,policy=name,strategy=a.strategy,seed=seed,sampling_seed=seed_for(seed,item['image_id']),checkpoint=str(base_path(a.checkpoint).resolve()),checkpoint_sha256=base_hash,checkpoint_epoch=checkpoint['epoch'],rho=model.rho,ablation=model.ablation,offset_checkpoint=str(Path(a.offset_checkpoint).resolve()),offset_checkpoint_sha256=head_hash,offset_base_checkpoint_sha256=head_checkpoint['base_checkpoint_sha256'],offset_architecture=architecture,relevance_power=relevance_power,original_proposal_state_sha256=head_checkpoint.get('original_proposal_state_sha256'),offset_strength=strength,max_offset_crops=head_checkpoint['config']['max_offset_crops'],max_crop_iou=nms,source_grid_indices_only=True,proposal_source='original_occupancy',annotation_used=False,annotations_used=False,gaze_used=False,observed_gaze_used=False,labels_used=False,output_role='Continuous ROI-search coordinates; separate human gaze NLL target grid remains24x48',geometry=dict(valid=geometry['geometry_valid'],error=geometry['geometry_error']),gaze_component_used=bool(any(not flag for flag in result['reservation_flags'])),conditional_gaze_steps=sum(not flag for flag in result['reservation_flags']),forced_roi_path_nll_reported=False,**result)
                                sink.write(json.dumps(trace,allow_nan=False)+'\n');sink.flush();summary={k:trace[k] for k in ('image_id','policy','seed','strategy','coordinates','source_anchor_indices','reservation_flags','actual_crop_rectangles','checkpoint_sha256','offset_checkpoint_sha256')};(output/f'{name}_image{item["image_id"]}_seed{seed}.json').write_text(json.dumps(summary));completed+=1
                    status('running');print(json.dumps(dict(image_id=item['image_id'],seed=seed,completed_paths=completed)),flush=True)
        status('complete')
    except Exception as exc:status('failed',str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--base-checkpoint-override');p.add_argument('--offset-checkpoint',required=True);p.add_argument('--manifest',default='manifests/all.csv');p.add_argument('--split',default='val');p.add_argument('--image-ids',nargs='+',type=int);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=4);p.add_argument('--budgets',nargs='+',type=int,default=[8,12,16]);p.add_argument('--strengths',nargs='+',type=float,default=[0,.5,1]);p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--strategy',choices=['sample','argmax'],default='sample');p.add_argument('--nms-thresholds',nargs='+',type=float);p.add_argument('--policy-nms-label',action='store_true');p.add_argument('--skip-half-nms03',action='store_true');p.add_argument('--max-iou',type=float,default=.3);predict(p.parse_args())
