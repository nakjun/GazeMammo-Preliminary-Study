"""Frozen image-only capture-aware scoring of fixedcontinuous ROI candidates."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.data import whole_tensor,seed_for
from src.guided_trajectory_1011 import candidate_geometry
from src.proposal_center_offset_1011 import load_center_offset_head,predict_center_offsets,shifted_candidate_coords,center_offset_rollout
from src.proposal_capture_1011 import load_capture_head,predict_capture
from src.proposal_capture_scoring_1011 import sample_capture_field,geometric_capture_scores
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
        base_hash=digest(base_path(a.checkpoint));offset_head,offset_checkpoint=load_center_offset_head(a.offset_checkpoint,base_hash,a.device);offset_hash=digest(a.offset_checkpoint)
        for parameter in offset_head.parameters():parameter.requires_grad_(False)
        if offset_checkpoint['relevance_power']!=1 or offset_checkpoint['epoch']!=15:raise ValueError('Scoring experiment requires fixedp1Spatial epoch15 offset head')
        paths=dict(absolute_utility_linear=ROOT/'runs/dual_target_1011/capture_absolute_linear_seed42/last.pt',capture50_linear=ROOT/'runs/dual_target_1011/capture_linear_seed42/last.pt');heads={};head_metadata={}
        for name in a.capture_sources:
            head,ck=load_capture_head(paths[name],base_hash,a.device)
            if ck['config']['architecture']!='linear' or ck['epoch']!=15 or ck['validation_read'] or ck['test_read']:raise ValueError('Capture scorehead training provenance mismatch')
            for parameter in head.parameters():parameter.requires_grad_(False)
            heads[name]=head;head_metadata[name]=dict(checkpoint=str(paths[name].resolve()),checkpoint_sha256=digest(paths[name]),head_state_sha256=ck['head_state_sha256'],initial_state_sha256=ck['initial_state_sha256'],target=ck['target'],feature_cache_sha256=ck['feature_cache_sha256'],training_image_ids=ck['training_image_ids'],training_patient_keys=ck['training_patient_keys'],validation_read=False,test_read=False,selection=ck['selection'])
        (output/'head_provenance.json').write_text(json.dumps(head_metadata,indent=2))
        with (output/'traces.jsonl').open('w') as sink:
            for item in records:
                with Image.open(item['image']) as source:image=source.convert('L')
                geometry=candidate_geometry(image,model.grid,224)
                with torch.no_grad():
                    memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device));offsets=predict_center_offsets(offset_head,memory)[0];coords=shifted_candidate_coords(model.cell_xy,offsets,image.size,.5);original=model.proposal(memory['keys']).sigmoid().squeeze(-1);fields={name:predict_capture(head,memory)[0] for name,head in heads.items()}
                for seed in a.seeds:
                    for source_name in a.capture_sources:
                        raw_field=fields[source_name];field_valid=bool(torch.isfinite(raw_field).all() and (raw_field>=0).all() and (raw_field<=1).all());field=torch.nan_to_num(raw_field,nan=0.,posinf=1.,neginf=0.).clamp(0,1);sampled=sample_capture_field(field,coords,model.grid)
                        for alpha in a.alphas:
                            score,valid=geometric_capture_scores(original,sampled,alpha,a.epsilon,field_valid)
                            for budget in a.budgets:
                                name=f'C_{source_name}_a{str(alpha).replace(".","p")}_K{budget}'
                                result=center_offset_rollout(model,memory,image,geometry,offsets,budget,.5,16,seed_for(seed,item['image_id']),'sample',.1,original_relevance=original,ranking_relevance=score)
                                recipe=dict(kind='originalR_capture_geometric',alpha=alpha,epsilon=a.epsilon,capture_source=source_name,sampling=dict(mode='bilinear',align_corners=False,padding_mode='border',grid=[24,48]),capture_valid=valid,invalid_capture_fallback='originalR',alpha1_exact_identity=True)
                                trace=dict(schema_version='proposal_center_offset_1011_v1',coordinate_protocol='center_offset_continuous_roi_v1',image_id=item['image_id'],image=str(Path(item['image']).resolve()),image_size=list(image.size),grid=list(model.grid),crop_size=224,policy=name,strategy='sample',seed=seed,sampling_seed=seed_for(seed,item['image_id']),checkpoint=str(base_path(a.checkpoint).resolve()),checkpoint_sha256=base_hash,checkpoint_epoch=checkpoint['epoch'],rho=model.rho,ablation=model.ablation,offset_checkpoint=str(Path(a.offset_checkpoint).resolve()),offset_checkpoint_sha256=offset_hash,offset_base_checkpoint_sha256=offset_checkpoint['base_checkpoint_sha256'],offset_architecture='spatial',relevance_power=1,original_proposal_state_sha256=offset_checkpoint['original_proposal_state_sha256'],offset_strength=.5,max_offset_crops=offset_checkpoint['config']['max_offset_crops'],max_crop_iou=.1,source_grid_indices_only=True,proposal_source='capture_aware_geometric',score_recipe=recipe,capture_field=field.cpu().tolist(),capture_scores=sampled.cpu().tolist(),capture_checkpoint_sha256=head_metadata[source_name]['checkpoint_sha256'],capture_head_state_sha256=head_metadata[source_name]['head_state_sha256'],capture_initial_state_sha256=head_metadata[source_name]['initial_state_sha256'],annotation_used=False,annotations_used=False,gaze_used=False,observed_gaze_used=False,labels_used=False,output_role='Continuous ROI search scoring; human gaze NLL remains a separate output',geometry=dict(valid=geometry['geometry_valid'],error=geometry['geometry_error']),gaze_component_used=bool(any(not f for f in result['reservation_flags'])),conditional_gaze_steps=sum(not f for f in result['reservation_flags']),forced_roi_path_nll_reported=False,**result)
                                sink.write(json.dumps(trace,allow_nan=False)+'\n');sink.flush();summary={k:trace[k] for k in ('image_id','policy','seed','coordinates','source_anchor_indices','reservation_flags','actual_crop_rectangles','checkpoint_sha256','offset_checkpoint_sha256','capture_checkpoint_sha256')};(output/f'{name}_image{item["image_id"]}_seed{seed}.json').write_text(json.dumps(summary));completed+=1
                    status('running');print(json.dumps(dict(image_id=item['image_id'],seed=seed,completed_paths=completed)),flush=True)
        status('complete')
    except Exception as exc:status('failed',str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--base-checkpoint-override');p.add_argument('--offset-checkpoint',default='runs/dual_target_1011/center_offset_spatial_p1_seed42/last.pt');p.add_argument('--manifest',default='manifests/all.csv');p.add_argument('--split',default='val');p.add_argument('--image-ids',nargs='+',type=int);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=4);p.add_argument('--capture-sources',nargs='+',choices=['absolute_utility_linear','capture50_linear'],default=['absolute_utility_linear','capture50_linear']);p.add_argument('--alphas',nargs='+',type=float,default=[.5,.75]);p.add_argument('--epsilon',type=float,default=1e-6);p.add_argument('--budgets',nargs='+',type=int,default=[8,12,16]);p.add_argument('--seeds',nargs='+',type=int,default=[42]);predict(p.parse_args())
