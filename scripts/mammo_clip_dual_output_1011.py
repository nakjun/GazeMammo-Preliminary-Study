"""One hashed runtime: actual-image canonical probability and independent ROI14."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.data import whole_tensor,patch_tensor,seed_for
from src.dual_output_1011 import config_sha256,unified_rollout
from src.mammo_clip_dual_output_1011 import load_mammo_clip_dual
from src.guided_trajectory_1011 import candidate_geometry
from src.proposal_utility_1011 import file_sha256,state_sha256
from proposal_policy_1011 import image_records


def teacher_vectors(probability,image,coords,mask,device):
    vectors=torch.zeros(1,coords.shape[1],sum(probability.vector_dims),device=device);n=int(mask.sum())
    if probability.foveal and n>1:
        crops=torch.stack([patch_tensor(image,p.numpy()*image.size,224) for p in coords[0,:n-1].cpu()])
        for start in range(0,len(crops),8):vectors[:,start+1:start+1+len(crops[start:start+8])]=probability.visual.patch_vector(crops[start:start+8].to(device))[None]
    return vectors


def run(a):
    torch.set_num_threads(a.threads);payload=json.loads(Path(a.config).read_text());config=payload['config'];tag=config_sha256(config)
    if tag!=payload['sha256']:raise ValueError('Combined config hash mismatch')
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);(output/'config.json').write_text(json.dumps(payload,indent=2));probability,guided,head=load_mammo_clip_dual(config,a.device);records=image_records(a);started=time.monotonic();results=[];completed=0
    def status(state,error=None):(output/'status.json').write_text(json.dumps(dict(state=state,phase=a.phase,images=len(records),completed=completed,seconds=time.monotonic()-started,config_sha256=tag,error=error)))
    status('running')
    caches=[];cache_maps=[]
    if a.phase in ('smoke','nll'):
        for dependency in config['reference_caches']:
            if file_sha256(dependency['raw_cache'])!=dependency['raw_cache_sha256']:raise ValueError('Reference raw cache hash mismatch')
            cache=torch.load(dependency['raw_cache'],map_location='cpu',weights_only=False);caches.append(cache);cache_maps.append({int(r['image_id']):r for r in cache['records']})
    try:
        sink=(output/'traces.jsonl').open('w') if a.phase=='generate' else None
        for item in records:
            image_id=item['image_id']
            with Image.open(item['image']) as source:image=source.convert('L')
            with torch.no_grad():human=probability.maps(image)
            if a.phase in ('smoke','nll'):
                reference=cache_maps[0][image_id]
                if reference['split']!='val':raise ValueError('Teacher evaluation outside validation')
                for mapping in cache_maps[1:]:
                    for key in ('coords','loss_mask','target'):
                        if not torch.equal(reference[key],mapping[image_id][key]):raise ValueError('Parent teacher alignment mismatch')
                coords=reference['coords'][None].to(a.device);mask=reference['loss_mask'].to(a.device).bool()
                with torch.no_grad():vectors=teacher_vectors(probability,image,coords,mask,a.device);logp=probability.teacher_logits(human,coords,vectors)[0]
                target=reference['target'].to(a.device);nll=float(-logp[mask,target[mask]].mean());parent_cached=torch.logaddexp(cache_maps[0][image_id]['raw_logp']+np.log(.75),cache_maps[1][image_id]['raw_logp']+np.log(.25));parent_cached=parent_cached-parent_cached.logsumexp(-1,keepdim=True);cached=torch.logaddexp(parent_cached+np.log(.75),cache_maps[2][image_id]['raw_logp']+np.log(.25));cached=cached-cached.logsumexp(-1,keepdim=True);cached=torch.logaddexp(cached+np.log(.75),cache_maps[3][image_id]['raw_logp']+np.log(.25));cached=cached-cached.logsumexp(-1,keepdim=True);error=float((logp.cpu()-cached).abs().max());embedding_reference=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{image_id}.pt',map_location='cpu',weights_only=False)['features'][-1].mean((-2,-1));embedding_error=float((human['embedding'][0].cpu()-embedding_reference).abs().max());
                if embedding_error>1e-5:raise ValueError('Originalcontrol nativeembedding parity failure')
                result=dict(embedding_reference_max_error=embedding_error,image_id=image_id,patient_key=reference['patient_key'],nll=nll,valid_steps=int(mask.sum()),cached_reference_max_logp_error=error)
                if error>1e-4:raise ValueError(f'Actual/reference logp error{error} exceeds1e-4; backend/layout audit required')
                if a.phase=='smoke':
                    state=None;step=[]
                    with torch.no_grad():
                        for t in range(coords.shape[1]):q,state,_=probability.step_history(human,coords[:,:t],state,vectors[:,t]);step.append(q)
                    if not torch.allclose(torch.stack(step,1)[0],logp,atol=1e-5,rtol=0):raise ValueError('Teacher versus causalstep mismatch')
                results.append(result)
            if a.phase in ('smoke','generate'):
                with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):roi_memory=guided.maps(whole_tensor(image,guided.input_size)[None].to(a.device))
                geometry=candidate_geometry(image,guided.grid,224);geometry_hash=state_sha256({k:geometry[k] for k in ('coords','rectangles','foreground','inbounds','background','edge')});image_hash=file_sha256(item['image'])
                for seed in a.seeds:
                    result=unified_rollout(probability,guided,head,human,roi_memory,image,geometry,config,seed_for(seed,image_id));previous=json.loads((ROOT/f'runs/dual_target_1011/center_offset_K14_confirmation/O_spatial_p1_s0p5_K14_nms0p15_image{image_id}_seed{seed}.json').read_text())
                    if result['coordinates'][:14]!=previous['coordinates'][:14] or result['source_anchor_indices'][:14]!=previous['source_anchor_indices'][:14]:raise ValueError('UnifiedoriginalR14 prefix mismatch')
                    for step_record in result['steps']:step_record['probability_role']='RAW.75currentCrossView+.25MammoUPMCwhole/controlpast; noR/geometrybias'
                    record=dict(schema_version='proposal_center_offset_1011_v1',coordinate_protocol='center_offset_continuous_roi_v1',image_id=image_id,image=str(Path(item['image']).resolve()),image_sha256=image_hash,image_size=list(image.size),grid=[24,48],crop_size=224,policy='DUAL_ROI14_RAW_MAMMO025',strategy='sample',seed=seed,sampling_seed=seed_for(seed,image_id),checkpoint=config['roi']['guided']['path'],checkpoint_sha256=config['roi']['guided']['sha256'],offset_checkpoint=config['roi']['offset']['path'],offset_checkpoint_sha256=config['roi']['offset']['sha256'],offset_base_checkpoint_sha256=config['roi']['guided']['sha256'],offset_architecture='spatial',relevance_power=1,offset_strength=.5,max_offset_crops=1,max_crop_iou=.15,source_grid_indices_only=True,proposal_source='original_occupancy',annotation_used=False,annotations_used=False,gaze_used=False,observed_gaze_used=False,labels_used=False,combined_config_sha256=tag,probability_dependencies=[dict(role=p['role'],sha256=p['sha256']) for p in config['probability']['parents']],probability_role='RAW.75currentCrossView+.25MammoUPMCwhole/controlpast,T1,gamma0; noR/geometrybias',geometry_sha256=geometry_hash,geometry=dict(valid=geometry['geometry_valid'],error=geometry['geometry_error']),reserved_prefix_matches_original_ROI14=True,forced_roi_path_nll_reported=False,**result)
                    if sink:sink.write(json.dumps(record,allow_nan=False)+'\n');sink.flush();(output/f'image{image_id}_seed{seed}.json').write_text(json.dumps({k:record[k] for k in ('image_id','seed','coordinates','source_anchor_indices','reservation_flags','combined_config_sha256')}))
                    completed+=1
            else:completed+=1
            status('running');print(json.dumps(dict(image_id=image_id,completed=completed)),flush=True)
        if sink:sink.close()
        if results:
            pd.DataFrame(results).to_csv(output/'per_image_nll.csv',index=False);summary=dict(images=len(results),canonical_legacy16_image_macro_nll=float(np.mean([r['nll'] for r in results])),maximum_reference_logp_error=max(r['cached_reference_max_logp_error'] for r in results),combined_config_sha256=tag,observed_gaze_used_for_teacher_only=True,annotations_used=False,physical_crop_size=224,target_grid=[24,48],forced_path_nll='Not applicable');(output/'summary.json').write_text(json.dumps(summary,indent=2))
        status('complete')
    except Exception as exc:status('failed',str(exc));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','nll','generate'],required=True);p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:1');p.add_argument('--threads',type=int,default=4);p.add_argument('--manifest',default='manifests/all.csv');p.add_argument('--split',default='val');p.add_argument('--image-ids',nargs='+',type=int);p.add_argument('--seeds',nargs='+',type=int,default=[42,43,44]);run(p.parse_args())
