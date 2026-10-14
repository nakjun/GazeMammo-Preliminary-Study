"""One frozen final CBIS TEST evaluation. No fitting, cached validation targets or INbreast payloads."""
import argparse,ast,csv,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.data import patch_tensor,whole_tensor,seed_for
from src.trajectory_1011 import segments,window_sample
from src.dual_output_1011 import config_sha256,unified_rollout
from src.mammo_clip_dual_output_1011 import load_mammo_clip_dual
from src.guided_trajectory_1011 import candidate_geometry
from src.proposal_utility_1011 import file_sha256,state_sha256

def metadata():
 with (ROOT/'manifests/all.csv').open() as f:allrows=list(csv.DictReader(f))
 with (ROOT/'manifests/trajectory.csv').open() as f:boxed={int(r['image_id']) for r in csv.DictReader(f) if r['split']=='test' and r['source']=='CBIS-DDSM'}
 rows=[r for r in allrows if r['split']=='test' and r['source']=='CBIS-DDSM'];ids={int(r['image_id']) for r in rows};patients={r['patient_key'] for r in rows};protected={r['patient_key'] for r in allrows if r['split'] in ('train','val','quarantine')}
 if len(rows)!=150 or len(ids)!=150 or len(boxed)!=131 or not boxed<=ids or patients&protected:raise ValueError('TEST cohort/patient protection mismatch')
 return rows,boxed

def canonical(filename,image_size):
 # Parse payloads ONLY for explicitly allowed TEST filenames; never parse other cohorts.
 found=None
 with (ROOT/'data/gaze_data/merged_gaze_data.csv').open() as f:
  for line in f:
   name,payload=line.rstrip('\n').split(';',1)
   if name==filename:
    if found is not None:raise ValueError('Duplicate gaze filename')
    found=np.array([np.asarray(s[0],dtype=float)/float(s[1]) for s in ast.literal_eval(payload)])
 if found is None:return None,'missing_gaze'
 parts=[p/image_size for p in segments(found,*image_size,32.)]
 if not parts:return None,'no_valid_gaze_segment'
 sample=window_sample(parts,int(next(r['image_id'] for r in metadata()[0] if r['file_name']==filename)),42,0,'legacy16',16,8,.5)
 return dict(coords=sample['coords'][:16],mask=sample['loss_mask'][:16]),None

def vectors(model,image,coords,mask,device):
 result=torch.zeros(1,16,sum(model.vector_dims),device=device);n=int(mask.sum())
 if n>1:
  crops=torch.stack([patch_tensor(image,p.numpy()*image.size,224) for p in coords[:n-1]])
  for start in range(0,len(crops),8):result[:,start+1:start+1+len(crops[start:start+8])]=model.visual.patch_vector(crops[start:start+8].to(device))[None]
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['freeze','smoke','nll','generate'],required=True);p.add_argument('--config',default='runs/dual_target_1011/mammo_clip_unified_runtime/config.json');p.add_argument('--protocol',default='runs/frozen_current_test_1012/protocol.json');p.add_argument('--output');p.add_argument('--device',default='cuda:1');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;rows,boxed=metadata();payload=json.loads(Path(a.config).read_text());tag=config_sha256(payload['config'])
 if tag!=payload['sha256']:raise ValueError('Frozen current config hash mismatch')
 protocolpath=Path(a.protocol)
 if a.phase=='freeze':
  protocolpath.parent.mkdir(parents=True,exist_ok=True)
  if protocolpath.exists():raise FileExistsError(protocolpath)
  audit=[];test_patients={r['patient_key'] for r in rows}
  for dependency in payload['config']['probability']['parents']+[payload['config']['roi']['guided'],payload['config']['roi']['offset']]:
   if file_sha256(dependency['path'])!=dependency['sha256']:raise ValueError('Frozen checkpoint SHA changed')
   ck=torch.load(dependency['path'],map_location='cpu',weights_only=False);training=ck.get('training_patient_keys',ck.get('config',{}).get('training_patient_keys',[]));selection=ck.get('selection_patient_keys',ck.get('config',{}).get('selection_patient_keys',[]))
   if test_patients&(set(training)|set(selection)):raise ValueError('TEST patient leaked into source training/selection')
   audit.append(dict(path=dependency['path'],sha256=dependency['sha256'],training_patients=len(training),selection_patients=len(selection),test_overlap=0,patient_metadata_present=bool(training)))
  protocol=dict(source_patient_audit=audit,kind='frozen_final_cbis_test_1012',model_config=str(Path(a.config).resolve()),model_config_sha256=tag,model_config_file_sha256=file_sha256(a.config),evaluator_sha256=file_sha256(__file__),all_manifest_sha256=file_sha256(ROOT/'manifests/all.csv'),trajectory_manifest_sha256=file_sha256(ROOT/'manifests/trajectory.csv'),gaze_source_sha256=file_sha256(ROOT/'data/gaze_data/merged_gaze_data.csv'),test_ids=[int(r['image_id']) for r in rows],annotated_ids=sorted(boxed),unannotated_ids=sorted({int(r['image_id']) for r in rows}-boxed),test_patient_keys=sorted({r['patient_key'] for r in rows}),patient_disjoint_train_val_quarantine=True,source='CBIS-DDSM ONLY; INbreast prohibited',gaze_protocol='segments min_distance32; window_sample legacy16 seed42 epoch0 length16 burnin8 mixedratio.5; hardcell24x48; image-macro; valid mask',teacher_past='actual224 original rounding; frozen three parent crop encoders; batch8; true observed prefix ONLY for NLL',generation='all150 image-only; originalR14 offsetstrength.5 NMS.15+2RAW; 16actual224 crops; seeds42,43,44',detection_confidence='original guided relevance[source_anchor_index] for every reserved/free actual crop; reserved shifted crop inherits anchor score; proposal ranking score only, not calibrated malignancy probability',boxes='No annotation inputs to this evaluator; independent posthoc131 annotated; 19unannotated never negative',selection='ONE final evaluation; no tuning, improvement, thresholds or checkpoint changes; goal remains paused',compute=payload['config']['compute'])
  protocolpath.write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol));return
 protocol=json.loads(protocolpath.read_text())
 for actual,expected in [(file_sha256(__file__),protocol['evaluator_sha256']),(file_sha256(a.config),protocol['model_config_file_sha256']),(tag,protocol['model_config_sha256']),(file_sha256(ROOT/'manifests/all.csv'),protocol['all_manifest_sha256']),(file_sha256(ROOT/'manifests/trajectory.csv'),protocol['trajectory_manifest_sha256']),(file_sha256(ROOT/'data/gaze_data/merged_gaze_data.csv'),protocol['gaze_source_sha256'])]:
  if actual!=expected:raise ValueError('Frozen protocol dependency changed')
 if not a.output:raise ValueError('Unique output required')
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);(out/'protocol.json').write_text(json.dumps(protocol,indent=2));model,guided,head=load_mammo_clip_dual(payload['config'],a.device);model.requires_grad_(False);guided.requires_grad_(False);head.requires_grad_(False);records=[];excluded=[];started=time.monotonic();count=0;rows=rows[:1] if a.phase=='smoke' else rows;sink=(out/'traces.jsonl').open('w') if a.phase in ('smoke','generate') else None
 def status(state): (out/'state.json').write_text(json.dumps(dict(state=state,phase=a.phase,completed=count,images=len(rows),seconds=time.monotonic()-started,model_config_sha256=tag)))
 status('running')
 with torch.no_grad():
  for row in rows:
   iid=int(row['image_id']);imagepath=ROOT/'data/image'/row['file_name'];image=Image.open(imagepath).convert('L');memory=model.maps(image)
   if a.phase in ('smoke','nll'):
    sample,error=canonical(row['file_name'],image.size)
    if error:excluded.append(dict(image_id=iid,reason=error))
    else:
     coords=sample['coords'][None].to(a.device);mask=sample['mask'];past=vectors(model,image,sample['coords'],mask,a.device);q=model.teacher_logits(memory,coords,past)[0]
     if not torch.isfinite(q).all() or q.logsumexp(-1).abs().max()>1e-5:raise ValueError('TEST q normalization')
     target=(sample['coords'][:,1]*48).long().clamp(0,47)*24+(sample['coords'][:,0]*24).long().clamp(0,23)
     if a.phase=='smoke':
      state=None;step=[]
      for t in range(16):z,state,_=model.step_history(memory,coords[:,:t],state,past[:,t]);step.append(z)
      step_error=float((q-torch.stack(step,1)[0]).abs().max());future=coords.clone();future[:,8:]=1-future[:,8:];future_vectors=past.clone();future_vectors[:,9:]+=100;future_error=float((q[:9]-model.teacher_logits(memory,future,future_vectors)[0,:9]).abs().max())
      if max(step_error,future_error)>1e-5:raise ValueError('TEST teacher-step/causality engineering mismatch')
      (out/'engineering_proof.json').write_text(json.dumps(dict(image_id=iid,teacher_step_error=step_error,future_coords_vectors_error=future_error,normalization_max_error=float(q.logsumexp(-1).abs().max()),no_cache_or_validation_reference_used=True),indent=2))
     records.append(dict(image_id=iid,patient_key=row['patient_key'],split='test',file_name=row['file_name'],image_sha256=file_sha256(imagepath),image_size=list(image.size),coords=sample['coords'],valid=mask,loss_mask=mask,target=target,raw_logp=q.cpu(),nll=float(-q.cpu()[mask,target[mask]].mean()),annotation_status='annotated' if iid in boxed else 'unannotated'))
   if sink:
    with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):roi_memory=guided.maps(whole_tensor(image,guided.input_size)[None].to(a.device))
    geometry=candidate_geometry(image,guided.grid,224)
    for seed in ([42] if a.phase=='smoke' else [42,43,44]):
     result=unified_rollout(model,guided,head,memory,roi_memory,image,geometry,payload['config'],seed_for(seed,iid));roi=payload['config']['roi']
     record=dict(schema_version='proposal_center_offset_1011_v1',coordinate_protocol='center_offset_continuous_roi_v1',evaluation_split='test',image_id=iid,patient_key=row['patient_key'],image=str(imagepath.resolve()),image_sha256=file_sha256(imagepath),image_size=list(image.size),grid=[24,48],crop_size=224,policy='FROZEN_CURRENT_ROI14_RAW',strategy='sample',seed=seed,sampling_seed=seed_for(seed,iid),checkpoint=roi['guided']['path'],checkpoint_sha256=roi['guided']['sha256'],offset_checkpoint=roi['offset']['path'],offset_checkpoint_sha256=roi['offset']['sha256'],offset_base_checkpoint_sha256=roi['guided']['sha256'],offset_architecture='spatial',relevance_power=1,offset_strength=.5,max_offset_crops=1,max_crop_iou=.15,source_grid_indices_only=True,proposal_source='original_occupancy',annotation_used=False,annotations_used=False,gaze_used=False,observed_gaze_used=False,labels_used=False,combined_config_sha256=tag,protocol_sha256=file_sha256(protocolpath),geometry=dict(valid=geometry['geometry_valid'],error=geometry['geometry_error']),forced_roi_path_nll_reported=False,**result);sink.write(json.dumps(record,allow_nan=False)+'\n');sink.flush()
   count+=1;status('running');print(json.dumps(dict(image_id=iid,completed=count)),flush=True)
 if sink:sink.close()
 if file_sha256(__file__)!=protocol['evaluator_sha256'] or file_sha256(a.config)!=protocol['model_config_file_sha256']:raise ValueError('Frozen evaluator/config changed during run')
 if records:torch.save(dict(kind='frozen_test_raw_legacy16_1012',grid=[24,48],records=records,protocol=protocol,excluded=excluded),out/'raw_probabilities.pt')
 (out/'summary.json').write_text(json.dumps(dict(phase=a.phase,model_config_sha256=tag,images_attempted=len(rows),gaze_images_valid=len(records),excluded=excluded,canonical_image_macro_nll=float(np.mean([r['nll'] for r in records])) if records else None,generation_images=len(rows) if sink else 0,boxes_read=False,normalization_tolerance=1e-5,goal_remains_paused=True),indent=2));status('complete')
if __name__=='__main__':main()
