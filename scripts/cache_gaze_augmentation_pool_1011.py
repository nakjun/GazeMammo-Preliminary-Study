"""Train-only chosen-segment crop vectors; canonical val93 cache is never resampled."""
import argparse,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gaze_nll_adaptation_1011 import cache_records,load_base
from src.proposal_utility_1011 import state_sha256
from src.data import patch_tensor
from train_attention_1011 import digest
from guided_trajectory_1011 import base_path


def canonical_proof(pool_coords,canonical,source):
    n=int(source['mask'].sum())
    if n!=len(canonical) or not torch.equal(pool_coords[canonical],source['coords'][:n]):raise ValueError('Canonical pool coords mismatch')
    expected=torch.linspace(0,len(pool_coords)-1,16).round().long() if len(pool_coords)>16 else torch.arange(len(pool_coords))
    if not torch.equal(torch.as_tensor(canonical),expected):raise ValueError('Canonical indices not exact legacy linspace')
    if not torch.equal(source['mask'],torch.arange(16)<n):raise ValueError('Canonical mask mismatch')
    return n


def fill_vectors(model,image,coords,canonical,source,device,batch_size=8):
    n=canonical_proof(coords,canonical,source);vectors=torch.empty(len(coords),source['vectors'].shape[-1]);filled=torch.zeros(len(coords),dtype=torch.bool)
    for t in range(n-1):vectors[canonical[t]]=source['vectors'][t+1];filled[canonical[t]]=True
    for t in range(n-1):
        first=patch_tensor(image,coords[canonical[t]].numpy()*image.size,224);second=patch_tensor(image,source['coords'][t].numpy()*image.size,224)
        if not torch.equal(first,second):raise ValueError('Canonical crop pixel mismatch')
    missing=torch.nonzero(~filled,as_tuple=False).flatten();started=time.monotonic()
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    with torch.no_grad():
        for start in range(0,len(missing),batch_size):
            indices=missing[start:start+batch_size];crops=torch.stack([patch_tensor(image,coords[index].numpy()*image.size,224) for index in indices]).to(device);encoded=model.visual.patch_vector(crops).detach().cpu();vectors[indices]=encoded;filled[indices]=True
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)
    elapsed=time.monotonic()-started
    if not filled.all() or not torch.isfinite(vectors).all():raise ValueError('Incomplete/nonfinite pool vectors')
    error=float((vectors[canonical[:n-1]]-source['vectors'][1:n]).abs().max()) if n>1 else 0.
    if error!=0:raise ValueError('Copied canonical vectors changed')
    return vectors,dict(tolerance=0.,max_abs_error=error,passed=True,crop_tensor_equality=True,reused_vectors=n-1,new_crop_encodes=len(missing),crop_encode_seconds=elapsed,crops_per_second=len(missing)/elapsed if elapsed else None)


def extract(a):
    torch.set_num_threads(a.threads);source_args=argparse.Namespace(cache=a.source_cache,base=a.base);source_records,source_provenance=cache_records(source_args);source_map={int(r['image_id']):r for r in source_records};base,checkpoint=load_base(a.base);visual_sha=state_sha256(base.visual.state_dict())
    if visual_sha!=source_provenance['visual_state_sha256'] or digest(base_path(a.base))!=source_provenance['base_checkpoint_sha256']:raise ValueError('Exact original control encoder/parent mismatch')
    labels=torch.load(a.pool_labels,map_location='cpu',weights_only=False);
    if labels.get('kind')!='gaze_chosen_segment_pool_labels_1011':raise ValueError('Wrong audited pool label kind')
    if labels['provenance']['source_parent_sha256']!=source_provenance['base_checkpoint_sha256']:raise ValueError('Pool labels parent mismatch')
    for key in ('gaze_sha256','manifest_sha256','all_manifest_sha256'):
        if labels['provenance'].get(key)!=source_provenance[key]:raise ValueError(f'Pool label source mismatch{key}')
    pool_records=labels['records']
    if len({int(r['image_id']) for r in pool_records})!=len(pool_records):raise ValueError('Duplicate pool labels')
    train_ids={int(r['image_id']) for r in source_records if r['split']=='train'}
    if len(pool_records)!=412 or {int(r['image_id']) for r in pool_records}!=train_ids or any(r['split']!='train' for r in pool_records):raise ValueError('Pool labels must exactly cover train412 only')
    if a.image_ids:
        if not set(a.image_ids)<=train_ids:raise ValueError('No validation/test all-point encoding allowed')
        pool_records=[r for r in pool_records if int(r['image_id']) in a.image_ids]
    encoder_parity=json.loads(Path(a.encoder_parity_proof).read_text())
    if not encoder_parity or any(not p.get('passed') or p.get('canonical_fresh_max_error')!=0 or p.get('novel_saved_fresh_max_error')!=0 or p.get('novel_repeat_max_error')!=0 for p in encoder_parity.values()):raise ValueError('Fresh encoder parity proof failed')
    base.to(a.device).eval()
    for parameter in base.parameters():parameter.requires_grad_(False)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);started=time.monotonic();completed=0;headers=[];validations=[]
    provenance=dict(kind='gaze_stratum_pool_cache_1011',base_checkpoint=str(base_path(a.base).resolve()),base_checkpoint_sha256=source_provenance['base_checkpoint_sha256'],visual_state_sha256=visual_sha,source_cache=str(Path(a.source_cache).resolve()),fresh_encoder_validation=dict(proof_path=str(Path(a.encoder_parity_proof).resolve()),proof_sha256=digest(a.encoder_parity_proof),log_path=str(Path(a.encoder_parity_proof).with_suffix('.log').resolve()),production_batch_size=8,tolerance=0.,cases=encoder_parity,passed=True),source_cache_provenance_sha256=digest(Path(a.source_cache)/'provenance.json'),gaze_sha256=source_provenance['gaze_sha256'],manifest_sha256=source_provenance['manifest_sha256'],all_manifest_sha256=source_provenance['all_manifest_sha256'],pool_labels_sha256=digest(a.pool_labels),pool_audit_provenance=labels['provenance'],annotation_used=False,zoom_labels_role='Unused supervised targets only; no GT zoom model inputs',smoke=bool(a.image_ids),input_size=[256,512],grid=[24,48],crop_size=224,records=headers,protocol='All min32-filtered points from exactly chosen original legacy16 segment; train412 only; original control frozen224crop encoder; old canonical preceding vectors reused exactly; whole-image maps/validation reference old guarded source cache',allowed_train_image_ids=sorted(train_ids),allowed_train_patient_keys=sorted({r['patient_key'] for r in source_records if r['split']=='train'}),training_patient_keys=sorted({r['patient_key'] for r in pool_records}),inherited_parent_selection_patient_keys=checkpoint['selection_patient_keys'])
    def write_json(path,value):temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
    def status(state,error=None):write_json(output/'status.json',dict(state=state,images=len(pool_records),completed=completed,seconds=time.monotonic()-started,error=error))
    write_json(output/'provenance.json',provenance);write_json(output/'config.json',vars(a));status('running')
    try:
        for pool in pool_records:
            image_id=int(pool['image_id']);source=source_map[image_id];coords=pool['pool_coords'];canonical=pool['canonical_pool_indices'];raw_indices=pool['pool_raw_indices']
            if pool['patient_key']!=source['patient_key'] or pool['split']!=source['split'] or coords.shape!=(len(raw_indices),2) or not torch.isfinite(coords).all() or ((coords<0)|(coords>1)).any() or np.any(np.diff(raw_indices.numpy())<=0):raise ValueError('Malformed pool mapping')
            canonical_proof(coords,canonical,source);image_path=ROOT/'data/image'/source['file_name']
            if digest(image_path)!=source['image_sha256']:raise ValueError('Source image changed')
            with Image.open(image_path) as image:vectors,proof=fill_vectors(base,image.convert('L'),coords,canonical,source,a.device,a.batch_size)
            record=dict(image_id=image_id,patient_key=pool['patient_key'],split='train',coords=coords,patch_vectors=vectors,raw_indices=raw_indices,canonical_pool_indices=canonical,canonical_raw_indices=raw_indices[canonical],canonical_coords=source['coords'],canonical_mask=source['mask'],zoom_log_scale_labels=pool['pool_zoom_log_scale_labels'],zoom_labels_role='Supervised targets only',image_sha256=source['image_sha256'],source_record_path=str((Path(a.source_cache)/f'{image_id}.pt').resolve()),visual_state_sha256=visual_sha,crop_size=224,canonical_vector_validation=proof)
            temporary=output/f'{image_id}.tmp';torch.save(record,temporary);temporary.replace(output/f'{image_id}.pt');reloaded=torch.load(output/f'{image_id}.pt',map_location='cpu',weights_only=False)
            if not torch.equal(reloaded['patch_vectors'][canonical[:-1]],source['vectors'][1:len(canonical)]):raise ValueError('Serialized canonical vector mismatch')
            headers.append(dict(image_id=image_id,patient_key=pool['patient_key'],split='train',file_sha256=digest(output/f'{image_id}.pt'),raw_indices_sha256=hashlib.sha256(raw_indices.numpy().tobytes()).hexdigest(),canonical_raw_indices=raw_indices[canonical].tolist(),pool_points=len(coords),canonical_vector_validation=proof));validations.append(proof);completed+=1;write_json(output/'provenance.json',provenance);status('running');print(json.dumps(dict(image_id=image_id,completed=completed,pool_points=len(coords),proof=proof)),flush=True)
        provenance['canonical_vector_validation']=dict(tolerance=0.,max_abs_error=max(v['max_abs_error'] for v in validations),passed=True,crop_tensor_equality=True);write_json(output/'provenance.json',provenance)
        summary=dict(state='complete',images=completed,points=sum(r['pool_points'] for r in headers),new_crops=sum(v['new_crop_encodes'] for v in validations),reused_crops=sum(v['reused_vectors'] for v in validations),crop_encode_seconds=sum(v['crop_encode_seconds'] for v in validations),seconds=time.monotonic()-started,canonical_vector_validation=provenance['canonical_vector_validation']);summary['measured_crop_throughput']=summary['new_crops']/summary['crop_encode_seconds'];write_json(output/'summary.json',summary);status('complete');print(json.dumps(summary),flush=True);return summary
    except Exception as error:status('failed',str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--source-cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--pool-labels',default='runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt');p.add_argument('--output',required=True);p.add_argument('--encoder-parity-proof',default='runs/dual_target_1011/train_gaze_pool_gpu_encoder_parity.json');p.add_argument('--image-ids',nargs='+',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4);p.add_argument('--batch-size',type=int,default=8);extract(p.parse_args())
