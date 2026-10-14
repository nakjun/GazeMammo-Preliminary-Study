"""Train-only original DINO point-vector cache for the audited chosen segments."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.data import patch_tensor
from gaze_dino_1011 import setup
from train_gaze_path_augmentation_1011 import load_pools
from train_attention_1011 import digest
from src.proposal_utility_1011 import state_sha256

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--pool-cache',default='runs/dual_target_1011/train_gaze_pool_cache');p.add_argument('--pool-labels',default='runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt');p.add_argument('--dino-cache',default='runs/dual_target_1011/dino/native_cache');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');a=p.parse_args();model,records,prov,source=setup(a);pools,poolheader=load_pools(a.pool_cache,records,digest(a.base),prov['visual_state_sha256'],a.cache,pool_labels=a.pool_labels);nativepath=Path(a.dino_cache);nativeheader=json.loads((nativepath/'provenance.json').read_text());nativebyid={int(r['image_id']):r for r in nativeheader['records']};out=Path(a.output);out.mkdir(parents=True,exist_ok=False);metadata=[];proof=[];started=time.monotonic();visualsha=state_sha256(model.visual.state_dict());newtotal=reusedtotal=0;encode_seconds=0.
 chosen=[r for r in records if r['split']=='train' and (not a.smoke or int(r['image_id']) in [62,65])]
 with torch.no_grad():
  for index,row in enumerate(chosen):
   identity=int(row['image_id']);pool=pools[identity];nativefile=nativepath/f'{identity}.pt'
   if digest(nativefile)!=nativebyid[identity]['sha256']:raise ValueError('Original DINO canonical cache changed')
   native=torch.load(nativefile,weights_only=False);canonical=torch.as_tensor(pool['canonical_pool_indices']);raw=torch.as_tensor(pool['raw_indices']);coords=pool['coords'];vectors=torch.zeros(len(coords),384);reused=set()
   if native['image_sha256']!=pool['image_sha256'] or not torch.equal(coords[canonical],native['coords']) or not torch.equal(pool['canonical_mask'],native['mask']) or not torch.equal(raw[canonical],torch.as_tensor(pool['canonical_raw_indices'])):raise ValueError('Canonical raw-index/coordinate/source mismatch')
   for t,position in enumerate(canonical[:-1]):vectors[int(position)]=native['vectors'][t+1];reused.add(int(position))
   new=[i for i in range(len(coords)) if i not in reused];path=ROOT/'data/image'/row['file_name']
   if digest(path)!=pool['image_sha256']:raise ValueError('Source image changed')
   with Image.open(path) as original:
    image=original.convert('L');torch.cuda.synchronize();begin=time.monotonic()
    for start in range(0,len(new),8):
     ids=new[start:start+8];crops=torch.stack([patch_tensor(image,coords[i].numpy()*image.size) for i in ids]);vectors[ids]=model.visual.patch_vector(crops.to(a.device)).cpu()
    torch.cuda.synchronize();elapsed=time.monotonic()-begin;encode_seconds+=elapsed
    if identity in [62,65]:
     positions=canonical[:-1][:8].tolist();crops=torch.stack([patch_tensor(image,coords[i].numpy()*image.size) for i in positions]);fresh=model.visual.patch_vector(crops.to(a.device)).cpu();error=float((fresh-vectors[positions]).abs().max());novel=new[:8];crops=torch.stack([patch_tensor(image,coords[i].numpy()*image.size) for i in novel]);freshnovel=model.visual.patch_vector(crops.to(a.device)).cpu();novelerror=float((freshnovel-vectors[novel]).abs().max());proof.append(dict(image_id=identity,canonical_fresh_error=error,novel_saved_fresh_error=novelerror,new_encodes=len(new),encode_seconds=elapsed,crops_per_second=len(new)/elapsed))
     if max(error,novelerror)>1e-5:raise ValueError('Actual DINO point crop parity failed')
   entry=dict(image_id=identity,patient_key=row['patient_key'],split='train',coords=coords,patch_vectors=vectors,raw_indices=raw,canonical_pool_indices=canonical,canonical_raw_indices=raw[canonical],canonical_coords=native['coords'],canonical_mask=native['mask'],image_sha256=pool['image_sha256'],source_pool_file_sha256=digest(Path(a.pool_cache)/f'{identity}.pt'),crop_size=224)
   torch.save(entry,out/f'{identity}.pt');metadata.append(dict(image_id=identity,patient_key=row['patient_key'],split='train',sha256=digest(out/f'{identity}.pt'),points=len(coords),new_encodes=len(new),reused=len(reused)));newtotal+=len(new);reusedtotal+=len(reused)
   if (index+1)%20==0:print(json.dumps(dict(completed=index+1,total=len(chosen),new_crops=newtotal)),flush=True)
 if state_sha256(model.visual.state_dict())!=visualsha:raise ValueError('Frozen DINO changed')
 header=dict(kind='dino_gaze_pool_cache_1011',records=metadata,source=source,visual_state_sha256=visualsha,source_pool_provenance_sha256=digest(Path(a.pool_cache)/'provenance.json'),source_pool_labels_sha256=digest(a.pool_labels),source_dino_cache_provenance_sha256=digest(nativepath/'provenance.json'),canonical_source_provenance_sha256=digest(Path(a.cache)/'provenance.json'),sampler_source_sha256=digest(ROOT/'src/gaze_path_augmentation_1011.py'),crop_size=224,annotation_used=False,smoke=a.smoke,proof=proof,new_crops=newtotal,reused_crops=reusedtotal,encode_seconds=encode_seconds,new_crops_per_second=newtotal/encode_seconds,seconds=time.monotonic()-started)
 (out/'provenance.json').write_text(json.dumps(header,indent=2));(out/'status.json').write_text(json.dumps(dict(state='complete',images=len(metadata)),indent=2));print(json.dumps(dict(images=len(metadata),new_crops=newtotal,reused=reusedtotal,seconds=header['seconds'],crops_per_second=header['new_crops_per_second'],proof=proof)),flush=True)
if __name__=='__main__':main()
