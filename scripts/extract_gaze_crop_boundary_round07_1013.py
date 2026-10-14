"""Resumable immutable crop boundary cache for three image-only K4 sources."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.data import patch_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest
from scripts.benchmark_convnext_lastblock_1012 import boundary_features

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round07_crop_block_adapt_pilot42');d=out/'boundary_records';d.mkdir(exist_ok=True);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');t=c['tensors'];encoder=GazePyramid('convnext_base').eval().requires_grad_(False);sha=state_digest(encoder.state_dict());assert sha==c['provenance']['visual_state_sha256'];m=dict(state='running',stage='extract',source_sha256=digest(__file__),cache_sha256=digest(cp),visual_sha256=sha,shape=[4,1024,7,7],dtype='float16',arms=['imitation','lesion_aux','random'],cohort=2393,completed=0,test_used=False,oracle_used=False,records={})
 def save():
  p=out/'extraction_manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m));tmp.replace(p)
 save()
 for i,row in enumerate(c['rows']):
  dest=d/f"{row['image_id']}.pt"
  if dest.exists():
   r=torch.load(dest,weights_only=False,map_location='cpu');assert r['image_id']==row['image_id'] and r['pixel_sha256']==row['image_sha256'] and r['visual_sha256']==sha and r['source_sha256']==m['source_sha256']
  else:
   image=Image.open(row['canonical_path']).convert('L');assert image.size==(1500,3000) and digest(row['canonical_path'])==row['image_sha256'];boundaries={};pooled={};coordinates={};masks={};sourcehash={}
   for arm in m['arms']:
    if arm=='random':xy=t['random_coordinates'][i,:4];mask=t['random_mask'][i,:4].clone()
    else:
     p=out.parent/'round06_lesion_trajectory_pilot42/predicted_records_k4'/f"{arm}_{row['image_id']}.pt";rec=torch.load(p,weights_only=False,map_location='cpu');assert rec['image_id']==row['image_id'] and rec['pixel_sha256']==row['image_sha256'];xy=rec['coordinates'];mask=rec['mask'].clone();sourcehash[arm]=digest(p)
    centers=xy.double().numpy()*[1500,3000];rounded=np.round(centers).astype(int);seen=set()
    for j,point in enumerate(rounded):
     key=tuple(point)
     if key in seen:mask[j]=False
     else:seen.add(key)
    x=torch.stack([patch_tensor(image,p,224) for p in centers])
    with torch.inference_mode():
     b=boundary_features(encoder,x).half();v=encoder.norm(encoder.features[7][-1](b.float()).mean((-1,-2),keepdim=True)).flatten(1)
    assert b.shape==(4,1024,7,7);boundaries[arm]=b;pooled[arm]=v;coordinates[arm]=xy;masks[arm]=mask
   r=dict(image_id=row['image_id'],patient_key=row['patient_key'],split=row['split'],features=boundaries,pooled=pooled,coordinates=coordinates,mask=masks,pixel_sha256=row['image_sha256'],visual_sha256=sha,source_sha256=m['source_sha256'],prediction_record_sha256=sourcehash);tmp=dest.with_suffix('.tmp');torch.save(r,tmp);tmp.replace(dest)
  m['records'][str(row['image_id'])]=digest(dest);m['completed']=i+1;save()
 assert state_digest(encoder.state_dict())==sha;m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();print(json.dumps(dict(extraction='complete',count=m['completed'],minutes=m['elapsed_minutes'])),flush=True)
if __name__=='__main__':main()
