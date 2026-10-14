"""Immutable ConvNeXt full-image spatial feature extraction, resumable TRAIN/VAL only."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round03_spatial_pilot42');out.mkdir(parents=True,exist_ok=True);records=out/'spatial_records';records.mkdir(exist_ok=True);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);model=GazePyramid('convnext_base').eval().requires_grad_(False);sha=state_digest(model.state_dict());assert sha==c['provenance']['visual_state_sha256'];m=dict(state='running',stage='extract',source_sha256=digest(__file__),cache_sha256=digest(cp),encoder_sha256=sha,shape=[1024,16,8],dtype='float16',input=[256,512],test_used=False,completed=0,cohort=2393,records={});start=time.time();summary=[]
 def save():
  p=out/'extraction_manifest.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(m));temp.replace(p)
 save()
 for i,row in enumerate(c['rows']):
  assert row['split'] in ['train','val'];p=records/f"{row['image_id']}.pt"
  if p.exists():
   r=torch.load(p,weights_only=False,map_location='cpu');assert r['image_id']==row['image_id'] and r['encoder_sha256']==sha and r['pixel_sha256']==digest(row['canonical_path'])
  else:
   pixelsha=digest(row['canonical_path']);x=whole_tensor(Image.open(row['canonical_path']).convert('L'))[None]
   with torch.inference_mode():
    z=model(x)[-1];assert z.shape==(1,1024,16,8);bins=torch.nn.functional.adaptive_avg_pool2d(z,(2,2));v=model.norm(bins).flatten(1);pooled=model.norm(z.mean((-1,-2),keepdim=True)).flatten(1);old=c['tensors']['global_features'][i:i+1];cos=float(torch.nn.functional.cosine_similarity(pooled,old).item())
   r=dict(image_id=row['image_id'],patient_key=row['patient_key'],split=row['split'],encoder_sha256=sha,pixel_sha256=pixelsha,spatial=z[0].half(),summary=v[0],old_global_cosine=cos);temp=p.with_suffix('.tmp');torch.save(r,temp);temp.replace(p)
  summary.append(r['summary']);m['records'][str(row['image_id'])]=digest(p);m['completed']=i+1;save()
 torch.save(dict(image_ids=[r['image_id'] for r in c['rows']],features=torch.stack(summary),encoder_sha256=sha),out/'spatial_summary.pt');assert state_digest(model.state_dict())==sha;m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();print(json.dumps(dict(extraction='complete',minutes=m['elapsed_minutes'],count=m['completed'])),flush=True)
if __name__=='__main__':main()
