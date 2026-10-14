"""Immutable ConvNeXt full-image spatial feature extraction, resumable TRAIN/VAL only."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_backbones import GazePyramid
from scripts.benchmark_convnext_lastblock_1012 import boundary_features
from src.official_cbis_classifier_1012 import digest,state_digest

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round17_high_global_adapt_pilot42');out.mkdir(parents=True,exist_ok=True);records=out/'boundary_records';records.mkdir(exist_ok=True);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);model=GazePyramid('convnext_base').eval().requires_grad_(False);sha=state_digest(model.state_dict());assert sha==c['provenance']['visual_state_sha256'];m=dict(state='running',stage='extract',source_sha256=digest(__file__),cache_sha256=digest(cp),encoder_sha256=sha,shape=[1024,32,16],dtype='float16',input=[512,1024],test_used=False,completed=0,cohort=2393,records={});start=time.time();summary=[]
 def save():
  p=out/'extraction_manifest.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(m));temp.replace(p)
 save()
 for i,row in enumerate(c['rows']):
  assert row['split'] in ['train','val'];p=records/f"{row['image_id']}.pt"
  if p.exists():
   r=torch.load(p,weights_only=False,map_location='cpu');assert r['image_id']==row['image_id'] and r['encoder_sha256']==sha and r['pixel_sha256']==digest(row['canonical_path'])
  else:
   pixelsha=digest(row['canonical_path']);x=whole_tensor(Image.open(row['canonical_path']).convert('L'),(512,1024))[None]
   with torch.inference_mode():
    z=boundary_features(model,x);assert z.shape==(1,1024,32,16);v=torch.zeros(1);cos=0.
   r=dict(image_id=row['image_id'],patient_key=row['patient_key'],split=row['split'],encoder_sha256=sha,pixel_sha256=pixelsha,spatial=z[0].half(),summary=v[0],old_global_cosine=cos);temp=p.with_suffix('.tmp');torch.save(r,temp);temp.replace(p)
  summary.append(r['summary']);m['records'][str(row['image_id'])]=digest(p);m['completed']=i+1;save()
 torch.save(dict(image_ids=[r['image_id'] for r in c['rows']],features=torch.stack(summary),encoder_sha256=sha),out/'boundary_index.pt');assert state_digest(model.state_dict())==sha;m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();print(json.dumps(dict(extraction='complete',minutes=m['elapsed_minutes'],count=m['completed'])),flush=True)
if __name__=='__main__':main()
