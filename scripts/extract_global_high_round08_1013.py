"""Frozen same-ConvNeXt global512x1024; low spatial maps reused; identical normalized mean/max."""
import sys,json,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round08_global_resolution_pilot42');d=out/'high_spatial_records';d.mkdir(exist_ok=True);batch=json.load(open(out/'benchmark_batchchoice.json'))['selected_batch'];cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');rows=c['rows'];encoder=GazePyramid('convnext_base').eval().requires_grad_(False);sha=state_digest(encoder.state_dict());assert sha==c['provenance']['visual_state_sha256'];m=dict(state='running',stage='global_feature_extract',source_sha256=digest(__file__),cache_sha256=digest(cp),visual_sha256=sha,batch=batch,input_high=[512,1024],input_low=[256,512],spatial_high_shape=[1024,32,16],dtype='float16',global_summary='norm(mean spatial) concat norm(max spatial),2048; both resolutions same FP16maps→FP32pool',low_source='round03 spatial_records immutable',test_used=False,oracle_used=False,cohort=2393,completed=0,records={})
 def save():
  p=out/'extraction_manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m));tmp.replace(p)
 def summary(z):return torch.cat([encoder.norm(z.mean((-1,-2),keepdim=True)).flatten(1),encoder.norm(z.amax((-1,-2),keepdim=True)).flatten(1)],1)
 save();low=[];high=[]
 for startidx in range(0,len(rows),batch):
  rs=rows[startidx:startidx+batch];missing=[r for r in rs if not (d/f"{r['image_id']}.pt").exists()]
  if missing:
   x=torch.stack([whole_tensor(Image.open(r['canonical_path']).convert('L'),(512,1024)) for r in missing])
   with torch.inference_mode():z=encoder(x)[-1].half();assert z.shape==(len(missing),1024,32,16) and torch.isfinite(z).all()
   for j,r in enumerate(missing):
    assert digest(r['canonical_path'])==r['image_sha256'];record=dict(image_id=r['image_id'],patient_key=r['patient_key'],split=r['split'],pixel_sha256=r['image_sha256'],visual_sha256=sha,source_sha256=m['source_sha256'],spatial=z[j]);dest=d/f"{r['image_id']}.pt";tmp=dest.with_suffix('.tmp');torch.save(record,tmp);tmp.replace(dest)
  for r in rs:
   p=d/f"{r['image_id']}.pt";hi=torch.load(p,weights_only=False,map_location='cpu');lo_path=out.parent/'round03_spatial_pilot42/spatial_records'/f"{r['image_id']}.pt";lo=torch.load(lo_path,weights_only=False,map_location='cpu');assert hi['image_id']==lo['image_id']==r['image_id'] and hi['pixel_sha256']==lo['pixel_sha256']==r['image_sha256'];assert hi['visual_sha256']==lo['encoder_sha256']==sha and hi['source_sha256']==m['source_sha256'];assert lo['spatial'].shape==(1024,16,8)
   with torch.inference_mode():low.append(summary(lo['spatial'][None].float())[0]);high.append(summary(hi['spatial'][None].float())[0])
   m['records'][str(r['image_id'])]=dict(high_sha256=digest(p),low_sha256=digest(lo_path));m['completed']+=1
  save()
 atomic=out/'global_meanmax.pt';tmp=atomic.with_suffix('.tmp');torch.save(dict(image_ids=[r['image_id'] for r in rows],low=torch.stack(low),high=torch.stack(high),visual_sha256=sha),tmp);tmp.replace(atomic);assert state_digest(encoder.state_dict())==sha;m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();print(json.dumps(dict(extraction='complete',count=m['completed'],minutes=m['elapsed_minutes'])),flush=True)
if __name__=='__main__':main()
