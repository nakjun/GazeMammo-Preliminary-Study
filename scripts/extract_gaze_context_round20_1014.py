"""Same image-only fixed centers, window224/448/896, resize224; atomic new scale features."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.data import patch_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);root=Path('runs/gaze_focus_auroc88_1013');out=root/'round20_gaze_context_pilot42';out.mkdir(exist_ok=True);cpath=Path('runs/official_cbis_v2_experiment_1012/development.pt');coordsfile=root/'round18_backbone_comparison_pilot42/fixed_coordinates.pt';oldfile=root/'round18_backbone_comparison_pilot42/convnext_base/features.pt';assert all(p.exists() for p in [cpath,coordsfile,oldfile]);c=torch.load(cpath,weights_only=False,map_location='cpu');fixed=torch.load(coordsfile,weights_only=False,map_location='cpu');old=torch.load(oldfile,weights_only=False,map_location='cpu');ids=[r['image_id'] for r in c['rows']];assert fixed['image_ids']==old['image_ids']==ids and fixed['valid_mask'].all();assert torch.equal(fixed['random_indices'],torch.arange(8)[None].expand(2393,-1));coords={'gaze8':fixed['gaze_coordinates'],'random8':fixed['random_coordinates']};encoder=GazePyramid('convnext_base').eval().requires_grad_(False);sha=state_digest(encoder.state_dict());assert sha==c['provenance']['visual_state_sha256'];batch=8;start=time.time();records=out/'records';records.mkdir(exist_ok=True);mp=out/'extraction_manifest.json';deps={str(p):digest(p) for p in [Path(__file__),Path('src/data.py'),Path('src/gaze_backbones.py'),coordsfile,oldfile,cpath]};m=dict(state='running',stage='new_context_feature_extract',visual_sha256=sha,source_sha256=digest(__file__),dependencies=deps,crop_sizes=[448,896],resize=224,arms=list(coords),cohort=2393,batch=batch,test_used=False,oracle_used=False,completed=0,records={},old_scale224_reuse=str(oldfile),historical_comparison='Earlier round6 448 grid/oldgaze16 is different coordinates/head and no matchedrandom; this is freshOOF gaze8/random8 with strongfrozen global')
 def save():
  tmp=mp.with_suffix('.tmp');tmp.write_text(json.dumps(m));tmp.replace(mp)
 if a.benchmark:
  row=c['rows'][0];image=Image.open(row['canonical_path']).convert('L');patches224=torch.stack([patch_tensor(image,xy.double().numpy()*[1500,3000],224) for arm in coords for xy in coords[arm][0]]);reference=torch.cat([old['gaze_vectors'][0],old['random_vectors'][0]])
  with torch.inference_mode():new=torch.cat([encoder.patch_vector(z) for z in patches224.split(4)])
  rel=float((new-reference).norm()/reference.norm());cos=float(torch.nn.functional.cosine_similarity(new.double(),reference.double()).min());assert rel<.001 and cos>.99999;crops=torch.stack([patch_tensor(image,xy.double().numpy()*[1500,3000],size) for size in [448,896] for arm in coords for xy in coords[arm][0]]);bench=[]
  for size in [4,8]:
   with torch.inference_mode():
    [encoder.patch_vector(z) for z in crops.split(size)];tick=time.time();[encoder.patch_vector(z) for z in crops.split(size)]
   seconds=time.time()-tick;bench.append(dict(batch=size,seconds_per_image32crops=seconds,full_minutes=seconds*2393/60))
  result=dict(benchmark=bench,selected_batch=min(bench,key=lambda r:r['seconds_per_image32crops'])['batch'],old224_patch_relative_L2=rel,old224_patch_min_cos_float64=cos,centers_rounding='np.round/PIL blackpadding/BILINEARresize224 same helper; window only changes');(out/'benchmark.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
 bm=json.load(open(out/'benchmark.json'));batch=bm['selected_batch'];m['batch']=batch;save();allfeatures={size:{arm:[] for arm in coords} for size in [448,896]}
 for i,row in enumerate(c['rows']):
  p=records/f"{row['image_id']}.pt"
  if p.exists():record=torch.load(p,weights_only=False,map_location='cpu');assert record['image_id']==row['image_id'] and record['visual_sha256']==sha and record['source_sha256']==m['source_sha256'] and record['pixel_sha256']==row['image_sha256']
  else:
   assert digest(row['canonical_path'])==row['image_sha256'];image=Image.open(row['canonical_path']).convert('L');features={};rectangles={}
   for size in [448,896]:
    features[size]={};rectangles[size]={}
    for arm in coords:
     xy=coords[arm][i];physical=xy.double().numpy()*[1500,3000];crops=torch.stack([patch_tensor(image,p,size) for p in physical]);centers=torch.from_numpy(np.round(physical).astype(int));rectangles[size][arm]=torch.cat([centers-size//2,centers+size//2],1)
     with torch.inference_mode():features[size][arm]=torch.cat([encoder.patch_vector(z) for z in crops.split(batch)])
     assert features[size][arm].shape==(8,1024) and torch.isfinite(features[size][arm]).all()
   record=dict(image_id=row['image_id'],patient_key=row['patient_key'],split=row['split'],features=features,coordinates={arm:coords[arm][i] for arm in coords},rectangles=rectangles,pixel_sha256=row['image_sha256'],visual_sha256=sha,source_sha256=m['source_sha256']);atomic(record,p)
  for size in allfeatures:
   for arm in coords:allfeatures[size][arm].append(record['features'][size][arm])
  m['records'][str(row['image_id'])]=digest(p);m['completed']=i+1;save()
 payload=dict(image_ids=ids,features={224:dict(gaze8=old['gaze_vectors'],random8=old['random_vectors']),**{size:{arm:torch.stack(v) for arm,v in values.items()} for size,values in allfeatures.items()}},mask=fixed['valid_mask'],coordinate_sha256=digest(coordsfile),visual_sha256=sha);atomic(payload,out/'all_scale_features.pt');assert state_digest(encoder.state_dict())==sha and all(digest(p)==h for p,h in deps.items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();print(json.dumps(dict(extraction='complete',images=m['completed'],minutes=m['elapsed_minutes'])),flush=True)
if __name__=='__main__':main()
