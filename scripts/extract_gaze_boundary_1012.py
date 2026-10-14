"""Resumable per-image TRAIN/VAL final-block boundary extraction with pinned encoder/centers."""
import argparse,json,sys,time,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.data import patch_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest
from scripts.benchmark_convnext_lastblock_1012 import boundary_features

def validate_record(path,expected):
 if not Path(path).is_file() or digest(path)!=expected:raise ValueError("Record missing or hash mismatch")

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);ap.add_argument('--arms',nargs='+',default=['grid','gaze_baseline']);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=True);records=out/'records';records.mkdir(exist_ok=True);cp='runs/official_cbis_v2_experiment_1012/development.pt';cachehash=digest(cp);sourcehash=digest(__file__);c=torch.load(cp,map_location='cpu',weights_only=False);assert c['provenance']['split']=='trainval' and len(c['rows'])==2393 and set(r['split'] for r in c['rows'])=={'train','val'};encoder=GazePyramid('convnext_base',pretrained=True).eval().requires_grad_(False);visualhash=state_digest(encoder.state_dict());assert visualhash==c['provenance']['visual_state_sha256']
 mpath=out/'manifest.json';m=json.loads(mpath.read_text()) if mpath.exists() else dict(state='running',stage='feature_extraction',crop_size=224,resize=224,boundary_shape=[1024,7,7],dtype='float16',threads=4,batch_size=4,test_used=False,arms=a.arms,cohort=2393,cache_sha256=cachehash,source_sha256=sourcehash,visual_state_sha256=visualhash,source_dependencies={p:digest(p) for p in ['src/data.py','src/gaze_backbones.py','src/official_cbis_classifier_1012.py','scripts/benchmark_convnext_lastblock_1012.py']},records={})
 assert m['source_sha256']==sourcehash and m['cache_sha256']==cachehash and m['arms']==a.arms;m['state']='running';start=time.monotonic()
 def save():
  m['completed_records']=len(m['records']);m['elapsed_current_session_seconds']=time.monotonic()-start;tmp=mpath.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(mpath)
 save()
 for index,row in enumerate(c['rows']):
  key=str(row['image_id']);path=records/f'{key}.pt'
  if key in m['records']:
   validate_record(path,m['records'][key]['record_sha256']);continue
  pixelhash=digest(row['canonical_path']);assert pixelhash==row['image_sha256'];features={};coords={};rounded={}
  with Image.open(row['canonical_path']) as image:
   image=image.convert('L');assert image.size==(1500,3000)
   for arm in a.arms:
    xy=c['tensors'][arm+'_coordinates'][index];centers=xy.double().numpy()*np.array(image.size);crops=torch.stack([patch_tensor(image,p,224) for p in centers]);
    with torch.inference_mode():features[arm]=torch.cat([boundary_features(encoder,batch).half().cpu() for batch in crops.split(4)])
    coords[arm]=xy;rounded[arm]=torch.tensor(np.round(centers).astype(int));assert features[arm].shape==(16,1024,7,7) and torch.isfinite(features[arm]).all()
  record=dict(kind='official_cbis_finalblock_boundary_1012',image_id=row['image_id'],patient_key=row['patient_key'],split=row['split'],pixel_sha256=pixelhash,cache_sha256=cachehash,source_sha256=sourcehash,visual_state_sha256=visualhash,features=features,coordinates=coords,rounded_centers=rounded,crop_size=224,resize=224,boundary_shape=[1024,7,7],dtype='float16')
  tmp=path.with_suffix('.tmp');torch.save(record,tmp);tmp.replace(path);m['records'][key]=dict(record_sha256=digest(path),pixel_sha256=pixelhash,split=row['split']);save()
  if (index+1)%25==0:print(json.dumps(dict(completed=index+1,total=2393,elapsed_seconds=time.monotonic()-start)),flush=True)
 assert digest(cp)==cachehash and digest(__file__)==sourcehash and state_digest(encoder.state_dict())==visualhash and all(digest(p)==h for p,h in m['source_dependencies'].items());m['state']='complete';save();print(json.dumps(dict(state='complete',completed=len(m['records']))),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if '--output' in sys.argv:
   out=Path(sys.argv[sys.argv.index('--output')+1]);out.mkdir(exist_ok=True);(out/'failure.json').write_text(json.dumps(dict(state='failed',error=repr(e))))
  raise
