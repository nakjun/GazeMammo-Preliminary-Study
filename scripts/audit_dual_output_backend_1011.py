"""Isolate live/cached visual features, projection layouts and raw predictions."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.dual_output_1011 import load_runtime
from src.data import whole_tensor,patch_tensor
from dual_output_1011 import teacher_vectors

torch.set_num_threads(4);config=json.loads((ROOT/'runs/dual_target_1011/unified_runtime/config.json').read_text())['config'];prob,guide,head=load_runtime(config,'cuda:1');results=[]
for j,(parent,dep,directory) in enumerate(zip(prob.parents,config['probability']['parents'],['control_visual_cache','frozen_visual_cache'])):
 cache=torch.load(dep['raw_cache'],map_location='cpu',weights_only=False);raws={int(r['image_id']):r for r in cache['records']}
 for image_id in (64,73):
  r=torch.load(ROOT/'runs/gaze_nll_adaptation_trials_1011_v2'/directory/f'{image_id}.pt',map_location='cpu',weights_only=False);image=Image.open(ROOT/'data/image'/r['file_name']).convert('L');coords=r['coords'][None].to('cuda:1');n=int(r['mask'].sum());vectors=torch.zeros(1,16,parent.patch.in_features,device='cuda:1');crops=torch.stack([patch_tensor(image,p.numpy()*image.size,224) for p in r['coords'][:n-1]])
  with torch.no_grad():
   for start in range(0,len(crops),8):vectors[:,start+1:start+1+len(crops[start:start+8])]=parent.visual.patch_vector(crops[start:start+8].to('cuda:1'))[None]
   live=parent.visual(whole_tensor(image,parent.input_size)[None].to('cuda:1'));cached=[f[None].to('cuda:1') for f in r['features']];variants={}
   for name,features in [('live_native',live),('live_contiguous',[f.contiguous() for f in live]),('cached_contiguous',[f.contiguous() for f in cached])]:
    memory=parent.maps_features(features);q=parent.teacher_logits(memory,coords,vectors)[0];variants[name]=dict(logp=q,max_cpu_raw_error=float((q.cpu()-raws[image_id]['raw_logp']).abs().max()))
  results.append(dict(role=dep['role'],image_id=image_id,live_strides=[list(f.stride()) for f in live],cached_strides=[list(f.stride()) for f in cached],feature_errors=[float((l-c).abs().max()) for l,c in zip(live,cached)],patch_vector_error=float((vectors[0].cpu()-r['vectors']).abs().max()),variant_errors={k:v['max_cpu_raw_error'] for k,v in variants.items()},same_gpu_live_contiguous_vs_cached=float((variants['live_contiguous']['logp']-variants['cached_contiguous']['logp']).abs().max())))
  print(json.dumps(results[-1]),flush=True)
Path('runs/dual_target_1011/unified_runtime/backend_audit.json').write_text(json.dumps(dict(results=results,cudnn_tf32=torch.backends.cudnn.allow_tf32,matmul_tf32=torch.backends.cuda.matmul.allow_tf32,tolerance_unchanged=1e-4),indent=2))
