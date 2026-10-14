import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.dual_output_1011 import load_runtime
from src.data import whole_tensor,patch_tensor
from dual_output_1011 import teacher_vectors

torch.set_num_threads(4);c=json.load(open('runs/dual_target_1011/unified_runtime/config_fp32_projection.json'))['config'];p,g,h=load_runtime(c,'cuda:1');records=[]
for i in [64,73]:
 r=torch.load(f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{i}.pt',weights_only=False);img=Image.open(ROOT/'data/image'/r['file_name']).convert('L');coords=r['coords'][None].to('cuda:1');mask=r['mask'].to('cuda:1')
 with torch.no_grad():mem=p.maps([whole_tensor(img,s)[None].to('cuda:1') for s in p.input_sizes]);v=teacher_vectors(p,img,coords,mask,'cuda:1');mix=p.teacher_logits(mem,coords,v)[0];parts=[];start=0;detail=[]
 for j,(m,dep,d) in enumerate(zip(p.parents,c['probability']['parents'],['control_visual_cache','frozen_visual_cache'])):
  rr=torch.load(f'runs/gaze_nll_adaptation_trials_1011_v2/{d}/{i}.pt',weights_only=False);cache=torch.load(dep['raw_cache'],weights_only=False);old=next(x for x in cache['records'] if int(x['image_id'])==i);dim=p.vector_dims[j];vv=v[...,start:start+dim];start+=dim
  with torch.no_grad(),torch.backends.cudnn.flags(allow_tf32=False):q=m.teacher_logits(mem['parents'][j],coords,vv)[0]
  parts.append(q);detail.append(dict(role=dep['role'],vector_error=float((vv[0].cpu()-rr['vectors']).abs().max()),actual_parent_error=float((q.cpu()-old['raw_logp']).abs().max())))
 expected=torch.logaddexp(parts[0]+__import__('math').log(.75),parts[1]+__import__('math').log(.25));expected-=expected.logsumexp(-1,keepdim=True);item=dict(image_id=i,parents=detail,wrapper_vs_direct_parent_mix=float((mix-expected).abs().max()));records.append(item);print(json.dumps(item),flush=True)
Path('runs/dual_target_1011/unified_runtime/wrapper_audit.json').write_text(json.dumps(records,indent=2))
