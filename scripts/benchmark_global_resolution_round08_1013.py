"""Same frozen ConvNeXt full-image resolution CPU measurement."""
import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.gaze_backbones import GazePyramid
from src.data import whole_tensor
from src.official_cbis_classifier_1012 import digest,state_digest

torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round08_global_resolution_pilot42');out.mkdir(exist_ok=True);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');model=GazePyramid('convnext_base').eval().requires_grad_(False);assert state_digest(model.state_dict())==c['provenance']['visual_state_sha256'];images=[Image.open(r['canonical_path']).convert('L') for r in c['rows'][:4]];results=[]
with torch.inference_mode():
 for size in [(256,512),(512,1024)]:
  x=torch.stack([whole_tensor(im,size) for im in images]);start=time.monotonic();z=model(x)[-1];seconds=time.monotonic()-start;results.append(dict(width=size[0],height=size[1],batch=4,seconds=seconds,spatial_shape=list(z.shape),full2393_forward_minutes=seconds/4*2393/60));assert torch.isfinite(z).all()
r=dict(state='complete',test_used=False,results=results,hi_spatial_fp16_gib=2393*1024*32*16*2/1024**3,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,source_sha256=digest(__file__),cuda_available=torch.cuda.is_available());(out/'benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
