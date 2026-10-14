import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest

torch.set_num_threads(4);torch.set_num_interop_threads(1);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');x=torch.stack([whole_tensor(Image.open(r['canonical_path']).convert('L'),(512,1024)) for r in c['rows'][:4]]);model=GazePyramid('convnext_base').eval().requires_grad_(False);results=[]
with torch.inference_mode():
 for batch in [1,2,4]:
  model(x[:batch]);times=[]
  for repeat in range(2):
   tick=time.monotonic();model(x[:batch]);times.append(time.monotonic()-tick)
  seconds=sum(times)/len(times);results.append(dict(batch=batch,mean_seconds=seconds,full2393_forward_minutes=seconds/batch*2393/60))
p=Path('runs/gaze_focus_auroc88_1013/round08_global_resolution_pilot42/benchmark_batchchoice.json');r=dict(results=results,selected_batch=min(results,key=lambda r:r['mean_seconds']/r['batch'])['batch'],source_sha256=digest(__file__),peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2);p.write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
