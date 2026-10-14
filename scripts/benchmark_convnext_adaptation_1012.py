"""Cost-only ConvNeXtBase adaptation benchmark; no cohort/model selection."""
import json,time,sys,resource,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image
from src.gaze_backbones import GazePyramid
from src.data import patch_tensor
from src.official_cbis_classifier_1012 import digest,state_digest

def benchmark(mode,crops,batch):
 encoder=GazePyramid('convnext_base',pretrained=True);head=torch.nn.Linear(1024,1)
 if mode!='full_backbone':
  encoder.requires_grad_(False)
  for layer in [encoder.features[6],encoder.features[7],encoder.norm]:layer.requires_grad_(True)
 else:encoder.requires_grad_(True)
 encoder.train();params=[p for p in list(encoder.parameters())+list(head.parameters()) if p.requires_grad];opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.01);x=crops[:batch];target=torch.ones(1)
 if mode=='cached_boundary':
  with torch.inference_mode():boundary=encoder.features[:6](x).detach().clone()
  # Clone outside inference_mode produces a normal tensor usable by autograd.
  boundary=boundary.clone()
 def step():
  opt.zero_grad()
  if mode=='cached_boundary':z=encoder.features[7](encoder.features[6](boundary));v=encoder.norm(z.mean((-1,-2),keepdim=True)).flatten(1)
  else:v=encoder.patch_vector(x)
  logit=head(v).mean(0);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,target);loss.backward();opt.step();return float(loss.detach())
 start=time.monotonic();warm=step();warmsec=time.monotonic()-start;start=time.monotonic();losses=[step(),step()];seconds=(time.monotonic()-start)/2
 result=dict(mode=mode,batch_patch_count=batch,seconds_per_step=seconds,warmup_seconds=warmsec,trainable_parameters=sum(p.numel() for p in params),peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,estimated_one_arm2037train16patch10epochs_hours=2037*16*10/batch*seconds/3600,estimated_twoarms10epochs_hours=2037*16*10*2/batch*seconds/3600,bag_loss='BCE on mean patch logits, no patch target forcing',benchmark_only_losses=losses)
 del encoder,head,opt;gc.collect();return result

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/convnext_adaptation_1012_benchmark');out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);row=c['rows'][0];image=Image.open(row['canonical_path']).convert('L');xy=c['tensors']['grid_coordinates'][0].double().numpy()*np.array(image.size);crops=torch.stack([patch_tensor(image,p,224) for p in xy]);del c;gc.collect();results=[]
 for mode,batch in [('full_backbone',4),('last_stage_raw',16),('cached_boundary',16)]:
  r=benchmark(mode,crops,batch);results.append(r);(out/'progress.json').write_text(json.dumps(results,indent=2));print(json.dumps(r),flush=True)
 report=dict(state='complete',threads=4,backbone='ConvNeXtBase unchanged architecture, newly trainable params only in temporary benchmark',no_big_job_started=True,test_used=False,source_sha256=digest(__file__),results=results,boundary_shape=[512,14,14],boundary_float32_twoarms2393_gib=2393*16*2*512*14*14*4/1024**3,boundary_float16_twoarms2393_gib=2393*16*2*512*14*14*2/1024**3,limitations='Throughput extrapolation from one image,2 measured steps, batch4/16; actual bag batching/validation/IO costs extra. PeakRSS is process-lifetime maximum, not independent per mode. FP16 cache requires parity audit before use.')
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
