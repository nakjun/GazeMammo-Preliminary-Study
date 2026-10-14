"""Final CN-block boundary cache cost, quantization parity, bag-gradient audit."""
import json,time,sys,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,torch
from PIL import Image
from src.gaze_backbones import GazePyramid
from src.data import patch_tensor
from src.official_cbis_classifier_1012 import digest,state_digest

def boundary_features(encoder,x):
 for layer in list(encoder.features):
  if layer is encoder.features[7]:
   for block in list(layer)[:-1]:x=block(x)
  else:x=layer(x)
 return x

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/convnext_lastblock_1012_benchmark');out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);row=c['rows'][0];image=Image.open(row['canonical_path']).convert('L');model=GazePyramid('convnext_base',pretrained=True).eval().requires_grad_(False);assert state_digest(model.state_dict())==c['provenance']['visual_state_sha256'];boundary={};parity={};times={}
 with torch.inference_mode():
  for arm in ['grid','gaze_baseline']:
   xy=c['tensors'][arm+'_coordinates'][0].double().numpy()*np.array(image.size);x=torch.stack([patch_tensor(image,p,224) for p in xy]);start=time.monotonic();z=torch.cat([boundary_features(model,b) for b in x.split(4)]);seconds=time.monotonic()-start;times[arm]=dict(crops=16,seconds=seconds,crops_per_second=16/seconds);assert z.shape==(16,1024,7,7);direct=torch.cat([model.patch_vector(b) for b in x.split(4)]);quant=z.half().float();restored=model.norm(model.features[7][-1](quant).mean((-1,-2),keepdim=True)).flatten(1);e=(restored-direct).abs();parity[arm]=dict(max_abs=float(e.max()),relative_l2=float(e.norm()/direct.norm()),min_cosine=float(torch.nn.functional.cosine_similarity(restored,direct).min()));boundary[arm]=z.half()
 frozen={k:v for k,v in model.state_dict().items() if not k.startswith('features.7.2.') and not k.startswith('norm.')};frozen_sha=state_digest(frozen);block=model.features[7][-1].requires_grad_(True);norm=model.norm.requires_grad_(True);head=torch.nn.Linear(2048,1);params=list(block.parameters())+list(norm.parameters())+list(head.parameters());opt=torch.optim.AdamW(params,lr=1e-5,weight_decay=.05);results=[]
 for bags in [1,4]:
  x=torch.cat([boundary['grid'],boundary['gaze_baseline']]*((bags+1)//2),0)[:bags*16].float();g=c['tensors']['global_features'][0].repeat(bags,1);y=(torch.arange(bags)%2).float()
  def step():
   opt.zero_grad();z=block(x);v=norm(z.mean((-1,-2),keepdim=True)).flatten(1).reshape(bags,16,1024).mean(1);logits=head(torch.cat([g,v],1)).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y);loss.backward();grad=float(sum(p.grad.abs().sum() for p in block.parameters() if p.grad is not None));opt.step();return float(loss.detach()),grad
  step();start=time.monotonic();values=[step(),step()];seconds=(time.monotonic()-start)/2;assert values[-1][1]>0;assert state_digest({k:v for k,v in model.state_dict().items() if not k.startswith('features.7.2.') and not k.startswith('norm.')})==frozen_sha;results.append(dict(bags_per_batch=bags,crops_per_batch=bags*16,step_seconds=seconds,gradient_abs_sum_finalblock=values[-1][1],twoarms2037train5epochs_hours=2037*2*5/bags*seconds/3600,trainable_params=sum(p.numel() for p in params),frozen_prefix_sha_unchanged=True))
 rate=sum(t['crops'] for t in times.values())/sum(t['seconds'] for t in times.values());report=dict(state='complete',test_used=False,no_big_job_started=True,source_sha256=digest(__file__),cache_sha256=digest(cp),pretrained_sha256=digest('/home/njsung/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth'),boundary=[1024,7,7],float16_cache_twoarms2393_gib=2393*16*2*1024*7*7*2/1024**3,extract=times,estimated_twoarms_full2393_extract_hours=2393*32/rate/3600,quantization_parity=parity,training=results,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,configuration=dict(last_CN_block_only=True,norm_trainable=True,bag_mean=True,joint_global_linear=True,epochs=5,lr=1e-5,weight_decay=.05),limitations='Two measured training steps and one image per arm; extraction IO/validation overhead not included. Quantization comparison uses same CPU encoder; old CUDA drift is separately audited.')
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
