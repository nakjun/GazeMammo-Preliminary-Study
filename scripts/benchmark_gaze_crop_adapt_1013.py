"""Round07 same actual OOF crop final-block boundary parity/runtime benchmark."""
import sys,json,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.data import patch_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest
from scripts.benchmark_convnext_lastblock_1012 import boundary_features
from scripts.train_gaze_focus_selected_1013 import LowJoint,local_summary

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round07_crop_block_adapt_pilot42');out.mkdir(exist_ok=True);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');t=c['tensors'];r=c['rows'][0];image=Image.open(r['canonical_path']).convert('L');encoder=GazePyramid('convnext_base').eval().requires_grad_(False);assert state_digest(encoder.state_dict())==c['provenance']['visual_state_sha256'];results={};bs=[]
 with torch.inference_mode():
  for arm in ['imitation','lesion_aux','random']:
   if arm=='random':coords=t['random_coordinates'][0,:4];ref=t['random_features'][0,:4]
   else:
    rec=torch.load(Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42/predicted_records_k4')/f"{arm}_{r['image_id']}.pt",weights_only=False,map_location='cpu');coords=rec['coordinates'];ref=rec['features']
   x=torch.stack([patch_tensor(image,p,224) for p in coords.double().numpy()*[1500,3000]]);start=time.monotonic();b=boundary_features(encoder,x);seconds=time.monotonic()-start;v=encoder.norm(encoder.features[7][-1](b.half().float()).mean((-1,-2),keepdim=True)).flatten(1);e=v-ref;results[arm]=dict(seconds=seconds,shape=list(b.shape),max_abs=float(e.abs().max()),relative_l2=float(e.norm()/ref.norm()),min_cosine=float(torch.nn.functional.cosine_similarity(v,ref).min()));bs.append(b.half())
 block=encoder.features[7][-1].requires_grad_(True);norm=encoder.norm.requires_grad_(True);head=LowJoint();opt=torch.optim.AdamW([dict(params=list(block.parameters())+list(norm.parameters()),lr=1e-5),dict(params=head.parameters(),lr=3e-4)],weight_decay=.05);b=bs[0].repeat(8,1,1,1).clone();g=t['global_features'][0].repeat(8,1);mask=torch.ones(8,4,dtype=torch.bool);y=torch.arange(8).remainder(2).float()
 def step():
  opt.zero_grad();v=norm(block(b.float()).mean((-1,-2),keepdim=True)).flatten(1).reshape(8,4,1024);z=local_summary(g,v,mask);loss=torch.nn.functional.binary_cross_entropy_with_logits(head(z),y);loss.backward();grad=sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None);opt.step();return grad
 step();start=time.monotonic();grad=step();seconds=time.monotonic()-start;assert grad>0;report=dict(arms=results,extract_full2393_minutes=sum(v['seconds'] for v in results.values())*2393/60,train_8bags_seconds=seconds,paired2037_epoch_minutes=2037*2/8*seconds/60,cache_fp16_gib=2393*3*4*1024*7*7*2/1024**3,gradient_abs_sum=grad,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,cuda_available=torch.cuda.is_available(),source_sha256=digest(__file__));(out/'benchmark.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
