"""True pre-final-CN-block highresolution extraction/restoration and full bag gradient timing."""
import sys,time,json,copy,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_backbones import GazePyramid
from src.gaze_spatial_mil_round09_1013 import SpatialGazeMIL
from scripts.benchmark_convnext_lastblock_1012 import boundary_features
from src.official_cbis_classifier_1012 import digest,state_digest

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);root=Path('runs/gaze_focus_auroc88_1013');out=root/'round17_high_global_adapt_pilot42';out.mkdir(exist_ok=True);cp=Path('runs/official_cbis_v2_experiment_1012/development.pt');headp=root/'round09_spatial_mil_pilot42/global_only/inner_recovery.pt';assert cp.exists() and headp.exists();c=torch.load(cp,weights_only=False,map_location='cpu');ck=torch.load(headp,weights_only=False,map_location='cpu');assert {'best_state','gm','gs','be'}<=ck.keys() and ck['be']==12;row=c['rows'][0];encoder=GazePyramid('convnext_base').eval().requires_grad_(False);assert state_digest(encoder.state_dict())==c['provenance']['visual_state_sha256'];x=whole_tensor(Image.open(row['canonical_path']).convert('L'),(512,1024))[None]
 with torch.inference_mode():boundary_features(encoder,x)
 tick=time.time()
 with torch.inference_mode():b=boundary_features(encoder,x).half();z=encoder.features[7][-1](b.float());old=torch.load(root/'round08_global_resolution_pilot42/high_spatial_records'/f"{row['image_id']}.pt",weights_only=False,map_location='cpu')['spatial'][None].float();cos=float(torch.nn.functional.cosine_similarity(z.flatten(1),old.flatten(1)).min());rel=float((z-old).norm()/old.norm())
 seconds=time.time()-tick;assert b.shape==(1,1024,32,16) and cos>.99999 and rel<.001;block=copy.deepcopy(encoder.features[7][-1]).requires_grad_(True).eval();model=SpatialGazeMIL(local=False);model.load_state_dict(ck['best_state']);bb=b.float();opt=torch.optim.AdamW([dict(params=block.parameters(),lr=1e-6),dict(params=model.parameters(),lr=3e-5)],weight_decay=.05);tick=time.time();z=block(bb).flatten(2).transpose(1,2);z=((z-ck['gm'])/ck['gs']).clamp(-5,5);zeros=torch.zeros(1,4,1024);coords=torch.zeros(1,4,2);mask=torch.zeros(1,4,dtype=torch.bool);logit=model(z,zeros,coords,mask,z,zeros,coords,mask,torch.ones(1,dtype=torch.bool));loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor([float(row['label'])]));loss.backward();grad=sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None);assert grad>0;opt.step();step=time.time()-tick;r=dict(batch=1,input=[512,1024],boundary_shape=list(b.shape),extract_seconds=seconds,extract_minutes=seconds*2393/60,paired_shared_step_seconds=step,full_epoch_minutes=step*2037/60,max_inner_full_fourarms_upper_minutes=step*2037/60*10*2*4,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,restore_cos=cos,restore_relative_L2=rel,CNblock_gradient_sum=grad,original_norm='Strong global MIL consumes pre-classifier-norm spatial maps; original norm is outside this branch and remains unchanged.');(out/'benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()
