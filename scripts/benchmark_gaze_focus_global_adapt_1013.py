"""Real full-image final-block boundary + bag backward benchmark; no model mutations persisted."""
import sys,time,json,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_backbones import GazePyramid
from scripts.benchmark_convnext_lastblock_1012 import boundary_features
from src.official_cbis_classifier_1012 import state_digest

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');model=GazePyramid('convnext_base').eval().requires_grad_(False);assert state_digest(model.state_dict())==c['provenance']['visual_state_sha256'];x=torch.stack([whole_tensor(Image.open(r['canonical_path']).convert('L')) for r in c['rows'][:8]])
 with torch.inference_mode():
  start=time.monotonic();b=boundary_features(model,x);ex=time.monotonic()-start;reference=model(x)[-1];restored=model.features[7][-1](b.half().float());cos=float(torch.nn.functional.cosine_similarity(restored.flatten(1),reference.flatten(1)).min())
 b=b.half().clone();block=model.features[7][-1].requires_grad_(True);norm=model.norm.requires_grad_(True);head=torch.nn.Sequential(torch.nn.LayerNorm(6144),torch.nn.Linear(6144,32),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(32,1));opt=torch.optim.AdamW([dict(params=list(block.parameters())+list(norm.parameters()),lr=1e-5),dict(params=head.parameters(),lr=3e-4)],weight_decay=.05);local=torch.randn(8,2048);y=torch.arange(8).remainder(2).float()
 def step():
  opt.zero_grad();z=block(b.float());g=norm(torch.nn.functional.adaptive_avg_pool2d(z,(2,2))).flatten(1);loss=torch.nn.functional.binary_cross_entropy_with_logits(head(torch.cat([g,local],1)).flatten(),y);loss.backward();grad=sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None);opt.step();return grad
 step();start=time.monotonic();grad=step();seconds=time.monotonic()-start;assert grad>0
 r=dict(cuda_available=torch.cuda.is_available(),extract_8_seconds=ex,extract_full2393_minutes=ex/8*2393/60,boundary_shape=list(b.shape),float16_full_gib=2393*1024*16*8*2/1024**3,restore_min_cosine=cos,train_8_image_step_seconds=seconds,estimated_paired2037_epoch_minutes=2037*2/8*seconds/60,gradient_abs_sum=grad,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
 out=Path('runs/gaze_focus_auroc88_1013/round05_global_block_adapt_pilot42');out.mkdir(exist_ok=True);(out/'benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
if __name__=='__main__':main()
