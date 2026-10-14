"""Small real-pixel boundary/head parity,gradient and frozen-prefix audit."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.gaze_backbones import GazePyramid
from src.data import patch_tensor
from src.official_cbis_classifier_1012 import restore,state_digest,digest
from src.gaze_attention_1011 import attention_loss
from scripts.benchmark_convnext_lastblock_1012 import boundary_features
from scripts.train_gaze_boundary_adaptation_1012 import Adapted,prefix_state

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_boundary_adaptation_1012_round9');out.mkdir(exist_ok=True);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',map_location='cpu',weights_only=False);t=c['tensors'];heads=Path('runs/official_cbis_v2_experiment_1012/heads');gc=torch.load(heads/'global_seed42/best.pt',map_location='cpu',weights_only=False);report={}
 for arm in ['grid','gaze_baseline']:
  head=restore(torch.load(heads/f'{arm}_seed42/best.pt',map_location='cpu',weights_only=False),gc);encoder=GazePyramid('convnext_base',pretrained=True).eval().requires_grad_(False);assert state_digest(encoder.state_dict())==c['provenance']['visual_state_sha256'];image=Image.open(c['rows'][0]['canonical_path']).convert('L');xy=t[arm+'_coordinates'][0].double().numpy()*np.array(image.size);x=torch.stack([patch_tensor(image,p,224) for p in xy])
  with torch.inference_mode():b=torch.cat([boundary_features(encoder,p) for p in x.split(4)]).half()
  b=b.clone()[None];model=Adapted(encoder,head).eval();prefix=state_digest(prefix_state(encoder));global_sha=state_digest(head.global_model.state_dict())
  with torch.inference_mode():ref=head(t['global_features'][:1],t[arm+'_features'][:1],t[arm+'_coordinates'][:1],t[arm+'_mask'][:1]);outp=model(t['global_features'][:1],b,t[arm+'_coordinates'][:1],t[arm+'_mask'][:1]);delta=float((ref['logits'].sigmoid()-outp['logits'].sigmoid()).abs().max())
  assert delta<1e-3
  opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5);model.train();loss=attention_loss(model(t['global_features'][:1],b,t[arm+'_coordinates'][:1],t[arm+'_mask'][:1]),torch.tensor([float(c['rows'][0]['label'])]))[0];loss.backward();gradient=float(sum(p.grad.abs().sum() for p in encoder.features[7][-1].parameters() if p.grad is not None));assert gradient>0;opt.step();assert state_digest(prefix_state(encoder))==prefix and state_digest(head.global_model.state_dict())==global_sha;report[arm]=dict(epoch0_probability_max_abs_error=delta,gradient_abs_sum_finalblock=gradient,frozen_prefix_sha256=prefix,frozen_global_sha256=global_sha,boundary_shape=list(b.shape))
 (Path('runs/gaze_boundary_adaptation_1012_round9')/'smoke_report.json').write_text(json.dumps(dict(state='passed',test_used=False,source_sha256=digest(__file__),checks=report),indent=2));print(json.dumps(report))
if __name__=='__main__':main()
