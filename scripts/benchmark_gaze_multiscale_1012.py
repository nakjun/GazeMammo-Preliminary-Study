"""CPU benchmark using exact cached centers and original patch_tensor helper."""
import json,time,sys,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.gaze_backbones import GazePyramid
from src.data import patch_tensor
from src.official_cbis_classifier_1012 import digest,state_digest

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_multiscale_1012_benchmark');out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);row=c['rows'][0];image=Image.open(row['canonical_path']).convert('L');centers=c['tensors']['grid_coordinates'][0].double().numpy()*np.array(image.size);encoder=GazePyramid('convnext_base',pretrained=True).eval().requires_grad_(False);actual=state_digest(encoder.state_dict());expected=c['provenance']['visual_state_sha256'];assert actual==expected
 crops=torch.stack([patch_tensor(image,point,448) for point in centers]);start=time.monotonic()
 with torch.inference_mode():encoder.patch_vector(crops[:4])
 warmup=time.monotonic()-start;start=time.monotonic()
 with torch.inference_mode():features=torch.cat([encoder.patch_vector(batch) for batch in crops.split(4)])
 seconds=time.monotonic()-start;torch.save(dict(image_id=row['image_id'],arm='grid',crop_size=448,features=features,pixel_sha256=row['image_sha256'],center_coordinates=c['tensors']['grid_coordinates'][0],visual_state_sha256=actual),out/'sample_448.pt');rate=len(crops)/seconds
 report=dict(state='complete',threads=4,crop_size=448,network_input=224,benchmark_crops=len(crops),batch_size=4,seconds=seconds,crops_per_second=rate,warmup_seconds=warmup,estimated_2393_grid_gaze_hours=2393*32/rate/3600,estimated_with_random_hours=2393*48/rate/3600,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,visual_state_sha256=actual,visual_matches_pinned_cache=True,source_sha256=digest(__file__),cache_sha256=digest(cp),test_used=False)
 (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
