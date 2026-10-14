"""Image-only inference for the grid global/local fusion backbone comparison."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import torch
from PIL import Image
from src.grid_selector import grid_points
from src.spatial import JointGlobalLocal
from src.data import whole_tensor,patch_tensor

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4);device=torch.device(a.device)
    ckpt=torch.load(Path(a.checkpoint)/'best.pt',map_location='cpu',weights_only=False)
    model=JointGlobalLocal(pretrained=False,**ckpt.get('model_config',{})).to(device).eval();model.load_state_dict(ckpt['state'])
    with Image.open(a.image) as original:image=original.convert('L')
    if image.size!=(1500,3000):raise ValueError('Requires dataset-sized 1500x3000 image')
    points=grid_points(image);centers=points*image.size
    with torch.inference_mode():
        patches=torch.stack([patch_tensor(image,p) for p in centers])[None].to(device)
        result=model(whole_tensor(image)[None].to(device),patches,torch.ones(1,len(points),dtype=torch.bool,device=device))
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    record=dict(backbone=model.model_config['backbone_name'],checkpoint=str(Path(a.checkpoint).resolve()),image=str(Path(a.image).resolve()),
                selector='image_only_grid_control',normalized_points=points.tolist(),patch_centers_pixels=centers.tolist(),
                probabilities={key:float(result[key].sigmoid().item()) for key in ['global_logits','local_logits','fusion_logits']},
                patch_attention=result['attention'][0].cpu().tolist())
    out.write_text(json.dumps(record,indent=2,allow_nan=False));print(out.resolve())
