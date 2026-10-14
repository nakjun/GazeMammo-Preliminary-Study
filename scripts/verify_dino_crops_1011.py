"""Separate actual new224crop encoding proof; no mutations of extractor cache."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.data import patch_tensor
from gaze_dino_1011 import setup

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--dino-cache',default='runs/dual_target_1011/dino/native_cache');p.add_argument('--device',default='cuda:0');a=p.parse_args();model,records,prov,source=setup(a);proof=[]
 with torch.no_grad():
  for identity in [62,64]:
   original=next(r for r in records if int(r['image_id'])==identity);saved=torch.load(Path(a.dino_cache)/f'{identity}.pt',weights_only=False)
   if not torch.equal(original['coords'],saved['coords']) or not torch.equal(original['mask'],saved['mask']):raise ValueError('Original canonical mapping changed')
   with Image.open(ROOT/'data/image'/original['file_name']) as image:
    image=image.convert('L');crops=torch.stack([patch_tensor(image,c.numpy()*image.size) for c in original['coords'][:-1]]);fresh=torch.zeros(16,384)
    for start in range(0,15,8):fresh[start+1:start+9]=model.visual.patch_vector(crops[start:start+8].to(a.device)).cpu()
   error=float((fresh-saved['vectors']).abs().max());proof.append(dict(image_id=identity,split=original['split'],fresh224vector_max_error=error,mapping='original canonicalcoord[t-1]→previousvector[t]; BOSzero',mask_equal=True))
   if error>1e-5:raise ValueError('Fresh DINO crop parity failed')
 out=Path(a.dino_cache)/'fresh_crop_proof.json';out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
if __name__=='__main__':main()
