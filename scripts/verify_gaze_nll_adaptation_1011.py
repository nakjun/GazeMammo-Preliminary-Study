"""Actual-image compact restore versus source cache, without annotation inputs."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from gaze_nll_adaptation_1011 import load_adapted
from src.data import whole_tensor,patch_tensor


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);model,_=load_adapted(a.checkpoint);model=model.to(a.device).eval();results=[]
    for image_id in (64,73):
        r=torch.load(Path(a.cache)/f'{image_id}.pt',weights_only=False);image=Image.open(ROOT/'data/image'/r['file_name']).convert('L');coords=r['coords'][None].to(a.device);features=[f[None].to(a.device) for f in r['features']];mc=model.maps_features(features);qc=model.teacher_logits(mc,coords,r['vectors'][None].to(a.device));ml=model.maps(whole_tensor(image,model.input_size)[None].to(a.device));v=torch.zeros_like(r['vectors']).to(a.device);n=int(r['mask'].sum());crops=torch.stack([patch_tensor(image,p.numpy()*image.size) for p in r['coords'][:n-1]])
        for start in range(0,len(crops),8):v[start+1:start+9]=model.visual.patch_vector(crops[start:start+8].to(a.device))
        ql=model.teacher_logits(ml,coords,v[None]);error=float((qc-ql).abs().max());vector_error=float((v-r['vectors'].to(a.device)).abs().max());key_error=float((mc['keys']-ml['keys']).abs().max());results.append(dict(image_id=image_id,max_logp_error=error,max_patch_vector_error=vector_error,max_key_error=key_error));print(json.dumps(results[-1]),flush=True)
        if error>=1e-4:raise ValueError('Cached/live probability reproduction exceeds original threshold')
    Path(a.output).write_text(json.dumps(dict(results=results,compact_restore=True,GT_zoom_inputs=False,GT_boxes_inputs=False,frozen_visual_dependency=True),indent=2))
if __name__=='__main__':main()
