"""Actual-image updated-head compact restore and native-cache causal parity."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from train_deepgaze_clinical_1011 import load_clinical
from src.deepgaze_clinical_1011 import history_batch,prepare_saliency,forward_saliency
from src.deepgaze_context_1011 import dense_centerbias
from evaluate_deepgaze3_1011 import cell_mass


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--native-cache',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;model,c=load_clinical(a.checkpoint,a.device);size=c['input_size'];proof=[]
    for name in ('62','62_mirror','64'):
        item=torch.load(Path(a.native_cache)/f'{name}.pt',map_location='cpu',weights_only=False);row=torch.load(f"runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{item['image_id']}.pt",map_location='cpu',weights_only=False)
        with Image.open(ROOT/'data/image'/row['file_name']) as source:image=source.convert('L')
        if item['flipped']:image=ImageOps.mirror(image)
        rgb=torch.from_numpy(np.asarray(image.resize(tuple(size),Image.Resampling.BILINEAR)).copy()).float()[None,None].repeat(1,3,1,1).to(a.device);coords=item['coords'][None].to(a.device);features=[f[None].to(a.device) for f in item['features']];saliency=prepare_saliency(model,features,size);bias=dense_centerbias(c['centerbias'],size,a.device)
        if item['flipped']:bias=bias.flip(-1)
        for t in (1,8):
            hx,hy=history_batch(coords,t,size);cached=cell_mass(forward_saliency(model,saliency,size,bias,hx,hy));live=cell_mass(model(rgb,bias,hx,hy));error=float((cached-live).abs().max());changed=coords.clone();changed[:,t:]=1-changed[:,t:];cx,cy=history_batch(changed,t,size);history_equal=all(torch.equal(torch.isnan(u),torch.isnan(v)) and torch.equal(torch.nan_to_num(u),torch.nan_to_num(v)) for u,v in ((hx,cx),(hy,cy)))
            if not history_equal:raise ValueError('Future change altered actual history inputs')
            repeat=cell_mass(forward_saliency(model,saliency,size,bias,hx,hy));repeat_error=float((cached-repeat).abs().max())
            future=cell_mass(forward_saliency(model,saliency,size,bias,cx,cy));invariance=float((cached-future).abs().max());result=dict(name=name,step=t,cached_live_max_logp_error=error,current_future_invariance=invariance,identical_history_inputs=history_equal,repeated_identical_max_logp_error=repeat_error,mass=float(cached.exp().sum()));proof.append(result);print(json.dumps(result),flush=True)
            if error>=1e-4 or invariance>1e-5 or repeat_error>1e-5 or abs(result['mass']-1)>1e-5:raise ValueError('Actual clinical DG restore/causality/mass proof failed')
    Path(a.output).write_text(json.dumps(dict(proof=proof,t0='Frozen actual arithmetic BOS; DG never called with empty history',roi_crop_size=224,target_grid=[24,48]),indent=2))
if __name__=='__main__':main()
