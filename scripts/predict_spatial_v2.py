"""Standalone V2 path/patch rendering and optional global/local classification."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.spatial import SpatialTrajectory,JointGlobalLocal
from src.spatial_inference import predict_path
from src.data import whole_tensor,patch_tensor

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True);p.add_argument('--output',required=True)
    p.add_argument('--classifier');p.add_argument('--prior-only',action='store_true');p.add_argument('--strategy',choices=['sample','argmax'],default='sample')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);device=torch.device(a.device)
    c=torch.load(Path(a.checkpoint)/'best.pt',map_location='cpu',weights_only=False)
    model=SpatialTrajectory(**c['model_config'],pretrained=False).to(device).eval();model.load_state_dict(c['state'])
    with Image.open(a.image) as source:image=source.convert('L')
    if image.size!=(1500,3000):raise ValueError('Requires matching dataset preprocessing')
    result=predict_path(model,image,device,a.seed,a.strategy,a.prior_only);points=result['points'];centers=points*[image.width,image.height]
    record=dict(kind='spatial_v2',checkpoint=str(Path(a.checkpoint).resolve()),image=str(Path(a.image).resolve()),model_config=c['model_config'],
                normalized_trajectory=points.tolist(),patch_centers_pixels=centers.tolist(),relaxed_steps=result['relaxations'],strategy=a.strategy,
                prior_only=a.prior_only,seed=a.seed,note='Image-only input; estimated breast mask; no gaze, boxes, or diagnosis labels loaded')
    if a.classifier:
        classifier_state=torch.load(Path(a.classifier)/'best.pt',map_location='cpu',weights_only=False)
        classifier=JointGlobalLocal(pretrained=False,**classifier_state.get('model_config',{})).to(device).eval();classifier.load_state_dict(classifier_state['state'])
        with torch.inference_mode():
            crops=torch.stack([patch_tensor(image,x) for x in centers])[None].to(device)
            output=classifier(whole_tensor(image)[None].to(device),crops,torch.ones(1,len(centers),dtype=torch.bool,device=device))
            record['malignant_probabilities']={k:float(output[k].sigmoid().item()) for k in ['global_logits','local_logits','fusion_logits']}
            record['patch_attention']=output['attention'][0].cpu().tolist()
    (out/'prediction.json').write_text(json.dumps(record,indent=2))
    fig,ax=plt.subplots(figsize=(7,13));ax.imshow(image,cmap='gray');ax.contour(result['breast_mask'],levels=[.5],colors=['lime'],linewidths=.5)
    ax.plot(centers[:,0],centers[:,1],color='cyan',alpha=.5);ax.scatter(centers[:,0],centers[:,1],c=np.arange(len(points)),cmap='turbo',s=20);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(10,10))
    for i,ax in enumerate(axes.ravel()):
        ax.axis('off')
        if i<len(centers):
            x,y=np.round(centers[i]).astype(int);patch=image.crop((x-112,y-112,x+112,y+112));ax.imshow(patch,cmap='gray');ax.set_title(str(i+1));patch.save(out/f'patch_{i+1:02d}.png')
    fig.tight_layout();fig.savefig(out/'patches.png',dpi=160);plt.close(fig)
    print(out.resolve())
