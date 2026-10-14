"""Render an image-only predicted trajectory and the resulting 224px patches."""
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
from matplotlib.collections import LineCollection
from src.models import trajectory_from_checkpoint
from src.data import whole_tensor,select_centers
from src.constraints import breast_mask,grid_mask,rollout

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--image',required=True)
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--strategy',choices=['argmax','sample'],default='argmax');p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--constraint',choices=['none','breast','hard','soft'],default='none')
    p.add_argument('--radius',type=float,default=112);p.add_argument('--penalty',type=float,default=3.)
    a=p.parse_args();torch.manual_seed(a.seed);torch.set_num_threads(4)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);device=torch.device(a.device)
    c=torch.load(Path(a.checkpoint)/'best.pt',map_location=device,weights_only=False)
    if c['method']!='trajectory':raise ValueError('Trajectory checkpoint required')
    model=trajectory_from_checkpoint(c).to(device).eval()
    with Image.open(a.image) as source:image=source.convert('L')
    if image.size!=(1500,3000):raise ValueError('Requires matching 1500x3000 mammogram preprocessing')
    torch.manual_seed(a.seed)
    with torch.inference_mode():
        if a.constraint=='none':trajectory=model.generate(whole_tensor(image)[None].to(device),a.strategy,a.temperature)[0].cpu().numpy()
        else:
            memory=model.memory(whole_tensor(image)[None].to(device));mask=breast_mask(image)
            allowed=torch.tensor(grid_mask(mask,c['grid'])[None],device=device)
            result=rollout(model,memory,allowed,torch.tensor([[image.width,image.height]],device=device),a.constraint,a.radius,a.penalty,a.strategy,a.temperature)
            trajectory=result['coords'][0][result['valid'][0]].cpu().numpy()
    centers=select_centers(image,'trajectory',c['k'],a.seed,prediction=trajectory) if a.constraint=='none' else trajectory*[image.width,image.height]
    (out/'prediction.json').write_text(json.dumps(dict(backbone=c.get('backbone','resnet50'),model_config=c.get('model_config',{}),checkpoint=str(Path(a.checkpoint).resolve()),image=str(Path(a.image).resolve()),strategy=a.strategy,
            temperature=a.temperature,seed=a.seed,constraint=a.constraint,radius=a.radius,normalized_trajectory=trajectory.tolist(),raw_pixel_trajectory=(trajectory*[image.width,image.height]).tolist(),
            patch_centers_pixels=centers.tolist(),note='Stored-order trajectory, no timestamp prediction; no actual gaze or boxes loaded'),indent=2))
    fig,ax=plt.subplots(figsize=(8,14));ax.imshow(image,cmap='gray');p=trajectory*[image.width,image.height]
    colors=np.arange(len(p));lines=LineCollection(np.stack([p[:-1],p[1:]],1),cmap='turbo',linewidths=1.5);lines.set_array(colors[:-1]);ax.add_collection(lines)
    sc=ax.scatter(p[:,0],p[:,1],c=colors,cmap='turbo',s=25,zorder=3)
    for i,(x,y) in enumerate(p):ax.annotate(str(i+1),(x,y),color='white',fontsize=8)
    ax.set_title(f'{c.get("backbone","resnet50")} predicted trajectory ({a.strategy})');ax.axis('off')
    fig.colorbar(sc,ax=ax,label='Generated sample index');fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=160);plt.close(fig)
    columns=4;rows=(len(centers)+columns-1)//columns;fig,axs=plt.subplots(rows,columns,figsize=(12,3*rows),squeeze=False)
    for i,ax in enumerate(axs.ravel()):
        ax.axis('off')
        if i<len(centers):
            x,y=np.round(centers[i]).astype(int);patch=image.crop((x-112,y-112,x+112,y+112))
            ax.imshow(patch,cmap='gray');ax.set_title(f'#{i+1} ({x}, {y})');patch.save(out/f'patch_{i+1:02d}.png')
    fig.tight_layout();fig.savefig(out/'patches.png',dpi=160);plt.close(fig)
    print(out.resolve())
