"""Render actual selected ConvNeXt-Base classifier inputs and local MIL attention."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib import patheffects
from src.data import ROOT

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--image-id',type=int);parser.add_argument('--device',default='cuda:0');a=parser.parse_args()
    root=ROOT/'runs/fusion_hpo_seed42/convnext_base';selection=json.loads((root/'selected_final.json').read_text());checkpoint=root/selection['refit_dir']
    frame=pd.read_csv(ROOT/'manifests/classification.csv');val=frame[frame.split=='val'].sort_values('image_id')
    row=val[val.image_id==a.image_id].iloc[0] if a.image_id is not None else val[val.label==1].iloc[0]
    out=ROOT/'results/convnext_cases'/f'image_{int(row.image_id)}';out.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/predict_grid_fusion.py'),'--checkpoint',str(checkpoint),'--image',str(ROOT/'data/image'/row.file_name),'--output',str(out/'prediction.json'),'--device',a.device],check=True)
    prediction=json.loads((out/'prediction.json').read_text());centers=np.array(prediction['patch_centers_pixels']);attention=np.array(prediction['patch_attention'])
    with Image.open(ROOT/'data/image'/row.file_name) as raw:image=raw.convert('L')
    coco=json.loads((ROOT/'manifests/instances_clipped.json').read_text());boxes=[b['bbox'] for b in coco['annotations'] if b['image_id']==int(row.image_id)]
    norm=Normalize(vmin=0,vmax=max(float(attention.max()),1e-6));cmap=plt.get_cmap('plasma')
    fig=plt.figure(figsize=(13,16));grid=fig.add_gridspec(2,2,height_ratios=[2.2,1.3],hspace=.18,wspace=.04)
    left=fig.add_subplot(grid[0,0]);right=fig.add_subplot(grid[0,1])
    for ax in [left,right]:ax.imshow(image,cmap='gray',vmin=0,vmax=255);ax.set_xlim(0,1500);ax.set_ylim(3000,0);ax.axis('off')
    left.set_title('Original + annotated lesion',fontsize=13)
    for box in boxes:left.add_patch(Rectangle(box[:2],box[2],box[3],fill=False,edgecolor='red',linewidth=2))
    right.set_title('16 grid patches + MIL attention',fontsize=13)
    right.plot(centers[:,0],centers[:,1],color='cyan',linestyle='--',alpha=.35,linewidth=1)
    for i,((x,y),weight) in enumerate(zip(centers,attention)):
        color=cmap(norm(weight));right.add_patch(Rectangle((x-112,y-112),224,224,fill=False,edgecolor=color,linewidth=1.5))
        right.scatter([x],[y],s=18,color=[color])
        text=right.text(x+13,y,str(i+1),color='white',fontsize=10,weight='bold');text.set_path_effects([patheffects.withStroke(linewidth=2,foreground='black')])
    for box in boxes:right.add_patch(Rectangle(box[:2],box[2],box[3],fill=False,edgecolor='red',linewidth=2))
    sub=grid[1,:].subgridspec(4,4,hspace=.35,wspace=.05)
    for i,(x,y) in enumerate(centers):
        x,y=np.round([x,y]).astype(int);patch=image.crop((x-112,y-112,x+112,y+112));patch.save(out/f'patch_{i+1:02d}.png')
        ax=fig.add_subplot(sub[i//4,i%4]);ax.imshow(patch,cmap='gray',vmin=0,vmax=255);ax.axis('off');ax.set_title(f'{i+1:02d} | attention {attention[i]:.3f}',fontsize=9,color='black')
    score=prediction['probabilities']['fusion_logits'];truth='MALIGNANT' if row.label==1 else 'BENIGN'
    fig.suptitle(f'ConvNeXt-Base | validation image {row.image_id} | GT: {truth}\nFusion malignant score: {score:.3f}',fontsize=15,y=.98)
    fig.text(.5,.026,'Red: annotated lesion, displayed after inference only. Patch color: local MIL attention.',ha='center',fontsize=10)
    fig.text(.5,.012,'Cyan dashed line: deterministic grid selection order; this model does not predict a gaze trajectory.',ha='center',fontsize=10)
    fig.subplots_adjust(top=.92,bottom=.055);fig.savefig(out/'case.png',dpi=160,bbox_inches='tight');plt.close(fig)
    record=dict(image_id=int(row.image_id),file_name=row.file_name,split='val',ground_truth=truth,selection_rule='First malignant validation image by image_id' if a.image_id is None else 'Explicit image_id',selector='image-only grid16',fusion_score=score,boxes_display_only=boxes,checkpoint=str(checkpoint))
    (out/'case_info.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2));print(out/'case.png')
