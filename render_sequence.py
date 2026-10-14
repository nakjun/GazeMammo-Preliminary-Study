"""Render the stored gaze order on one mammogram (not elapsed time)."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gaze_mammo_mpl')
import ast
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

root = Path(__file__).resolve().parent
data = root / 'data'
coco = json.loads((data / 'annotations/instances_all.json').read_text())
image = next(im for im in coco['images'] if im['id'] == 642)
for line in (data / 'gaze_data/merged_gaze_data.csv').read_text().splitlines():
    name, payload = line.split(';', 1)
    if name == image['file_name']:
        gaze = ast.literal_eval(payload)
        break
p = np.array([np.array(g[0], dtype=float)/g[1] for g in gaze])
valid = np.isfinite(p).all(axis=1) & (p[:,0]>=0) & (p[:,0]<image['width']) & (p[:,1]>=0) & (p[:,1]<image['height'])
indices = np.arange(len(p))
norm = Normalize(1,len(p))
fig,ax = plt.subplots(figsize=(9,15),facecolor='white')
with Image.open(data/'image'/image['file_name']) as im:
    ax.imshow(im,cmap='gray')
# Connect only adjacent valid samples; never bridge a missing/out-of-bounds sample.
adjacent = valid[:-1] & valid[1:]
segments = np.stack([p[:-1],p[1:]],axis=1)[adjacent]
lines = LineCollection(segments,cmap='turbo',norm=norm,linewidths=1.2,alpha=.65)
lines.set_array((indices[:-1]+1)[adjacent])
ax.add_collection(lines)
sc = ax.scatter(p[valid,0],p[valid,1],c=indices[valid]+1,cmap='turbo',norm=norm,s=12,alpha=.9,zorder=3)
for a in coco['annotations']:
    if a['image_id'] == image['id']:
        x,y,w,h = a['bbox']
        ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='white',linestyle='--',linewidth=1.5))
for idx,label,marker in [(0,'START','o'),(len(p)-1,'END','X')]:
    if valid[idx]:
        ax.scatter(*p[idx],s=130,marker=marker,c='white',edgecolor='black',linewidth=1.5,zorder=5)
        ax.annotate(f'{label} #{idx+1}',p[idx],xytext=(12,-20 if idx==0 else 20),textcoords='offset points',color='white',fontsize=11,weight='bold',bbox=dict(facecolor='black',alpha=.7,pad=3),zorder=6)
for idx in range(49,len(p),50):
    if valid[idx]:
        ax.annotate(str(idx+1),p[idx],xytext=(5,5),textcoords='offset points',color='white',fontsize=9,bbox=dict(facecolor='black',alpha=.6,pad=1),zorder=4)
ax.set_title(f'Gaze sequence — stored sample order\nImage ID {image["id"]} | {len(p)} samples | dashed box: malignant lesion',fontsize=13,pad=14)
ax.axis('off')
cbar=fig.colorbar(sc,ax=ax,fraction=.035,pad=.025,shrink=.7)
cbar.set_label('Sample order (not elapsed time)')
cbar.set_ticks([1,50,100,150,200,250,len(p)])
fig.text(.5,.025,'Coordinates restored per sample: (x / scale, y / scale).\nLines connect consecutive in-bounds samples; timestamps are unavailable.',ha='center',fontsize=10)
fig.subplots_adjust(bottom=.065,top=.93,left=.025,right=.9)
out=root/'eda'/'gaze_sequence_image_642.png'
fig.savefig(out,dpi=180,facecolor='white')
print(out)
