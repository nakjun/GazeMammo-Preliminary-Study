"""CPU, image-only checkpoint rollouts for one fixed validation case."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.data import load_gazes,reduce_gaze,whole_tensor,patch_tensor,seed_for
from src.trajectory_1011 import restore,segments
from src.metrics import dtw_distance


def stable_checkpoint(path,attempts=5):
    last_error=None
    for _ in range(attempts):
        try:
            before=path.stat();payload=path.read_bytes();after=path.stat()
            signature=lambda stat:(stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if signature(before)!=signature(after):raise RuntimeError('Checkpoint changed during read')
            checkpoint=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
            digest=hashlib.sha256(payload).hexdigest()
            return checkpoint,digest
        except Exception as error:
            last_error=error;time.sleep(.2)
    raise RuntimeError(f'Could not load a stable checkpoint: {path}: {last_error}')


def normalized_to_pixels(points,size):return np.asarray(points,float)*np.asarray(size,float)

@torch.no_grad()
def predict_image_only(model,image,image_id,seed=42,strategy='sample',steps=16):
    """No gaze argument or lookup; crop centers come exclusively from predictions."""
    model.eval();memory=model.maps(whole_tensor(image,model.input_size)[None]);prefix=torch.zeros(1,0,2);hidden=None
    generator=torch.Generator().manual_seed(seed_for(seed,image_id));gates=[]
    for index in range(steps):
        vector=model.visual.patch_vector(patch_tensor(image,normalized_to_pixels(prefix[0,-1].numpy(),image.size))[None]) if model.foveal and index else None
        logp,hidden,diag=model.step_history(memory,prefix,hidden,vector)
        cell=int(logp.argmax(-1)) if strategy=='argmax' else int(torch.multinomial(logp.softmax(-1),1,generator=generator))
        prefix=torch.cat([prefix,model.cell_xy[cell][None,None]],1);gates.append(diag['gates'][0].tolist())
    return prefix[0].numpy(),gates


def diagnostics(points,truth,size):
    pixels=normalized_to_pixels(points,size);distances=np.linalg.norm(np.diff(pixels,axis=0),axis=1)
    visits=[]
    for index,p in enumerate(pixels):visits.append(int((np.linalg.norm(pixels[:index]-p,axis=1)<112).sum()))
    return dict(normalized_dtw_to_legacy16=dtw_distance(truth,points),mean_step_distance_pixels=float(distances.mean()) if len(distances) else 0.,exact_repeat_count=len(points)-len(np.unique(points,axis=0)),mean_previous_visits_within112px=float(np.mean(visits)),return_fraction112px=float(np.mean(np.asarray(visits[1:])>0)) if len(points)>1 else 0.,step_distances_pixels=distances.tolist())


def draw_panel(axis,image,points,title):
    from matplotlib.collections import LineCollection
    from matplotlib.patheffects import withStroke
    pixels=normalized_to_pixels(points,image.size);values=np.linspace(0,1,len(points))
    axis.imshow(image,cmap='gray',vmin=0,vmax=255)
    if len(points)>1:
        lines=LineCollection(np.stack([pixels[:-1],pixels[1:]],axis=1),cmap='turbo',linewidths=1.8,alpha=.85);lines.set_array(values[:-1]);lines.set_clim(0,1);axis.add_collection(lines)
    axis.scatter(pixels[:,0],pixels[:,1],c=values,cmap='turbo',vmin=0,vmax=1,s=22,edgecolors='white',linewidths=.4)
    axis.scatter(*pixels[0],marker='o',facecolors='none',edgecolors='#00ff90',s=100,linewidths=2)
    axis.scatter(*pixels[-1],marker='X',facecolors='#ff2879',edgecolors='white',s=75,linewidths=.6)
    for step,(x,y) in enumerate(pixels):
        axis.annotate(str(step+1),(x,y),xytext=(4,3),textcoords='offset points',color='white',fontsize=7,path_effects=[withStroke(linewidth=1.8,foreground='black')])
    axis.set_title(title,fontsize=10);axis.set_xlim(0,image.width);axis.set_ylim(image.height,0);axis.set_aspect('equal');axis.set_xticks([]);axis.set_yticks([])


def render(suite,image_id=64,output=None,tag=None,threads=2):
    torch.set_num_threads(threads)
    manifest=pd.read_csv(ROOT/'manifests/all.csv');rows=manifest[(manifest.image_id==image_id)&(manifest.split=='val')&(manifest.label>=0)]
    if len(rows)!=1:raise ValueError('Case must be one labelled validation image')
    row=rows.iloc[0]
    with Image.open(ROOT/'data/image'/row.file_name) as src:image=src.convert('L')
    size=image.size;metadata=dict(image_id=image_id,patient_key=row.patient_key,image_size=list(size),generation_seed=42,training_seed=42,policy='none',prediction_inputs='image and past predicted coordinates/crops only',models={})
    generated={};latest=[]
    for name,kind in [('control','best'),('improved','best'),('improved','last')]:
        variant=f'{name}_{kind}'
        path=suite/name/f'{kind}.pt';checkpoint,digest=stable_checkpoint(path)
        if row.patient_key in checkpoint['training_patient_keys']:raise ValueError('Validation patient is in training provenance')
        history=json.loads((suite/name/'history.json').read_text()) if (suite/name/'history.json').exists() else []
        completed=max([r['epoch'] for r in history]+[checkpoint['epoch']]);latest.append(completed)
        model=restore(checkpoint).eval()
        info=dict(checkpoint=str(path),checkpoint_sha256=digest,checkpoint_epoch=checkpoint['epoch'],latest_completed_epoch=completed,diagnostics={})
        for strategy in (('sample','argmax') if variant=='improved_last' else ('sample',)):
            points,gates=predict_image_only(model,image,image_id,strategy=strategy)
            generated[(variant,strategy)]=points;info['diagnostics'][strategy]=dict(gates=gates,predicted_normalized_coordinates=points.tolist())
        metadata['models'][variant]=info
        del model,checkpoint
    # Truth is read only after every prediction has been generated.
    raw=load_gazes()[row.file_name];legacy=reduce_gaze(raw,*size,k=16,min_distance=32);selected_segment=max(range(len(legacy)),key=lambda index:len(legacy[index]));truth=legacy[selected_segment]/size
    full=segments(raw,*size,min_distance=32);contiguous=full[selected_segment][:16]/size
    metadata['truth']=dict(legacy16=truth.tolist(),contiguous_first16=contiguous.tolist(),continuous_segment=selected_segment,min_distance_pixels=32)
    for (name,strategy),points in generated.items():metadata['models'][name]['diagnostics'][strategy].update(diagnostics(points,truth,size))
    destination=Path(output) if output else suite/'visual_checks';destination.mkdir(parents=True,exist_ok=True)
    tag=tag or f'epoch_{min(latest):02d}';stem=f'{tag}_case_{image_id}';png=destination/f'{stem}.png';json_path=destination/f'{stem}.metadata.json'
    if png.exists() or json_path.exists():raise FileExistsError(f'Visual check exists; use a new --tag: {stem}')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,13))
    panels=[('Observed legacy16',truth,None),('Observed contiguous first16',contiguous,None),
            ('control best sample',generated[('control_best','sample')],'control_best'),
            ('improved best sample',generated[('improved_best','sample')],'improved_best'),
            ('improved LAST sample',generated[('improved_last','sample')],'improved_last'),
            ('improved LAST argmax',generated[('improved_last','argmax')],'improved_last')]
    for axis,(label,points,variant) in zip(axes.flat,panels):
        if variant:
            info=metadata['models'][variant];title=f'{label}, seed42\ncheckpoint epoch{info["checkpoint_epoch"]}; latest complete{info["latest_completed_epoch"]}'
        else:title=f'{label} ({len(points)} samples)'
        draw_panel(axis,image,points,title)
    fig.suptitle(f'Validation case{image_id}: ongoing training visual check\nTime color: blue→red; green circle=start, pink X=end; numbered gaze samples',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(png,dpi=170);plt.close(fig)
    metadata['output_image']=str(png);metadata['created_utc']=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat();json_path.write_text(json.dumps(metadata,indent=2,allow_nan=False))
    print(json.dumps(dict(image=str(png),metadata=str(json_path)),ensure_ascii=False),flush=True)
    return png,json_path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suite',type=Path,default=ROOT/'runs/full_1011_seed42');parser.add_argument('--image-id',type=int,default=64);parser.add_argument('--output',type=Path);parser.add_argument('--tag');parser.add_argument('--threads',type=int,default=2);a=parser.parse_args()
    render(a.suite,a.image_id,a.output,a.tag,a.threads)
if __name__=='__main__':main()
