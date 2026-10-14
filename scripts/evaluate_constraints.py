"""Validation-only comparison of image-only masks and visit exclusion policies."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT,whole_tensor,seed_for
from src.models import trajectory_from_checkpoint
from src.constraints import breast_mask,grid_mask,rollout,spatial_metrics
from src.metrics import dtw_distance

POLICIES={'none':('none',0),'breast':('breast',0),'hard64':('hard',64),'hard112':('hard',112),'hard224':('hard',224),'soft112':('soft',112)}

def worker(args):
    torch.set_num_threads(4);device=torch.device(args.device);base=Path(args.checkpoints)
    c=torch.load(base/args.backbone/'best.pt',map_location='cpu',weights_only=False)
    model=trajectory_from_checkpoint(c).to(device).eval()
    frame=pd.read_csv(ROOT/'manifests/trajectory.csv');frame=frame[frame.split=='val'].sort_values('image_id').reset_index(drop=True)
    if set(frame.patient_key)&set(c['training_patient_keys']):raise ValueError('Validation patient leakage')
    if args.limit:frame=frame.head(args.limit)
    truth=json.loads((ROOT/'manifests/gaze_sequences.json').read_text())
    coco=json.loads((ROOT/'manifests/instances_clipped.json').read_text());ann={int(r.image_id):[] for _,r in frame.iterrows()}
    for box in coco['annotations']:
        if box['image_id'] in ann:ann[box['image_id']].append(box['bbox'])
    out=Path(args.output)/args.backbone;out.mkdir(parents=True,exist_ok=True)
    predictions={};records=[];qc=[]
    strategies=['argmax','sample']
    for offset in range(0,len(frame),args.batch_size):
        batch=frame.iloc[offset:offset+args.batch_size];images=[];masks=[];inputs=[]
        for _,r in batch.iterrows():
            with Image.open(ROOT/'data/image'/r.file_name) as source:image=source.convert('L')
            mask=breast_mask(image);images.append(image);masks.append(mask);inputs.append(whole_tensor(image))
            target=np.array(max(truth[str(r.image_id)]['segments'],key=len));xy=np.clip((target*[r.width,r.height]).astype(int),[0,0],[r.width-1,r.height-1])
            qc.append(dict(image_id=int(r.image_id),mask_fraction=float(mask.mean()),observed_gaze_outside=float((~mask[xy[:,1],xy[:,0]]).mean())))
        allowed=torch.tensor(np.stack([grid_mask(m,c['grid']) for m in masks]),device=device)
        if not allowed.any(1).all():raise ValueError('Breast mask has no representable grid cell')
        sizes=torch.tensor(batch[['width','height']].to_numpy(),device=device,dtype=torch.float32)
        with torch.inference_mode():memory=model.memory(torch.stack(inputs).to(device))
        for strategy in strategies:
            for policy,(kind,radius) in POLICIES.items():
                # Same random seed per batch across policies; model and mask are fixed.
                torch.manual_seed(seed_for(args.seed,offset))
                result=rollout(model,memory,allowed,sizes,kind,radius,strategy=strategy)
                key=f'{strategy}_{policy}';predictions.setdefault(key,{})
                for i,(_,r) in enumerate(batch.iterrows()):
                    p=result['coords'][i][result['valid'][i]].cpu().numpy()
                    if not len(p):raise ValueError('No valid predicted points')
                    predictions[key][str(r.image_id)]=p.tolist()
                    target=np.array(max(truth[str(r.image_id)]['segments'],key=len))
                    record=dict(backbone=args.backbone,strategy=strategy,policy=policy,image_id=int(r.image_id),patient_key=r.patient_key,
                                normalized_dtw=dtw_distance(target,p),relaxed_steps=int(result['relaxations'][i]),**spatial_metrics(p,masks[i],ann[int(r.image_id)]))
                    records.append(record)
        print(f'{args.backbone}: {min(offset+args.batch_size,len(frame))}/{len(frame)}',flush=True)
    df=pd.DataFrame(records);df.to_csv(out/'per_image_metrics.csv',index=False);pd.DataFrame(qc).to_csv(out/'mask_audit.csv',index=False)
    provenance=dict(method='trajectory',backbone=args.backbone,model_config=c.get('model_config',{}),training_patient_keys=c['training_patient_keys'],selection_patient_keys=c['selection_patient_keys'],generated_image_ids=frame.image_id.tolist(),
                    checkpoint=str((base/args.backbone).resolve()),checkpoint_sha256=hashlib.sha256((base/args.backbone/'best.pt').read_bytes()).hexdigest(),seed=args.seed,
                    note='Inference-only constraints; boxes and observed gaze used only for post-hoc evaluation')
    for key,data in predictions.items():
        (out/f'{key}.json').write_text(json.dumps(data));(out/f'{key}.provenance.json').write_text(json.dumps(dict(provenance,condition=key),indent=2))
    summary=df.groupby(['backbone','strategy','policy']).mean(numeric_only=True).drop(columns='image_id').reset_index();summary.to_csv(out/'summary.csv',index=False)
    # Fixed first 3 images, no selection by improvement.
    fig,axs=plt.subplots(3,3,figsize=(11,17))
    for row,(_,r) in enumerate(frame.head(3).iterrows()):
        with Image.open(ROOT/'data/image'/r.file_name) as source:image=source.convert('L')
        mask=breast_mask(image)
        for col,policy in enumerate(['none','breast','hard112']):
            ax=axs[row,col];ax.imshow(image,cmap='gray');ax.contour(mask,levels=[.5],colors=['lime'],linewidths=.5)
            p=np.array(predictions[f'sample_{policy}'][str(r.image_id)])*[r.width,r.height]
            ax.plot(p[:,0],p[:,1],color='cyan',alpha=.5,linewidth=.7);ax.scatter(p[:,0],p[:,1],c=np.arange(len(p)),cmap='turbo',s=14)
            ax.set_title(f'{policy} | image {r.image_id}');ax.axis('off')
    fig.tight_layout();fig.savefig(out/'examples.png',dpi=140);plt.close(fig)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoints',default=str(ROOT/'runs/trajectory_large_seed42'))
    p.add_argument('--output',default=str(ROOT/'runs/trajectory_constraints_seed42'));p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cuda:0');p.add_argument('--backbone',default='swin_b');p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--all-models',action='store_true');a=p.parse_args()
    if a.all_models:
        models=['resnet50','resnet101','convnext_base','swin_b'];base=Path(a.output);base.mkdir(parents=True,exist_ok=True)
        def launch(index):
            result=[]
            for name in models[index::2]:
                out=base/name;out.mkdir(exist_ok=True)
                with (out/'run.log').open('w') as log:
                    subprocess.run([sys.executable,__file__,'--checkpoints',a.checkpoints,'--output',a.output,'--backbone',name,'--device',f'cuda:{index}',
                                    '--seed',str(a.seed),'--batch-size',str(a.batch_size)],check=True,stdout=log,stderr=subprocess.STDOUT)
                result.append(pd.read_csv(out/'summary.csv'))
            return result
        with ThreadPoolExecutor(max_workers=2) as pool:tables=[t for batch in pool.map(launch,range(2)) for t in batch]
        pd.concat(tables).to_csv(base/'comparison.csv',index=False)
    else:worker(a)
