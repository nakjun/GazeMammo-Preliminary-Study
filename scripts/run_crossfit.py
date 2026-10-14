"""Build train-heldout and validation predictions for leakage-safe stage D."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def call(*args):subprocess.run([sys.executable,str(ROOT/'scripts/pipeline.py'),*map(str,args)],check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['trajectory','heatmap'],default='trajectory')
    p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--folds',type=int,default=3);p.add_argument('--k',type=int,default=16);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--manifest',default=str(ROOT/'manifests/trajectory.csv'))
    p.add_argument('--grid-x',type=int,default=24);p.add_argument('--grid-y',type=int,default=48)
    p.add_argument('--backbone',choices=['resnet50','densenet121','efficientnet_b0','convnext_tiny','swin_t','resnet101','convnext_base','swin_b'],default='resnet50')
    p.add_argument('--cache-visual',action='store_true')
    p.add_argument('--decoder-dim',type=int,default=128);p.add_argument('--decoder-layers',type=int,default=2)
    p.add_argument('--decoder-heads',type=int,default=4);p.add_argument('--ff-multiplier',type=int,default=2);p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--batch-size',type=int,default=8)
    a=p.parse_args();base=Path(a.output);base.mkdir(parents=True,exist_ok=True);merged={};provenance=[]
    import pandas as pd
    frame=pd.read_csv(a.manifest)
    actual=set(frame.loc[frame.split=='train','crossfit_fold'])
    if actual!=set(range(a.folds)):raise ValueError(f'Manifest fold IDs {actual} do not match --folds {a.folds}')
    # No final test generation here. Hyperparameters must be locked first.
    for fold in [*range(a.folds),None]:
        name=f'fold{fold}' if fold is not None else 'full_train';model=base/name;generated=base/(name+'_predictions')
        common=['--method',a.method,'--output',model,'--device',a.device,'--epochs',a.epochs,'--k',a.k,'--seed',a.seed,
                '--manifest',a.manifest,'--grid-x',a.grid_x,'--grid-y',a.grid_y,'--backbone',a.backbone,
                '--decoder-dim',a.decoder_dim,'--decoder-layers',a.decoder_layers,'--decoder-heads',a.decoder_heads,
                '--ff-multiplier',a.ff_multiplier,'--dropout',a.dropout,'--lr',a.lr,'--batch-size',a.batch_size]
        if a.cache_visual:common+=['--cache-visual']
        if fold is not None:common+=['--holdout-fold',fold]
        call('train-gaze',*common)
        options=['--checkpoint',model,'--output',generated,'--device',a.device,'--seed',a.seed,
                 '--splits','train' if fold is not None else 'val','--manifest',a.manifest]
        if fold is not None:options+=['--holdout-fold',fold]
        call('generate',*options)
        new=json.loads((generated/'predictions.json').read_text())
        if set(new)&set(merged):raise ValueError('Duplicated prediction IDs')
        merged.update(new);provenance.append(json.loads((generated/'predictions.provenance.json').read_text()))
    (base/'predictions.json').write_text(json.dumps(merged))
    (base/'predictions.provenance.json').write_text(json.dumps({'artifacts':provenance},indent=2))
