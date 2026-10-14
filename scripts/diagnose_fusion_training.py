"""Plot observed optimization/generalization curves from the original comparison."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT

if __name__=='__main__':
    original=ROOT/'runs/grid_fusion_backbones_seed42';out=ROOT/'runs/fusion_hpo_seed42';out.mkdir(parents=True,exist_ok=True)
    names=['densenet161','efficientnet_b5','convnext_base','swin_b','vit_b_16'];fig,axes=plt.subplots(5,2,figsize=(10,13));rows=[]
    for i,name in enumerate(names):
        history=json.loads((original/name/'history.json').read_text());epochs=[r['epoch'] for r in history];loss=[r['train_loss'] for r in history];auc=[r['validation']['fusion_logits']['auroc'] for r in history]
        best=max(history,key=lambda r:r['validation']['fusion_logits']['auroc'])
        axes[i,0].plot(epochs,loss,'o-',color='steelblue');axes[i,0].set_title(name+' / train BCE');axes[i,0].set_ylim(0,.8)
        axes[i,1].plot(epochs,auc,'o-',color='darkorange');axes[i,1].set_title(name+' / validation AUROC (n=93)');axes[i,1].set_ylim(.5,.85)
        for ax in axes[i]:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
        rows.append(dict(backbone=name,first_train_loss=loss[0],last_train_loss=loss[-1],first_val_auroc=auc[0],last_val_auroc=auc[-1],peak_val_auroc=best['validation']['fusion_logits']['auroc'],peak_epoch=best['epoch']))
    fig.tight_layout();fig.savefig(out/'original_learning_curves.png',dpi=150);plt.close(fig);pd.DataFrame(rows).to_csv(out/'original_diagnosis.csv',index=False)
