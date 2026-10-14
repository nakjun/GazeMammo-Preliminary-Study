"""Confirm actual finetuning, fixed BN statistics, patient split, and inference parity."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import re
import torch
import pandas as pd
from src.data import ROOT
from src.fusion_backbones import SPECS

if __name__=='__main__':
    torch.set_num_threads(4);base=ROOT/'runs/grid_fusion_backbones_seed42';report={}
    frame=pd.read_csv(ROOT/'manifests/classification.csv')
    train=set(frame[frame.split=='train'].patient_key);forbidden=set(frame[frame.split!='train'].patient_key)
    example=frame[frame.split=='val'].sort_values('image_id').iloc[0]
    for name,(_,weight) in SPECS.items():
        c=torch.load(base/name/'best.pt',map_location='cpu',weights_only=False)
        original=torch.load(Path(torch.hub.get_dir())/'checkpoints'/weight.url.rsplit('/',1)[-1],map_location='cpu',weights_only=True)
        if name=='densenet161':
            pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            original={pattern.sub(r'\1\2',key):value for key,value in original.items()}
        key='features.conv0.weight' if name=='densenet161' else 'conv_proj.weight' if name=='vit_b_16' else 'features.0.0.weight'
        delta=float((c['state']['visual.'+key]-original[key]).norm());assert delta>0
        differences=[]
        for key,value in c['state'].items():
            if key.startswith('visual.') and key.endswith(('running_mean','running_var')):
                assert key[len('visual.'):] in original
                differences.append(float((value-original[key[len('visual.'):]]).abs().max()))
        assert all(v==0 for v in differences)
        assert set(c['training_patient_keys'])==train and not train&forbidden
        single=json.loads((ROOT/'results'/f'grid_{name}_single_image.json').read_text())
        preds=pd.read_csv(base/name/'fusion_logits_predictions.csv')
        expected=float(preds.loc[preds.image_id==example.image_id,'probability'].iloc[0])
        diff=abs(single['probabilities']['fusion_logits']-expected);assert diff<1e-4
        report[name]=dict(selected_epoch=c['epoch'],first_conv_weight_change_l2=delta,bn_stat_buffers_compared=len(differences),
            bn_running_stats_max_change=max(differences,default=0),training_patients=len(train),heldout_patients_disjoint=True,
            singleton_prediction_absolute_difference=diff)
        print(name,report[name],flush=True)
    (base/'finetuning_audit.json').write_text(json.dumps(report,indent=2)+'\n')
