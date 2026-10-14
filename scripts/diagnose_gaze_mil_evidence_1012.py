"""Read-only TRAIN/VAL component and evidence-concentration diagnostics."""
import json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.train_gaze_mil_evidence_1012 import EvidenceMIL
from scripts.train_gaze_contextual_improvement_1012 import build_bags
from src.official_cbis_classifier_1012 import metrics,digest

def main():
    torch.set_num_threads(4);out=Path(sys.argv[1]);cache=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors'];bags=build_bags(t);results=[]
    for path in sorted(out.glob('*/best.pt')):
        c=torch.load(path,map_location='cpu',weights_only=False);n=c['normalization'];g=((t['global_features']-n['global_mean'])/n['global_std']).clamp(-10,10);p=((bags[c['arm']][0]-n['patch_mean'])/n['patch_std']).clamp(-10,10);m=EvidenceMIL(temperature=c['temperature']);m.load_state_dict(c['state']);m.eval()
        with torch.inference_mode():
            e=m.patch_head(p).squeeze(-1);pool=c['temperature']*(torch.logsumexp(e/c['temperature'],1)-math.log(16));gl=m.global_head(g).squeeze(-1);weights=(e/c['temperature']).softmax(1)
        record=dict(arm=c['arm'],temperature=c['temperature'],seed=c['seed'],splits={})
        for split in ['train','val']:
            ids=np.flatnonzero(f.split==split);labels=f.iloc[ids].label.values
            record['splits'][split]=dict(joint_auroc=metrics(labels,(gl[ids]+pool[ids]).sigmoid().numpy())['auroc'],global_component_auroc=metrics(labels,gl[ids].sigmoid().numpy())['auroc'],patch_component_auroc=metrics(labels,pool[ids].sigmoid().numpy())['auroc'],effective_patch_count=float((1/weights[ids].square().sum(1)).mean()),mean_max_patch_weight=float(weights[ids].max(1).values.mean()))
        results.append(record)
    (out/'component_diagnostics.json').write_text(json.dumps(dict(source_sha256=digest(__file__),results=results),indent=2));print(json.dumps(results))
if __name__=='__main__':main()
