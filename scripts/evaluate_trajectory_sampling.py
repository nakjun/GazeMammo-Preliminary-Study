"""Evaluate probabilistic rollout separately from the argmax backbone comparison."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',default=str(ROOT/'runs/trajectory_backbones_seed42'))
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1']);a=p.parse_args();base=Path(a.runs)
    if not (base/'comparison.csv').exists():raise ValueError('Complete the argmax comparison first')
    c=json.loads((base/'experiment_config.json').read_text())
    def worker(index):
        summaries=[]
        for name in c['backbones'][index::len(a.devices)]:
            out=base/name/'validation_sample';out.mkdir(parents=True,exist_ok=True)
            with (out/'run.log').open('w') as log:
                subprocess.run([sys.executable,str(ROOT/'scripts/pipeline.py'),'generate','--checkpoint',str(base/name),
                   '--manifest',str(ROOT/'manifests/trajectory.csv'),'--device',a.devices[index],'--output',str(out),
                   '--splits','val','--batch-size',str(c['batch_size']),'--with-gaze-evaluation','--strategy','sample',
                   '--temperature','1.0','--seed',str(c['seed'])],check=True,stdout=log,stderr=subprocess.STDOUT)
            metrics=pd.read_csv(out/'trajectory_metrics.csv');summaries.append(dict(backbone=name,**metrics.drop(columns='image_id').mean().to_dict()))
        return summaries
    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:rows=[r for batch in pool.map(worker,range(len(a.devices))) for r in batch]
    pd.DataFrame(rows).to_csv(base/'sampling_comparison.csv',index=False)
    print(json.dumps(rows,indent=2))
