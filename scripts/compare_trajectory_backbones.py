"""Train all backbone variants under the same trajectory protocol and validate."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def execute(model,device,c,base):
    out=base/model;out.mkdir(parents=True,exist_ok=True)
    pipeline=ROOT/'scripts/pipeline.py'
    with (out/'run.log').open('w') as log:
        cmd=[sys.executable,str(pipeline),'train-gaze','--method','trajectory','--backbone',model,'--cache-visual',
             '--manifest',str(ROOT/'manifests/trajectory.csv'),'--device',device,'--output',str(out)]
        for key in ['seed','k','grid_x','grid_y','batch_size','epochs','patience','lr']:
            cmd += ['--'+key.replace('_','-'),str(c[key])]
        for key in ['decoder_dim','decoder_layers','decoder_heads','ff_multiplier','dropout']:
            if key in c:cmd+=['--'+key.replace('_','-'),str(c[key])]
        subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,str(pipeline),'generate','--checkpoint',str(out),'--manifest',str(ROOT/'manifests/trajectory.csv'),
                        '--device',device,'--output',str(out/'validation'),'--splits','val','--batch-size',str(c['batch_size']),
                        '--with-gaze-evaluation'],check=True,stdout=log,stderr=subprocess.STDOUT)
    summary=json.loads((out/'training_summary.json').read_text())
    metrics=pd.read_csv(out/'validation/trajectory_metrics.csv')
    summary.update(metrics.drop(columns='image_id').mean().to_dict())
    (out/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/trajectory_backbones.json'))
    p.add_argument('--devices',nargs='+',default=['cuda:0','cuda:1']);p.add_argument('--output',default=str(ROOT/'runs/trajectory_backbones_seed42'))
    a=p.parse_args();c=json.loads(Path(a.config).read_text());base=Path(a.output);base.mkdir(parents=True,exist_ok=True)
    (base/'experiment_config.json').write_text(json.dumps(c,indent=2))
    # A worker owns one GPU for its sequence of models.
    def worker(index):return [execute(name,a.devices[index],c,base) for name in c['backbones'][index::len(a.devices)]]
    with ThreadPoolExecutor(max_workers=len(a.devices)) as pool:
        results=[result for batch in pool.map(worker,range(len(a.devices))) for result in batch]
    pd.DataFrame(results).sort_values('normalized_dtw').to_csv(base/'comparison.csv',index=False)
    (base/'comparison.json').write_text(json.dumps(results,indent=2))
