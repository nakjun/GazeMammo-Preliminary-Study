"""Run the stage-B development experiment; never reads test images."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def run(command):
    print(' '.join(map(str,command)),flush=True)
    subprocess.run(list(map(str,command)),check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(ROOT/'configs/pilot.json'))
    p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0)
    p.add_argument('--name',default='baseline_seed42');a=p.parse_args();c=json.loads(Path(a.config).read_text())
    base=ROOT/'runs'/a.name;base.mkdir(parents=True,exist_ok=True)
    pipeline=ROOT/'scripts/pipeline.py';summary=[]
    for method in c['methods']:
        features=base/f'{method}_features';training=base/method
        if (features/'features.pt').exists():
            cached_config=json.loads((features/'config.json').read_text())
            for key,expected in dict(method=method,k=c['k'],crop_size=c['crop_size'],min_distance=c['min_distance'],seed=c['seed'],limit=a.limit).items():
                if cached_config[key]!=expected:raise ValueError(f'Existing feature cache differs in {key}; choose a new --name')
        if not (features/'features.pt').exists():
            run([sys.executable,pipeline,'extract','--method',method,'--output',features,'--device',a.device,
                 '--seed',c['seed'],'--k',c['k'],'--crop-size',c['crop_size'],'--min-distance',c['min_distance'],'--limit',a.limit])
        run([sys.executable,pipeline,'train-classifier','--features',features,'--output',training,'--device',a.device,
             '--seed',c['seed'],'--epochs',c['epochs'],'--patience',c['patience'],'--batch-size',c['batch_size'],
             '--lr',c['lr'],'--aggregation',c['aggregation'],'--bootstrap',c['bootstrap']])
        result=json.loads((training/'validation_metrics.json').read_text());summary.append(dict(method=method,**result))
    (base/'comparison.json').write_text(json.dumps(summary,indent=2))
    import pandas as pd
    pd.DataFrame([dict(method=x['method'],**x['metrics'],ci_low=x['auroc_ci']['low'],ci_high=x['auroc_ci']['high']) for x in summary]).to_csv(base/'comparison.csv',index=False)
    print(json.dumps(summary,indent=2),flush=True)
