"""Small real-data integration run, never a performance experiment."""
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'runs/integration_smoke';out.mkdir(parents=True,exist_ok=True)
f=pd.read_csv(ROOT/'manifests/classification.csv')
train=f[f.split=='train'].groupby(['crossfit_fold','label'],group_keys=False).head(2)
val=f[f.split=='val'].groupby('label',group_keys=False).head(4)
small=pd.concat([train,val]).sort_values('image_id');manifest=out/'manifest.csv';small.to_csv(manifest,index=False)

def run(script,*args):
    cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)]
    print(' '.join(cmd),flush=True);subprocess.run(cmd,check=True)

run('run_crossfit.py','--manifest',manifest,'--device','cpu','--epochs',1,'--k',4,'--grid-x',8,'--grid-y',16,'--output',out/'crossfit')
run('pipeline.py','extract','--method','trajectory','--manifest',manifest,'--device','cpu','--k',4,'--predictions',out/'crossfit/predictions.json','--output',out/'features')
for aggregation in ['mil','sequence','shuffled']:
    run('pipeline.py','train-classifier','--manifest',manifest,'--features',out/'features','--device','cpu','--epochs',1,'--bootstrap',20,'--aggregation',aggregation,'--output',out/aggregation)
run('pipeline.py','evaluate','--manifest',manifest,'--features',out/'features','--checkpoint',out/'mil','--device','cpu','--bootstrap',20,'--output',out/'evaluation')
run('analyze_patches.py','--features',out/'features','--output',out/'coverage')
(out/'completed.json').write_text(json.dumps({'status':'passed','train_images':len(train),'validation_images':len(val),'warning':'one-epoch integration smoke; not research performance'},indent=2))
