"""Ordered spatial-head ablations followed by leakage-safe static/foveal fusion comparison."""
import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PIPE=ROOT/'scripts/spatial_pipeline.py'

def run(command,log):subprocess.run(list(map(str,command)),check=True,stdout=log,stderr=subprocess.STDOUT)
def training(base,name,device,phase,fpn=False,foveal=False,init=None,fold=None,sigma=96,epochs=15):
    out=base/name;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.log').open('w') as log:
        cmd=[sys.executable,PIPE,'train','--output',out,'--device',device,'--phase',phase,'--sigma',sigma,'--epochs',epochs,'--batch-size',2 if foveal else 8]
        if fpn:cmd+=['--fpn']
        if foveal:cmd+=['--foveal']
        if init:cmd+=['--init',init]
        if fold is not None:cmd+=['--holdout-fold',fold]
        run(cmd,log)
    return out
def generation(out,model,device,fold=None,prior=False):
    out.mkdir(parents=True,exist_ok=True)
    with (out/'run.log').open('w') as log:
        cmd=[sys.executable,PIPE,'generate','--checkpoint',model,'--output',out,'--device',device]
        if fold is not None:cmd+=['--split','train','--holdout-fold',fold]
        if prior:cmd+=['--prior-only']
        run(cmd,log)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'runs/spatial_v2_seed42'));a=p.parse_args();base=Path(a.output);base.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        pres=list(pool.map(lambda x:training(base,x[0],x[1],'pretrain',fpn=x[2],epochs=8),[('pretrain_plain','cuda:0',False),('pretrain_fpn','cuda:1',True)]))
    configs=[('spatial_hard',False,False,0,pres[0]),('spatial_soft',False,False,96,pres[0]),('fpn_soft',True,False,96,pres[1]),('foveal_soft',True,True,96,pres[1])]
    def sequence_worker(index):
        for name,fpn,foveal,sigma,init in configs[index::2]:
            model=training(base,name,f'cuda:{index}','sequence',fpn,foveal,init,sigma=sigma,epochs=15)
            generation(model/'validation',model,f'cuda:{index}')
    with ThreadPoolExecutor(2) as pool:list(pool.map(sequence_worker,range(2)))
    generation(base/'static_validation',pres[1],'cuda:0',prior=True)
    records=[]
    for name,_,_,_,_ in configs:
        metrics=pd.read_csv(base/name/'validation/metrics.csv');records.append(dict(model=name,**metrics.drop(columns=['image_id','patient_key']).mean().to_dict()))
    metrics=pd.read_csv(base/'static_validation/metrics.csv');records.append(dict(model='static_heatmap',**metrics.drop(columns=['image_id','patient_key']).mean().to_dict()))
    pd.DataFrame(records).to_csv(base/'trajectory_comparison.csv',index=False)
    # Repeat pretraining within each fold; never warm-start from a model that saw held-out patients.
    def cf_worker(index):
        for fold in range(index,3,2):
            root=base/f'fold{fold}';pre=training(root,'pretrain',f'cuda:{index}','pretrain',fpn=True,fold=fold,epochs=8)
            model=training(root,'foveal',f'cuda:{index}','sequence',fpn=True,foveal=True,init=pre,fold=fold,epochs=15)
            generation(root/'static',pre,f'cuda:{index}',fold=fold,prior=True)
            generation(root/'sequence',model,f'cuda:{index}',fold=fold)
    with ThreadPoolExecutor(2) as pool:list(pool.map(cf_worker,range(2)))
    for policy in ['static','sequence']:
        merged={};artifacts=[]
        paths=[base/f'fold{fold}/{policy}/predictions.json' for fold in range(3)]+[base/('static_validation' if policy=='static' else 'foveal_soft/validation')/'predictions.json']
        for path in paths:
            points=json.loads(path.read_text());meta=json.loads(path.with_suffix('.provenance.json').read_text())
            if set(points)&set(merged):raise ValueError('Overlapping prediction IDs')
            merged.update(points);artifacts.append(meta)
        if len(merged)!=469:raise ValueError('Missing train/validation predictions')
        (base/f'{policy}.json').write_text(json.dumps(merged));(base/f'{policy}.provenance.json').write_text(json.dumps({'artifacts':artifacts},indent=2))
    def classify(index):
        policy=['static','sequence'][index];out=base/(policy+'_classification');out.mkdir(exist_ok=True)
        with (out/'run.log').open('w') as log:
            run([sys.executable,ROOT/'scripts/pipeline.py','extract','--method','trajectory','--predictions',base/f'{policy}.json','--keep-predicted-centers','--generation-policy','hard',
                 '--generation-radius',112,'--generation-strategy','sample','--output',out/'features','--device',f'cuda:{index}'],log)
            run([sys.executable,PIPE,'fusion','--features',out/'features','--output',out/'fusion','--epochs',12,'--patience',4,'--device',f'cuda:{index}'],log)
    with ThreadPoolExecutor(2) as pool:list(pool.map(classify,range(2)))
    save={'status':'complete','test_used':False,'fold_pretraining_excludes_heldout_patients':True}
    (base/'completed.json').write_text(json.dumps(save,indent=2))
