"""Leakage-safe three-policy classification comparison using large Swin-B cross-fitting."""
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
from src.data import ROOT,whole_tensor,seed_for
from src.models import trajectory_from_checkpoint
from src.constraints import breast_mask,grid_mask,rollout
from src.metrics import cluster_bootstrap
POLICIES={'none':'none','breast':'breast','hard112':'hard'}

def execute(command,log):subprocess.run(list(map(str,command)),check=True,stdout=log,stderr=subprocess.STDOUT)

def fold_worker(a):
    torch.set_num_threads(4);base=Path(a.output);fold=base/f'fold{a.worker_fold}';fold.mkdir(parents=True,exist_ok=True)
    model_path=(Path(a.reuse_folds_from)/f'fold{a.worker_fold}/model') if a.reuse_folds_from else fold/'model';pipeline=ROOT/'scripts/pipeline.py'
    if not a.reuse_folds_from:
        with (fold/'run.log').open('w') as log:
            command=[sys.executable,pipeline,'train-gaze','--method','trajectory','--backbone','swin_b','--decoder-dim',512,'--decoder-layers',6,
                 '--decoder-heads',8,'--ff-multiplier',4,'--dropout',.2,'--lr',.0003,'--batch-size',16,'--holdout-fold',a.worker_fold,
                 '--manifest',ROOT/'manifests/trajectory.csv','--output',model_path,'--device',a.device]
            if a.finetune:
                command+=['--finetune-backbone','--backbone-lr',.00001,'--lr',.0001,'--batch-size',8,'--epochs',20,'--patience',6,'--amp','--cache-images',
                          '--init-checkpoint',Path(a.frozen_folds_source)/f'fold{a.worker_fold}/model']
            else:command+=['--cache-visual']
            execute(command,log)
    c=torch.load(model_path/'best.pt',map_location='cpu',weights_only=False);device=torch.device(a.device)
    model=trajectory_from_checkpoint(c).to(device).eval()
    f=pd.read_csv(ROOT/'manifests/trajectory.csv');f=f[(f.split=='train')&(f.crossfit_fold==a.worker_fold)].sort_values('image_id')
    if set(f.patient_key)&set(c['training_patient_keys']):raise ValueError('Cross-fitting patient leak')
    data={p:{} for p in POLICIES}
    for offset in range(0,len(f),8):
        frame=f.iloc[offset:offset+8];images=[];allowed=[]
        for _,r in frame.iterrows():
            with Image.open(ROOT/'data/image'/r.file_name) as source:image=source.convert('L')
            images.append(whole_tensor(image));allowed.append(grid_mask(breast_mask(image),c['grid']))
        masks=torch.tensor(np.stack(allowed),device=device);sizes=torch.tensor(frame[['width','height']].to_numpy(),device=device,dtype=torch.float32)
        with torch.inference_mode():memory=model.memory(torch.stack(images).to(device))
        for policy,kind in POLICIES.items():
            torch.manual_seed(seed_for(a.seed,offset));result=rollout(model,memory,masks,sizes,kind=kind,radius=112,strategy=a.strategy)
            for i,(_,r) in enumerate(frame.iterrows()):data[policy][str(r.image_id)]=result['coords'][i][result['valid'][i]].cpu().tolist()
    metadata=dict(method='trajectory',backbone='swin_b',training_patient_keys=c['training_patient_keys'],selection_patient_keys=c['selection_patient_keys'],
                  generated_image_ids=f.image_id.tolist(),checkpoint=str(model_path.resolve()),checkpoint_sha256=hashlib.sha256((model_path/'best.pt').read_bytes()).hexdigest(),seed=a.seed)
    for policy,points in data.items():
        (fold/f'{policy}.json').write_text(json.dumps(points));(fold/f'{policy}.provenance.json').write_text(json.dumps(dict(metadata,policy=POLICIES[policy],radius=112,strategy=a.strategy),indent=2))
    print(f'Fold {a.worker_fold} completed ({len(f)} held-out images)',flush=True)

def main(a):
    if a.seed!=42:raise ValueError('This experiment reuses seed42 full-validation artifacts; generate matching artifacts before changing seed')
    base=Path(a.output);base.mkdir(parents=True,exist_ok=True)
    def folds(index):
        for fold in range(index,3,2):
            with (base/f'worker_fold{fold}.log').open('w') as log:
                command=[sys.executable,__file__,'--output',base,'--worker-fold',fold,'--device',f'cuda:{index}','--seed',a.seed,'--strategy',a.strategy]
                if a.finetune:command+=['--finetune','--frozen-folds-source',a.frozen_folds_source]
                if a.reuse_folds_from:command+=['--reuse-folds-from',a.reuse_folds_from]
                execute(command,log)
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(folds,range(2)))
    valbase=Path(a.validation_source)
    for policy in POLICIES:
        merged={};artifacts=[]
        paths=[base/f'fold{fold}/{policy}.json' for fold in range(3)]+[valbase/f'{a.strategy}_{policy}.json']
        for path in paths:
            data=json.loads(path.read_text());meta=json.loads(path.with_suffix('.provenance.json').read_text())
            if set(data)&set(merged):raise ValueError('Duplicate IDs in cross-fit predictions')
            meta.update(generation_policy=POLICIES[policy],generation_radius=112,generation_strategy=a.strategy)
            merged.update(data);artifacts.append(meta)
        if len(merged)!=469:raise ValueError('Expected all 376 train and 93 validation images')
        (base/f'{policy}.json').write_text(json.dumps(merged));(base/f'{policy}.provenance.json').write_text(json.dumps({'artifacts':artifacts},indent=2))
    def classifiers(index):
        result=[]
        for policy in list(POLICIES)[index::2]:
            out=base/policy;out.mkdir(exist_ok=True)
            with (out/'run.log').open('w') as log:
                execute([sys.executable,ROOT/'scripts/pipeline.py','extract','--method','trajectory','--keep-predicted-centers','--generation-policy',POLICIES[policy],
                         '--generation-strategy',a.strategy,'--generation-radius',112,'--predictions',base/f'{policy}.json','--output',out/'features','--device',f'cuda:{index}'],log)
                execute([sys.executable,ROOT/'scripts/pipeline.py','train-classifier','--features',out/'features','--output',out/'classifier','--device',f'cuda:{index}',
                         '--batch-size',32,'--epochs',50,'--patience',10],log)
            result.append(dict(policy=policy,**json.loads((out/'classifier/validation_metrics.json').read_text())))
        return result
    with ThreadPoolExecutor(max_workers=2) as pool:results=[r for batch in pool.map(classifiers,range(2)) for r in batch]
    (base/'comparison.json').write_text(json.dumps(results,indent=2))
    pd.DataFrame([dict(policy=r['policy'],**r['metrics'],auroc_ci_low=r['auroc_ci']['low'],auroc_ci_high=r['auroc_ci']['high']) for r in results]).to_csv(base/'comparison.csv',index=False)
    reference=pd.read_csv(base/'none/classifier/validation_predictions.csv');comparisons={}
    for policy in ['breast','hard112']:
        data=pd.read_csv(base/policy/'classifier/validation_predictions.csv');comparisons[policy]=cluster_bootstrap(data,reference,repetitions=1000,seed=a.seed)
    (base/'paired_auroc_difference.json').write_text(json.dumps(comparisons,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'runs/constraint_classification_seed42'))
    p.add_argument('--worker-fold',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--strategy',choices=['argmax','sample'],default='sample');p.add_argument('--reuse-folds-from')
    p.add_argument('--finetune',action='store_true');p.add_argument('--frozen-folds-source',default=str(ROOT/'runs/constraint_classification_seed42'))
    p.add_argument('--validation-source',default=str(ROOT/'runs/trajectory_constraints_seed42/swin_b'));a=p.parse_args()
    fold_worker(a) if a.worker_fold is not None else main(a)
