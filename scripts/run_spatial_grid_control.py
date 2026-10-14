"""Image-only grid control with the same breast mask, K, grid, and fusion classifier."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
from PIL import Image
import subprocess
from src.data import ROOT
from src.constraints import breast_mask,grid_mask

if __name__=='__main__':
    base=ROOT/'runs/spatial_v2_seed42/grid_control';base.mkdir(parents=True,exist_ok=True)
    f=pd.read_csv(ROOT/'manifests/classification.csv');f=f[f.split.isin(['train','val'])]
    points={}
    ids=np.arange(24*48);normalized=np.stack([((ids%24)+.5)/24,((ids//24)+.5)/48],-1)
    for _,r in f.iterrows():
        with Image.open(ROOT/'data/image'/r.file_name) as source:mask=breast_mask(source)
        candidates=normalized[grid_mask(mask,(24,48))]*[r.width,r.height]
        index=int(((candidates-[r.width/2,r.height/2])**2).sum(1).argmin());chosen=[index]
        distance=((candidates-candidates[index])**2).sum(1)
        for _ in range(15):
            index=int(distance.argmax());chosen.append(index);distance=np.minimum(distance,((candidates-candidates[index])**2).sum(1))
        selected=candidates[chosen];d=np.linalg.norm(selected[:,None]-selected[None,:],axis=-1)
        if d[np.triu_indices(16,1)].min()<112:raise ValueError('Grid control does not satisfy the common distance budget')
        points[str(r.image_id)]=(selected/[r.width,r.height]).tolist()
    (base/'predictions.json').write_text(json.dumps(points))
    (base/'predictions.provenance.json').write_text(json.dumps(dict(method='trajectory',kind='image_only_grid_control',training_patient_keys=[],generated_image_ids=f.image_id.tolist(),
                    generation_policy='hard',generation_radius=112,generation_strategy='argmax',seed=42),indent=2))
    with (base/'run.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'scripts/pipeline.py'),'extract','--method','trajectory','--predictions',str(base/'predictions.json'),
                        '--keep-predicted-centers','--generation-policy','hard','--generation-radius','112','--generation-strategy','argmax',
                        '--output',str(base/'features'),'--device','cuda:0'],check=True,stdout=log,stderr=subprocess.STDOUT)
        subprocess.run([sys.executable,str(ROOT/'scripts/spatial_pipeline.py'),'fusion','--features',str(base/'features'),
                        '--output',str(base/'fusion'),'--epochs','12','--patience','4','--device','cuda:0'],check=True,stdout=log,stderr=subprocess.STDOUT)
