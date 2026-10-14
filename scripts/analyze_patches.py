"""Post-hoc lesion coverage of already selected patches; never selects with boxes."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import pandas as pd
import torch
from src.data import ROOT

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cache=torch.load(Path(a.features)/'features.pt',weights_only=False,map_location='cpu')
    if cache['config']['method']=='whole':raise ValueError('Whole-image coverage is not patch selection')
    frame=pd.read_csv(Path(a.features)/'manifest.csv')
    coco=json.loads((ROOT/'manifests/instances_clipped.json').read_text());rows=[]
    size=cache['config']['crop_size']
    for idx,r in frame.iterrows():
        centers=cache['centers'][idx][cache['mask'][idx]].numpy()*[r.width,r.height]
        patches=np.c_[centers-size/2,centers+size/2];lesions=[x for x in coco['annotations'] if x['image_id']==r.image_id]
        hits=[];coverage=[]
        for ann in lesions:
            x,y,w,h=ann['bbox'];intersection=np.maximum(0,np.minimum(patches[:,2],x+w)-np.maximum(patches[:,0],x))*np.maximum(0,np.minimum(patches[:,3],y+h)-np.maximum(patches[:,1],y))
            hits.append(bool((intersection>0).any()));coverage.append(float(intersection.max()/(w*h)))
        overlap=[]
        for i in range(len(patches)):
            for j in range(i):
                intersection=np.maximum(0,np.minimum(patches[i,2:],patches[j,2:])-np.maximum(patches[i,:2],patches[j,:2])).prod()
                overlap.append(float(intersection/size**2))
        rows.append(dict(image_id=r.image_id,patient_key=r.patient_key,split=r.split,label=r.label,
                         any_lesion_hit=bool(any(hits)),lesion_hit_rate=float(np.mean(hits)),
                         mean_best_single_patch_lesion_coverage=float(np.mean(coverage)),
                         mean_pairwise_patch_overlap=float(np.mean(overlap)) if overlap else 0))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);d=pd.DataFrame(rows);d.to_csv(out/'patch_coverage.csv',index=False)
    (out/'summary.json').write_text(json.dumps(d.groupby('split')[['any_lesion_hit','lesion_hit_rate','mean_best_single_patch_lesion_coverage','mean_pairwise_patch_overlap']].mean().to_dict(),indent=2))
