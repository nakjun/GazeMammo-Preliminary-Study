"""Posthoc geometry feasibility bounds; never selects generation actions or model settings."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_proposal_policy_1011 import rectangles,coverage


def analyze(output):
    frame=pd.read_csv(ROOT/'manifests/trajectory.csv');val=frame[frame.split=='val'].set_index('image_id')
    boxes={int(i):[] for i in val.index}
    for a in json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations']:
        if int(a['image_id']) in boxes:boxes[int(a['image_id'])].append(a['bbox'])
    grid=(24,48);ids=np.arange(np.prod(grid));coords=np.stack([((ids%grid[0])+.5)/grid[0],((ids//grid[0])+.5)/grid[1]],axis=1)
    results={};records=[]
    for crop in (224,256,320,384,448):
        per_image=[];lesion_hits=[];unreachable=[]
        for image_id,row in val.iterrows():
            if not boxes[image_id]:continue
            candidates=rectangles(coords,[row.width,row.height],crop);hits=[]
            for lesion_index,box in enumerate(boxes[image_id]):
                maximum=float(coverage(candidates,box).max());hit=bool(maximum>=.5);hits.append(hit);lesion_hits.append(hit)
                records.append(dict(crop_size=crop,image_id=int(image_id),patient_key=row.patient_key,lesion_index=lesion_index,max_achievable_coverage=maximum,actual50_reachable=int(hit)))
                if not hit:unreachable.append(dict(image_id=int(image_id),lesion_index=lesion_index,max_achievable_coverage=maximum))
            per_image.append(float(np.mean(hits)))
        bound=float(np.mean(per_image));results[str(crop)]=dict(image_macro_actual50_upper_bound=bound,lesion_macro_actual50_upper_bound=float(np.mean(lesion_hits)),known_images=len(per_image),known_lesions=len(lesion_hits),unreachable_lesions=unreachable,required_fraction_of_upper_bound_for85pct=float(.85/bound),crop_area_relative_to224=float((crop/224)**2))
    result=dict(protocol='Posthoc GT-only feasibility bound on fixed val93; boxless unknown; same24x48grid. Each lesion independently receives its optimal single crop. This is an optimistic geometry upper bound, not achieved16patch recall or independent validation; model inputs/generation are unchanged.',goal=dict(physical_actual50_image_macro=.85,legacy16_validation_nll=5.3),crop_sizes=results)
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'oracle_bounds.json').write_text(json.dumps(result,indent=2));pd.DataFrame(records).to_csv(output/'oracle_per_lesion.csv',index=False)
    print(json.dumps({c:{k:v for k,v in r.items() if k!='unreachable_lesions'} for c,r in results.items()},indent=2));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='runs/proposal_visit_1011/oracle_feasibility');args=parser.parse_args();analyze(args.output)
