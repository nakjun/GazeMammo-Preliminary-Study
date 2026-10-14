"""Posthoc diagnostic only: reachable GT lesions, source ranking and hardNMS."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from evaluate_proposal_policy_1011 import rectangles,coverage,iou
from src.proposal_utility_1011 import file_sha256


def audit(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Audit output already contains artifacts')
    targets={(83,0),(367,0),(509,1),(772,0)};ids={i for i,_ in targets};boxes={i:[] for i in ids};annotation=ROOT/'manifests/instances_clipped.json'
    for r in json.loads(annotation.read_text())['annotations']:
        if int(r['image_id']) in boxes:boxes[int(r['image_id'])].append(r['bbox'])
    records=[]
    with Path(a.traces).open() as source:
        for line in source:
            t=json.loads(line);image_id=int(t['image_id']);budget=t['requested_roi_budget']
            if image_id not in ids or budget not in (12,16) or t['offset_strength']!=.5 or t['max_crop_iou']!=.1:continue
            if t['offset_architecture']!='spatial' or t.get('relevance_power',0)!=0 or t['seed']!=42:continue
            candidates=rectangles(t['candidate_coords'],np.array(t['image_size']),224);actual=rectangles(t['coordinates'],np.array(t['image_size']),224);scores=np.asarray(t['relevance']);order=np.argsort(-scores,kind='stable');ranks=np.empty(len(order),dtype=int);ranks[order]=np.arange(1,len(order)+1);chosen=[];status={};first_blocker={};survivors=[]
            for index in order:
                index=int(index);blocking=[j for j in chosen if iou(candidates[index],candidates[j])>.1]
                if blocking:status[index]='suppressed';first_blocker[index]=blocking[0]
                else:chosen.append(index);status[index]='kept';survivors.append(index)
            if survivors[:budget]!=t['reserved_candidate_indices']:raise ValueError('Saved reservation does not match reconstructed actual shiftedNMS')
            selected=t['reserved_candidate_indices'];last_rank=int(max(ranks[selected]))
            for _,lesion in sorted(x for x in targets if x[0]==image_id):
                box=boxes[image_id][lesion];cand_cov=coverage(candidates,box);actual_cov=coverage(actual,box);adequate=[int(j) for j in order if cand_cov[j]>=.5];first=adequate[0] if adequate else None;blocker=first_blocker.get(first);prior_selected=[j for j in selected if ranks[j]<ranks[first]] if first is not None else [];blocking_selected=[j for j in prior_selected if iou(candidates[first],candidates[j])>.1] if first is not None else [];kept_adequate=[j for j in survivors if cand_cov[j]>=.5]
                row=dict(image_id=image_id,lesion_index=lesion,policy=t['policy'],budget=budget,box=box,actual_max_coverage=float(actual_cov.max()),actual_path_hit50=bool(actual_cov.max()>=.5),shifted_pool_oracle=float(cand_cov.max()),adequate_candidate_count=len(adequate),first_adequate_index=first,first_adequate_raw_rank=int(ranks[first]) if first is not None else None,first_adequate_score=float(scores[first]) if first is not None else None,first_adequate_coverage=float(cand_cov[first]) if first is not None else None,first_adequate_selected=first in selected if first is not None else False,first_adequate_suppressed=bool(blocking_selected),first_adequate_budget_exhausted=bool(first is not None and ranks[first]>last_rank),first_suppressor_index=blocking_selected[0] if blocking_selected else None,first_suppressor_reserved_step=(selected.index(blocking_selected[0])+1) if blocking_selected else None,first_suppressor_raw_rank=int(ranks[blocking_selected[0]]) if blocking_selected else None,first_suppressor_iou=float(iou(candidates[first],candidates[blocking_selected[0]])) if blocking_selected else None,first_suppressor_gt_coverage=float(cand_cov[blocking_selected[0]]) if blocking_selected else None,unlimited_nms_first_adequate_kept_index=kept_adequate[0] if kept_adequate else None,unlimited_nms_first_adequate_kept_position=survivors.index(kept_adequate[0])+1 if kept_adequate else None,adequate_before_budget_suppressed_count=int(sum(ranks[j]<=last_rank and status[j]=='suppressed' for j in adequate)),adequate_raw_candidates_after_budget_count=int(sum(ranks[j]>last_rank for j in adequate)),selected_gt_coverages=[float(cand_cov[j]) for j in selected],selected_source_indices=selected,selected_raw_ranks=[int(ranks[j]) for j in selected])
                records.append(row)
    if len(records)!=8:raise ValueError(f'Expected4lesionsx2budgets, found{len(records)}')
    payload=dict(protocol='Posthoc source-rank and hardNMS audit, not an inference policy; no image-specific rule created',source_trace_sha256=file_sha256(a.traces),annotation_sha256=file_sha256(annotation),threshold=.5,crop_size=224,nms_threshold=.1,records=records);(output/'diagnostic.json').write_text(json.dumps(payload,indent=2));pd.DataFrame(records).to_csv(output/'diagnostic.csv',index=False)
    print(pd.DataFrame(records)[['image_id','lesion_index','budget','actual_max_coverage','shifted_pool_oracle','first_adequate_raw_rank','first_adequate_suppressed','first_suppressor_index','first_suppressor_gt_coverage','unlimited_nms_first_adequate_kept_position']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--traces',default='runs/dual_target_1011/center_offset_spatial_calibration_seed42/traces.jsonl');p.add_argument('--output',default='runs/dual_target_1011/center_offset_p0_reachable_misses_audit');audit(p.parse_args())
