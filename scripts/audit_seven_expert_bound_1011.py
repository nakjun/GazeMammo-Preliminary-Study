"""Read-only target-aware scalar bound; no oracle predictions or weights saved."""
import json,time,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from audit_expert_convex_bounds_1011 import sha,check
from gaze_nll_adaptation_1011 import routing

def main():
 torch.set_num_threads(4);started=time.monotonic();out=ROOT/'runs/dual_target_1011/seven_expert_scalar_bound';out.mkdir(exist_ok=True);check(not (out/'report.json').exists(),'Completed audit must not be overwritten')
 paths=dict(control='human_control_raw_cache_for_ensemble.pt',improved='human_improved_raw_cache.pt',clinical_DG='deepgaze_clinical_best_raw_cache.pt',teacherKL_aug_DINO='dino/gaze_augmented_teacherkl_seed42/best_raw_cache.pt',MDN_product='gaze_mdn_product_seed42/corrected_raw_cache.pt',crossview='gaze_crossview_seed42/fixed_export/best_raw_cache.pt',Mammo='mammo_clip_public/gaze_seed42/best_raw_cache.pt')
 allowed={int(r['image_id']):r for r in routing()};check(len(allowed)==505,'Canonical505');maps=[];sources=[];normmax=0.
 for name,relative in paths.items():
  p=ROOT/'runs/dual_target_1011'/relative;h=sha(p);d=torch.load(p,map_location='cpu',weights_only=False);check(d['grid']==[24,48] and len(d['records'])==505,'Grid/count');m={int(r['image_id']):r for r in d['records']};check(set(m)==set(allowed) and len(m)==505,'Exactunique505');check(sha(d['base_checkpoint'])==d['base_checkpoint_sha256'],'SourcecheckpointSHA');maps.append(m);sources.append(dict(name=name,path=str(p.relative_to(ROOT)),sha256=h,checkpoint=d['base_checkpoint'],checkpoint_sha256=d['base_checkpoint_sha256']))
 trainpatients={r['patient_key'] for r in allowed.values() if r['split']=='train'};valpatients={r['patient_key'] for r in allowed.values() if r['split']=='val'};check(not trainpatients.intersection(valpatients),'Patientoverlap');losses={'train':[],'val':[]};steps={'train':0,'val':0}
 for identity,route in allowed.items():
  refs=[m[identity] for m in maps];r=refs[0];coords=r['coords'];mask=r['loss_mask'].bool();target=r['target'];check(coords.shape==(16,2) and mask.shape==(16,) and target.shape==(16,) and mask.any(),'Canonicalshape');check(torch.isfinite(coords).all() and (coords>=0).all() and (coords<=1).all(),'Coordinates');expected=(coords[:,1]*48).long().clamp(0,47)*24+(coords[:,0]*24).long().clamp(0,23);check(torch.equal(expected,target),'Hardtargetformula')
  selected=[]
  for ref in refs:
   check(ref['split']==route['split'] and ref['patient_key']==route['patient_key'],'Protectedpatientroute');check(all(torch.equal(ref[k],r[k]) for k in ['coords','loss_mask','target']),'Exactsourcealignment');q=ref['raw_logp'];check(q.shape==(16,1152) and torch.isfinite(q).all(),'Finiteq');err=float(q.logsumexp(-1).abs().max());normmax=max(normmax,err);check(err<=1e-5,'Normalizedq');selected.append(q[mask,target[mask]].double())
  scalar=float(-torch.stack(selected).max(0).values.mean());losses[route['split']].append(scalar);steps[route['split']]+=int(mask.sum())
 for source in sources:check(sha(ROOT/source['path'])==source['sha256'],'Immutablecache')
 splits={s:dict(images=len(losses[s]),valid_steps=steps[s],patient_clusters=len(trainpatients if s=='train' else valpatients),target_aware_perstep_convex_lower_bound_image_macro=float(np.mean(losses[s]))) for s in losses};check(splits['train']['images']==412 and splits['val']['images']==93,'Exactsplitcounts');bound=splits['val']['target_aware_perstep_convex_lower_bound_image_macro']
 result=dict(kind='TARGET_AWARE_SCALAR_DIAGNOSTIC_ONLY',formula='image-macro mean of -log(max_i q_i(current_target)); only valid canonical steps',sources=sources,splits=splits,convex_gating_only_rejected=bound>5.3,goal_achieved=False,interpretation='Bound above5.3 rules out convex mixtures of these seven fixed expert distributions; below5.3 is feasibility only, not an achieved causal model.',verification=dict(exact505=True,protected_patient_disjoint=True,coords_masks_targets_grid_sources_aligned=True,max_logsumexp_error=normmax,no_oracle_weights_or_predictions_saved=True,no_test_quarantine_ROI_access=True,all_manifest_sha256=sha(ROOT/'manifests/all.csv'),trajectory_manifest_sha256=sha(ROOT/'manifests/trajectory.csv')),seconds=time.monotonic()-started)
 (out/'report.json').write_text(json.dumps(result,indent=2));(out/'report.md').write_text(f"Target-aware scalar diagnostic only. Seven fixed experts, including current teacher-KL augmented DINO and Mammo.\n\nTrain412 bound: {splits['train']['target_aware_perstep_convex_lower_bound_image_macro']:.9f}; val93 bound: {bound:.9f}.\n\nConvex-gating-only rejected at5.3: {bound>5.3}. No oracle predictions, weights, or inference maps saved. This is not an achieved model score.\n");print(json.dumps(splits))
if __name__=='__main__':main()
