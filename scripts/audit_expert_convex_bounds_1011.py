"""TARGET-AWARE DIAGNOSTIC bounds only; never produces prediction distributions."""
import json,hashlib,csv,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def check(value,message):
 if not value:raise ValueError(message)

def main():
 torch.set_num_threads(4);out=ROOT/'runs/dual_target_1011/expert_convex_bounds_diagnostic';out.mkdir(exist_ok=True);check(not (out/'DIAGNOSTIC_bounds.json').exists(),'Do not overwrite completed diagnostic');started=time.monotonic()
 names=['control','improved','clinical_DG','DINO','MDN_product']
 paths=['human_control_raw_cache_for_ensemble.pt','human_improved_raw_cache.pt','deepgaze_clinical_best_raw_cache.pt','dino/gaze_seed42/best_raw_cache.pt','gaze_mdn_product_seed42/corrected_raw_cache.pt']
 payloads=[torch.load(ROOT/'runs/dual_target_1011'/p,map_location='cpu',weights_only=False) for p in paths];maps=[];sources=[]
 allframe=pd.read_csv(ROOT/'manifests/all.csv');trajectory=pd.read_csv(ROOT/'manifests/trajectory.csv');manifest=pd.concat([allframe[allframe.split=='train'],trajectory[trajectory.split=='val']],ignore_index=True);allowed={int(r.image_id):(r.split,r.patient_key) for _,r in manifest.iterrows()};check(len(allowed)==505,'Exact505routing')
 for name,path,p in zip(names,paths,payloads):
  check(p['grid']==[24,48] and len(p['records'])==505,'Grid/count');mapping={int(r['image_id']):r for r in p['records']};check(len(mapping)==505 and set(mapping)==set(allowed),'Expert505allowlist')
  checkpoint=p['base_checkpoint'];check(sha(checkpoint)==p['base_checkpoint_sha256'],'CheckpointSHA')
  sources.append(dict(expert=name,raw_cache=str((ROOT/'runs/dual_target_1011'/path).relative_to(ROOT)),raw_cache_sha256=sha(ROOT/'runs/dual_target_1011'/path),checkpoint=checkpoint,checkpoint_sha256=p['base_checkpoint_sha256']))
  maps.append(mapping)
 records=[];wins={s:np.zeros(5,dtype=np.int64) for s in ['train','val']};step_counts={s:np.zeros((16,5),dtype=np.int64) for s in ['train','val']};deltas={s:[[] for _ in names] for s in ['train','val']};normmax=0.;kktmax=0.;solver_iterations=[]
 for image_id in sorted(allowed):
  refs=[mapping[image_id] for mapping in maps];reference=refs[0];split,patient=allowed[image_id];check(reference['split']==split and reference['patient_key']==patient,'Canonicalpatient/split')
  coords=reference['coords'];mask=reference['loss_mask'].bool();target=reference['target'];check(coords.shape==(16,2) and mask.shape==(16,) and target.shape==(16,) and mask.any(),'Canonicalshape')
  recomputed=(coords[:,1]*48).long().clamp(0,47)*24+(coords[:,0]*24).long().clamp(0,23);check(torch.equal(recomputed,target),'Hardtarget')
  check(torch.isfinite(coords).all() and (coords>=0).all() and (coords<=1).all(),'Coordsrange')
  for ref in refs:
   check(ref['split']==split and ref['patient_key']==patient,'Expertpatient/split');check(all(torch.equal(ref[k],reference[k]) for k in ['coords','loss_mask','target']),'Expertcoords/mask/targetalignment')
   raw=ref['raw_logp'];check(raw.shape==(16,1152) and torch.isfinite(raw).all(),'Rawshape/finite');error=float(raw.logsumexp(-1).abs().max());normmax=max(normmax,error);check(error<=1e-5,'Normalizedrawexpert')
  logp=torch.stack([ref['raw_logp'][mask,target[mask]] for ref in refs]).double().numpy().T;loss=-logp;standalone=loss.mean(0);best_idx=loss.argmin(1);step_bound=loss.min(1).mean();constant_expert=standalone.min()
  # The optimum of a convex objective over the simplex is a target-aware bound,
  # not a learned gate. Stabilize target probabilities with one common step factor.
  shift=logp.max(1);prob=np.exp(logp-shift[:,None])
  def objective(w):return float(-np.log(prob@w).mean()-shift.mean())
  def grad(w):return -(prob/(prob@w)[:,None]).mean(0)
  fit=minimize(objective,np.ones(5)/5,jac=grad,method='SLSQP',bounds=[(0,1)]*5,constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(5))],options=dict(ftol=1e-12,maxiter=1000))
  check(fit.success,'Convexoptimizerfailure');w=np.maximum(fit.x,0);w/=w.sum();bound=objective(w);g=grad(w);active=w>1e-7;level=float((g*w).sum());kkt=max(float(np.abs(g[active]-level).max()),float(np.maximum(level-g[~active],0).max()) if (~active).any() else 0.);kktmax=max(kktmax,kkt);solver_iterations.append(fit.nit)
  check(step_bound<=bound+1e-8 and bound<=constant_expert+1e-8,'Boundordering')
  baseline_logp=np.logaddexp(logp[:,0]+np.log(.75),logp[:,1]+np.log(.25));clinical_parent=np.logaddexp(baseline_logp+np.log(.75),logp[:,2]+np.log(.25))
  row=dict(image_id=image_id,patient_key=patient,split=split,valid_steps=int(mask.sum()),diagnostic_target_aware_perstep_convex_lower_bound=step_bound,diagnostic_target_aware_perimage_constant_convex_optimum=bound,diagnostic_target_aware_perimage_best_single_expert=constant_expert,perimage_best_single_expert=names[int(standalone.argmin())],fixed_arithmetic075_nll=float(-baseline_logp.mean()),fixed_clinical025_nll=float(-clinical_parent.mean()))
  row.update({name+'_nll':float(v) for name,v in zip(names,standalone)});records.append(row)
  wins[split]+=np.bincount(best_idx,minlength=5)
  for t,k in zip(torch.where(mask)[0].tolist(),best_idx):step_counts[split][t,k]+=1;deltas[split][k].append(float(loss[list(torch.where(mask)[0].tolist()).index(t),4]-loss.min(1)[list(torch.where(mask)[0].tolist()).index(t)]))
 pd.DataFrame(records).to_csv(out/'per_image_DIAGNOSTIC_bounds.csv',index=False)
 summary={}
 for split in ['train','val']:
  rows=[r for r in records if r['split']==split];count=412 if split=='train' else 93;check(len(rows)==count,'Splitdenominator');n=sum(r['valid_steps'] for r in rows);table={name:float(np.mean([r[name+'_nll'] for r in rows])) for name in names};table.update({k:float(np.mean([r[k] for r in rows])) for k in ['fixed_arithmetic075_nll','fixed_clinical025_nll','diagnostic_target_aware_perimage_best_single_expert','diagnostic_target_aware_perimage_constant_convex_optimum','diagnostic_target_aware_perstep_convex_lower_bound']});oracle=table['diagnostic_target_aware_perstep_convex_lower_bound'];perimage=table['diagnostic_target_aware_perimage_constant_convex_optimum']
  summary[split]=dict(images=len(rows),patient_clusters=len({r['patient_key'] for r in rows}),valid_steps=n,NLL_image_macro=table,perstep_oracle_best_expert_frequencies={name:dict(steps=int(wins[split][i]),fraction=float(wins[split][i]/n),mean_conditional_delta_loss_vs_current_MDN=float(np.mean(deltas[split][i])) if deltas[split][i] else None) for i,name in enumerate(names)},perstep_frequencies_by_t={str(t):{name:int(step_counts[split][t,i]) for i,name in enumerate(names)} for t in range(16)},perimage_best_single_expert_frequencies={name:sum(r['perimage_best_single_expert']==name for r in rows) for name in names},diagnostic_gap_from_current_MDN=table['MDN_product']-oracle,diagnostic_perstep_bound_below5p3=oracle<=5.3,diagnostic_perimage_bound_below5p3=perimage<=5.3)
 result=dict(kind='TARGET_AWARE_DIAGNOSTIC_CONVEX_BOUNDS_ONLY',scope='These scalars use observed TARGETS to select experts/weights. They are not normalized model predictions, deployable gates, achieved results or evidence that a causal gate can learn the oracle. Bound applies only to convex mixtures of these five frozen expert distributions; excludes new models and learned products. MDNproduct is merely one fixed expert in the audited convex set.',math='Perstep bound=-log(max_k p_k(y_t)); anyconvexmixture target probability<=max_k. Perimage constantweights optimum=min_w mean_t[-log(sum_k w_k p_k(y_t))] withsimplex constraints; also show bestsingleexpert perimage. Aggregation is imagemacro withcanonical validmask.',sources=sources,verification=dict(exact505=True,train412=True,val93=True,patients_coords_masks_targets_grid_aligned=True,all_source_checkpoint_SHA_match=True,max_raw_logsumexp_error=normmax,convex_solver_all_success=True,max_KKT_residual=kktmax,max_solver_iterations=max(solver_iterations),no_new_inference_or_training=True,no_target_gating_predictions_saved=True,no_ROI_or_test_access=True,manifest_sha256=sha(ROOT/'manifests/trajectory.csv'),all_manifest_sha256=sha(ROOT/'manifests/all.csv')),splits=summary,seconds=time.monotonic()-started)
 (out/'DIAGNOSTIC_bounds.json').write_text(json.dumps(result,indent=2,allow_nan=False));lines=['# Target-aware convex expert bounds — diagnostic only','','No learned gate or normalized prediction is produced. Targets are used only to compute optimistic scalar bounds. These bounds apply to convex mixtures of five frozen experts, not new models or variable product corrections. Per-step selection can access the current target; a causal inference gate cannot. Per-image optimization likewise uses the full target trajectory.','','| Expert or diagnostic bound | Train412 NLL | Val93 NLL |','|---|---:|---:|']
 for key in summary['train']['NLL_image_macro']:lines.append(f"| {key} | {summary['train']['NLL_image_macro'][key]:.6f} | {summary['val']['NLL_image_macro'][key]:.6f} |")
 lines.extend(['',f"Patient clusters: train{summary['train']['patient_clusters']}, validation{summary['val']['patient_clusters']}. All IDs, patients, coordinates, masks, hard targets, grid and checkpoint hashes verified. Maximum raw normalization error{normmax:.3g}; convex solver maximum KKT residual{kktmax:.3g}.",'','Per-step expert frequencies and conditional loss gaps are in DIAGNOSTIC_bounds.json. Per-image scalar bounds are in per_image_DIAGNOSTIC_bounds.csv. No prediction map, per-target gate or oracle weight vector is saved. A bound below5.3 is mathematical headroom only; a bound above5.3 would rule out any convex causal gate of these fixed experts on this cohort.'])
 (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps({split:summary[split]['NLL_image_macro'] for split in summary},indent=2))
if __name__=='__main__':main()
