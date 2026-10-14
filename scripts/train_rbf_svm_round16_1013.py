"""TRAIN-only blockwise PCA64 + nonlinear RBF SVM; no probability calibration."""
import sys,json,time,itertools,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from src.official_cbis_classifier_1012 import digest
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');tmp.write_bytes(pickle.dumps(v));tmp.replace(p)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);threadpool_limits(4);tick=time.time();root=Path('runs/gaze_focus_auroc88_1013');out=root/'round16_rbf_svm_pilot42';out.mkdir(exist_ok=True);cp=Path('runs/official_cbis_v2_experiment_1012/development.pt');gp=root/'round08_global_resolution_pilot42/global_meanmax.pt';sp=root/'round12_gaze_extent_pilot42/diverse_selections.pt';assert all(p.exists() for p in [cp,gp,sp]);c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);ids=f.image_id.tolist();y=f.label.values;tr=np.flatnonzero(f.split=='train');va=np.flatnonzero(f.split=='val');assert len(tr)==2037 and len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);partners,_,_=pair_views(f);other=np.array([p[0] if p else i for i,p in enumerate(partners)]);assert set(other[fit])<=set(fit) and set(other[hold])<=set(hold) and set(other[va])<=set(va);g=torch.load(gp,weights_only=False,map_location='cpu');assert g['image_ids']==ids and g['high'].shape==(2393,2048);globalraw=g['high'].numpy();sel=torch.load(sp,weights_only=False,map_location='cpu');assert sel['image_ids']==ids;idx=sel['selection']['8']['indices'];mask=sel['selection']['8']['mask'];rs=[torch.load(root/'round12_gaze_extent_pilot42/predicted_records_k16'/f"imitation_{i}.pt",weights_only=False,map_location='cpu') for i in ids];v=torch.stack([r['features'] for r in rs]).gather(1,idx[:,:,None].expand(-1,-1,1024));rmask=torch.arange(8)[None]<mask.sum(1)[:,None];assert torch.equal(mask.sum(1),rmask.sum(1));rv=c['tensors']['random_features'][:,:8]
 def summary(v,m):return torch.cat([(v*m[:,:,None]).sum(1)/m.sum(1).clamp_min(1)[:,None],v.masked_fill(~m[:,:,None],-1e9).max(1).values],1).numpy()
 locals={'global_only':None,'gaze8':summary(v,mask),'random8':summary(rv,rmask)};m=dict(state='running',stage='TRAIN_inner_RBF_selection',seed=42,test_used=False,oracle_used=False,CNN_extraction=False,backbone_frozen=True,PCA_components_per_block=64,PCA_whiten=False,PCA_solver='randomized',configs=[dict(C=C,gamma=gamma) for C,gamma in itertools.product([.1,1,10],['scale',.01])],paired_rule='mean per-view PCA features; label-free fixed samebreast CC/MLO partner',selection='same TRAIN patient internal holdout AUROC only; fullTRAIN refit; fullVAL356 once',probability_calibration=False,VALID_effective_K=float(mask[va].sum(1).float().mean()),fit_patients=sorted(set(f.patient_key.iloc[fit])),hold_patients=sorted(set(f.patient_key.iloc[hold])),dependencies={str(p):digest(p) for p in [Path(__file__),cp,gp,sp,Path('scripts/train_gaze_multiview_1012.py')]},results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save()
 def block(raw,train):
  scaler=StandardScaler().fit(raw[train]);z=scaler.transform(raw);pca=PCA(64,whiten=False,svd_solver='randomized',random_state=42,iterated_power=3).fit(z[train]);return pca.transform(z).astype(np.float32),dict(scaler=scaler,pca=pca,fit_image_ids=f.iloc[train].image_id.tolist())
 # Reuse global transformations across all arms while keeping inner/fullTRAIN fit separate.
 transforms={tag:block(globalraw,train) for tag,train in [('inner',fit),('fullTRAIN',tr)]}
 for arm,raw in locals.items():
  folder=out/arm;folder.mkdir(exist_ok=True);selected=None;best=-1;hist=[];xg,gb=transforms['inner'];x=xg;lb=None
  if raw is not None:xl,lb=block(raw,fit);x=np.concatenate([xg,xl],1)
  paired=.5*(x+x[other]);assert np.isfinite(paired).all()
  for config in m['configs']:
   model=SVC(C=config['C'],gamma=config['gamma'],kernel='rbf',probability=False,random_state=42,cache_size=256);model.fit(paired[fit],y[fit]);score=float(roc_auc_score(y[hold],model.decision_function(paired[hold])));hist.append(dict(**config,inner_auroc=score))
   if score>best:best=score;selected=config.copy()
  xg,gb=transforms['fullTRAIN'];x=xg;lb=None
  if raw is not None:xl,lb=block(raw,tr);x=np.concatenate([xg,xl],1)
  paired=.5*(x+x[other]);model=SVC(C=selected['C'],gamma=selected['gamma'],kernel='rbf',probability=False,random_state=42,cache_size=256);model.fit(paired[tr],y[tr]);atomic(dict(model=model,global_block=gb,local_block=lb,selected_config=selected,inner_history=hist,fit_image_ids=f.iloc[tr].image_id.tolist(),paired_rule=m['paired_rule']),folder/'selected.pkl');single=model.decision_function(x[va]);pairedscore=model.decision_function(paired[va]);mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(arm=arm,selected_config=selected,inner_auroc=best,inner_history=hist,train_paired_auroc=float(roc_auc_score(y[tr],model.decision_function(paired[tr]))))
  for mode,score in [('singleview',single),('paired_feature_mean',pairedscore),('fixed_decision_score_average',fixed)]:df=f.iloc[va][['image_id','patient_key','label']].copy();df['decision_score']=score;df.to_csv(folder/f'{mode}_scores.csv',index=False);r[mode]=dict(auroc=float(roc_auc_score(y[va],score)))
  m['results'].append(r);save();print(json.dumps(dict(arm=arm,config=selected,paired=r['paired_feature_mean']['auroc'],train=r['train_paired_auroc'])),flush=True)
 assert all(digest(p)==sha for p,sha in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-tick)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
