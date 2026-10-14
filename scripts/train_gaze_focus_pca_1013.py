"""TRAIN patient-CV block PCA64 + regularized logistic diagnostic."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='4'
import sys,json,time,pickle
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from src.official_cbis_classifier_1012 import digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views
from scripts.train_gaze_focus_selected_1013 import local_summary

def main():
 torch.set_num_threads(4);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round04_traincv_pca_pilot42');out.mkdir(exist_ok=False);cp=Path('runs/official_cbis_v2_experiment_1012/development.pt');c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=np.flatnonzero(f.split=='train');va=np.flatnonzero(f.split=='val');y=f.label.to_numpy();assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=np.array([p[0] if p else i for i,p in enumerate(partners)]);sp=Path('runs/gaze_focus_auroc88_1013/round03_spatial_pilot42/spatial_summary.pt');r=torch.load(sp,weights_only=False,map_location='cpu');assert r['image_ids']==f.image_id.tolist();globalx=r['features'].numpy();local={};deps={str(p):digest(p) for p in [cp,sp,Path(__file__),Path('scripts/train_gaze_multiview_1012.py')]}
 for source in ['gaze_baseline','random']:
  p=Path('runs/gaze_focus_auroc88_1013/round02_selected_meanmax_pilot42')/f'{source}_selection.pt';r=torch.load(p,weights_only=False,map_location='cpu');assert r['image_ids']==f.image_id.tolist();deps[str(p)]=digest(p);ids=r['indices'][:,:4];mask=r['mask'][:,:4];z=t[source+'_features'][torch.arange(len(f))[:,None],ids];local[source]=local_summary(torch.zeros(len(f),0),z,mask).numpy()
 m=dict(state='running',stage='train_cv',source_sha256=digest(__file__),dependencies=deps,test_used=False,oracle_used=False,seed=42,pca_components_per_block=64,C_grid=[.01,.1,1],selection='mean AUROC across TRAIN3fold patient stratified CV',normalization='StandardScaler and PCA fitted only fold training patients; final fullTRAIN refit',paired_mode='fixed same-breast opposite-view PCA feature average, classifier trained paired; single and fixedprobavg separate',results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 def fitblock(x,fit):
  sc=StandardScaler();z=sc.fit_transform(x[fit]);pca=PCA(64,svd_solver='randomized',random_state=42);pca.fit(z);return (sc,pca),pca.transform(sc.transform(x))
 def paired(x,ids):return .5*(x[ids]+x[other[ids]])
 folds=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));oof={arm:{C:np.zeros(len(tr)) for C in [.01,.1,1]} for arm in ['global_only','gaze_baseline','random']};scores={arm:{C:[] for C in [.01,.1,1]} for arm in oof};foldaudit=[];save()
 for fold,(fitidx,holdidx) in enumerate(folds):
  fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold]).issubset(set(hold));foldaudit.append(dict(fold=fold,fit_patient_count=f.patient_key.iloc[fit].nunique(),heldout_patient_count=f.patient_key.iloc[hold].nunique(),patient_disjoint=True,paired_partner_within_fold=True));_,g=fitblock(globalx,fit)
  for arm in oof:
   x=g if arm=='global_only' else np.concatenate([g,fitblock(local[arm],fit)[1]],1)
   for C in [.01,.1,1]:
    model=LogisticRegression(C=C,solver='lbfgs',max_iter=1000,random_state=42);model.fit(paired(x,fit),y[fit]);p=model.predict_proba(paired(x,hold))[:,1];oof[arm][C][holdidx]=p;scores[arm][C].append(float(roc_auc_score(y[hold],p)))
   m.update(current_fold=fold,current_arm=arm);save()
 chosen={arm:max([.01,.1,1],key=lambda C:(np.mean(scores[arm][C]),-C)) for arm in oof};gfit,g=fitblock(globalx,tr);m.update(stage='final_train_and_val',fold_audit=foldaudit,chosen_C=chosen,cv_scores=scores);save()
 for arm in oof:
  C=chosen[arm];lf=None;x=g
  if arm!='global_only':lf,l=fitblock(local[arm],tr);x=np.concatenate([g,l],1)
  model=LogisticRegression(C=C,solver='lbfgs',max_iter=1000,random_state=42);model.fit(paired(x,tr),y[tr]);single=model.predict_proba(x[va])[:,1];pp=model.predict_proba(paired(x,va))[:,1];mapping=dict(zip(va,single));fixed=np.array([.5*(mapping[i]+mapping[other[i]]) for i in va]);d=out/arm;d.mkdir();record=dict(arm=arm,C=C,oof_auc=float(roc_auc_score(y[tr],oof[arm][C])),train_paired_auroc=float(roc_auc_score(y[tr],model.predict_proba(paired(x,tr))[:,1])),n_features=x.shape[1])
  for mode,p in [('singleview',single),('learned_paired_feature_average',pp),('fixed_probability_average',fixed)]:
   assert np.isfinite(p).all();df=f.iloc[va][['image_id','patient_key','label']].copy();df['probability']=p;df.to_csv(d/f'{mode}_predictions.csv',index=False);record[mode]=metrics(y[va],p)
  pickle.dump(dict(global_transform=gfit,local_transform=lf,classifier=model),open(d/'model.pkl','wb'));df=f.iloc[tr][['image_id','patient_key','label']].copy();df['probability']=oof[arm][C];df.to_csv(d/'train_oof_predictions.csv',index=False);m['results'].append(record);save();print(json.dumps(record),flush=True)
 baseline=m['results'][0]['oof_auc'];m['train_oof_delta_vs_global']={r['arm']:r['oof_auc']-baseline for r in m['results']};assert all(digest(p)==sha for p,sha in deps.items());m.update(state='complete',stage='complete',elapsed_minutes=(time.time()-start)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
