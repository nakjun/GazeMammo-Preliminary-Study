import numpy as np,pandas as pd,torch
from scripts.train_gaze_oof_fusion_1012 import patient_folds,fit_base,fit_meta

def test_patient_fold_complete_and_no_overlap():
 f=pd.DataFrame([dict(patient_key=str(i//2),label=(i//2)%2,split='train') for i in range(12)]+[dict(patient_key='val',label=0,split='val')]);folds=patient_folds(f,42)
 assert sorted(np.concatenate([b for _,b in folds]).tolist())==list(range(12))
 for a,b in folds:assert not set(f.iloc[a].patient_key)&set(f.iloc[b].patient_key) and 12 not in a and 12 not in b

def test_heldout_labels_do_not_change_predictions_or_model():
 torch.set_num_threads(1);x=torch.arange(30,dtype=torch.float32).reshape(10,3)/10;y=torch.tensor([0.,1.]*5);fit=np.arange(6);hold=np.arange(6,10)
 a,ck=fit_base(x,y,fit,hold,42,epochs=2,dim=4);alter=y.clone();alter[hold]=float('nan');b,other=fit_base(x,alter,fit,hold,42,epochs=2,dim=4)
 assert np.array_equal(a,b) and all(torch.equal(ck['state'][k],other['state'][k]) for k in ck['state'])
 assert ck['epochs']==2 and ck['fit_indices']==fit.tolist()

def test_meta_coefficients_and_normalizer_do_not_fit_val():
 tx=np.array([[0.,1.],[1.,2.],[2.,1.],[3.,2.],[4.,1.],[5.,2.]]);y=np.array([0,0,1,0,1,1]);vx=np.array([[1.,2.],[2.,3.]])
 _,a=fit_meta(tx,y,vx);_,b=fit_meta(tx,y,vx+1000)
 assert a==b
if __name__=='__main__':
 test_patient_fold_complete_and_no_overlap();test_heldout_labels_do_not_change_predictions_or_model();test_meta_coefficients_and_normalizer_do_not_fit_val();print('3 patient OOF/no-heldout-label/train-only-meta checks passed')
