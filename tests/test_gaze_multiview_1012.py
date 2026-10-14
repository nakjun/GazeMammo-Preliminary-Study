import pandas as pd,torch
from scripts.train_gaze_multiview_1012 import pair_views,pooled_summary

def fixture():
 return pd.DataFrame([dict(image_id=i,patient_key=p,split='train',source_sn=s,image_sha256=str(i),label=i%2) for i,p,s in [(1,'p','Mass-Training_P_LEFT_CC'),(2,'p','Mass-Training_P_LEFT_MLO'),(3,'p','Calc-Training_P_LEFT_MLO'),(4,'p','Mass-Training_P_RIGHT_MLO'),(5,'q','Mass-Training_Q_LEFT_CC')]])
def test_pairing_ignores_labels_and_other_breast():
 f=fixture();a,_,_=pair_views(f);f.label=1-f.label;b,_,_=pair_views(f)
 assert a==b and a[0]==[1] and a[3]==[] and a[4]==[]
 assert all(len(x)<=1 for x in a)
def test_missing_view_fallback_and_mask():
 f=fixture();a,_,_=pair_views(f);g=torch.rand(5,2);p=torch.rand(5,16,2);x=pooled_summary(g,p,a)
 assert torch.equal(x[4,:6],x[4,6:12]) and x[4,-1]==0 and x[0,-1]==1
if __name__=='__main__':
 test_pairing_ignores_labels_and_other_breast();test_missing_view_fallback_and_mask();print('2 meaningful pairing checks passed')
