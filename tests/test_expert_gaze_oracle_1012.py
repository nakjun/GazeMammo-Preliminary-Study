import torch
from scripts.diagnose_expert_gaze_oracle_1012 import expert_bag

def test_current_vectors_not_previous_point_and_padding_mask():
 r={'image_id':1,'parts':[torch.tensor([[.2,.3],[.4,.5]])],'vectors':[torch.stack([torch.full((1024,),7.),torch.full((1024,),9.)])]}
 v,c,m,a=expert_bag(r)
 assert torch.equal(v[0],r['vectors'][0][0]) and torch.equal(v[1],r['vectors'][0][1])
 assert m.sum()==2 and v[2:].abs().sum()==0 and a['valid']==2
 assert torch.equal(c[:2],r['parts'][0])
def test_legacy_selection_is_label_free_and_longest_segment():
 r={'image_id':2,'label':0,'parts':[torch.tensor([[.1,.2]]),torch.rand(20,2)],'vectors':[torch.ones(1,1024),torch.arange(20.)[:,None].repeat(1,1024)]}
 v,c,m,a=expert_bag(r);r['label']=1;vv,cc,mm,aa=expert_bag(r)
 assert torch.equal(v,vv) and torch.equal(c,cc) and torch.equal(m,mm) and a==aa and a['segment']==1 and m.sum()==16
if __name__=='__main__':
 test_current_vectors_not_previous_point_and_padding_mask();test_legacy_selection_is_label_free_and_longest_segment();print('2 expert alignment/no-label-selection checks passed')
