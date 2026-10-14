import torch,numpy as np
from scripts.train_gaze_diagnosis_selector_1012 import fit_selector,patch_scores,select_hard

def test_fit_excludes_heldout_labels_and_selection_invariant():
 torch.set_num_threads(1);torch.manual_seed(8);g=torch.rand(8,4);p=torch.rand(8,32,4);y=torch.tensor([0.,1.]*4);fit=np.arange(6);hold=np.arange(6,8);a,_=fit_selector(g,p,y,fit,42,epochs=2,hidden=3);alter=y.clone();alter[hold]=float('nan');b,_=fit_selector(g,p,alter,fit,42,epochs=2,hidden=3)
 sa=patch_scores(a,g,p,hold);sb=patch_scores(b,g,p,hold);assert torch.equal(sa,sb) and all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict());coords=torch.rand(32,2);mask=torch.ones(32,dtype=torch.bool);assert torch.equal(select_hard(sa[0],coords,mask),select_hard(sb[0],coords,mask))

def test_selection_budget_alignment_and_duplicate_removal():
 scores=torch.arange(32,dtype=torch.float32);coords=torch.stack([torch.linspace(.05,.95,32),torch.linspace(.05,.95,32)],1);coords[30]=coords[31];selected=select_hard(scores,coords,torch.ones(32,dtype=torch.bool));assert selected.shape==(16,) and len(torch.unique(torch.round(coords[selected]*torch.tensor([1500.,3000.])),dim=0))==16
 assert selected[0]==31 and 30 not in selected
if __name__=='__main__':
 test_fit_excludes_heldout_labels_and_selection_invariant();test_selection_budget_alignment_and_duplicate_removal();print('2 OOFlabel/rankinvariance and alignedhard16dedup checks passed')
