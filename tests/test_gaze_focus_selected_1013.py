import torch
from scripts.train_gaze_focus_selected_1013 import select_source,local_summary

def test_selection_label_free_dedup_and_masked_negative_max():
 coords=torch.rand(16,2);coords[15]=coords[14];ids=select_source(torch.arange(16.),coords,torch.ones(16,dtype=torch.bool));assert ids[0]==15 and 14 not in ids
 g=torch.ones(1,2);p=torch.tensor([[[-2.,-3.],[0.,0.]]]);mask=torch.tensor([[True,False]]);x=local_summary(g,p,mask);assert torch.equal(x,torch.tensor([[1.,1.,-2.,-3.,-2.,-3.]]))
if __name__=='__main__':
 test_selection_label_free_dedup_and_masked_negative_max();print('OOFscorelabel-freepixeldup and maskedmean/max check passed')
