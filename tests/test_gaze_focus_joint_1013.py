import torch
from src.gaze_focus_joint_1013 import JointGazeFocus

def test_joint_global_and_local_gradients():
 torch.manual_seed(3);m=JointGazeFocus(input_dim=4,dim=8);g=torch.rand(2,2,4);p=torch.rand(2,2,3,4);c=torch.rand(2,2,3,2);mask=torch.ones(2,2,3,dtype=torch.bool);torch.nn.functional.binary_cross_entropy_with_logits(m(g,p,c,mask,torch.tensor([True,False])),torch.tensor([0.,1.])).backward();assert m.global_encoder[1].weight.grad.abs().sum()>0 and m.patch_encoder[1].weight.grad.abs().sum()>0

def test_singleview_ignores_partner_and_padded_patches():
 m=JointGazeFocus(input_dim=4,dim=8).eval();g=torch.rand(2,2,4);p=torch.rand(2,2,3,4);c=torch.rand(2,2,3,2);mask=torch.ones(2,2,3,dtype=torch.bool);use=torch.zeros(2,dtype=torch.bool);a=m(g,p,c,mask,use);g[:,1]=100;p[:,1]=-100;c[:,1]=.99;b=m(g,p,c,mask,use);assert torch.equal(a,b)
if __name__=='__main__':
 test_joint_global_and_local_gradients();test_singleview_ignores_partner_and_padded_patches();print('Jointglobal/localgradient and absentpartnerinvariance checks passed')
