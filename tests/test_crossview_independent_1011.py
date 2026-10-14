import numpy as np,torch
from src.gaze_crossview_1011 import physical_rectangles,negative_indices,contrastive_aux

def test_negatives_physical_geometry_and_deterministic_limit():
    coords=np.stack([np.linspace(0,1,80),np.full(80,.5)],1);first=negative_indices(coords,[1500,3000],40,3,62,8);assert first==negative_indices(coords,[1500,3000],40,3,62,8) and len(first)<=16 and 40 not in first
    rectangles,centers=physical_rectangles(coords,[1500,3000]);positive=rectangles[40]
    for i in first:
        overlap=np.maximum(np.minimum(rectangles[i,2:],positive[2:])-np.maximum(rectangles[i,:2],positive[:2]),0).prod();assert overlap==0 and np.linalg.norm(centers[i]-centers[40])>=448

def test_aux_uses_current_positive_detached_targets_and_skips_zero_support():
    torch.manual_seed(42);coords=torch.stack([torch.linspace(0,1,80),torch.full((80,),.5)],1);targets=torch.randn(80,1024,requires_grad=True);queries=torch.randn(1,2,1024,requires_grad=True);group=[dict(image_id=62,image_size=[1500,3000],selected_pool_indices=[20,50])];pools={62:dict(coords=coords,patch_vectors=targets)};loss,details=contrastive_aux(queries,group,pools,1);loss.backward();assert torch.isfinite(loss) and queries.grad.abs().sum()>0 and targets.grad is None and details['eligible_anchors']==2
    zero=queries.detach().clone().requires_grad_(True);pools[62]['coords']=torch.full((80,2),.5);loss,details=contrastive_aux(zero,group,pools,1);assert loss==0 and details['skipped_anchors']==2;loss.backward();assert not zero.grad.any()
