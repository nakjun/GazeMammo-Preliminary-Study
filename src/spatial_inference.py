"""Image-only closed-loop prediction; no gaze/box/label files are read."""
import numpy as np
import torch
from .data import whole_tensor,patch_tensor
from .constraints import breast_mask,grid_mask,VisitConstraint

@torch.no_grad()
def predict_path(model,image,device,seed=42,strategy='sample',prior_only=False,input_tensor=None,policy='hard',radius=112,penalty=3.):
    if strategy not in ('argmax','sample'):raise ValueError(strategy)
    tissue=breast_mask(image);allowed=torch.tensor(grid_mask(tissue,model.grid)[None],device=device)
    if not allowed.any():raise ValueError('Empty candidate grid')
    tensor=whole_tensor(image,size=getattr(model,'input_size',(256,512))) if input_tensor is None else input_tensor
    memory=model.maps(tensor[None].to(device))
    sizes=torch.tensor([[image.width,image.height]],device=device,dtype=torch.float32)
    state=VisitConstraint(allowed,sizes,model.grid,policy,radius,penalty)
    hidden=None;previous=torch.zeros(1,2,device=device);older=previous;vectors=None;points=[]
    torch.manual_seed(seed)
    for step in range(model.k):
        if prior_only:logits=memory['prior']
        else:logits,hidden=model.step(memory,previous,older,vectors,hidden,step)
        masked,valid=state.apply(logits)
        if not valid.item():break
        idx=masked.argmax(-1) if strategy=='argmax' else torch.multinomial(masked.softmax(-1),1).squeeze(-1)
        state.update(idx,valid);xy=state.coords[idx];points.append(xy[0].cpu().tolist())
        if model.foveal and not prior_only:
            crop=patch_tensor(image,xy[0].cpu().numpy()*[image.width,image.height])
            vectors=model.visual.patch_vector(crop[None].to(device))
        older,previous=previous,xy
    return dict(points=np.array(points),memory=memory,breast_mask=tissue,relaxations=int(state.relaxations.item()))
