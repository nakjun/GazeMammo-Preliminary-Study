"""Image-only physical-crop ROI reservations; distinct from gaze likelihood output."""
import torch
from .data import patch_tensor
from .proposal_policy_1011 import policy_distribution,proposal_budget_indices


def reservation_steps(budget,steps=16,layout='first'):
    if not 0<=budget<=steps or steps<1 or layout not in ('first','interleaved'):raise ValueError('Invalid reservation schedule')
    return list(range(budget)) if layout=='first' else [i*steps//budget for i in range(budget)]


@torch.no_grad()
def budget_rollout(model,memory,image,geometry,budget=3,layout='first',crop_size=224,steps=16,seed=42,strategy='sample',max_iou=.3,proposal_relevance=None,candidate_indices=None,original_relevance=None):
    """Every decoder update follows the actually chosen crop, including ROI steps."""
    schedule=reservation_steps(budget,steps,layout)
    if crop_size<1 or not 0<=max_iou<=1 or strategy not in ('sample','argmax'):raise ValueError('Invalid crop or strategy')
    if geometry.get('crop_size',crop_size)!=crop_size:raise ValueError('Geometry/action crop size mismatch')
    if original_relevance is None:
        proposal=model.proposal(memory['keys']).squeeze(-1);valid=bool(model.proposal_valid and torch.isfinite(proposal).all());relevance=proposal.sigmoid() if valid else torch.zeros_like(proposal)
    else:
        supplied=original_relevance.to(memory['prior']).reshape(1,-1)
        if supplied.shape[-1]!=model.cell_xy.shape[0]:raise ValueError('Original relevance shape mismatch')
        valid=bool(model.proposal_valid and torch.isfinite(supplied).all());relevance=supplied if valid else torch.zeros_like(supplied)
    roi=relevance if proposal_relevance is None else proposal_relevance.to(relevance).reshape_as(relevance)
    roi_valid=bool(torch.isfinite(roi).all() and valid)
    if candidate_indices is not None:
        supplied=[int(i) for i in candidate_indices]
        if len(supplied)!=len(set(supplied)) or any(i<0 or i>=model.cell_xy.shape[0] for i in supplied):raise ValueError('Invalid image-only candidate indices')
        ranked=supplied[:budget] if roi_valid else []
    else:ranked=proposal_budget_indices(roi,geometry['rectangles'],budget,max_iou) if budget and roi_valid else []
    reserved=dict(zip(schedule,ranked));generator=torch.Generator(device='cpu').manual_seed(seed);prefix=memory['prior'].new_zeros(1,0,2);hidden=None;vector=None;traces=[];indices=[]
    for t in range(steps):
        raw,hidden,_=model.step_history(memory,prefix,hidden,vector)
        dist=policy_distribution(raw,relevance,geometry,'P0',rho=model.rho,ablation=model.ablation,proposal_valid=valid)
        sampled=int(torch.multinomial(dist['q'].cpu(),1,generator=generator)) if strategy=='sample' else int(dist['q'].argmax(-1))
        index=reserved.get(t,sampled);q=dist['q']
        if t in reserved:q=torch.zeros_like(q);q[:,index]=1
        traces.append(dict(step=t,index=index,q_entropy=float(-(q*q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum()),chosen_probability=float(q[0,index]),original_chosen_probability=float(dist['q_original'][0,index]),raw_chosen_logp=float(raw[0,index]),reserved_proposal=t in reserved,crop_size=crop_size))
        indices.append(index);prefix=torch.cat([prefix,model.cell_xy[index][None,None]],1)
        if model.foveal:vector=model.visual.patch_vector(patch_tensor(image,prefix[0,-1].cpu().numpy()*image.size,crop_size)[None].to(prefix.device))
    return dict(coordinates=prefix[0].cpu().tolist(),indices=indices,relevance=roi[0].cpu().tolist(),original_relevance=relevance[0].cpu().tolist(),proposal_valid=roi_valid,steps=traces,reserved_candidate_indices=ranked,reserved_steps=list(reserved),requested_roi_budget=budget,actual_roi_budget=len(ranked))
