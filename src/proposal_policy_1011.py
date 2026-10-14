"""Fixed-checkpoint image-only proposal policies; no annotation input."""
import torch
from .data import patch_tensor
from .guided_trajectory_1011 import crop_iou
POLICIES=('P0','P1','P2','P3','P4','P5')


def policy_distribution(raw,relevance,geometry,policy,alpha=.125,tau=.5,rho=1.,ablation='combined',proposal_valid=True,roi_relevance=None):
    if policy not in POLICIES or not 0<=alpha<=1 or tau<=0:raise ValueError('Invalid policy parameters')
    r=relevance.to(raw);zero=torch.zeros_like(raw)
    proposal_valid=bool(proposal_valid and torch.isfinite(r).all())
    if not proposal_valid:r=zero
    roi_r=r if roi_relevance is None else roi_relevance.to(raw)
    roi_valid=bool(proposal_valid and torch.isfinite(roi_r).all())
    if not roi_valid:roi_r=zero
    geo=zero
    if geometry['geometry_valid']:
        geo=-3*geometry['background'].to(raw)-.5*geometry['edge'].to(raw)*(1-r)
    original_bias=(2*r if ablation in ('combined','lesion') else zero)+(geo if ablation in ('combined','geometry') else zero)
    original=(raw+rho*original_bias).softmax(-1)
    bias=original_bias
    if policy=='P1':bias=2*r
    if policy=='P2':bias=6*r+geo
    q=(raw+rho*bias).softmax(-1) if policy in ('P1','P2') else original
    roi_logits=torch.log(roi_r.clamp(0,1)+1e-6)/tau
    if geometry['geometry_valid']:
        roi_logits=roi_logits-(.5 if policy=='P4' else 3)*geometry['background'].to(raw)
        if policy!='P4':roi_logits=roi_logits-.5*geometry['edge'].to(raw)*(1-roi_r)
    roi=roi_logits.softmax(-1)
    if policy in ('P3','P4') and alpha and roi_valid:q=(1-alpha)*original+alpha*roi
    return dict(q=q,q_original=original,p_roi=roi,bias=bias)


def proposal_budget_indices(relevance,rectangles,budget=3,max_iou=.3):
    selected=[]
    for index in torch.argsort(relevance.flatten(),descending=True,stable=True).tolist():
        if not selected or float(crop_iou(rectangles[index:index+1],rectangles[selected])[0])<=max_iou:
            selected.append(index)
            if len(selected)==budget:break
    return selected


@torch.no_grad()
def rollout(model,memory,image,geometry,policy='P0',steps=16,seed=42,alpha=.125,tau=.5,strategy='sample',roi_relevance=None):
    if strategy not in ('sample','argmax'):raise ValueError('Invalid strategy')
    if steps<1:raise ValueError('steps must be positive')
    prefix=memory['prior'].new_zeros(1,0,2);hidden=None;vector=None;traces=[];indices=[]
    generator=torch.Generator(device='cpu').manual_seed(seed)
    proposal=model.proposal(memory['keys']).squeeze(-1)
    valid=bool(model.proposal_valid and torch.isfinite(proposal).all())
    relevance=proposal.sigmoid() if valid else torch.zeros_like(proposal)
    roi_r=relevance if roi_relevance is None else roi_relevance.to(relevance).reshape_as(relevance)
    roi_valid=bool(valid and torch.isfinite(roi_r).all())
    budget=proposal_budget_indices(roi_r,geometry['rectangles'],min(3,steps)) if policy=='P5' and roi_valid else []
    for t in range(steps):
        raw,hidden,_=model.step_history(memory,prefix,hidden,vector)
        dist=policy_distribution(raw,relevance,geometry,policy,alpha,tau,model.rho,model.ablation,valid,roi_r)
        # One CPU categorical draw each step, including reserved steps, maintains RNG alignment.
        sampled=int(torch.multinomial(dist['q'].cpu(),1,generator=generator)) if strategy=='sample' else int(dist['q'].argmax(-1))
        index=budget[t] if t<len(budget) else sampled
        q=dist['q']
        if t<len(budget):q=torch.zeros_like(q);q[:,index]=1
        traces.append(dict(step=t,index=index,raw_logp=raw[0].cpu().tolist(),bias=dist['bias'][0].cpu().tolist(),q_original=dist['q_original'][0].cpu().tolist(),p_roi=dist['p_roi'][0].cpu().tolist(),q=q[0].cpu().tolist(),reserved_proposal=t<len(budget)))
        indices.append(index);prefix=torch.cat([prefix,model.cell_xy[index][None,None]],1)
        if model.foveal:vector=model.visual.patch_vector(patch_tensor(image,prefix[0,-1].cpu().numpy()*image.size)[None].to(prefix.device))
    return dict(coordinates=prefix[0].cpu().tolist(),indices=indices,relevance=(roi_r if policy in ('P3','P4','P5') and roi_valid else relevance)[0].cpu().tolist(),original_relevance=relevance[0].cpu().tolist(),proposal_valid=valid,steps=traces)
