"""Train-only box-center offsets from frozen proposal anchors, fixed224 crops."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .proposal_utility_1011 import state_sha256


def center_offset_targets(image_id,patient_key,boxes,image_size,allowlist,grid=(24,48),crop_size=224,max_offset_crops=1.,source_relevance=None,relevance_power=0):
    if allowlist.get(int(image_id))!=patient_key:raise ValueError('Offset target outside train image/patient allowlist')
    if crop_size!=224 or max_offset_crops<=0 or relevance_power not in (0,1,2):raise ValueError('Invalid offset bounds/relevance power')
    w,h=image_size;gx,gy=grid;ids=np.arange(gx*gy);low=np.stack([(ids%gx)*w/gx,(ids//gx)*h/gy],-1);high=low+[w/gx,h/gy]
    normalized=np.stack([((ids%gx)+.5)/gx,((ids//gx)+.5)/gy],-1).astype(np.float32);anchors=normalized.astype(np.float64)*[w,h];rectangles=[]
    for x,y,bw,bh in boxes:
        if not np.isfinite([x,y,bw,bh]).all() or bw<=0 or bh<=0:raise ValueError('Invalid training box')
        r=[max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)]
        if r[2]<=r[0] or r[3]<=r[1]:raise ValueError('Training box outside image')
        rectangles.append(r)
    targets=np.zeros((gx*gy,2));weights=np.zeros(gx*gy);assignment=np.full(gx*gy,-1,dtype=np.int64);clip=np.zeros((gx*gy,2),dtype=bool);assigned_boxes=[];relevance_fallback_boxes=[]
    scores=None
    if relevance_power and source_relevance is not None:
        scores=np.asarray(source_relevance.detach().cpu() if isinstance(source_relevance,torch.Tensor) else source_relevance,dtype=float).reshape(-1)
        if scores.shape!=(gx*gy,):raise ValueError('Source relevance shape mismatch')
    if rectangles:
        rectangles=np.asarray(rectangles);overlap=np.maximum(0,np.minimum(high[:,None],rectangles[None,:,2:])-np.maximum(low[:,None],rectangles[None,:,:2])).prod(-1);positive=overlap.max(-1)>0;assignment[positive]=overlap[positive].argmax(-1);assigned_boxes=np.unique(assignment[positive]).tolist()
        centers=(rectangles[:,:2]+rectangles[:,2:])/2;raw=(centers[assignment[positive]]-anchors[positive])/crop_size;targets[positive]=np.clip(raw,-max_offset_crops,max_offset_crops);clip[positive]=np.abs(raw)>max_offset_crops
        for box_index in assigned_boxes:
            cells=assignment==box_index
            if relevance_power and scores is not None:
                values=scores[cells]
                if np.isfinite(values).all() and (values>=0).all() and (values<=1).all():importance=values**relevance_power
                else:importance=np.zeros(cells.sum())
            else:importance=np.zeros(cells.sum())
            if relevance_power and importance.sum()>0:weights[cells]=importance/(importance.sum()*len(assigned_boxes))
            else:
                weights[cells]=1./(cells.sum()*len(assigned_boxes))
                if relevance_power:relevance_fallback_boxes.append(box_index)
    positives=assignment>=0
    return dict(known=bool(len(boxes)),targets=torch.tensor(targets,dtype=torch.float32),weights=torch.tensor(weights,dtype=torch.float32),assignment=torch.tensor(assignment),anchor_coords=torch.tensor(normalized),assigned_boxes=assigned_boxes,unassigned_box_indices=[i for i in range(len(boxes)) if i not in assigned_boxes],clipped_component_fraction=float(clip[positives].mean()) if positives.any() else 0.,weighted_clipped_component_fraction=float((clip.mean(-1)*weights).sum()),positive_cells=int(positives.sum()),box_count=len(boxes),relevance_power=relevance_power,relevance_fallback_box_indices=relevance_fallback_boxes)


class CenterOffsetHead(nn.Module):
    def __init__(self,dim,architecture='linear',grid=(24,48),max_offset_crops=1.):
        super().__init__();self.dim=dim;self.architecture=architecture;self.grid=tuple(grid);self.max_offset_crops=max_offset_crops
        if architecture=='linear':self.layers=nn.Linear(dim,2);final=self.layers
        elif architecture=='spatial':
            self.layers=nn.Sequential(nn.Conv2d(dim,64,3,padding=1),nn.GELU(),nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,2,1));final=self.layers[-1]
        else:raise ValueError('Unknown offset architecture')
        nn.init.zeros_(final.weight);nn.init.zeros_(final.bias)
    def forward(self,keys):
        single=keys.ndim==2
        if single:keys=keys[None]
        gx,gy=self.grid
        if keys.ndim!=3 or keys.shape[1:]!=(gx*gy,self.dim):raise ValueError('Offset head key shape mismatch')
        if self.architecture=='linear':raw=self.layers(keys)
        else:raw=self.layers(keys.transpose(1,2).reshape(keys.shape[0],self.dim,gy,gx)).flatten(2).transpose(1,2)
        offsets=raw.tanh()*self.max_offset_crops
        return offsets[0] if single else offsets


def predict_center_offsets(head,memory):return head(memory['keys'] if isinstance(memory,dict) else memory)


def offset_loss(prediction,target,weights,known=True):
    if not known:return prediction.sum()*0
    weights=weights.to(prediction);target=target.to(prediction)
    if prediction.shape!=target.shape or weights.shape!=prediction.shape[:-1]:raise ValueError('Offset loss shape mismatch')
    if (weights<0).any() or not torch.isfinite(weights).all():raise ValueError('Invalid offset weights')
    return (F.smooth_l1_loss(prediction,target,reduction='none').mean(-1)*weights).sum()


def shifted_candidate_coords(anchor_coords,offsets,image_size,strength=1.,crop_size=224):
    """Continuous coordinates, clamped inside image; actual crops round pixel centers."""
    if not 0<=strength<=1 or crop_size!=224:raise ValueError('Invalid offset inference protocol')
    if anchor_coords.shape!=offsets.shape or not torch.isfinite(offsets).all():raise ValueError('Invalid offset candidates')
    if strength==0:return anchor_coords.clone()
    size=anchor_coords.new_tensor(image_size);shifted=anchor_coords+strength*crop_size*offsets.to(anchor_coords)/size
    return shifted.clamp_min(0).minimum(1-1/size)


def offset_rectangles(coords,image_size,crop_size=224):
    size=torch.as_tensor(image_size,dtype=torch.float64,device=coords.device);centers=(coords.double()*size).round();low=centers-crop_size//2
    return torch.cat([low,low+crop_size],-1).to(coords.dtype)


def load_center_offset_head(path,base_sha256,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('kind')!='proposal_center_offset_1011' or checkpoint.get('base_checkpoint_sha256')!=base_sha256:raise ValueError('Offset head base hash mismatch')
    if state_sha256(checkpoint['head_state'])!=checkpoint['head_state_sha256']:raise ValueError('Offset head state hash mismatch')
    head=CenterOffsetHead(**checkpoint['config']);head.load_state_dict(checkpoint['head_state'],strict=True)
    return head.to(device).eval(),checkpoint


@torch.no_grad()
def center_offset_rollout(model,memory,image,geometry,offsets,budget=8,strength=1.,steps=16,seed=42,strategy='sample',max_iou=.3,original_relevance=None,ranking_relevance=None):
    """Refine reserved ROI centers only; free gaze actions stay on original grid."""
    from .proposal_budget_1011 import reservation_steps
    from .proposal_policy_1011 import policy_distribution,proposal_budget_indices
    from .data import patch_tensor
    if strategy not in ('sample','argmax') or not 0<=max_iou<=1:raise ValueError('Invalid offset rollout protocol')
    schedule=reservation_steps(budget,steps,'first');anchors=model.cell_xy;offsets=offsets.to(anchors).reshape_as(anchors);candidates=shifted_candidate_coords(anchors,offsets,image.size,strength)
    if original_relevance is None:
        logits=model.proposal(memory['keys']).squeeze(-1);valid=bool(model.proposal_valid and torch.isfinite(logits).all());relevance=logits.sigmoid() if valid else torch.zeros_like(logits)
    else:
        relevance=original_relevance.to(memory['prior']).reshape(1,-1);valid=bool(model.proposal_valid and torch.isfinite(relevance).all())
        if relevance.shape[-1]!=len(anchors):raise ValueError('Original proposal shape mismatch')
        if not valid:relevance=torch.zeros_like(relevance)
    rank_scores=relevance if ranking_relevance is None else ranking_relevance.to(relevance).reshape_as(relevance)
    if not torch.isfinite(rank_scores).all() or (rank_scores<0).any() or (rank_scores>1).any():raise ValueError('Invalid ROI ranking scores')
    legacy_zero=strength==0 or not bool(offsets.any());rectangles=geometry['rectangles'].to(anchors) if legacy_zero else offset_rectangles(candidates,image.size)
    ranked=proposal_budget_indices(rank_scores,rectangles,budget,max_iou) if budget and valid else [];reserved=dict(zip(schedule,ranked));prefix=memory['prior'].new_zeros(1,0,2);hidden=None;vector=None;generator=torch.Generator(device='cpu').manual_seed(seed);traces=[];indices=[]
    for t in range(steps):
        raw,hidden,_=model.step_history(memory,prefix,hidden,vector);dist=policy_distribution(raw,relevance,geometry,'P0',rho=model.rho,ablation=model.ablation,proposal_valid=valid)
        sampled=int(torch.multinomial(dist['q'].cpu(),1,generator=generator)) if strategy=='sample' else int(dist['q'].argmax(-1));index=reserved.get(t,sampled);point=candidates[index] if t in reserved else anchors[index];q=dist['q']
        if t in reserved:q=torch.zeros_like(q);q[:,index]=1
        traces.append(dict(step=t,source_anchor_index=index,reserved_proposal=t in reserved,q_entropy=float(-(q*q.clamp_min(torch.finfo(q.dtype).tiny).log()).sum()),chosen_source_probability=float(q[0,index]),raw_chosen_source_logp=float(raw[0,index])))
        indices.append(index);prefix=torch.cat([prefix,point[None,None]],1)
        if model.foveal:vector=model.visual.patch_vector(patch_tensor(image,prefix[0,-1].cpu().numpy()*image.size,224)[None].to(prefix.device))
    return dict(coordinates=prefix[0].cpu().tolist(),indices=indices,source_anchor_indices=indices,candidate_coords=candidates.cpu().tolist(),offset_predicted=offsets.cpu().tolist(),relevance=rank_scores[0].cpu().tolist(),original_relevance=relevance[0].cpu().tolist(),proposal_valid=valid,reserved_candidate_indices=ranked,reserved_steps=list(reserved),reservation_flags=[t in reserved for t in range(steps)],actual_crop_rectangles=offset_rectangles(prefix[0],image.size).cpu().tolist(),steps=traces,nms_geometry_protocol='legacy_zero_offset_baseline' if legacy_zero else 'rounded_actual_shifted224_rectangles',requested_roi_budget=budget,actual_roi_budget=len(ranked))
