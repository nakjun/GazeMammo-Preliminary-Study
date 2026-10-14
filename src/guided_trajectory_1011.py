"""Weak-box guided gaze adaptation; discrete predicted histories are detached."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage
from .trajectory_1011 import Trajectory1011
from .constraints import breast_mask
from .data import patch_tensor
ABLATIONS=('gaze_only','geometry','lesion','combined')


def candidate_geometry(image,grid=(24,48),crop_size=224,mask=None):
    w,h=image.size;gx,gy=grid;ids=np.arange(gx*gy);xy=np.stack([((ids%gx)+.5)/gx,((ids//gx)+.5)/gy],-1)
    centers=np.round(xy*np.array([w,h])).astype(int);rect=np.concatenate([centers-crop_size//2,centers-crop_size//2+crop_size],-1)
    clipped=np.concatenate([np.maximum(rect[:,:2],[0,0]),np.minimum(rect[:,2:],[w,h])],-1);area=np.maximum(0,clipped[:,2:]-clipped[:,:2]).prod(-1)/crop_size**2
    valid=True;error=None;fraction=np.zeros(len(ids));edge=np.zeros(len(ids));distance=np.zeros(len(ids))
    try:
        if mask is None:mask=breast_mask(image)
        else:
            raw_mask=np.asarray(mask)
            if not np.isfinite(raw_mask).all():raise ValueError('Nonfinite estimated breast mask')
            mask=raw_mask.astype(bool)
        if mask.shape!=(h,w) or not mask.any():raise ValueError('Empty/invalid estimated breast mask')
        integral=np.pad(mask.astype(np.int64),((1,0),(1,0))).cumsum(0).cumsum(1)
        x0,y0,x1,y1=clipped.T;fraction=(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])/crop_size**2
        edt=ndimage.distance_transform_edt(np.pad(mask,1,constant_values=False))[1:-1,1:-1]
        cx=np.clip(centers[:,0],0,w-1);cy=np.clip(centers[:,1],0,h-1);inside=mask[cy,cx];distance=edt[cy,cx];edge=inside*np.exp(-distance/112.)
    except Exception as exc:valid=False;error=str(exc)
    return dict(coords=torch.tensor(xy,dtype=torch.float32),rectangles=torch.tensor(rect,dtype=torch.float32),inbounds=torch.tensor(area,dtype=torch.float32),foreground=torch.tensor(fraction,dtype=torch.float32),background=torch.tensor(1-fraction if valid else np.zeros_like(fraction),dtype=torch.float32),edge=torch.tensor(edge,dtype=torch.float32),distance=torch.tensor(distance,dtype=torch.float32),geometry_valid=valid,geometry_error=error,image_size=(w,h))


def union_area(rectangles):
    if not rectangles:return 0.
    xs=sorted({float(r[0]) for r in rectangles}|{float(r[2]) for r in rectangles});area=0.
    for a,b in zip(xs,xs[1:]):
        intervals=sorted((r[1],r[3]) for r in rectangles if r[0]<b and r[2]>a);covered=0.;last=None
        for lo,hi in intervals:
            if last is None:covered+=hi-lo;last=hi
            elif hi>last:covered+=hi-max(last,lo);last=hi
        area+=(b-a)*covered
    return area


def training_box_targets(image_id,patient_key,boxes,image_size,allowlist,grid=(24,48),crop_size=224):
    if allowlist.get(int(image_id))!=patient_key:raise ValueError('Training target image/patient outside train allowlist')
    w,h=image_size;gx,gy=grid;rectangles=[]
    for x,y,bw,bh in boxes:
        if not np.isfinite([x,y,bw,bh]).all() or bw<=0 or bh<=0:raise ValueError('Invalid training box')
        r=[max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)]
        if r[2]<=r[0] or r[3]<=r[1]:raise ValueError('Box outside image')
        rectangles.append(r)
    ids=np.arange(gx*gy);lo=np.stack([(ids%gx)*w/gx,(ids//gx)*h/gy],-1);hi=lo+[w/gx,h/gy];occupancy=[]
    for low,high in zip(lo,hi):
        intersections=[]
        for box in rectangles:
            r=[max(low[0],box[0]),max(low[1],box[1]),min(high[0],box[2]),min(high[1],box[3])]
            if r[2]>r[0] and r[3]>r[1]:intersections.append(r)
        occupancy.append(union_area(intersections)/(w*h/(gx*gy)))
    centers=np.round((lo+hi)/2);crop_lo=centers-crop_size//2;crop_hi=crop_lo+crop_size;coverage=[]
    for box in rectangles:
        overlap=np.maximum(0,np.minimum(crop_hi,box[2:])-np.maximum(crop_lo,box[:2])).prod(-1)/((box[2]-box[0])*(box[3]-box[1]));best=float(overlap.max())
        if best<=0:raise ValueError('No candidate crop covers this training box')
        coverage.append(overlap/best)
    return dict(known=bool(rectangles),occupancy=torch.tensor(occupancy,dtype=torch.float32),coverage=torch.tensor(np.asarray(coverage).reshape(-1,gx*gy),dtype=torch.float32),boxes=rectangles)


def balanced_proposal_loss(logits,target,known=True):
    if not known:return logits.sum()*0
    y=target.to(logits);positive=y.sum();negative=(1-y).sum();loss=logits.sum()*0
    if positive>0:loss=loss-(y*F.logsigmoid(logits)).sum()/positive
    if negative>0:loss=loss-((1-y)*F.logsigmoid(-logits)).sum()/negative
    return .5*loss


def visit_loss(probabilities,coverage,valid=None):
    if not len(coverage) or probabilities.shape[-2]==0 or (valid is not None and not bool(valid.any())):return probabilities.sum()*0
    q=probabilities if valid is None else probabilities*valid[...,None]
    u=(q@coverage.to(q).T).clamp(0,1-1e-7)
    logmiss=torch.log1p(-u).sum(-2);hit=-torch.expm1(logmiss)
    return -hit.clamp_min(torch.finfo(q.dtype).tiny).log().mean()


def crop_iou(rectangles,past_rectangles):
    if not len(past_rectangles):return rectangles.new_zeros(len(rectangles))
    intersection=(torch.minimum(rectangles[:,None,2:],past_rectangles[None,:,2:])-torch.maximum(rectangles[:,None,:2],past_rectangles[None,:,:2])).clamp_min(0).prod(-1)
    areas=(rectangles[:,2:]-rectangles[:,:2]).prod(-1)[:,None];past=(past_rectangles[:,2:]-past_rectangles[:,:2]).prod(-1)[None]
    return (intersection/(areas+past-intersection).clamp_min(1)).max(-1).values


def free_losses(probabilities,geometry,relevance,coverage,sampled_indices,valid=None):
    # Single-image rollout; both distributions and losses remain differentiable.
    q=probabilities;valid=torch.ones(len(q),dtype=torch.bool,device=q.device) if valid is None else valid.bool();r=relevance.detach().to(q)
    background=geometry['background'].to(q);edge=geometry['edge'].to(q);den=valid.sum().clamp_min(1)
    geo=((q*(background+.25*edge*(1-r))).sum(-1)*valid).sum()/den if geometry['geometry_valid'] else q.sum()*0
    repeat=q.sum()*0;rect=geometry['rectangles'].to(q);ids=sampled_indices.detach()
    previous=[];count=0
    for t in range(len(q)):
        if not valid[t]:continue
        if previous:
            cost=crop_iou(rect,rect[torch.tensor(previous,device=q.device)])*(1-.75*r);repeat=repeat+(q[t]*cost).sum();count+=1
        previous.append(int(ids[t]))
    repeat=repeat/max(count,1) if geometry['geometry_valid'] else q.sum()*0
    return dict(geometry=geo,visit=visit_loss(q,coverage,valid),repeat=repeat)

class GuidedTrajectory1011(Trajectory1011):
    def __init__(self,ablation='combined',rho=1.,**config):
        super().__init__(**config)
        if ablation not in ABLATIONS or not 0<=rho<=1:raise ValueError('Invalid guided configuration')
        self.ablation=ablation;self.rho=rho;self.proposal=nn.Linear(self.bos.numel(),1);self.proposal_valid=True
        self.model_config.update(ablation=ablation,rho=rho)
        self.set_backbone('frozen')
    def set_backbone(self,mode):super().set_backbone('frozen')
    def train(self,mode=True):
        super().train(mode);self.visual.eval();return self
    def guide_step(self,memory,prefix,geometry,hidden=None,patch_vector=None,rho=None):
        base,hidden,diagnostic=super().step_history(memory,prefix,hidden,patch_vector);prop=self.proposal(memory['keys']).squeeze(-1);r=prop.sigmoid().detach()
        lesion=self.ablation in ('lesion','combined');geo=self.ablation in ('geometry','combined');bias=torch.zeros_like(base)
        if lesion and self.proposal_valid and torch.isfinite(prop).all():bias=bias+2*r
        else:r=torch.zeros_like(r)
        if geo and geometry['geometry_valid']:bias=bias-3*geometry['background'].to(base)-.5*geometry['edge'].to(base)*(1-r)
        strength=self.rho if rho is None else rho
        guided=(base+strength*bias).log_softmax(-1);diagnostic.update(base_logp=base,guided_logp=guided,proposal_logits=prop,relevance=r,guidance_bias=bias)
        return guided,hidden,diagnostic
    def teacher_guided(self,memory,coords,geometry,patch_vectors=None,rho=None):
        hidden=None;outputs=[]
        for t in range(coords.shape[1]):
            logp,hidden,_=self.guide_step(memory,coords[:,:t],geometry,hidden,patch_vectors[:,t] if patch_vectors is not None else None,rho);outputs.append(logp)
        return torch.stack(outputs,1)
    def predicted_rollout(self,memory,image,geometry,steps=16,seed=42,strategy='sample',rho=None):
        prefix=memory['prior'].new_zeros(1,0,2);hidden=None;vector=None;outputs=[];base=[];ids=[];generator=torch.Generator().manual_seed(seed)
        for t in range(steps):
            logp,hidden,diag=self.guide_step(memory,prefix,geometry,hidden,vector,rho);index=int(logp.argmax(-1)) if strategy=='argmax' else int(torch.multinomial(logp.exp().detach().cpu(),1,generator=generator))
            outputs.append(logp[0]);base.append(diag['base_logp'][0]);ids.append(index);prefix=torch.cat([prefix,self.cell_xy[index][None,None].detach()],1)
            if self.foveal:
                with torch.no_grad():vector=self.visual.patch_vector(patch_tensor(image,prefix[0,-1].detach().cpu().numpy()*image.size)[None].to(prefix.device))
        return dict(logp=torch.stack(outputs),base_logp=torch.stack(base),points=prefix[0].detach(),indices=torch.tensor(ids,device=prefix.device),relevance=diag['relevance'][0].detach())


def rho_schedule(epoch):return 0. if epoch<=2 else min(1.,(epoch-2)/3.)
def nonvisual_state(model):return {key:value for key,value in model.state_dict().items() if not key.startswith('visual.')}
def initialize_from_base(checkpoint,ablation='combined',rho=1.):
    config=dict(checkpoint['model_config']);config.pop('ablation',None);config.pop('rho',None)
    model=GuidedTrajectory1011(**config,ablation=ablation,rho=rho,pretrained=False);missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    if unexpected or set(missing)!={'proposal.weight','proposal.bias'}:raise ValueError('Base checkpoint mismatch')
    return model

def restore_guided(checkpoint,base_checkpoint):
    if checkpoint['kind']!='guided_trajectory_1011':raise ValueError('Wrong guided checkpoint')
    config=dict(checkpoint['model_config']);model=GuidedTrajectory1011(**config,pretrained=False)
    missing,unexpected=model.load_state_dict(base_checkpoint['state'],strict=False)
    if unexpected or set(missing)!={'proposal.weight','proposal.bias'}:raise ValueError('Base restore mismatch')
    missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    if unexpected or any(not key.startswith('visual.') for key in missing):raise ValueError('Nonvisual guided restore mismatch')
    return model
