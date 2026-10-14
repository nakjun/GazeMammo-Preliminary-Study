"""CXR-inspired dual-source relational scoring; raw patch values stay unmixed."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .gaze_classifier_1011 import GazeDiagnosisHead
from .gaze_attention_1011 import attention_loss,head_state,validate_sources
VARIANTS=('dual_mean','dual_cross','dual_cross_drop','dual_cross_drop_quality')


def inbounds_fraction(coordinates,image_size=(1500,3000),crop_size=224):
    """Exact PIL crop intersection area using the existing rounded-center rule."""
    is_tensor=torch.is_tensor(coordinates)
    xy=coordinates if is_tensor else torch.as_tensor(coordinates,dtype=torch.float64)
    # numpy float32 coordinates times integer image.size promote to float64 in patch_tensor callers.
    work=xy.to(dtype=torch.float64);size=work.new_tensor(image_size);center=torch.round(work*size);lo=center-crop_size//2;hi=lo+crop_size
    overlap=(torch.minimum(hi,size)-torch.maximum(lo,torch.zeros_like(lo))).clamp_min(0)
    quality=overlap.prod(-1)/float(crop_size**2)
    return quality.to(xy.dtype) if is_tensor and xy.is_floating_point() else (quality if is_tensor else quality.numpy())


def masked_probability(scores,mask):
    scores=scores.masked_fill(~mask,-1e9);prob=scores.softmax(-1)*mask
    return prob/prob.sum(-1,keepdim=True).clamp_min(1e-12)

def counts(coordinates,mask):
    equal=((coordinates[:,:,None]-coordinates[:,None,:]).abs().amax(-1)<=1e-6)&mask[:,:,None]&mask[:,None,:]
    return equal.sum(-1).clamp_min(1).to(coordinates.dtype)

class GazeCXR1011(nn.Module):
    def __init__(self,global_model,variant='dual_cross_drop_quality',input_dim=1024,dim=64,dropout=.2):
        super().__init__()
        if variant not in VARIANTS or global_model.mode!='global':raise ValueError('Invalid CXR/global configuration')
        self.config=dict(variant=variant,input_dim=input_dim,dim=dim,dropout=dropout);self.variant=variant
        self.global_model=global_model.requires_grad_(False).eval();gdim=global_model.config['dim']
        self.value=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,dim),nn.GELU())
        self.cross_enabled=variant!='dual_mean';self.quality_enabled=variant.endswith('quality');self.modality_dropout=.3 if '_drop' in variant else 0.
        if self.cross_enabled:
            self.position=nn.Sequential(nn.Linear(2,dim),nn.Tanh());self.query=nn.Linear(dim,dim,bias=False);self.key=nn.Linear(dim,dim,bias=False)
            self.self_score=nn.Linear(dim,dim);self.cross_score=nn.Linear(dim,dim,bias=False);self.global_score=nn.Linear(gdim,dim,bias=False);self.score_gate=nn.Linear(dim,dim);self.score=nn.Linear(dim,1,bias=False)
        self.source_gate=nn.Linear(gdim+2*dim,2);self.dropout=nn.Dropout(dropout);self.gate=nn.Linear(gdim+dim,1);self.delta=nn.Linear(dim,1);self.local_head=nn.Linear(dim,1)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9));nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def train(self,mode=True):super().train(mode);self.global_model.eval();return self
    def cross_context(self,query_values,query_coords,query_mask,key_values,key_coords,key_mask,key_counts,key_quality):
        q=self.query(query_values+self.position(query_coords));k=self.key(key_values+self.position(key_coords))
        scores=torch.einsum('bid,bjd->bij',q,k)/math.sqrt(q.shape[-1])-key_counts.log()[:,None]
        if self.quality_enabled:scores=scores+key_quality.clamp_min(1e-12).log()[:,None]
        weights=masked_probability(scores,key_mask[:,None].expand_as(scores))
        return torch.einsum('bij,bjd->bid',weights,key_values)*query_mask[...,None],weights
    def forward(self,global_features,gaze_features,gaze_coordinates,grid_features,grid_coordinates,gaze_mask=None,grid_mask=None,force_gaze_absent=False,force_grid_absent=False,uniform=False):
        sources=[(gaze_features,gaze_coordinates,gaze_mask),(grid_features,grid_coordinates,grid_mask)];values=[];coords=[];masks=[];quality=[];duplicates=[]
        for source,(features,xy,mask) in enumerate(sources):
            if features.shape[:2]!=xy.shape[:2] or xy.shape[-1]!=2:raise ValueError('Patch/coordinate shape mismatch')
            mask=torch.ones(xy.shape[:2],dtype=torch.bool,device=xy.device) if mask is None else mask.bool().clone()
            if (source==0 and force_gaze_absent) or (source==1 and force_grid_absent):mask[:]=False
            if source==0 and self.training and self.modality_dropout:mask=mask&~(torch.rand(len(mask),device=xy.device)<self.modality_dropout)[:,None]
            xy=torch.where(mask[...,None],xy,torch.zeros_like(xy));features=torch.where(mask[...,None],features,torch.zeros_like(features))
            q=inbounds_fraction(xy)
            if self.quality_enabled:mask=mask&(q>0)
            xy=torch.where(mask[...,None],xy,torch.zeros_like(xy));features=torch.where(mask[...,None],features,torch.zeros_like(features))
            coords.append(xy);values.append(self.value(features));masks.append(mask);quality.append(q);duplicates.append(counts(xy,mask))
        with torch.no_grad():g=self.global_model.global_project(global_features);global_logit=self.global_model.head(g).squeeze(-1)
        contexts=[torch.zeros_like(v) for v in values];cross_weights=[None,None]
        if self.cross_enabled:
            for i in (0,1):contexts[i],cross_weights[i]=self.cross_context(values[i],coords[i],masks[i],values[1-i],coords[1-i],masks[1-i],duplicates[1-i],quality[1-i])
        attention=[];pooled=[]
        for i in (0,1):
            score=values[i].new_zeros(masks[i].shape)
            if self.cross_enabled and not uniform:
                h=self.self_score(values[i])+self.position(coords[i])+self.cross_score(contexts[i])+self.global_score(g)[:,None]
                score=self.score(h.tanh()*self.score_gate(values[i]).sigmoid()).squeeze(-1)
            score=score-duplicates[i].log()
            if self.quality_enabled:score=score+quality[i].clamp_min(1e-12).log()
            a=masked_probability(score,masks[i]);attention.append(a);pooled.append((a[...,None]*values[i]).sum(1))
        available=torch.stack([mask.any(-1) for mask in masks],-1);beta=masked_probability(self.source_gate(torch.cat([g,*pooled],-1)),available)
        z=beta[:,0,None]*pooled[0]+beta[:,1,None]*pooled[1];dropped=self.dropout(z);valid=available.any(-1)
        alpha=self.gate(torch.cat([g,dropped],-1)).sigmoid().squeeze(-1)*valid;residual=alpha*self.delta(dropped).squeeze(-1)
        signed=[alpha[:,None]*beta[:,i,None]*attention[i]*F.linear(values[i],self.delta.weight,None).squeeze(-1) for i in (0,1)]
        return dict(logits=global_logit+residual,global_logits=global_logit,local_logits=self.local_head(dropped).squeeze(-1)*valid,alpha=alpha,residual=residual,residual_bias=alpha*self.delta.bias[0],valid_bag=valid,beta=beta,gaze_attention=attention[0],grid_attention=attention[1],gaze_signed=signed[0],grid_signed=signed[1],gaze_quality=quality[0],grid_quality=quality[1],gaze_mask=masks[0],grid_mask=masks[1],gaze_context=contexts[0],grid_context=contexts[1])


def restore_head(checkpoint,global_checkpoint):
    if checkpoint['kind']!='gaze_cxr_1011':raise ValueError('Wrong CXR checkpoint')
    global_model=GazeDiagnosisHead(**global_checkpoint['model_config']);global_model.load_state_dict(global_checkpoint['state'])
    model=GazeCXR1011(global_model,**checkpoint['model_config']);missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    if unexpected or any(not key.startswith('global_model.') for key in missing):raise ValueError('CXR state mismatch')
    return model
