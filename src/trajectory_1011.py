"""Causal location history experiments. Coordinates are gaze samples, not fixations."""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .spatial import SpatialTrajectory, gaussian_targets
from .data import reduce_gaze


def segments(points, width, height, min_distance=32):
    return reduce_gaze(points, width, height, k=max(len(points), 1), min_distance=min_distance)


def window_sample(parts, image_id, seed=42, epoch=0, mode='contiguous', length=16,
                  burn_in=8, mixed_ratio=.5, start=None, segment=None):
    if mode not in ('legacy16', 'contiguous', 'mixed'): raise ValueError(mode)
    if length < 1 or burn_in < 0 or not 0 <= mixed_ratio <= 1: raise ValueError('Invalid window settings')
    if not parts: raise ValueError('No valid gaze segment')
    digest=hashlib.sha256(f'{seed}:{epoch}:{image_id}'.encode()).digest()
    rng=np.random.default_rng(int.from_bytes(digest[:8], 'little'))
    selected='legacy16' if mode=='mixed' and rng.random()<mixed_ratio else ('contiguous' if mode=='mixed' else mode)
    if selected=='legacy16':
        # Exactly preserve reduce_gaze(..., k=16), including its tie selection.
        reduced=[p[np.linspace(0,len(p)-1,16).round().astype(int)] if len(p)>16 else p for p in parts]
        segment=max(range(len(reduced)), key=lambda i:len(reduced[i])); p=reduced[segment]; start=0; prefix=0
        positions=np.linspace(0,len(parts[segment])-1,16).round().astype(int) if len(parts[segment])>16 else np.arange(len(p))
    else:
        segment=int(rng.integers(len(parts))) if segment is None else segment
        source=parts[segment]
        start=int(rng.integers(max(1,len(source)-length+1))) if start is None else start
        if not 0 <= start < len(source): raise ValueError('Invalid start')
        prefix=min(start,burn_in); lo=start-prefix; p=source[lo:start+length]; positions=np.arange(lo,lo+len(p))
    total=max(16,length)+burn_in
    coords=torch.zeros(total,2); valid=torch.zeros(total,dtype=torch.bool); loss_mask=valid.clone()
    coords[:len(p)]=torch.as_tensor(p,dtype=torch.float32); valid[:len(p)]=True; loss_mask[prefix:len(p)]=True
    pos=torch.full((total,),-1,dtype=torch.long); pos[:len(p)]=torch.as_tensor(positions)
    return dict(coords=coords,valid=valid,loss_mask=loss_mask,positions=pos,segment=int(segment),start=int(start),
                prefix=prefix,truncated_history=bool(start>prefix),mode=selected)


class Trajectory1011(SpatialTrajectory):
    def __init__(self, spatial_history=False, behavior_mixture=False, history_length=4, **kwargs):
        super().__init__(**kwargs)
        if history_length not in (1,2,4): raise ValueError("history_length must be 1, 2, or 4")
        self.history_length=history_length
        self.spatial_history=spatial_history; self.behavior_mixture=behavior_mixture
        self.model_config.update(spatial_history=spatial_history,behavior_mixture=behavior_mixture,history_length=history_length)
        dim=self.bos.numel()
        if spatial_history: self.history_head=nn.Sequential(nn.Linear(dim+4*history_length,dim),nn.GELU(),nn.Linear(dim,1))
        if behavior_mixture:
            self.behavior_gate=nn.Linear(dim,3); self.local_scale=nn.Parameter(torch.tensor(-2.)); self.return_scale=nn.Parameter(torch.tensor(-3.)); self.return_decay=nn.Parameter(torch.tensor(-2.))
        gx,gy=self.grid; ids=torch.arange(self.cells)
        self.register_buffer('cell_xy',torch.stack([((ids%gx)+.5)/gx,((ids//gx)+.5)/gy],-1),persistent=False)

    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        """Prediction at len(prefix); uses only past positions and past crops."""
        b=memory['prior'].shape[0]; index=prefix.shape[1]; zero=memory['prior'].new_zeros(b,2)
        previous=prefix[:,-1] if index else zero; older=prefix[:,-2] if index>1 else zero
        logits,hidden=super().step(memory,previous,older,patch_vector,hidden,index)
        if self.spatial_history:
            maps=[]
            for lag in range(1,self.history_length+1):
                delta=self.cell_xy[None]-prefix[:,-lag,None] if index>=lag else zero[:,None].expand(-1,self.cells,-1)
                valid=logits.new_full((b,self.cells,1),float(index>=lag))
                maps.extend([delta,delta.square().sum(-1,keepdim=True).sqrt(),valid])
            logits=logits+self.history_head(torch.cat([memory['keys'],*maps],-1)).squeeze(-1)
        global_logp=logits.log_softmax(-1)
        if not self.behavior_mixture:return global_logp,hidden,dict(gates=logits.new_tensor([1.,0.,0.]).expand(b,-1),branches=global_logp[:,None])
        gate=self.behavior_gate(hidden[-1]); branches=[global_logp]
        if index:
            local=-((self.cell_xy[None]-previous[:,None]).square().sum(-1))/(2*(.01+F.softplus(self.local_scale))**2)
            ages=torch.arange(index-1,-1,-1,device=logits.device,dtype=logits.dtype)
            ret=-((self.cell_xy[None,:,None]-prefix[:,None]).square().sum(-1))/(2*(.01+F.softplus(self.return_scale))**2)
            ret=(ret-F.softplus(self.return_decay)*ages).logsumexp(-1)
            branches.extend([local.log_softmax(-1),ret.log_softmax(-1)])
        else:
            gate=gate.clone(); gate[:,1:]=-torch.inf; branches.extend([global_logp,global_logp])
        branches=torch.stack(branches,1); loggate=gate.log_softmax(-1)
        return (branches+loggate[:,:,None]).logsumexp(1),hidden,dict(gates=loggate.exp(),branches=branches)

    def teacher_logits(self,memory,coords,patch_vectors=None,order='original',return_diagnostics=False):
        hidden=None; outputs=[]; gates=[]
        for index in range(coords.shape[1]):
            vectors=patch_vectors[:,index] if patch_vectors is not None else None
            logits,hidden,diag=self.step_history(memory,coords[:,:index],hidden,vectors)
            outputs.append(logits); gates.append(diag['gates'])
        result=torch.stack(outputs,1)
        return (result,torch.stack(gates,1)) if return_diagnostics else result


def objective(logits,coords,mask,nll_weight=.5,sigma=96,sizes=(1500,3000),grid=(24,48)):
    if not 0<=nll_weight<=1:raise ValueError('nll_weight must be in [0,1]')
    gx,gy=grid; ids=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1)
    logp=logits.log_softmax(-1); nll=-logp.gather(-1,ids[...,None]).squeeze(-1)
    soft=-(gaussian_targets(coords,grid,sigma,sizes)*logp).sum(-1)
    # Per-image mean maintains uniform image weighting even for short segments.
    counts=mask.sum(-1).clamp_min(1)
    return (((nll_weight*nll+(1-nll_weight)*soft)*mask).sum(-1)/counts).mean(), nll


def check_provenance(checkpoint,allowed_train,allowed_selection=None,evaluation=None,independent=False):
    if 'training_patient_keys' not in checkpoint or 'selection_patient_keys' not in checkpoint:raise ValueError('Missing patient provenance')
    training=set(checkpoint['training_patient_keys']); selection=set(checkpoint['selection_patient_keys'])
    if training-set(allowed_train):raise ValueError('Initializer training patient leakage')
    if allowed_selection is not None and selection-set(allowed_selection):raise ValueError('Initializer selection patient leakage')
    if training&selection:raise ValueError('Training/selection patient overlap')
    if evaluation is not None and training&set(evaluation):raise ValueError('Evaluation/training overlap')
    if independent and evaluation is not None and selection&set(evaluation):raise ValueError('Evaluation patients selected this checkpoint; not independent holdout')


def initialize(model,checkpoint):
    if checkpoint.get('kind') not in ('spatial_v2','trajectory_1011'):raise ValueError('Expected gaze checkpoint')
    config=checkpoint['model_config']
    for key in ('grid','dim','backbone_name','fpn','motion_bias'):
        if key=='motion_bias' and not config.get(key,False): continue
        if config.get(key, 'resnet50' if key=='backbone_name' else False)!=model.model_config.get(key):raise ValueError(f'Initializer architecture mismatch: {key}')
    missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    allowed=('history_head.','behavior_gate.','local_scale','return_scale','return_decay','motion.')
    if unexpected or any(not name.startswith(allowed) and name!='motion_scale' for name in missing):raise ValueError(f'Architecture mismatch: {missing}, {unexpected}')
    return dict(missing=missing,unexpected=unexpected)


def restore(checkpoint):
    if checkpoint.get('kind')!='trajectory_1011':raise ValueError('Expected trajectory_1011 checkpoint')
    model=Trajectory1011(**checkpoint['model_config'],pretrained=False); model.load_state_dict(checkpoint['state']); return model
