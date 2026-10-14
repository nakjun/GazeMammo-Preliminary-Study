"""Frozen-key crop utility regression; outputs are not clinical probabilities."""
import hashlib
import random
import numpy as np
import torch
from torch import nn
from .guided_trajectory_1011 import training_box_targets


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def state_sha256(state):
    h=hashlib.sha256()
    for key,value in sorted(state.items()):
        value=value.detach().cpu().contiguous(); h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


class ProposalUtilityHead(nn.Linear):
    def __init__(self,dim): super().__init__(dim,1)
    def forward(self,keys): return super().forward(keys).squeeze(-1).sigmoid()


def predict_utility(head,memory):
    return head(memory['keys'] if isinstance(memory,dict) else memory)


def utility_targets(image_id,patient_key,boxes,image_size,allowlist,grid=(24,48)):
    result=training_box_targets(image_id,patient_key,boxes,image_size,allowlist,grid,224)
    result['utility']=result['coverage'].amax(0) if result['known'] else torch.zeros(grid[0]*grid[1])
    return result


def balanced_mse(prediction,target,known=True):
    """Image-level class means, weighted .5 each; absent classes contribute zero."""
    zero=prediction.sum()*0
    if not known:return zero,dict(positive=None,negative=None)
    target=target.to(prediction); residual=(prediction-target).square(); positive=target>0; negative=target==0
    pos=residual[positive].mean() if positive.any() else zero
    neg=residual[negative].mean() if negative.any() else zero
    return .5*(pos+neg),dict(positive=float(pos.detach()) if positive.any() else None,negative=float(neg.detach()) if negative.any() else None)


def freeze_model(model):
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def validate_cache(cache,base_sha256,allowlist,annotation_sha256=None):
    if cache.get('kind')!='proposal_utility_features_1011' or cache.get('base_checkpoint_sha256')!=base_sha256:raise ValueError('Feature cache base checkpoint hash mismatch')
    if annotation_sha256 is not None and cache.get('annotation_sha256')!=annotation_sha256:raise ValueError('Feature cache annotation hash mismatch')
    records=cache['records']; ids=[int(r['image_id']) for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(allowlist):raise ValueError('Feature cache IDs must exactly equal train allowlist')
    dim=records[0]['keys'].shape[-1] if records else None
    for r in records:
        if allowlist.get(int(r['image_id']))!=r['patient_key']:raise ValueError('Feature cache patient outside train allowlist')
        keys=r['keys']
        if keys.requires_grad or keys.ndim!=2 or keys.shape[0]!=1152 or keys.shape[-1]!=dim or not torch.isfinite(keys).all():raise ValueError('Invalid frozen feature keys')
        for name in ('utility','occupancy'):
            t=r[name]
            if t.shape!=(1152,) or not torch.isfinite(t).all() or (t<0).any() or (t>1.000001).any():raise ValueError('Invalid cached targets')
    return records


def load_utility_head(path,base_sha256,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('kind')!='proposal_utility_1011' or checkpoint.get('base_checkpoint_sha256')!=base_sha256:raise ValueError('Utility head base checkpoint hash mismatch')
    if state_sha256(checkpoint['head_state'])!=checkpoint['head_state_sha256']:raise ValueError('Utility head state hash mismatch')
    head=ProposalUtilityHead(checkpoint['config']['dim']);head.load_state_dict(checkpoint['head_state'],strict=True)
    return head.to(device).eval(),checkpoint
