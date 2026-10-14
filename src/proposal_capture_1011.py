"""Predict whether a fixed224 action captures >=50% of a marked training box."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .proposal_utility_1011 import state_sha256,validate_cache


def capture_targets(image_id,patient_key,boxes,image_size,allowlist,grid=(24,48),crop_size=224):
    if allowlist.get(int(image_id))!=patient_key:raise ValueError('Capture target outside train image/patient allowlist')
    w,h=image_size;gx,gy=grid;rectangles=[]
    for x,y,bw,bh in boxes:
        if not np.isfinite([x,y,bw,bh]).all() or bw<=0 or bh<=0:raise ValueError('Invalid training box')
        rect=[max(0,x),max(0,y),min(w,x+bw),min(h,y+bh)]
        if rect[2]<=rect[0] or rect[3]<=rect[1]:raise ValueError('Training box outside image')
        rectangles.append(rect)
    ids=np.arange(gx*gy);centers=np.round(np.stack([((ids%gx)+.5)*w/gx,((ids//gx)+.5)*h/gy],-1));lo=centers-crop_size//2;hi=lo+crop_size;absolute=[]
    for box in rectangles:absolute.append(np.maximum(0,np.minimum(hi,box[2:])-np.maximum(lo,box[:2])).prod(-1)/((box[2]-box[0])*(box[3]-box[1])))
    raw=np.asarray(absolute).reshape(-1,gx*gy);raw_utility=raw.max(0) if rectangles else np.zeros(gx*gy);coverage=torch.tensor(raw,dtype=torch.float32);utility=torch.tensor(raw_utility,dtype=torch.float32);binary=torch.tensor(raw_utility>=.5,dtype=torch.float32)
    return dict(known=bool(rectangles),capture50=binary,absolute_utility=utility,absolute_coverage=coverage,attainable_capture50=bool(binary.any()))


class CaptureHead(nn.Module):
    def __init__(self,dim,architecture='linear',grid=(24,48)):
        super().__init__();self.dim=dim;self.grid=tuple(grid);self.architecture=architecture
        if architecture=='linear':self.layers=nn.Linear(dim,1)
        elif architecture=='spatial':self.layers=nn.Sequential(nn.Conv2d(dim,64,3,padding=1),nn.GELU(),nn.Conv2d(64,32,3,padding=1),nn.GELU(),nn.Conv2d(32,1,1))
        else:raise ValueError('Unknown capture architecture')
    def logits(self,keys):
        single=keys.ndim==2
        if single:keys=keys[None]
        gx,gy=self.grid
        if keys.ndim!=3 or keys.shape[1:]!=(gx*gy,self.dim):raise ValueError('Capture head key shape mismatch')
        if self.architecture=='linear':value=self.layers(keys).squeeze(-1)
        else:value=self.layers(keys.transpose(1,2).reshape(keys.shape[0],self.dim,gy,gx)).flatten(1)
        return value[0] if single else value
    def forward(self,keys):return self.logits(keys).sigmoid()


def predict_capture(head,memory):return head(memory['keys'] if isinstance(memory,dict) else memory)


def balanced_capture_bce(logits,target,known=True):
    zero=logits.sum()*0
    if not known:return zero,dict(positive=None,negative=None)
    target=target.to(logits)
    if not torch.all((target==0)|(target==1)):raise ValueError('Capture50 requires binary targets')
    residual=F.binary_cross_entropy_with_logits(logits,target,reduction='none');positive=target>0;negative=target==0;pos=residual[positive].mean() if positive.any() else zero;neg=residual[negative].mean() if negative.any() else zero
    return .5*(pos+neg),dict(positive=float(pos.detach()) if positive.any() else None,negative=float(neg.detach()) if negative.any() else None)


def load_capture_head(path,base_sha256,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('kind')!='proposal_capture_1011' or checkpoint.get('base_checkpoint_sha256')!=base_sha256:raise ValueError('Capture head base hash mismatch')
    if state_sha256(checkpoint['head_state'])!=checkpoint['head_state_sha256']:raise ValueError('Capture head state hash mismatch')
    config=checkpoint['config'];head=CaptureHead(config['dim'],config['architecture'],config['grid']);head.load_state_dict(checkpoint['head_state'],strict=True)
    return head.to(device).eval(),checkpoint
