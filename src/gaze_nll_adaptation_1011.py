"""Hard human-gaze NLL adaptation with a frozen visual dependency."""
import torch
from torch.nn import functional as F
from .trajectory_1011 import Trajectory1011
from .spatial import fourier_xy


class GazeNLLAdaptation1011(Trajectory1011):
    def maps_features(self,features):
        if getattr(self,'feature_memory_format',None)=='contiguous_nchw':features=[f.contiguous() for f in features]
        size=(self.grid[1],self.grid[0])
        if self.fpn:
            top=self.lateral[2](features[2]);middle=self.lateral[1](features[1])+F.interpolate(top,size=features[1].shape[-2:],mode='bilinear',align_corners=False)
            fine=self.lateral[0](features[0])+F.interpolate(middle,size=features[0].shape[-2:],mode='bilinear',align_corners=False)
            spatial=self.refine(F.interpolate(fine,size=size,mode='bilinear',align_corners=False))
        else:spatial=self.refine(F.interpolate(self.lateral[2](features[2]),size=size,mode='bilinear',align_corners=False))
        keys=spatial.flatten(2).transpose(1,2)+self.position(fourier_xy(self.cell_xy))[None]
        return dict(keys=keys,prior=self.density(spatial).flatten(1),global_context=keys.mean(1))
    def maps(self,image):return self.maps_features(self.visual(image))
    def train(self,mode=True):
        super().train(mode);self.visual.eval();return self


def initialize(base_model):
    config={k:v for k,v in base_model.model_config.items() if k not in ('ablation','rho')}
    model=GazeNLLAdaptation1011(**config,pretrained=False)
    state={k:v for k,v in base_model.state_dict().items() if not k.startswith('proposal.')}
    model.load_state_dict(state,strict=True);model.set_backbone('frozen');return model


def hard_nll(logp,coords,mask,grid=(24,48)):
    gx,gy=grid;ids=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1)
    values=-logp.log_softmax(-1).gather(-1,ids[...,None]).squeeze(-1)
    return ((values*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()


def prior_targets(coords,mask,grid=(24,48)):
    gx,gy=grid;ids=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1)
    targets=F.one_hot(ids,gx*gy).to(coords)*mask[...,None]
    return targets.sum(1)/mask.sum(1,keepdim=True).clamp_min(1)


def restore_adaptation(checkpoint,base_model):
    if checkpoint['kind']!='gaze_nll_adaptation_1011':raise ValueError('Wrong adaptation checkpoint')
    model=initialize(base_model)
    model.input_size=tuple(checkpoint['model_config']['input_size']);model.model_config['input_size']=list(model.input_size)
    model.feature_memory_format=checkpoint.get('feature_memory_format')
    missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    if unexpected or any(not k.startswith('visual.') for k in missing):raise ValueError('Adaptation nonvisual state mismatch')
    return model
