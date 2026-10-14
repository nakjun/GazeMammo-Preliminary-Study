"""Clinical readout adaptation with frozen native visual features and honest history."""
import math
import torch
from torch.nn import functional as F


def freeze_features(model):
    for p in model.features.parameters():p.requires_grad_(False)
    model.features.eval();return model


def history_batch(coords,t,size):
    if not 1<=t<=coords.shape[1]:raise ValueError('DeepGaze history requires a real preceding fixation')
    result=coords.new_full((len(coords),4,2),float('nan'))
    for lag in range(min(t,4)):result[:,lag]=coords[:,t-1-lag]*coords.new_tensor(size)
    return result[:,:,0],result[:,:,1]


def prepare_saliency(model,features,size):
    w,h=size;shape=(math.ceil(h/model.downsample/model.readout_factor),math.ceil(w/model.downsample/model.readout_factor))
    readout=torch.cat([F.interpolate(f,shape) for f in features],1)
    return [network(readout) for network in model.saliency_networks]


def forward_saliency(model,saliency,size,centerbias,hx,hy):
    from deepgaze_pytorch.modules import encode_scanpath_features
    w,h=size;shape=saliency[0].shape[-2:];encoded=F.interpolate(encode_scanpath_features(hx,hy,(h,w),device=hx.device),shape);predictions=[]
    for static,scanpath,selection,finalizer in zip(saliency,model.scanpath_networks,model.fixation_selection_networks,model.finalizers):
        predictions.append(finalizer(selection((static,scanpath(encoded))),centerbias)[:,None])
    return (torch.cat(predictions,1)-math.log(len(predictions))).logsumexp(1,keepdim=True)


def head_state(model):return {k:v for k,v in model.state_dict().items() if not k.startswith('features.')}

def inventory(model):
    return dict(frozen_features=sum(p.numel() for p in model.features.parameters()),**{name:sum(p.numel() for p in getattr(model,name).parameters()) for name in ('saliency_networks','scanpath_networks','fixation_selection_networks','finalizers')},trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
