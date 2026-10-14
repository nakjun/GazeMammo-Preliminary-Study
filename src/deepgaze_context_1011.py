"""Train-only center bias and exact native DeepGaze feature caching."""
import math
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

def train_centerbias(records,manifest,grid=(24,48),sigma=1.,floor=1e-8):
    frame=pd.read_csv(manifest);allowed={int(r.image_id):str(r.patient_key) for _,r in frame[frame.split=='train'].iterrows()};train=[r for r in records if r['split']=='train']
    if len(train)!=412 or len({int(r['image_id']) for r in train})!=412 or set(int(r['image_id']) for r in train)!=set(allowed):raise ValueError('Exact412 train centerbias allowlist required')
    gx,gy=grid;density=np.zeros((gy,gx),dtype=np.float64)
    for record in train:
        if record['patient_key']!=allowed[int(record['image_id'])]:raise ValueError('Train patient mismatch')
        mask=record['loss_mask'].bool();coords=record['coords'][mask].numpy()
        if not len(coords) or not np.isfinite(coords).all() or (coords<0).any() or (coords>1).any():raise ValueError('Invalid canonical train gaze')
        x=np.minimum((coords[:,0]*gx).astype(int),gx-1);y=np.minimum((coords[:,1]*gy).astype(int),gy-1);np.add.at(density,(y,x),1/len(coords))
    density=gaussian_filter(density,sigma=sigma,mode='reflect');density=np.maximum(density,floor);density/=density.sum();return torch.tensor(density,dtype=torch.float32)

def dense_centerbias(probability,size,device):
    dense=F.interpolate(probability[None,None].to(device),size=(size[1],size[0]),mode='bilinear',align_corners=False)[:,0];dense=dense.clamp_min(1e-30);dense=dense/dense.sum((-1,-2),keepdim=True);return dense.log()

def native_features(model,rgb):return tuple(x.detach() for x in model.features(F.interpolate(rgb,scale_factor=1/model.downsample,recompute_scale_factor=False)))

def forward_native(model,features,image_size,centerbias,x_hist,y_hist):
    from deepgaze_pytorch.modules import encode_scanpath_features
    width,height=image_size;shape=[math.ceil(height/model.downsample/model.readout_factor),math.ceil(width/model.downsample/model.readout_factor)];readout=torch.cat([F.interpolate(x,shape) for x in features],1);predictions=[]
    for saliency,scanpath,selection,finalizer in zip(model.saliency_networks,model.scanpath_networks,model.fixation_selection_networks,model.finalizers):
        x=saliency(readout);encoded=encode_scanpath_features(x_hist,y_hist,size=(height,width),device=x.device);y=scanpath(F.interpolate(encoded,shape));predictions.append(finalizer(selection((x,y)),centerbias)[:,None])
    return (torch.cat(predictions,1)-math.log(len(predictions))).logsumexp(1,keepdim=True)
