"""Train-only horizontal image/path transforms with exact even-crop reflection."""
import numpy as np
import torch
from PIL import ImageOps
from .data import patch_tensor,seed_for


def mirror_coordinates(coords,mask=None):
    output=coords.clone();valid=torch.ones(coords.shape[:-1],dtype=torch.bool,device=coords.device) if mask is None else mask.bool()
    output[...,0]=torch.where(valid,1-coords[...,0],coords[...,0])
    if not torch.isfinite(output).all() or (output<0).any() or (output>1).any():raise ValueError('Mirror coordinates outside closed image extent')
    return output


def mirrored_crop_centers(coords,image_size):
    centers=np.round(coords.detach().cpu().numpy()*np.asarray(image_size)).astype(np.int64)
    centers[:,0]=image_size[0]-centers[:,0]
    return centers


def mirror_patch_tensor(image,coord,crop_size=224):
    if crop_size%2:raise ValueError('Exact center reflection requires even crop size')
    center=mirrored_crop_centers(coord[None],image.size)[0]
    return patch_tensor(ImageOps.mirror(image),center,crop_size)


def flip_for_epoch(image_id,epoch,seed=42,split='train'):
    if split!='train':raise ValueError('Validation/test image mirroring forbidden')
    return bool(np.random.default_rng(seed_for(seed+104729,int(image_id)+int(epoch)*100000)).random()<.5)
