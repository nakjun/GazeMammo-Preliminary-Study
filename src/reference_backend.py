"""Reference Mask2Former pixel encoder used by HAT/ChestSearch.

Loads released author weights. Their pure-PyTorch deformable-attention reference
is used instead of the compiled CUDA extension; this is not an FPN substitute.
Third-party algorithm classes are loaded from the archived upstream sources.
"""
import ast
import importlib.util
import math
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Callable,Dict,List,Optional,Tuple,Union
import copy
import logging
import numpy as np
import torch
from torch import nn,Tensor
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_,constant_,uniform_,normal_
from torchvision.models import resnet50
from .data import ROOT

UPSTREAM=ROOT/'literature/code/HAT/hat'

class Conv2d(nn.Conv2d):
    def __init__(self,*args,norm=None,activation=None,**kwargs):
        super().__init__(*args,**kwargs);self.norm=norm;self.activation=activation
    def forward(self,x):
        x=F.conv2d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)
        if self.norm is not None:x=self.norm(x)
        if self.activation is not None:x=self.activation(x)
        return x

def get_norm(name,channels):
    if not name:return None
    if name=='GN':return nn.GroupNorm(32,channels)
    raise ValueError('Unsupported reference normalization '+str(name))

def fill(module):
    nn.init.xavier_uniform_(module.weight)
    if module.bias is not None:nn.init.zeros_(module.bias)

weight_init=SimpleNamespace(c2_xavier_fill=fill)
ShapeSpec=lambda **kwargs:SimpleNamespace(**kwargs)

def definitions(path,namespace,names=None):
    """Keep upstream class/function definitions intact, replacing only imports."""
    tree=ast.parse(Path(path).read_text())
    tree.body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and (names is None or node.name in names)]
    exec(compile(tree,str(path),'exec'),namespace)
    return namespace

TRANSFORMER={**globals()}
definitions(UPSTREAM/'transformer_decoder/transformer.py',TRANSFORMER)
POSITION={**globals()}
definitions(UPSTREAM/'transformer_decoder/position_encoding.py',POSITION)
CORE={**globals()}
definitions(UPSTREAM/'pixel_decoder/ops/functions/ms_deform_attn_func.py',CORE,names=['ms_deform_attn_core_pytorch'])

def deform_apply(value,spatial_shapes,level_start_index,locations,weights,im2col_step):
    return CORE['ms_deform_attn_core_pytorch'](value,spatial_shapes,locations,weights)

OPS={**globals(),'MSDeformAttnFunction':SimpleNamespace(apply=deform_apply),'ms_deform_attn_core_pytorch':CORE['ms_deform_attn_core_pytorch']}
definitions(UPSTREAM/'pixel_decoder/ops/modules/ms_deform_attn.py',OPS)
PIXEL={**globals(),'MSDeformAttn':OPS['MSDeformAttn'],'PositionEmbeddingSine':POSITION['PositionEmbeddingSine'],
       '_get_clones':TRANSFORMER['_get_clones'],'_get_activation_fn':TRANSFORMER['_get_activation_fn'],
       'autocast':torch.cuda.amp.autocast}
definitions(UPSTREAM/'pixel_decoder/msdeformattn.py',PIXEL)

class ReferencePixelEncoder(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();self.backbone=resnet50(weights=None)
        # Remove ImageNet classifier; preserve all four convolutional levels.
        self.backbone.fc=nn.Identity()
        if pretrained:
            weights=torch.load(ROOT/'literature/weights/M2F_R50.pkl',map_location='cpu',weights_only=True)
            remapped={}
            for key,value in weights.items():
                key=key.removeprefix('stages.')
                if key.startswith('stem.conv1.norm.'):key='bn1.'+key[len('stem.conv1.norm.'):]
                elif key.startswith('stem.conv1.'):key='conv1.'+key[len('stem.conv1.'):]
                else:
                    stage=int(key[3]);key='layer'+str(stage-1)+key[4:]
                    for i in [1,2,3]:key=key.replace(f'conv{i}.norm.',f'bn{i}.')
                    key=key.replace('shortcut.norm.','downsample.1.').replace('shortcut.','downsample.0.')
                remapped[key]=value
            missing,unexpected=self.backbone.load_state_dict(remapped,strict=False)
            if unexpected or any(not key.endswith('num_batches_tracked') for key in missing):raise ValueError((missing,unexpected))
        shapes={f'res{i}':ShapeSpec(channels=2**(i+6),stride=2**i) for i in range(2,6)}
        self.pixel_decoder=PIXEL['MSDeformAttnPixelDecoder'](shapes,transformer_dropout=.0,transformer_nheads=8,
            transformer_dim_feedforward=1024,transformer_enc_layers=6,conv_dim=256,mask_dim=256,norm='GN',
            transformer_in_features=['res3','res4','res5'],common_stride=4)
        if pretrained:
            state=torch.load(ROOT/'literature/weights/M2F_R50_MSDeformAttnPixelDecoder.pkl',map_location='cpu',weights_only=True)
            remapped={}
            for key,value in state.items():
                if key.startswith('adapter'):key='lateral_convs.'+key
                elif key.startswith('layer'):key='output_convs.'+key
                remapped[key]=value
            self.pixel_decoder.load_state_dict(remapped,strict=True)
    def forward(self,image):
        b=self.backbone;x=b.maxpool(b.relu(b.bn1(b.conv1(image))));features={}
        for i in range(1,5):x=getattr(b,'layer'+str(i))(x);features['res'+str(i+1)]=x
        high,_,levels=self.pixel_decoder.forward_features(features)
        return high,levels[0],levels[1]
