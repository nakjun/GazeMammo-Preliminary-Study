"""Spatially shared gaze prediction with continuous history and native-patch feedback."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50,ResNet50_Weights
from .models import PatchClassifier

def fourier_xy(xy):
    frequencies=xy.new_tensor([1.,2.,4.,8.])*math.pi
    phase=xy[...,None]*frequencies
    return torch.cat([phase.sin(),phase.cos()],-1).flatten(-2)

def gaussian_targets(points,grid=(24,48),sigma_pixels=96,sizes=(1500,3000)):
    """Normalized Gaussian per point; sigma is measured in source PNG pixels."""
    gx,gy=grid;indices=torch.arange(gx*gy,device=points.device)
    cells=torch.stack([((indices%gx)+.5)/gx,((indices//gx)+.5)/gy],-1)
    if sigma_pixels==0:
        ids=(points[...,1]*gy).long().clamp(0,gy-1)*gx+(points[...,0]*gx).long().clamp(0,gx-1)
        return F.one_hot(ids,gx*gy).float()
    if sigma_pixels<0:raise ValueError('sigma must be nonnegative')
    distance=((points[...,None,:]-cells)*points.new_tensor(sizes)).square().sum(-1)
    return (-distance/(2*sigma_pixels**2)).softmax(-1)

class ImageEncoder(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__();r=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.stem=nn.Sequential(r.conv1,r.bn1,r.relu,r.maxpool)
        self.layer1=r.layer1;self.layer2=r.layer2;self.layer3=r.layer3;self.layer4=r.layer4
    def forward(self,image):
        x=self.layer1(self.stem(image));c2=self.layer2(x);c3=self.layer3(c2);c4=self.layer4(c3)
        return c2,c3,c4
    def patch_vector(self,image):return self(image)[-1].mean((-1,-2))

class SpatialTrajectory(nn.Module):
    def __init__(self,grid=(24,48),k=16,dim=128,foveal=False,fpn=False,pretrained=True,backbone_name='resnet50',motion_bias=False,input_size=(256,512)):
        super().__init__();self.grid=tuple(grid);self.k=k;self.cells=grid[0]*grid[1];self.foveal=foveal;self.fpn=fpn
        self.backbone_name=backbone_name;self.motion_bias=motion_bias;self.input_size=tuple(input_size)
        self.model_config=dict(grid=list(grid),k=k,dim=dim,foveal=foveal,fpn=fpn,backbone_name=backbone_name,motion_bias=motion_bias,input_size=list(input_size))
        from .gaze_backbones import GazePyramid
        self.visual=ImageEncoder(pretrained) if backbone_name=='resnet50' else GazePyramid(backbone_name,pretrained)
        channels=[512,1024,2048] if backbone_name=='resnet50' else self.visual.channels
        self.lateral=nn.ModuleList([nn.Conv2d(c,dim,1) for c in channels])
        self.refine=nn.Sequential(nn.Conv2d(dim,dim,3,padding=1),nn.GroupNorm(8,dim),nn.GELU())
        self.position=nn.Linear(16,dim);self.density=nn.Conv2d(dim,1,1)
        self.history=nn.Sequential(nn.Linear(16+2+1,dim),nn.GELU())
        self.patch=nn.Linear(channels[-1],dim);self.rnn=nn.GRU(dim,dim,batch_first=True)
        self.bos=nn.Parameter(torch.zeros(dim));self.initial=nn.Linear(dim,dim)
        self.dynamic_scale=nn.Parameter(torch.tensor(.1));self.backbone_mode='frozen'
        if motion_bias:
            self.motion=nn.Linear(dim,15)
            nn.init.zeros_(self.motion.weight);nn.init.zeros_(self.motion.bias)
            with torch.no_grad():self.motion.bias.reshape(3,5)[:,1].copy_(torch.tensor([-.1,0.,.1]))
            self.motion_scale=nn.Parameter(torch.tensor(math.log(math.expm1(.1))))
        self.set_backbone('frozen')
    def set_backbone(self,mode):
        if mode not in ('frozen','last','all'):raise ValueError(mode)
        self.backbone_mode=mode
        for name,p in self.visual.named_parameters():p.requires_grad=mode=='all' or (mode=='last' and self.is_last_parameter(name))
    def is_last_parameter(self,name):
        return name.startswith('layer4.') if self.backbone_name=='resnet50' else self.visual.is_last_parameter(name)
    def motion_logits(self,q,previous):
        shape=q.shape[:-1];params=self.motion(q).reshape(*shape,3,5)
        mixture=params[...,0].log_softmax(-1)
        means=previous[...,None,:]+params[...,1:3].tanh()
        scales=.02+.26*params[...,3:5].sigmoid()
        gx,gy=self.grid;ids=torch.arange(self.cells,device=q.device)
        xy=torch.stack([((ids%gx)+.5)/gx,((ids//gx)+.5)/gy],-1)
        distance=(xy-means[...,None,:])/scales[...,None,:]
        logits=(mixture[...,None]-.5*distance.square().sum(-1)-scales.log().sum(-1)[...,None]).logsumexp(-2)
        return F.softplus(self.motion_scale)*logits.log_softmax(-1)
    def train(self,mode=True):
        super().train(mode)
        # Fixed BN statistics avoid mixing future teacher crops through batch statistics.
        for module in self.visual.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self
    def maps(self,image):
        features=self.visual(image);size=(self.grid[1],self.grid[0])
        if self.fpn:
            top=self.lateral[2](features[2]);middle=self.lateral[1](features[1])+F.interpolate(top,size=features[1].shape[-2:],mode='bilinear',align_corners=False)
            fine=self.lateral[0](features[0])+F.interpolate(middle,size=features[0].shape[-2:],mode='bilinear',align_corners=False)
            spatial=self.refine(F.interpolate(fine,size=size,mode='bilinear',align_corners=False))
        else:spatial=self.refine(F.interpolate(self.lateral[2](features[2]),size=size,mode='bilinear',align_corners=False))
        ids=torch.arange(self.cells,device=image.device);gx,gy=self.grid
        xy=torch.stack([((ids%gx)+.5)/gx,((ids//gx)+.5)/gy],-1)
        keys=spatial.flatten(2).transpose(1,2)+self.position(fourier_xy(xy))[None]
        return dict(keys=keys,prior=self.density(spatial).flatten(1),global_context=keys.mean(1))
    def prior_logits(self,image):return self.maps(image)['prior']
    def teacher_logits(self,memory,coords,patch_vectors=None,order='original'):
        b,t,_=coords.shape
        previous=torch.cat([torch.zeros_like(coords[:,:1]),coords[:,:-1]],1)
        delta=torch.cat([torch.zeros_like(previous[:,:1]),previous[:,1:]-previous[:,:-1]],1)
        time=torch.arange(t,device=coords.device).float()[None,:,None].expand(b,-1,-1)/max(self.k-1,1)
        inputs=self.history(torch.cat([fourier_xy(previous),delta,time],-1))
        if self.foveal:
            if patch_vectors is None:raise ValueError('Foveal teacher features required')
            inputs=inputs+self.patch(patch_vectors)
        inputs=inputs+memory['global_context'][:,None]
        inputs=inputs.clone();inputs[:,0]=self.bos+memory['global_context']
        q,_=self.rnn(inputs,torch.tanh(self.initial(memory['global_context']))[None])
        logits=memory['prior'][:,None]+self.dynamic_scale*torch.einsum('btd,bcd->btc',q,memory['keys'])/math.sqrt(q.shape[-1])
        if self.motion_bias:
            bias=self.motion_logits(q,previous);bias=bias.clone();bias[:,0]=0;logits=logits+bias
        return logits
    def forward(self,image,coords,previous_patches=None,previous_valid=None):
        memory=self.maps(image);vectors=None
        if self.foveal:
            selected=self.visual.patch_vector(previous_patches[previous_valid])
            vectors=selected.new_zeros((*coords.shape[:2],self.patch.in_features));vectors[previous_valid]=selected
        return self.teacher_logits(memory,coords,vectors)
    def step(self,memory,previous,older,patch_vector,hidden,index):
        time=previous.new_full((len(previous),1),index/max(self.k-1,1))
        inputs=self.history(torch.cat([fourier_xy(previous),previous-older,time],-1))+memory['global_context']
        if index==0:inputs=self.bos+memory['global_context']
        elif self.foveal:inputs=inputs+self.patch(patch_vector)
        if hidden is None:hidden=torch.tanh(self.initial(memory['global_context']))[None]
        q,hidden=self.rnn(inputs[:,None],hidden)
        logits=memory['prior']+self.dynamic_scale*torch.einsum('bd,bcd->bc',q[:,0],memory['keys'])/math.sqrt(q.shape[-1])
        if self.motion_bias and index>0:logits=logits+self.motion_logits(q[:,0],previous)
        return logits,hidden

class GlobalLocalFusion(nn.Module):
    """Shared patch/whole ResNet50 features with explicit global/local/fusion heads."""
    def __init__(self,dim=128,input_dim=2048):
        super().__init__();self.local=PatchClassifier(dim=dim,input_dim=input_dim)
        self.global_project=nn.Sequential(nn.Linear(input_dim,dim),nn.ReLU(),nn.Dropout(.1))
        self.global_head=nn.Linear(dim,1);self.fusion=nn.Linear(2*dim,1)
    def forward(self,global_features,patch_features,mask):
        g=self.global_project(global_features);x=self.local.projection(patch_features)
        attention=self.local.attention(x).squeeze(-1).masked_fill(~mask,-torch.inf).softmax(1)
        z=(x*attention[:,:,None]).sum(1)
        return dict(global_logits=self.global_head(g).squeeze(-1),local_logits=self.local.head(z).squeeze(-1),
                    fusion_logits=self.fusion(torch.cat([g,z],1)).squeeze(-1),attention=attention)

class JointGlobalLocal(nn.Module):
    def __init__(self,pretrained=True,backbone_name='resnet50',head_dropout=.1):
        super().__init__()
        from .fusion_backbones import FusionEncoder
        self.visual=ImageEncoder(pretrained) if backbone_name=='resnet50' else FusionEncoder(backbone_name,pretrained)
        self.model_config=dict(backbone_name=backbone_name,head_dropout=head_dropout)
        self.head=GlobalLocalFusion(input_dim=2048 if backbone_name=='resnet50' else self.visual.channels)
        self.head.local.projection[2].p=head_dropout
        self.head.global_project[2].p=head_dropout
    def train(self,mode=True):
        super().train(mode)
        for module in self.visual.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self
    def forward(self,image,patches,mask):
        g=self.visual.patch_vector(image);local=self.visual.patch_vector(patches[mask])
        full=local.new_zeros((*mask.shape,local.shape[-1]));full[mask]=local
        return self.head(g,full,mask)
