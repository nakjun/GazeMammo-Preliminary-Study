"""ImageNet encoders for matched global/native-patch classification.

All use existing 512x256 whole-image and 224x224 local inputs. ViT positions
are interpolated at runtime so the pretrained square grid also supports rectangles.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

SPECS={
    'densenet161':(2208,models.DenseNet161_Weights.IMAGENET1K_V1),
    'efficientnet_b5':(2048,models.EfficientNet_B5_Weights.IMAGENET1K_V1),
    'convnext_base':(1024,models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
    'swin_b':(1024,models.Swin_B_Weights.IMAGENET1K_V1),
    'vit_b_16':(768,models.ViT_B_16_Weights.IMAGENET1K_V1),
}

def interpolate_positions(position,height,width):
    side=int((position.shape[1]-1)**.5)
    if side*side!=position.shape[1]-1:raise ValueError('Expected square pretrained ViT grid')
    if (height,width)==(side,side):return position
    spatial=position[:,1:].reshape(1,side,side,-1).permute(0,3,1,2)
    spatial=F.interpolate(spatial,size=(height,width),mode='bicubic',align_corners=True)
    return torch.cat([position[:,:1],spatial.flatten(2).transpose(1,2)],1)

class FusionEncoder(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__();self.name=name;self.channels,weights=SPECS[name]
        net=getattr(models,name)(weights=weights if pretrained else None)
        if name=='vit_b_16':
            self.conv_proj=net.conv_proj;self.class_token=net.class_token
            self.encoder=net.encoder;self.patch_size=net.patch_size
        else:
            self.features=net.features
            self.norm=net.norm if name=='swin_b' else net.classifier[0] if name=='convnext_base' else nn.Identity()
    def patch_vector(self,x):
        if self.name=='vit_b_16':
            if x.shape[-1]%self.patch_size or x.shape[-2]%self.patch_size:raise ValueError('ViT inputs must divide by patch size')
            x=self.conv_proj(x);h,w=x.shape[-2:];x=x.flatten(2).transpose(1,2)
            x=torch.cat([self.class_token.expand(x.shape[0],-1,-1),x],1)
            x=x+interpolate_positions(self.encoder.pos_embedding,h,w)
            return self.encoder.ln(self.encoder.layers(self.encoder.dropout(x)))[:,0]
        x=self.features(x)
        if self.name=='swin_b':return self.norm(x).mean((1,2))
        if self.name=='convnext_base':return self.norm(x.mean((-1,-2),keepdim=True)).flatten(1)
        if self.name=='densenet161':x=F.relu(x)
        return x.mean((-1,-2))
