"""Multi-level image features for gaze prediction, without diagnosis heads."""
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from .fusion_backbones import interpolate_positions

SPECS={
 'densenet161':([768,2112,2208],models.DenseNet161_Weights.IMAGENET1K_V1),
 'efficientnet_b5':([64,176,2048],models.EfficientNet_B5_Weights.IMAGENET1K_V1),
 'convnext_base':([256,512,1024],models.ConvNeXt_Base_Weights.IMAGENET1K_V1),
 'swin_b':([256,512,1024],models.Swin_B_Weights.IMAGENET1K_V1),
 'vit_b_16':([768,768,768],models.ViT_B_16_Weights.IMAGENET1K_V1),
}

class GazePyramid(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__();self.name=name;self.channels,weight=SPECS[name];self.patch_channels=self.channels[-1]
        net=getattr(models,name)(weights=weight if pretrained else None)
        if name=='vit_b_16':
            self.conv_proj=net.conv_proj;self.class_token=net.class_token;self.encoder=net.encoder;self.patch_size=net.patch_size
        else:
            self.features=net.features
            self.norm=net.norm if name=='swin_b' else net.classifier[0] if name=='convnext_base' else nn.Identity()
    def is_last_parameter(self,name):
        prefixes={'densenet161':('features.denseblock4.','features.norm5.'),
          'efficientnet_b5':('features.6.','features.7.','features.8.'),
          'convnext_base':('features.6.','features.7.','norm.'),
          'swin_b':('features.6.','features.7.','norm.'),
          'vit_b_16':('encoder.layers.encoder_layer_9.','encoder.layers.encoder_layer_10.','encoder.layers.encoder_layer_11.','encoder.ln.')}
        return name.startswith(prefixes[self.name])
    def vit_features(self,x):
        x=self.conv_proj(x);h,w=x.shape[-2:];x=x.flatten(2).transpose(1,2)
        x=torch.cat([self.class_token.expand(len(x),-1,-1),x],1)
        x=self.encoder.dropout(x+interpolate_positions(self.encoder.pos_embedding,h,w));levels=[]
        for i,layer in enumerate(self.encoder.layers):
            x=layer(x)
            if i in (3,7,11):levels.append(self.encoder.ln(x)[:,1:].transpose(1,2).reshape(len(x),self.patch_channels,h,w))
        return levels,self.encoder.ln(x)[:,0]
    def forward(self,x):
        if self.name=='vit_b_16':return tuple(self.vit_features(x)[0])
        levels=[]
        wanted={'densenet161':('denseblock2','denseblock3','norm5'),
                'efficientnet_b5':('3','5','8'),'convnext_base':('3','5','7'),'swin_b':('3','5','7')}[self.name]
        for name,layer in self.features.named_children():
            x=layer(x)
            if name in wanted:
                if self.name=='swin_b':level=(self.norm(x) if name=='7' else x).permute(0,3,1,2)
                elif self.name=='densenet161' and name=='norm5':level=F.relu(x)
                else:level=x
                levels.append(level)
        return tuple(levels)
    def patch_vector(self,x):
        if self.name=='vit_b_16':return self.vit_features(x)[1]
        features=self(x)[-1]
        if self.name=='convnext_base':return self.norm(features.mean((-1,-2),keepdim=True)).flatten(1)
        return features.mean((-1,-2))
