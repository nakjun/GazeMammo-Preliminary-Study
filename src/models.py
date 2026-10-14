import torch
from torch import nn
from torchvision.models import resnet50,ResNet50_Weights
from torchvision import models as vision

BACKBONES={'resnet50':2048,'densenet121':1024,'efficientnet_b0':1280,'convnext_tiny':768,'swin_t':768,
           'resnet101':2048,'convnext_base':1024,'swin_b':1024}

class SpatialBackbone(nn.Module):
    def __init__(self,name,pretrained=True):
        super().__init__();self.name=name;self.channels=BACKBONES[name]
        weights={'resnet50':vision.ResNet50_Weights.IMAGENET1K_V2,
                 'densenet121':vision.DenseNet121_Weights.DEFAULT,
                 'efficientnet_b0':vision.EfficientNet_B0_Weights.DEFAULT,
                 'convnext_tiny':vision.ConvNeXt_Tiny_Weights.DEFAULT,
                 'swin_t':vision.Swin_T_Weights.DEFAULT,
                 'resnet101':vision.ResNet101_Weights.IMAGENET1K_V2,
                 'convnext_base':vision.ConvNeXt_Base_Weights.DEFAULT,
                 'swin_b':vision.Swin_B_Weights.DEFAULT}[name]
        net=getattr(vision,name)(weights=weights if pretrained else None)
        self.features=nn.Sequential(*list(net.children())[:-2]) if name.startswith('resnet') else net.features
        self.norm=net.norm if name.startswith('swin') else nn.Identity()
    def forward(self,x):
        x=self.features(x)
        if self.name.startswith('swin'):x=self.norm(x).permute(0,3,1,2)
        if self.name=='densenet121':x=nn.functional.relu(x)
        return x

def backbone(pretrained=True):
    network=resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
    return nn.Sequential(*list(network.children())[:-2])

class PatchClassifier(nn.Module):
    def __init__(self,aggregation='mil',dim=128,k=32,input_dim=2048):
        super().__init__();self.aggregation=aggregation
        self.projection=nn.Sequential(nn.Linear(input_dim,dim),nn.ReLU(),nn.Dropout(.1))
        self.attention=nn.Sequential(nn.Linear(dim,64),nn.Tanh(),nn.Linear(64,1))
        if aggregation in ('sequence','shuffled'):
            self.position=nn.Parameter(torch.randn(1,k,dim)*.02)
            self.encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,4,dim*2,.1,batch_first=True),2)
        self.head=nn.Linear(dim,1)
    def forward(self,features,mask):
        x=self.projection(features)
        if self.aggregation=='shuffled':
            # Shuffle valid patches only; keep padding masked at its original positions.
            x=x.clone()
            for b in range(len(x)):
                ids=mask[b].nonzero().flatten();x[b,ids]=x[b,ids[torch.randperm(len(ids),device=x.device)]]
        if self.aggregation in ('sequence','shuffled'):
            x=self.encoder(x+self.position[:,:x.size(1)],src_key_padding_mask=~mask)
        scores=self.attention(x).squeeze(-1).masked_fill(~mask,-torch.inf)
        attention=scores.softmax(1)
        return self.head((x*attention[:,:,None]).sum(1)).squeeze(-1),attention

class TrajectoryModel(nn.Module):
    def __init__(self,grid=(24,48),k=16,dim=128,pretrained=True,backbone_name='resnet50',layers=2,heads=4,ff_multiplier=2,dropout=.1,finetune_backbone=False):
        if dim % heads:raise ValueError('Decoder dimension must be divisible by number of attention heads')
        if layers<1 or ff_multiplier<1:raise ValueError('Invalid decoder size')
        super().__init__();self.grid=grid;self.k=k;self.dim=dim;self.cells=grid[0]*grid[1]
        self.backbone_name=backbone_name
        self.finetune_backbone=finetune_backbone
        self.model_config=dict(dim=dim,layers=layers,heads=heads,ff_multiplier=ff_multiplier,dropout=dropout,finetune_backbone=finetune_backbone)
        # Keep old ResNet checkpoint keys compatible.
        self.visual=backbone(pretrained) if backbone_name=='resnet50' else SpatialBackbone(backbone_name,pretrained)
        self.projection=nn.Linear(BACKBONES[backbone_name],dim)
        self.spatial=nn.Linear(2,dim);self.tokens=nn.Embedding(self.cells+1,dim)
        self.position=nn.Parameter(torch.randn(1,k,dim)*.02)
        self.decoder=nn.TransformerDecoder(nn.TransformerDecoderLayer(dim,heads,dim*ff_multiplier,dropout,batch_first=True),layers)
        self.head=nn.Linear(dim,self.cells)
        for p in self.visual.parameters():p.requires_grad=finetune_backbone
    def train(self,mode=True):
        super().train(mode)
        if not self.finetune_backbone:self.visual.eval()
        else:
            for module in self.visual.modules():
                if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self
    def memory(self,image):
        if self.finetune_backbone:features=self.visual(image)
        else:
            with torch.no_grad():features=self.visual(image)
        return self.memory_features(features)
    def memory_features(self,features):
        b,c,h,w=features.shape
        y,x=torch.meshgrid(torch.linspace(0,1,h,device=features.device),torch.linspace(0,1,w,device=features.device),indexing='ij')
        pos=torch.stack([x,y],-1).reshape(1,h*w,2)
        return self.projection(features.flatten(2).transpose(1,2))+self.spatial(pos)
    def decode(self,memory,inputs):
        length=inputs.shape[1]
        x=self.tokens(inputs)+self.position[:,:length]
        causal=torch.triu(torch.ones(length,length,device=x.device,dtype=torch.bool),1)
        return self.head(self.decoder(x,memory,tgt_mask=causal))
    def forward(self,image,target):
        inputs=torch.cat([torch.full_like(target[:,:1],self.cells),target[:,:-1].clamp_min(0)],1)
        return self.decode(self.memory(image),inputs)
    def forward_features(self,features,target):
        if self.finetune_backbone:raise ValueError('Fine-tuning cannot use cached backbone features')
        inputs=torch.cat([torch.full_like(target[:,:1],self.cells),target[:,:-1].clamp_min(0)],1)
        return self.decode(self.memory_features(features),inputs)
    @torch.no_grad()
    def generate(self,image,strategy='argmax',temperature=1.):
        if strategy not in ('argmax','sample') or temperature<=0:raise ValueError('Invalid generation strategy or temperature')
        memory=self.memory(image);tokens=torch.full((len(image),1),self.cells,device=image.device,dtype=torch.long)
        generated=[]
        for _ in range(self.k):
            logits=self.decode(memory,tokens)[:,-1]
            next_token=logits.argmax(-1) if strategy=='argmax' else torch.multinomial((logits/temperature).softmax(-1),1).squeeze(-1)
            generated.append(next_token);tokens=torch.cat([tokens,next_token[:,None]],1)
        t=torch.stack(generated,1);gx,gy=self.grid
        return torch.stack([((t%gx)+.5)/gx,((t//gx)+.5)/gy],-1)

class JointPatchClassifier(nn.Module):
    """Trainable ResNet50 patch encoder and MIL head."""
    def __init__(self,k=16,pretrained=True):
        super().__init__();self.visual=backbone(pretrained);self.mil=PatchClassifier(k=k)
    def train(self,mode=True):
        super().train(mode)
        for module in self.visual.modules():
            if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()
        return self
    def forward(self,patches,mask):
        features=self.visual(patches[mask]).mean((-1,-2))
        full=features.new_zeros((*mask.shape,2048));full[mask]=features
        return self.mil(full,mask)

def trajectory_from_checkpoint(checkpoint):
    """Restore architecture; old checkpoints retain the original decoder defaults."""
    model=TrajectoryModel(checkpoint['grid'],checkpoint['k'],pretrained=False,
                          backbone_name=checkpoint.get('backbone','resnet50'),**checkpoint.get('model_config',{}))
    model.load_state_dict(checkpoint['state'])
    return model

class HeatmapModel(nn.Module):
    def __init__(self,grid=(24,48),pretrained=True):
        super().__init__();self.grid=grid;self.visual=backbone(pretrained)
        self.head=nn.Sequential(nn.Conv2d(2048,128,1),nn.ReLU(),nn.Conv2d(128,1,1))
        for p in self.visual.parameters():p.requires_grad=False
    def train(self,mode=True):
        super().train(mode);self.visual.eval();return self
    def forward(self,image):
        with torch.no_grad():features=self.visual(image)
        return nn.functional.interpolate(self.head(features),size=(self.grid[1],self.grid[0]),mode='bilinear',align_corners=False).flatten(1)
