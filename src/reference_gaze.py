"""Breast adaptation of released HAT/ChestSearch decoder implementations.

Preserves author decoder classes and released Mask2Former pixel features.
Adaptations: one image-only task, no duration, learned first fixation (BOS),
16 output locations, aspect-preserving mammography input, common scoring grid.
"""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from .data import ROOT
from .reference_backend import ReferencePixelEncoder,definitions,TRANSFORMER,weight_init
import src.reference_backend as backend

class CachedFeatures(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__();self.pixel_decoder=nn.Identity();self.cached=None
    def forward(self,image):return tuple(x.clone() for x in self.cached)

def decoder_class(family):
    repo=ROOT/'literature/code'/('HAT' if family=='hat' else 'GazeSearch/src')
    path=repo/('hat/models.py' if family=='hat' else 'chestsearch/models.py')
    positions={**vars(backend)};definitions(repo/'common/position_encoding.py',positions)
    name='HumanAttnTransformer' if family=='hat' else 'ExpertAttnTransformer'
    namespace={**vars(backend),'pe':SimpleNamespace(**positions),'ImageFeatureEncoder':CachedFeatures,
               'TransformerEncoder':TRANSFORMER['TransformerEncoder'],'TransformerEncoderLayer':TRANSFORMER['TransformerEncoderLayer']}
    definitions(path,namespace,names=['TokenEmbedding','MLP','LayerNorm2d','SelfAttentionLayer','CrossAttentionLayer','FFNLayer','_get_activation_fn',name])
    return namespace[name]

class ReferenceGaze(nn.Module):
    def __init__(self,family='hat',grid=(24,48),k=16,input_size=(256,512),pretrained=True):
        super().__init__();self.family=family;self.grid=tuple(grid);self.k=k;self.cells=grid[0]*grid[1];self.input_size=tuple(input_size)
        self.foveal=False # Native external crops are not used; author spatial indexing supplies foveal information.
        self.model_config=dict(family=family,grid=list(grid),k=k,input_size=list(input_size))
        self.visual=ReferencePixelEncoder(pretrained)
        pa=SimpleNamespace(im_w=input_size[0],im_h=input_size[1],backbone_config='',pixel_decoder='MSD',
            special_symbols=['<pad>','<eos>'],pad_idx=0,eos_idx=1,max_traj_length=k+1)
        self.core=decoder_class(family)(pa,num_decoder_layers=6,hidden_dim=256 if family=='hat' else 384,nhead=4,
            ntask=1,tgt_vocab_size=(pa.im_w//4)*(pa.im_h//4)+2,num_output_layers=3,train_encoder=False,
            train_pixel_decoder=False,dropout=0.,dim_feedforward=512,parallel_arch=True,
            dorsal_source=['P1','P2'] if family=='hat' else ['P1'],num_encoder_layers=3,
            output_feature_map_name='P4')
        self.set_backbone('pixel')
    def set_backbone(self,mode):
        for name,p in self.visual.named_parameters():
            p.requires_grad=mode=='all' or mode=='pixel' and name.startswith('pixel_decoder.') or mode=='last' and (name.startswith('pixel_decoder.') or name.startswith('backbone.layer4.'))
    def train(self,mode=True):
        super().train(mode)
        for m in self.visual.backbone.modules():
            if isinstance(m,nn.modules.batchnorm._BatchNorm):m.eval()
        return self
    def maps(self,image):
        features=self.visual(image);memory={'features':features}
        memory['prior']=self.prefix_logits(memory,image.new_empty(len(image),0,2))
        return memory
    def prefix_logits(self,memory,previous,padding=None,return_dense=False):
        b,t,_=previous.shape;high=memory['features'][0];h,w=high.shape[-2:]
        ids=(previous[...,1]*h).long().clamp(0,h-1)*w+(previous[...,0]*w).long().clamp(0,w-1)+2
        tokens=torch.cat([torch.zeros(b,1,dtype=torch.long,device=previous.device),ids],1)
        mask=torch.zeros_like(tokens,dtype=torch.bool)
        if padding is not None:mask[:,1:]=padding
        self.core.encoder.cached=memory['features']
        mode=self.core.training
        # Request the author's raw map branch without changing child eval/dropout modes.
        self.core.training=True
        result=self.core(torch.empty(b,3,1,1,device=previous.device),tokens,mask,tokens)
        self.core.training=mode
        raw=result['pred_fixation_map'][:,0]
        self.core.encoder.cached=None
        if return_dense:return raw
        probability=F.adaptive_avg_pool2d(raw.float().sigmoid()[:,None],(self.grid[1],self.grid[0])).flatten(1).clamp_min(1e-12)
        return probability.log()-probability.sum(-1,keepdim=True).log()
    def teacher_logits(self,memory,coords,patch_vectors=None):
        b,t,_=coords.shape
        features=tuple(x[:,None].expand(b,t,*x.shape[1:]).reshape(b*t,*x.shape[1:]) for x in memory['features'])
        previous=coords[:,None,:-1].expand(b,t,t-1,2).reshape(b*t,t-1,2)
        indices=torch.arange(t,device=coords.device)
        padding=(torch.arange(t-1,device=coords.device)[None]>=indices[:,None])[None].expand(b,-1,-1).reshape(b*t,t-1)
        return self.prefix_logits({'features':features},previous,padding).reshape(b,t,self.cells)
    def forward(self,image,coords):return self.teacher_logits(self.maps(image),coords)
    def step(self,memory,previous,older,patch_vector,hidden,index):
        if index==0:prefix=previous.new_empty(len(previous),0,2)
        else:prefix=torch.cat([hidden,previous[:,None]],1)
        return self.prefix_logits(memory,prefix),prefix
