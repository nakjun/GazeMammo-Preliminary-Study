"""Permutation-invariant duplicate-corrected MIL with a frozen global reference."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .gaze_classifier_1011 import GazeDiagnosisHead

VARIANTS=('mean_residual','gated_residual','contextual_residual','grid_contextual_residual')

class GazeAttention1011(nn.Module):
    def __init__(self,global_model,variant='contextual_residual',input_dim=1024,dim=64,dropout=.2):
        super().__init__()
        if variant not in VARIANTS or global_model.mode!='global':raise ValueError('Invalid attention/global configuration')
        self.config=dict(variant=variant,input_dim=input_dim,dim=dim,dropout=dropout)
        self.variant=variant;self.source='grid' if variant.startswith('grid_') else 'gaze'
        self.global_model=global_model.requires_grad_(False).eval()
        global_dim=global_model.config['dim']
        self.value=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,dim),nn.GELU())
        self.V=nn.Linear(dim,dim);self.W=nn.Linear(dim,dim);self.score=nn.Linear(dim,1,bias=False)
        self.U=nn.Linear(global_dim,dim,bias=False) if 'contextual' in variant else None
        self.dropout=nn.Dropout(dropout);self.gate=nn.Linear(global_dim+dim,1);self.delta=nn.Linear(dim,1);self.local_head=nn.Linear(dim,1)
        nn.init.zeros_(self.gate.weight);nn.init.constant_(self.gate.bias,math.log(.1/.9));nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias)
    def train(self,mode=True):
        super().train(mode);self.global_model.eval();return self
    def forward(self,global_features,patch_features,coordinates,mask=None,uniform=False):
        if patch_features.shape[:2]!=coordinates.shape[:2] or coordinates.shape[-1]!=2:raise ValueError('Patch/coordinate shape mismatch')
        if mask is None:mask=torch.ones(coordinates.shape[:2],dtype=torch.bool,device=coordinates.device)
        mask=mask.bool();available=mask.any(-1)
        with torch.no_grad():
            g=self.global_model.global_project(global_features);global_logit=self.global_model.head(g).squeeze(-1)
        clean=torch.where(mask[...,None],patch_features,torch.zeros_like(patch_features));z_i=self.value(clean)
        scores=torch.zeros_like(mask,dtype=z_i.dtype)
        if self.variant!='mean_residual' and not uniform:
            content=self.V(z_i)
            if self.U is not None:content=content+self.U(g)[:,None]
            scores=self.score(content.tanh()*self.W(z_i).sigmoid()).squeeze(-1)
        duplicate=((coordinates[:,:,None]-coordinates[:,None,:]).abs().amax(-1)<=1e-6)&mask[:,:,None]&mask[:,None,:]
        counts=duplicate.sum(-1).clamp_min(1).to(scores.dtype)
        adjusted=(scores-counts.log()).masked_fill(~mask,-1e9)
        attention=adjusted.softmax(-1)*mask;attention=attention/attention.sum(-1,keepdim=True).clamp_min(1e-12)
        pooled=(attention[...,None]*z_i).sum(1);dropped=self.dropout(pooled)
        alpha=self.gate(torch.cat([g,dropped],-1)).sigmoid().squeeze(-1)*available
        delta=self.delta(dropped).squeeze(-1);correction=alpha*delta
        local_logit=self.local_head(dropped).squeeze(-1)*available
        # Exact signed decomposition in eval; training dropout acts on pooled z.
        signed=alpha[:,None]*attention*F.linear(z_i,self.delta.weight,None).squeeze(-1)
        bias=alpha*self.delta.bias[0]
        return dict(logits=global_logit+correction,global_logits=global_logit,local_logits=local_logit,attention=attention,alpha=alpha,residual=correction,signed_contribution=signed,residual_bias=bias,valid_bag=available,values=z_i,duplicate_counts=counts)


def attention_loss(output,labels):
    main=F.binary_cross_entropy_with_logits(output['logits'],labels)
    valid=output['valid_bag'];aux=F.binary_cross_entropy_with_logits(output['local_logits'],labels,reduction='none')
    auxiliary=(aux*valid).sum()/valid.sum().clamp_min(1)
    regularizer=output['residual'].square().mean()
    return main+.25*auxiliary+.01*regularizer,dict(main=main,auxiliary=auxiliary,regularizer=regularizer)


def head_state(model):return {key:value for key,value in model.state_dict().items() if not key.startswith('global_model.')}
def restore_head(checkpoint,global_checkpoint):
    if checkpoint['kind']!='gaze_attention_1011' or global_checkpoint['model_config']['mode']!='global':raise ValueError('Wrong checkpoint kind')
    baseline=GazeDiagnosisHead(**global_checkpoint['model_config']);baseline.load_state_dict(global_checkpoint['state'])
    model=GazeAttention1011(baseline,**checkpoint['model_config'])
    missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    if unexpected or any(not key.startswith('global_model.') for key in missing):raise ValueError('Attention checkpoint architecture mismatch')
    return model


def validate_sources(payload,global_checkpoint,manifest):
    import pandas as pd
    frame=pd.DataFrame(payload['rows']);provenance=payload['provenance']
    expected=manifest[manifest.split.isin(['train','val'])&manifest.label.isin([0,1])]
    if frame.image_id.duplicated().any() or set(frame.image_id)!=set(expected.image_id):raise ValueError('Cache image population mismatch')
    if (frame.groupby('patient_key').split.nunique()>1).any():raise ValueError('Patient split overlap')
    lhs=frame.sort_values('image_id').reset_index(drop=True);rhs=expected.sort_values('image_id').reset_index(drop=True)
    for key in ('image_id','patient_key','split','label','file_name'):
        if lhs[key].tolist()!=rhs[key].tolist():raise ValueError(f'Cache provenance mismatch: {key}')
    allowed_train=set(manifest[manifest.split=='train'].patient_key);allowed_val=set(manifest[(manifest.split=='val')&(manifest.label>=0)].patient_key)
    for label,train,selection in [('gaze',provenance['gaze_training_patient_keys'],provenance['gaze_selection_patient_keys']),('global',global_checkpoint['training_patient_keys'],global_checkpoint['selection_patient_keys'])]:
        if set(train)-allowed_train or set(selection)-allowed_val or set(train)&set(selection):raise ValueError(f'{label} patient provenance mismatch')
    if set(global_checkpoint['classifier_training_patient_keys'])!=set(frame[frame.split=='train'].patient_key):raise ValueError('Global classifier patient mismatch')
    if global_checkpoint['gaze_provenance']['gaze_checkpoint_sha256']!=provenance['gaze_checkpoint_sha256']:raise ValueError('Global/cache gaze checkpoint mismatch')
    if provenance['final_test_used'] or provenance['seed']!=42 or provenance['steps']!=16 or provenance['strategy']!='sample' or provenance['policy']!='none' or provenance['patch_size']!=224:raise ValueError('Invalid source protocol')
    tensors=payload['tensors']
    shapes={'global_features':(len(frame),1024),'gaze_features':(len(frame),16,1024),'grid_features':(len(frame),16,1024),'gaze_coordinates':(len(frame),16,2),'grid_coordinates':(len(frame),16,2)}
    for key,shape in shapes.items():
        if key not in tensors or tuple(tensors[key].shape)!=shape:raise ValueError(f'Cache tensor shape mismatch: {key}')
        if key.endswith('coordinates') and ((tensors[key]<0).any() or (tensors[key]>=1).any()):raise ValueError(f'Coordinate range mismatch: {key}')
    if len(provenance['gaze_checkpoint_sha256'])!=64:raise ValueError('Invalid source gaze hash')
    for key,value in tensors.items():
        if len(value)!=len(frame) or not torch.isfinite(value).all():raise ValueError(f'Invalid cached tensor {key}')
    return frame
