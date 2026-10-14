"""Live clinical DG conditional expert with independently restored image-only BOS.

This module does not modify the original guided ROI memory or ROI14 policy.
PIL images are required by maps; maps_rgb takes explicitly raw RGB0..255 plus
independently computed BOS memory, never normalized whole_tensor as DG input.
"""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from .data import whole_tensor
from .deepgaze_context_1011 import native_features,dense_centerbias
from .deepgaze_clinical_1011 import history_batch,prepare_saliency,forward_saliency
from .proposal_utility_1011 import file_sha256,state_sha256

ROOT=Path(__file__).resolve().parents[1]

class ClinicalDGProbability(nn.Module):
    grid=(24,48)
    input_size=(512,1024)
    input_kind='PILgrayscale_to_RGB0_255'
    foveal=False
    def __init__(self,model,checkpoint,bos,metadata=None):
        super().__init__();self.model=model;self.bos=bos;self.metadata=metadata or {};self.cell_xy=bos.cell_xy
        if tuple(checkpoint['input_size'])!=self.input_size:raise ValueError('Clinical runtime requires512x1024')
        if list(checkpoint['bos_weights'])!=[.75,.25] or tuple(bos.weights)!=(.75,.25):raise ValueError('Clinical BOS recipe mismatch')
        bias=checkpoint['centerbias']
        if bias.shape!=(48,24) or not torch.isfinite(bias).all() or (bias<=0).any() or abs(float(bias.sum())-1)>1e-6:raise ValueError('Invalid trainKDE')
        self.register_buffer('centerbias_probability',bias.clone());self.register_buffer('dense_bias',dense_centerbias(bias,self.input_size,next(model.parameters()).device))
        self.requires_grad_(False);self.eval()

    @torch.no_grad()
    def maps(self,images):
        if isinstance(images,Image.Image):images=[images]
        if not isinstance(images,(list,tuple)) or not images or any(not isinstance(im,Image.Image) for im in images):raise ValueError('PILimages required; do not supply normalized mammography tensor')
        device=self.dense_bias.device;gray=[im.convert('L') for im in images]
        rgb=torch.stack([torch.from_numpy(np.asarray(im.resize(self.input_size,Image.Resampling.BILINEAR)).copy()).float()[None].repeat(3,1,1) for im in gray]).to(device)
        bos_memory=self.bos.maps([torch.stack([whole_tensor(im,size) for im in gray]).to(device) for size in self.bos.input_sizes])
        return self.maps_rgb(rgb,bos_memory)

    @torch.no_grad()
    def maps_rgb(self,rgb,bos_memory):
        if rgb.ndim!=4 or rgb.shape[1:]!=(3,1024,512) or not torch.isfinite(rgb).all() or (rgb<0).any() or (rgb>255).any():raise ValueError('Expected finite raw RGB0..255512x1024')
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):features=native_features(self.model,rgb)
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):saliency=prepare_saliency(self.model,features,self.input_size)
        return dict(saliency=saliency,bos=bos_memory,batch=len(rgb),prior=bos_memory.get('prior'),input_protocol='PILbilinear grayscale replicatedRGB0..255512x1024')

    @torch.no_grad()
    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        if prefix.ndim!=3 or prefix.shape[0]!=memory['batch'] or prefix.shape[-1]!=2 or not torch.isfinite(prefix).all() or (prefix<0).any() or (prefix>1).any():raise ValueError('Invalid normalized actual prefix')
        if patch_vector is not None:raise ValueError('DG expert uses observed coordinates; external crop vectors not consumed')
        if prefix.shape[1]==0:
            logp,_,_=self.bos.step_history(memory['bos'],prefix,None,None);role='image_only_arithmetic_BOS'
        else:
            hx,hy=history_batch(prefix,len(prefix[0]),self.input_size)
            with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):
                density=forward_saliency(self.model,memory['saliency'],self.input_size,self.dense_bias.expand(memory['batch'],-1,-1),hx,hy)
            from scripts.evaluate_deepgaze3_1011 import cell_mass
            logp=cell_mass(density);role='clinicalDG_true_last4past'
        if not torch.isfinite(logp).all() or not torch.allclose(logp.logsumexp(-1),torch.zeros(len(prefix),device=prefix.device),atol=1e-5):raise ValueError('Conditional mass normalization failed')
        return logp,None,dict(probability_role=role,history_length=prefix.shape[1],checkpoint_sha256=self.metadata.get('checkpoint_sha256'),ROI_dependency='separate original guided keys; this expert does not modify R14')

    def teacher_logits(self,memory,coords,patch_vectors=None):
        if patch_vectors is not None:raise ValueError('DG teacher does not consume patchvectors')
        return torch.stack([self.step_history(memory,coords[:,:t])[0] for t in range(coords.shape[1])],1)


def load_clinical_probability(checkpoint_path,bos_config_path,device='cpu'):
    sys.path.insert(0,str(ROOT/'scripts'))
    from train_deepgaze_clinical_1011 import load_clinical
    from .dual_output_1011 import load_runtime,config_sha256
    payload=json.loads(Path(bos_config_path).read_text())
    if payload['sha256']!=config_sha256(payload['config']):raise ValueError('BOS config SHA mismatch')
    model,checkpoint=load_clinical(checkpoint_path,device);bos,_,_=load_runtime(payload['config'],device)
    parents=payload['config']['probability']['parents']
    if len(parents)!=2 or any(a['sha256']!=b['sha256'] or str(Path(a['path']).resolve())!=str(Path(b['path']).resolve()) for a,b in zip(checkpoint['bos_sources'],parents)):raise ValueError('Clinical checkpoint BOS dependency mismatch')
    if state_sha256(model.features.state_dict())!=checkpoint['feature_state_sha256']:raise ValueError('Frozen native feature mismatch')
    metadata=dict(checkpoint_sha256=file_sha256(checkpoint_path),source_checkpoint_sha256=checkpoint['source_checkpoint_sha256'],feature_state_sha256=checkpoint['feature_state_sha256'],BOS_config_sha256=payload['sha256'],checkpoint_epoch=checkpoint['epoch'],centerbias='checkpoint train-only KDE48x24; bilinear dense normalization',whole_image_encoders=3,preceding_crop_encoders=0,preprocessing='DG rawRGB0..255512x1024; independent BOS256x512; no ROIencoder included',selection=checkpoint['selection'],numerics=dict(cudnn_enabled=True,frozen_DenseNet_visual_TF32=True,updated_readout_scanpath_selection_finalizer_TF32=False,matmul_TF32=False,parity_tolerance=1e-4))
    return ClinicalDGProbability(model,checkpoint,bos,metadata).to(device).eval(),metadata
