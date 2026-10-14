"""New dual runtime:75% frozen arithmetic ensemble plus25% clinical DG.

Original probability parent states/crop vectors remain independent. Clinical
DenseNet consumes original raw image and only actual observed coordinates.
The two BOS image memories are safely shared as the very same memory objects;
original guided ROI encoder and keys remain independent of every expert.
"""
import json,math,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from .data import whole_tensor
from .dual_output_1011 import load_runtime,config_sha256,ROOT
from .clinical_dg_probability_1011 import ClinicalDGProbability
from .proposal_utility_1011 import file_sha256,state_sha256

class ClinicalMixture(nn.Module):
    def __init__(self,baseline,clinical):
        super().__init__();self.baseline=baseline;self.clinical=clinical;self.grid=baseline.grid;self.cell_xy=baseline.cell_xy;self.foveal=baseline.foveal;self.visual=baseline.visual;self.vector_dims=baseline.vector_dims
    @torch.no_grad()
    def maps(self,image):
        if not isinstance(image,Image.Image):raise ValueError('Clinical mixture maps requires original PILimage')
        device=self.cell_xy.device;gray=image.convert('L');base=self.baseline.maps([whole_tensor(gray,size)[None].to(device) for size in self.baseline.input_sizes])
        rgb=torch.from_numpy(np.asarray(gray.resize((512,1024),Image.Resampling.BILINEAR)).copy()).float()[None,None].repeat(1,3,1,1).to(device)
        clinical=self.clinical.maps_rgb(rgb,base)
        return dict(baseline=base,clinical=clinical,prior=base['prior'])
    @torch.no_grad()
    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        base,state,diagnostic=self.baseline.step_history(memory['baseline'],prefix,hidden,patch_vector)
        clinic,_,clinical_diagnostic=self.clinical.step_history(memory['clinical'],prefix)
        q=torch.logaddexp(base+math.log(.75),clinic+math.log(.25));q=q-q.logsumexp(-1,keepdim=True)
        if not torch.isfinite(q).all():raise ValueError('Clinical mixture nonfinite')
        return q,state,dict(baseline=diagnostic,clinical=clinical_diagnostic,probability_role='RAW.75baseline(.75control+.25improved)+.25clinicalDG; noR/geometrybias')
    def teacher_logits(self,memory,coords,patch_vectors=None):
        hidden=None;outputs=[]
        for t in range(coords.shape[1]):
            q,hidden,_=self.step_history(memory,coords[:,:t],hidden,patch_vectors[:,t] if patch_vectors is not None else None);outputs.append(q)
        return torch.stack(outputs,1)


def load_clinical_dual(config,device='cpu'):
    if config.get('kind')!='clinical_dual_output_1011' or config['probability']['weights']!=[.5625,.1875,.25]:raise ValueError('Unsupported clinical mixture')
    for name,h in config['source_code_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('New clinical runtime source changed')
    baseline,guided,head=load_runtime(config['baseline_config']['config'],device)
    dependency=config['probability']['parents'][2]
    if file_sha256(dependency['path'])!=dependency['sha256']:raise ValueError('Clinical best SHA mismatch')
    sys.path.insert(0,str(ROOT/'scripts'));from train_deepgaze_clinical_1011 import load_clinical
    model,checkpoint=load_clinical(dependency['path'],device)
    parents=config['probability']['parents'][:2]
    if len(checkpoint['bos_sources'])!=2 or any(a['sha256']!=b['sha256'] for a,b in zip(checkpoint['bos_sources'],parents)):raise ValueError('Clinical BOS dependencies differ')
    clinic=ClinicalDGProbability(model,checkpoint,baseline,dict(checkpoint_sha256=dependency['sha256']))
    result=ClinicalMixture(baseline,clinic).to(device).eval();result.requires_grad_(False)
    return result,guided,head


def build_clinical_config():
    baseline=json.loads((ROOT/'runs/dual_target_1011/unified_runtime/config_cudnn_ieee.json').read_text());reference=json.loads((ROOT/'runs/dual_target_1011/deepgaze_clinical_fixed_mixtures/runtime_configs.json').read_text())[0]
    if reference['weights']!=[.5625,.1875,.25]:raise ValueError('Wrong approved mixture reference')
    parents=list(baseline['config']['probability']['parents']);d=reference['sources'][2];parents.append(dict(role='clinicalDG',path=d['checkpoint'],sha256=d['checkpoint_sha256'],raw_cache=d['raw_cache'],raw_cache_sha256=d['raw_cache_sha256']))
    source=['src/clinical_dual_output_1011.py','src/clinical_dg_probability_1011.py','scripts/clinical_dual_output_1011.py','src/deepgaze_clinical_1011.py','src/deepgaze_context_1011.py','scripts/train_deepgaze_clinical_1011.py']
    return dict(kind='clinical_dual_output_1011',baseline_config=baseline,source_code_sha256={p:file_sha256(ROOT/p) for p in source},probability=dict(parents=parents,weights=[.5625,.1875,.25],temperature=1,geometry_coefficient=0,recipe='RAW.75baseline+.25clinicalDG; effectiveBOScontrol.75/improved.25',numerics=dict(baseline=baseline['config']['probability']['numerics'],clinical=dict(cudnn_enabled=True,visual_TF32=True,head_TF32=False,matmul_TF32=False))),roi=baseline['config']['roi'],compute=dict(independent_whole_image_encoders=4,preceding_crop_encoders=2,clinical_BOS_same_baseline_memories_reused=True,original_ROI_encoder_independent=True),claims=dict(validation='Exploratory val93; fixed3mixtureselection; NLL5.3 not met',forced_path_nll_reported=False))
