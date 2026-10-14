"""New independent DINO teacher-KL expert plus preserved MDN dual parent."""
import json,math
import torch
from torch import nn
from .data import whole_tensor
from .mdn_dual_output_1011 import load_mdn_dual
from .gaze_dino_1011 import restore_dino
from .dual_output_1011 import ROOT,config_sha256
from .proposal_utility_1011 import file_sha256

class IndependentCropBundle(nn.Module):
    def __init__(self,parent,dino):super().__init__();self.parent=parent;self.dino=dino
    def patch_vector(self,images):
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):return torch.cat([self.parent.patch_vector(images),self.dino.patch_vector(images)],-1)

class DinoMDNProbability(nn.Module):
    def __init__(self,parent,dino):
        super().__init__();self.parent=parent;self.dino=dino;self.grid=parent.grid;self.cell_xy=parent.cell_xy;self.foveal=True;self.vector_dims=parent.vector_dims+[dino.patch.in_features];self.visual=IndependentCropBundle(parent.visual,dino.visual)
        if tuple(dino.grid)!=tuple(parent.grid) or not torch.equal(dino.cell_xy,parent.cell_xy):raise ValueError('DINOgrid mismatch')
    @torch.no_grad()
    def maps(self,image):
        parent=self.parent.maps(image)
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):dino=self.dino.maps(whole_tensor(image,self.dino.input_size)[None].to(self.cell_xy.device))
        return dict(parent=parent,dino=dino,prior=parent['prior'],embedding=parent['embedding'])
    @torch.no_grad()
    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        parent_state,dino_state=(None,None) if hidden is None else hidden;dim=sum(self.parent.vector_dims)
        if patch_vector is not None and patch_vector.shape[-1]!=sum(self.vector_dims):raise ValueError('Independentparent/DINOvector mismatch')
        pv=patch_vector[...,:dim] if patch_vector is not None else None;dv=patch_vector[...,dim:] if patch_vector is not None else None
        parent,ps,pdiag=self.parent.step_history(memory['parent'],prefix,parent_state,pv)
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):dino,ds,ddiag=self.dino.step_history(memory['dino'],prefix,dino_state,dv)
        dino=dino.log_softmax(-1);q=torch.logaddexp(parent+math.log(.75),dino+math.log(.25));q=q-q.logsumexp(-1,keepdim=True)
        if not torch.isfinite(q).all():raise ValueError('Nonfinite independentDINO mixture')
        return q,(ps,ds),dict(parent=pdiag,dino=ddiag,probability_role='RAW.75MDNparent+.25independentteacherKLDINO; noR/geometrybias')
    def teacher_logits(self,memory,coords,patch_vectors=None):
        state=None;outputs=[]
        for t in range(coords.shape[1]):q,state,_=self.step_history(memory,coords[:,:t],state,patch_vectors[:,t] if patch_vectors is not None else None);outputs.append(q)
        return torch.stack(outputs,1)


def load_dino_mdn_dual(config,device='cpu'):
    if config.get('kind')!='dino_mdn_dual_output_1011' or config['probability']['weights']!=[.75,.25]:raise ValueError('WrongDINOmixture')
    for name,h in config['source_code_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('DINOmixture source SHA')
    payload=config['parent_config']
    if payload['sha256']!=config_sha256(payload['config']):raise ValueError('NestedMDNconfigSHA')
    parent,guided,head=load_mdn_dual(payload['config'],device);d=config['probability']['parents'][-1]
    if file_sha256(d['path'])!=d['sha256']:raise ValueError('TeacherKLDINOcheckpointSHA')
    dino=restore_dino(torch.load(d['path'],map_location='cpu',weights_only=False),device);dino.requires_grad_(False)
    result=DinoMDNProbability(parent,dino).to(device).eval();result.requires_grad_(False)
    return result,guided,head


def build_dino_mdn_config():
    parent=json.loads((ROOT/'runs/dual_target_1011/mdn_unified_runtime/config.json').read_text());dpath=ROOT/'runs/dual_target_1011/dino/gaze_augmented_teacherkl_seed42/best.pt';dcache=dpath.parent/'best_raw_cache.pt';base=ROOT/'runs/dual_target_1011/gaze_mdn_product_seed42/corrected_raw_cache.pt';ck=torch.load(dpath,map_location='cpu',weights_only=False)
    deps=list(parent['config']['probability']['parents']);deps.append(dict(role='teacherKLDINO',path=str(dpath.resolve()),sha256=file_sha256(dpath)))
    source=['src/dino_mdn_dual_output_1011.py','scripts/dino_mdn_dual_output_1011.py','src/gaze_dino_1011.py']
    return dict(kind='dino_mdn_dual_output_1011',parent_config=parent,source_code_sha256={p:file_sha256(ROOT/p) for p in source},reference_caches=[dict(raw_cache=str(base.resolve()),raw_cache_sha256=file_sha256(base)),dict(raw_cache=str(dcache.resolve()),raw_cache_sha256=file_sha256(dcache))],probability=dict(parents=deps,weights=[.75,.25],temperature=1,geometry_coefficient=0,recipe='RAWnormalized arithmetic.75MDNparent+.25independentteacherKLDINO includingimageonlyBOS',DINOepoch=ck['epoch'],numerics=dict(parent=parent['config']['probability']['numerics'],DINO=dict(cudnn_enabled=True,cudnn_TF32=True,matmul_TF32=False))),roi=parent['config']['roi'],compute=dict(independent_whole_image_encoders=5,preceding_crop_encoders=3,DINOencoder_and_keys_independent=True,MDNembedding_existingcontrol_forward_reused=True,original_ROI_encoder_independent=True),claims=dict(validation='Exploratory fixedval93; NLL5.3 not met',forced_path_nll_reported=False))
