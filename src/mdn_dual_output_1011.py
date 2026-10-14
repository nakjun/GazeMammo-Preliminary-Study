"""Frozen clinical parent plus learned causal signed MDN product correction.

Captures actual original-control visual output from its existing forward call;
no encoder substitution, extra resize, cached validation feature or gaze input.
Independent original guided ROI14 memory remains unchanged.
"""
import json
import torch
from torch import nn
from .clinical_dual_output_1011 import load_clinical_dual
from .dual_output_1011 import ROOT,config_sha256
from .gaze_mdn_correction_1011 import GazeMDNCorrection
from .proposal_utility_1011 import file_sha256

class MDNProductProbability(nn.Module):
    def __init__(self,parent,correction):
        super().__init__();self.parent=parent;self.correction=correction;self.grid=parent.grid;self.cell_xy=parent.cell_xy;self.foveal=parent.foveal;self.visual=parent.visual;self.vector_dims=parent.vector_dims
    @torch.no_grad()
    def maps(self,image):
        captures=[]
        hook=self.parent.baseline.parents[0].visual.register_forward_hook(lambda module,args,output:captures.append(tuple(x.detach() for x in output)))
        try:memory=self.parent.maps(image)
        finally:hook.remove()
        if len(captures)!=1 or len(captures[0])!=3 or captures[0][-1].shape[1]!=1024:raise ValueError('Originalcontrol feature capture mismatch')
        embedding=captures[0][-1].mean((-2,-1))
        if not torch.isfinite(embedding).all():raise ValueError('Invalid actualcontrol embedding')
        return dict(parent=memory,embedding=embedding,original_control_features=captures[0],image_size=embedding.new_tensor([image.size]),prior=memory['prior'])
    @torch.no_grad()
    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        parent,state,diagnostic=self.parent.step_history(memory['parent'],prefix,hidden,patch_vector)
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):q=self.correction.step_history(memory['embedding'],prefix,memory['image_size'],parent)
        if not torch.isfinite(q).all():raise ValueError('Nonfinite product probability')
        return q,state,dict(parent=diagnostic,probability_role='RAWnormalizedclinical.25parent+tanh(alpha_t)*causalMDNlogmass; noR/geometrybias')
    def teacher_logits(self,memory,coords,patch_vectors=None):
        hidden=None;outputs=[]
        for t in range(coords.shape[1]):
            q,hidden,_=self.step_history(memory,coords[:,:t],hidden,patch_vectors[:,t] if patch_vectors is not None else None);outputs.append(q)
        return torch.stack(outputs,1)


def load_mdn_dual(config,device='cpu'):
    if config.get('kind')!='mdn_dual_output_1011':raise ValueError('Unsupported MDNproduct runtime')
    for name,h in config['source_code_sha256'].items():
        if file_sha256(ROOT/name)!=h:raise ValueError('MDNruntime source SHA mismatch')
    payload=config['parent_config']
    if payload['sha256']!=config_sha256(payload['config']):raise ValueError('Nestedclinical config SHA mismatch')
    parent,guided,head=load_clinical_dual(payload['config'],device);dependency=config['probability']['parents'][-1]
    if file_sha256(dependency['path'])!=dependency['sha256']:raise ValueError('MDNbest SHA mismatch')
    ck=torch.load(dependency['path'],map_location='cpu',weights_only=False)
    if ck.get('kind')!='gaze_mdn_correction_1011':raise ValueError('Wrong MDNcheckpoint')
    for d in payload['config']['probability']['parents']:
        if ck['config']['parent_cache_sha256'].get(d['raw_cache'])!=d['raw_cache_sha256']:raise ValueError('MDNtraining parentdependency mismatch')
    if ck['config']['base_sha256']!=payload['config']['probability']['parents'][0]['sha256']:raise ValueError('MDNembedding source mismatch')
    model=GazeMDNCorrection().to(device);model.load_state_dict(ck['state'],strict=True);model.eval();model.requires_grad_(False)
    result=MDNProductProbability(parent,model).to(device).eval();result.requires_grad_(False)
    return result,guided,head


def build_mdn_config():
    parent=json.loads((ROOT/'runs/dual_target_1011/clinical_unified_runtime/config.json').read_text());path=ROOT/'runs/dual_target_1011/gaze_mdn_product_seed42/best.pt';reference=path.parent/'corrected_raw_cache.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
    parents=list(parent['config']['probability']['parents']);parents.append(dict(role='MDNproduct',path=str(path.resolve()),sha256=file_sha256(path)))
    source=['src/mdn_dual_output_1011.py','scripts/mdn_dual_output_1011.py','src/gaze_mdn_correction_1011.py','src/gaze_mdn_1011.py']
    return dict(kind='mdn_dual_output_1011',parent_config=parent,source_code_sha256={p:file_sha256(ROOT/p) for p in source},reference_cache=dict(raw_cache=str(reference.resolve()),raw_cache_sha256=file_sha256(reference)),probability=dict(parents=parents,temperature=1,geometry_coefficient=0,recipe='normalizedclinical.25parent+tanh(alpha_t)*MDNlogmass',checkpoint_epoch=ck['epoch'],alpha_tanh=ck['state']['alpha'].tanh().tolist(),numerics=parent['config']['probability']['numerics']),roi=parent['config']['roi'],compute=dict(independent_whole_image_encoders=4,preceding_crop_encoders=2,MDNembedding_same_originalcontrol_forward_reused=True,original_ROI_encoder_independent=True),claims=dict(validation='Exploratory val93 checkpointselection; fixed15epochtrial; NLL5.3 not met',forced_path_nll_reported=False))
