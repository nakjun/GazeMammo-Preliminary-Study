"""Causal probability component plus independent frozen ROI source."""
import hashlib,json,math,sys
from pathlib import Path
import torch
from torch import nn
from .data import whole_tensor,patch_tensor
from .proposal_utility_1011 import file_sha256
from .proposal_center_offset_1011 import load_center_offset_head,predict_center_offsets,shifted_candidate_coords,offset_rectangles
from .proposal_policy_1011 import proposal_budget_indices
ROOT=Path(__file__).resolve().parents[1]


def config_sha256(config):return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class PatchBundle(nn.Module):
    def __init__(self,models):super().__init__();self.encoders=nn.ModuleList([m.visual for m in models]);self.enabled=[m.foveal for m in models]
    def patch_vector(self,images):
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):return torch.cat([encoder.patch_vector(images) for encoder,enabled in zip(self.encoders,self.enabled) if enabled],-1)


class ProbabilityComponent(nn.Module):
    """All parent states remain separate; coordinates/crops share the real past."""
    def __init__(self,parents,weights):
        super().__init__()
        if len(parents)!=len(weights) or not parents or any(w<0 for w in weights) or abs(sum(weights)-1)>1e-10:raise ValueError('Invalid probability mixture')
        if any(tuple(m.grid)!=tuple(parents[0].grid) or not torch.equal(m.cell_xy,parents[0].cell_xy) for m in parents):raise ValueError('Probability parent grid mismatch')
        self.parents=nn.ModuleList(parents);self.weights=tuple(weights);self.grid=tuple(parents[0].grid);self.cell_xy=parents[0].cell_xy;self.input_sizes=[tuple(m.input_size) for m in parents];self.foveal=any(m.foveal for m in parents);self.vector_dims=[m.patch.in_features if m.foveal else 0 for m in parents];self.visual=PatchBundle(parents)
        for parameter in self.parameters():parameter.requires_grad_(False)
        self.eval()
    def maps(self,images):
        if isinstance(images,torch.Tensor):
            if len(set(self.input_sizes))!=1:raise ValueError('Distinct parent resolutions require separate tensors')
            images=[images]*len(self.parents)
        if len(images)!=len(self.parents):raise ValueError('Parent image count mismatch')
        memories=[]
        for model,image in zip(self.parents,images):
            if hasattr(model,'maps_features'):
                with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):features=model.visual(image)
                with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):memory=model.maps_features(features)
            else:
                with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):memory=model.maps(image)
            memories.append(memory)
        return dict(parents=memories,prior=memories[0]['prior'])
    def step_history(self,memory,prefix,hidden=None,patch_vector=None):
        states=[None]*len(self.parents) if hidden is None else list(hidden)
        if len(states)!=len(self.parents):raise ValueError('Parent state count mismatch')
        if patch_vector is not None and patch_vector.shape[-1]!=sum(self.vector_dims):raise ValueError('Concatenated previous vector mismatch')
        logps=[];next_states=[];diagnostics=[];start=0
        for model,weight,mem,state,dim in zip(self.parents,self.weights,memory['parents'],states,self.vector_dims):
            vector=patch_vector[...,start:start+dim] if patch_vector is not None and dim else None;start+=dim
            with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):raw,new_state,diag=model.step_history(mem,prefix,state,vector)
            logps.append(raw.log_softmax(-1));next_states.append(new_state);diagnostics.append(diag)
        terms=[q+math.log(w) for q,w in zip(logps,self.weights) if w>0];mixed=torch.logsumexp(torch.stack(terms),0);mixed=mixed-mixed.logsumexp(-1,keepdim=True)
        if not torch.isfinite(mixed).all():raise ValueError('Nonfinite mixed probability')
        return mixed,tuple(next_states),dict(parent_diagnostics=diagnostics,parent_logp=logps)
    def teacher_logits(self,memory,coords,patch_vectors=None):
        state=None;outputs=[]
        for t in range(coords.shape[1]):
            logp,state,_=self.step_history(memory,coords[:,:t],state,patch_vectors[:,t] if patch_vectors is not None else None);outputs.append(logp)
        return torch.stack(outputs,1)


def validate_dependencies(config):
    if config.get('kind')!='dual_output_1011' or config['probability'].get('temperature')!=1 or config['probability'].get('geometry_coefficient')!=0:raise ValueError('Unsupported dual-output recipe')
    for relative,sha in config.get('source_code_sha256',{}).items():
        if file_sha256(ROOT/relative)!=sha:raise ValueError('Runtime source/geometry code SHA mismatch')
    for dependency in config['probability']['parents']+[config['roi']['guided'],config['roi']['offset']]:
        if file_sha256(dependency['path'])!=dependency['sha256']:raise ValueError('Runtime dependency SHA mismatch')


def load_runtime(config,device='cpu',parent_loader=None):
    sys.path.insert(0,str(ROOT/'scripts'))
    from gaze_nll_adaptation_1011 import load_base,apply_cache_resolution
    from src.gaze_nll_adaptation_1011 import initialize
    from guided_trajectory_1011 import load_guided
    validate_dependencies(config);torch.backends.cuda.matmul.allow_tf32=False;parents=[]
    for dependency in config['probability']['parents']:
        if parent_loader is None:
            source,_=load_base(dependency['path']);model=apply_cache_resolution(initialize(source),dependency['cache_layout'])
        else:model=parent_loader(dependency)
        parents.append(model.to(device).eval())
    probability=ProbabilityComponent(parents,config['probability']['weights']).to(device);guided,_=load_guided(config['roi']['guided']['path']);guided=guided.to(device).eval();head,checkpoint=load_center_offset_head(config['roi']['offset']['path'],config['roi']['guided']['sha256'],device)
    for module in (guided,head):
        for parameter in module.parameters():parameter.requires_grad_(False)
    if not torch.equal(probability.cell_xy,guided.cell_xy):raise ValueError('ROI and probability source grid mismatch')
    return probability,guided,head


def build_config():
    parents=[]
    for role,cache_name,weight in [('control','human_control_raw_cache_for_ensemble.pt',.75),('improved','human_improved_raw_cache.pt',.25)]:
        cache_path=ROOT/'runs/dual_target_1011'/cache_name;cache=torch.load(cache_path,map_location='cpu',weights_only=False);checkpoint=Path(cache['base_checkpoint']);layout={k:cache['cache_provenance'][k] for k in ('input_size','feature_memory_format') if k in cache['cache_provenance']}
        if file_sha256(checkpoint)!=cache['base_checkpoint_sha256']:raise ValueError('Source cache checkpoint mismatch')
        parents.append(dict(role=role,path=str(checkpoint.resolve()),sha256=cache['base_checkpoint_sha256'],raw_cache=str(cache_path.resolve()),raw_cache_sha256=file_sha256(cache_path),cache_layout=layout,visual_state_sha256=cache['cache_provenance']['visual_state_sha256']))
    guide=ROOT/'runs/guided_1011_combined_seed42/best.pt';head=ROOT/'runs/dual_target_1011/center_offset_spatial_p1_seed42/last.pt'
    return dict(source_code_sha256={str(p.relative_to(ROOT)):file_sha256(p) for p in [Path(__file__),ROOT/'src/data.py',ROOT/'src/constraints.py',ROOT/'src/proposal_center_offset_1011.py']},kind='dual_output_1011',probability=dict(kind='raw_arithmetic',parents=parents,weights=[.75,.25],temperature=1,geometry_coefficient=0,gaze_target_grid=[24,48],numerics=dict(visual_cudnn_tf32=True,projection_decoder_cudnn_tf32=False,matmul_tf32=False,cudnn_enabled=True,reason='ExplicitcuDNNon;visualTF32matchescachedvisuals,projectiondecoderIEEE; parentauditmaxerror<2e-5')),roi=dict(guided=dict(path=str(guide.resolve()),sha256=file_sha256(guide)),offset=dict(path=str(head.resolve()),sha256=file_sha256(head)),strength=.5,nms=.15,reserved=14,crop_size=224,total_steps=16),compute=dict(independent_whole_image_encoders=3,preceding_crop_encoders=2,safe_visual_reuse_applied=False),claims=dict(forced_path_nll_reported=False,validation='Exploratory fixedval93; NLL5.3 goal not achieved'))


@torch.no_grad()
def actual_memories(probability,guided,image,device):
    inputs=[whole_tensor(image,size)[None].to(device) for size in probability.input_sizes];human=probability.maps(inputs);roi=guided.maps(whole_tensor(image,guided.input_size)[None].to(device));return human,roi


@torch.no_grad()
def unified_rollout(probability,guided,head,human_memory,roi_memory,image,geometry,config,seed=42):
    roi=config['roi']
    with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):offsets=predict_center_offsets(head,roi_memory)[0];relevance=guided.proposal(roi_memory['keys']).sigmoid().squeeze(-1)
    anchors=guided.cell_xy;candidates=shifted_candidate_coords(anchors,offsets,image.size,roi['strength'])
    if not torch.isfinite(relevance).all():raise ValueError('Invalid original ROI proposal')
    selected=proposal_budget_indices(relevance,offset_rectangles(candidates,image.size),roi['reserved'],roi['nms']);prefix=human_memory['prior'].new_zeros(1,0,2);state=None;vector=None;generator=torch.Generator().manual_seed(seed);indices=[];trace=[]
    for t in range(roi['total_steps']):
        logp,state,_=probability.step_history(human_memory,prefix,state,vector);sample=int(torch.multinomial(logp.exp().cpu(),1,generator=generator));reserved=t<len(selected);index=selected[t] if reserved else sample;point=candidates[index] if reserved else probability.cell_xy[index];indices.append(index);prefix=torch.cat([prefix,point[None,None]],1);trace.append(dict(step=t,source_anchor_index=index,reserved_proposal=reserved,raw_probability_chosen_source=float(logp[0,index].exp()),raw_probability_entropy=float(-(logp.exp()*logp).sum()),probability_role='normalizedrawcontrol.75+improved.25; noR/geometrybias'))
        if probability.foveal:vector=probability.visual.patch_vector(patch_tensor(image,prefix[0,-1].cpu().numpy()*image.size,224)[None].to(prefix.device))
    return dict(coordinates=prefix[0].cpu().tolist(),indices=indices,source_anchor_indices=indices,candidate_coords=candidates.cpu().tolist(),offset_predicted=offsets.cpu().tolist(),relevance=relevance[0].cpu().tolist(),proposal_valid=True,reserved_candidate_indices=selected,reserved_steps=list(range(len(selected))),reservation_flags=[t<len(selected) for t in range(16)],actual_crop_rectangles=offset_rectangles(prefix[0],image.size).cpu().tolist(),steps=trace,requested_roi_budget=14,actual_roi_budget=len(selected),nms_geometry_protocol='rounded_actual_shifted224_rectangles',gaze_component_used=True,conditional_gaze_steps=16-len(selected))
