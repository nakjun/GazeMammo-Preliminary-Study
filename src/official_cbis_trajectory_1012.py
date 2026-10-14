"""Fresh ImageNet-only, causal official-CBIS trajectory experiment."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import whole_tensor, patch_tensor, seed_for
from .gaze_nll_adaptation_1011 import GazeNLLAdaptation1011, hard_nll
from .trajectory_1011 import segments, window_sample, objective

PRETRAINED = '/home/njsung/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth'
CONFIG = dict(grid=(24,48), k=16, dim=128, foveal=True, fpn=True,
              backbone_name='convnext_base', motion_bias=True, input_size=(256,512),
              spatial_history=True, history_length=4, behavior_mixture=False)
KIND = 'official_cbis_trajectory_1012'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def state_sha256(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        t=v.detach().cpu().contiguous();h.update(k.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode());h.update(t.numpy().tobytes())
    return h.hexdigest()

def build_fresh_model(device='cpu', pretrained_path=PRETRAINED, seed=42):
    torch.manual_seed(seed)
    model=GazeNLLAdaptation1011(**CONFIG, pretrained=False)
    source=torch.load(pretrained_path,map_location='cpu',weights_only=True)
    visual={k:v for k,v in source.items() if k.startswith('features.')}
    visual.update({k.replace('classifier.0.','norm.'):v for k,v in source.items() if k.startswith('classifier.0.')})
    model.visual.load_state_dict(visual,strict=True)
    model.set_backbone('frozen');model.visual.eval()
    model.original_pretrained_sha256=sha256(pretrained_path)
    model.original_pretrained_path=str(pretrained_path)
    model.feature_memory_format='contiguous_nchw'
    return model.to(device)

def allowed_payloads(csv_path, names):
    """Never parse an excluded row (including TEST and other datasets)."""
    names=set(names);found={}
    with open(csv_path) as f:
        for line in f:
            name,payload=line.rstrip('\n').split(';',1)
            if name not in names:continue
            if name in found:raise ValueError('Duplicate allowed gaze filename')
            samples=ast.literal_eval(payload)
            with np.errstate(divide='ignore',invalid='ignore'):
                points=np.asarray([np.asarray(s[0],dtype=float)/float(s[1]) for s in samples])
            if points.ndim!=2 or points.shape[1]!=2:raise ValueError('Invalid gaze coordinates')
            found[name]=points
    if set(found)!=names:raise ValueError('Missing allowed gaze payload')
    return found

def teacher_inputs(record, epoch=0, improved=False):
    w=window_sample([p.numpy() for p in record['parts']],record['image_id'],epoch=epoch,
                    mode='mixed' if improved else 'legacy16',burn_in=8)
    # Decoder at t gets the crop of t-1, never the target at t.
    vectors=torch.zeros(len(w['coords']),1024)
    for t in range(1,int(w['valid'].sum())):
        vectors[t]=record['vectors'][w['segment']][w['positions'][t-1]]
    length=len(w['coords']) if improved else 16
    return w['coords'][:length],w['loss_mask'][:length],vectors[:length]

def checkpoint_payload(model, cache, epoch, nll, arm):
    state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('visual.')}
    return dict(nonvisual_state_sha256=state_sha256(state),backbone_state_sha256=state_sha256(model.visual.state_dict()),kind=KIND, state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('visual.')},
        smoke=cache.get('smoke',False),source_hashes=cache.get('source_hashes',{}),runtime_flags=cache.get('runtime_flags',{}),model_config=model.model_config,original_pretrained_sha256=model.original_pretrained_sha256,
        original_pretrained_path=model.original_pretrained_path,protocol_sha256=cache['protocol_sha256'],
        gaze_manifest_sha256=cache['gaze_manifest_sha256'],source_sha256=sha256(__file__),
        training_patient_keys=cache['training_patient_keys'],selection_patient_keys=cache['selection_patient_keys'],
        test_patient_keys=cache['test_patient_keys'],epoch=epoch,canonical_val_hard_nll=nll,arm=arm,seed=42,feature_memory_format='contiguous_nchw')

def load_selected_trajectory(checkpoint, device='cpu'):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False) if isinstance(checkpoint,(str,Path)) else checkpoint
    if ck.get('kind')!=KIND:raise ValueError('Wrong trajectory checkpoint kind')
    for path,digest in ck.get('source_hashes',{}).items():
        if sha256(path)!=digest:raise ValueError('Trajectory dependency source changed')
    if ck['source_sha256']!=sha256(__file__):raise ValueError('Trajectory source changed')
    if ck['nonvisual_state_sha256']!=state_sha256(ck['state']):raise ValueError('Head state hash changed')
    if set(ck['training_patient_keys'])&set(ck['selection_patient_keys']):raise ValueError('Patient overlap')
    if (set(ck['training_patient_keys'])|set(ck['selection_patient_keys']))&set(ck['test_patient_keys']):raise ValueError('Official TEST patient leakage')
    if ck['model_config']!={k:list(v) if isinstance(v,tuple) else v for k,v in CONFIG.items()}:raise ValueError('Unexpected trajectory configuration')
    model=build_fresh_model(device,ck['original_pretrained_path'])
    if model.original_pretrained_sha256!=ck['original_pretrained_sha256']:raise ValueError('Original pretrained hash changed')
    if any(k.startswith('visual.') for k in ck['state']):raise ValueError('Compact head contains visual weights')
    missing,unexpected=model.load_state_dict(ck['state'],strict=False)
    if unexpected or any(not k.startswith('visual.') for k in missing):raise ValueError('Nonvisual state mismatch')
    if ck['backbone_state_sha256']!=state_sha256(model.visual.state_dict()):raise ValueError('Backbone state hash changed')
    model.requires_grad_(False);model.eval();model.selected_protocol_sha256=ck['protocol_sha256']
    return model

@torch.inference_mode()
def rollout(model, image, image_id, seed=42):
    if image.size!=(1500,3000):raise ValueError('Requires canonical 1500x3000 image')
    device=next(model.parameters()).device
    memory=model.maps(whole_tensor(image,model.input_size)[None].to(device))
    generator=torch.Generator(device='cpu').manual_seed(seed_for(seed,int(image_id)))
    prefix=torch.empty(1,0,2,device=device);hidden=None;vector=None
    for t in range(16):
        logp,hidden,_=model.step_history(memory,prefix,hidden,vector)
        cell=torch.multinomial(logp.exp().cpu(),1,generator=generator).squeeze(1).to(device)
        point=model.cell_xy[cell][:,None]
        prefix=torch.cat([prefix,point],1)
        if t<15:
            crop=patch_tensor(image,point[0,0].cpu().numpy()*[1500,3000],224)
            vector=model.visual.patch_vector(crop[None].to(device))
    return prefix[0].cpu().numpy()
