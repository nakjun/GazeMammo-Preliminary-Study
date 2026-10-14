"""Pixel-only frozen selectors and common original-control classification features."""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .data import ROOT, patch_tensor, whole_tensor, select_centers, seed_for
from .dual_output_1011 import config_sha256, unified_rollout
from .proposal_center_offset_1011 import offset_rectangles, predict_center_offsets, shifted_candidate_coords
from .proposal_policy_1011 import proposal_budget_indices
from .proposal_utility_1011 import file_sha256, state_sha256

ARMS = ('random', 'grid', 'proposal', 'gaze', 'hybrid')
SOURCE_SHA256 = '0bc7e224c6dafe36b8be8b12dbe3694af3b272f22cfedca3b39795d56522d239'
DEFAULT_CONFIG = ROOT/'runs/dual_target_1011/mammo_clip_unified_runtime/config.json'


def _dependencies(value):
    found = {}
    def walk(v):
        if isinstance(v, dict):
            if 'path' in v and 'sha256' in v:
                found[str(Path(v['path']).resolve())] = v['sha256']
            for x in v.values(): walk(x)
        elif isinstance(v, list):
            for x in v: walk(x)
    walk(value)
    return [dict(path=p, sha256=h) for p,h in sorted(found.items())]


def _patient_proof(value, train, val):
    if isinstance(value, dict):
        for key, items in value.items():
            if 'patient_keys' in key and isinstance(items, (list, tuple, set)):
                allowed = val if 'selection' in key else train if ('train' in key) else train|val
                if set(items)-allowed: raise ValueError('Nested source patient provenance outside allowed development split: '+key)
            _patient_proof(items, train, val)
    elif isinstance(value, (list, tuple)):
        for item in value: _patient_proof(item, train, val)


def validate_development_population(frame, payload):
    required = {'image_id','patient_key','split','label','source','file_name'}
    if not required <= set(frame): raise ValueError('Missing population columns')
    selected = frame[(frame.source=='CBIS-DDSM') & frame.split.isin(['train','val']) & frame.label.isin([0,1])].copy()
    if selected.image_id.duplicated().any(): raise ValueError('Duplicate image identity')
    train=set(frame.loc[(frame.source=='CBIS-DDSM') & (frame.split=='train'),'patient_key']);val=set(frame.loc[(frame.source=='CBIS-DDSM') & (frame.split=='val'),'patient_key'])
    excluded=set(frame.loc[frame.split.isin(['test','TEST','quarantine']),'patient_key'])
    if train&val or (train|val)&excluded: raise ValueError('Patient split overlap')
    counts=selected.groupby(['split','label']).size().to_dict()
    if counts!={('train',0):187,('train',1):189,('val',0):46,('val',1):47}: raise ValueError('Expected exact labeled development469 population')
    # Validate every frozen checkpoint, including nested initializer proof. Never load feature caches.
    proofs=[]
    for dep in _dependencies(payload):
        if file_sha256(dep['path'])!=dep['sha256']: raise ValueError('Source dependency changed')
        checkpoint=torch.load(dep['path'],map_location='cpu',weights_only=False)
        if 'training_patient_keys' not in checkpoint and 'training_patient_keys' not in checkpoint.get('config',{}): raise ValueError('Missing source training patient provenance')
        _patient_proof(checkpoint,train,val)
        proofs.append(dict(path=dep['path'],sha256=dep['sha256'],training_patient_keys=checkpoint.get('training_patient_keys',checkpoint.get('config',{}).get('training_patient_keys',[])),selection_patient_keys=checkpoint.get('selection_patient_keys',checkpoint.get('config',{}).get('selection_patient_keys',[]))))
    payload['source_patient_audit']=proofs
    return selected.reset_index(drop=True)


def classification_encoder(probability):
    """Unwrap independent DINO crop bundle to original control PatchBundle."""
    visual=probability.visual
    while not hasattr(visual,'encoders'):
        if not hasattr(visual,'parent'): raise ValueError('Original control encoder missing')
        visual=visual.parent
    return visual.encoders[0]


def load_frozen_source(config_path=DEFAULT_CONFIG, device='cpu'):
    from .mammo_clip_dual_output_1011 import load_mammo_clip_dual
    path=Path(config_path).resolve();payload=json.loads(path.read_text())
    if payload.get('sha256')!=SOURCE_SHA256 or config_sha256(payload['config'])!=SOURCE_SHA256: raise ValueError('Unapproved source config')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.allow_tf32=True
    probability,guided,offset=load_mammo_clip_dual(payload['config'],device)
    payload=copy.deepcopy(payload);payload['source_config_path']=str(path);payload['source_config_file_sha256']=file_sha256(path)
    for module in (probability,guided,offset): module.eval();module.requires_grad_(False)
    payload['classification_encoder_sha256']=state_sha256(classification_encoder(probability).state_dict())
    return probability,guided,offset,payload


def _sync(device):
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


@torch.no_grad()
def extract_image_features(bundle, image, image_id, seed=42, arms=ARMS):
    """Accept pixels and an opaque numeric identity only; never annotations or filenames."""
    if isinstance(image_id,bool) or not isinstance(image_id,(int,np.integer)): raise ValueError('Numeric opaque image identity required')
    if len(set(arms))!=len(arms) or set(arms)-set(ARMS): raise ValueError('Unknown/duplicate selector')
    probability,guided,offset,payload=bundle;device=next(probability.parameters()).device
    image=image.convert('L');pathseed=seed_for(seed,int(image_id));encoder=classification_encoder(probability)
    _sync(device);started=time.perf_counter()
    # Same encoder/preprocessing for global and all local features.
    with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):
        global_feature=encoder.patch_vector(whole_tensor(image,(256,512))[None].to(device))[0]
    _sync(device);global_seconds=time.perf_counter()-started
    memories=None;memory_seconds=0.;result={'global_features':global_feature.float().cpu(), 'timing':{'global_classification_seconds':global_seconds},'selection_metadata':{}}
    for arm in arms:
        _sync(device);started=time.perf_counter()
        if arm in ('random','grid'):
            pixels=select_centers(image,arm,16,pathseed)
            if len(pixels)<16: pixels=np.resize(pixels,(16,2))
            coords=torch.tensor(pixels/np.array(image.size),dtype=torch.float32)
            meta={'policy':arm,'requested_roi_budget':0,'actual_roi_budget':0}
        else:
            if arm=='proposal':
                with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):
                    roi=guided.maps(whole_tensor(image,guided.input_size)[None].to(device))
                offsets=predict_center_offsets(offset,roi)[0]
                relevance=guided.proposal(roi['keys']).sigmoid().squeeze(-1)
                candidates=shifted_candidate_coords(guided.cell_xy,offsets,image.size,payload['config']['roi']['strength'])
                selected=proposal_budget_indices(relevance,offset_rectangles(candidates,image.size),16,payload['config']['roi']['nms'])
                if len(selected)!=16: raise ValueError('Proposal NMS failed exact sixteen budget')
                coords=candidates[selected].float().cpu()
                meta={'policy':arm,'requested_roi_budget':16,'actual_roi_budget':16,'gaze_component_used':False,'conditional_gaze_steps':0,'reservation_flags':[True]*16}
            elif arm=='gaze':
                if memories is None:
                    before=time.perf_counter();human=probability.maps(image);_sync(device);memory_seconds=time.perf_counter()-before;memories=(human,None)
                human=memories[0];prefix=human['prior'].new_zeros(1,0,2);state=None;vector=None;generator=torch.Generator().manual_seed(pathseed)
                for step in range(16):
                    logp,state,_=probability.step_history(human,prefix,state,vector)
                    index=int(torch.multinomial(logp.exp().cpu(),1,generator=generator));point=probability.cell_xy[index]
                    prefix=torch.cat([prefix,point[None,None]],1)
                    if probability.foveal: vector=probability.visual.patch_vector(patch_tensor(image,prefix[0,-1].cpu().numpy()*image.size,224)[None].to(device))
                coords=prefix[0].float().cpu()
                meta={'policy':arm,'requested_roi_budget':0,'actual_roi_budget':0,'gaze_component_used':True,'conditional_gaze_steps':16,'reservation_flags':[False]*16,'probability_role':'frozen RAW MammoMixture; no ROI bias'}
            else:
                if memories is None:
                    before=time.perf_counter();human=probability.maps(image);_sync(device);memory_seconds=time.perf_counter()-before;memories=(human,None)
                with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):
                    roi=guided.maps(whole_tensor(image,guided.input_size)[None].to(device))
                meta=unified_rollout(probability,guided,offset,memories[0],roi,image,None,payload['config'],seed=pathseed)
                if meta['actual_roi_budget']!=14: raise ValueError('Hybrid NMS failed exact fourteen budget')
                meta['policy']=arm;meta['probability_role']='frozen RAW MammoMixture with original ROI14+RAW2 selection'
                coords=torch.tensor(meta['coordinates'],dtype=torch.float32)
        _sync(device);selector_seconds=time.perf_counter()-started
        patches=torch.stack([patch_tensor(image,c.double().numpy()*image.size,224) for c in coords]).to(device)
        _sync(device);before=time.perf_counter()
        with torch.backends.cudnn.flags(enabled=True,allow_tf32=True): features=encoder.patch_vector(patches)
        _sync(device)
        result[arm]={'features':features.float().cpu(),'coordinates':coords,'rectangles':offset_rectangles(coords,image.size).long().cpu(),'mask':torch.ones(16,dtype=torch.bool)}
        result['timing'][arm]={'selector_seconds':selector_seconds,'classification_seconds':time.perf_counter()-before,'patch_count':16}
        result['selection_metadata'][arm]=meta
    result['timing']['source_memory_seconds']=memory_seconds
    for v in [result['global_features']]+[result[a]['features'] for a in arms]:
        if v.shape[-1]!=1024 or not torch.isfinite(v).all(): raise ValueError('Invalid common classification features')
    return result


def validate_feature_cache(cache):
    if cache.get('kind')!='gaze_patch_classification_features_1012': raise ValueError('Wrong feature cache kind')
    rows=cache['rows'];p=cache['provenance'];t=cache['tensors'];n=len(rows)
    if p['source_config_sha256']!=SOURCE_SHA256 or p['crop_size']!=224 or p['feature_dim']!=1024: raise ValueError('Cache source/topology mismatch')
    if p['arms']!=list(ARMS) or p['path_seed']!=42: raise ValueError('Unapproved selector protocol')
    config_payload=json.loads(Path(p['source_config_path']).read_text())
    if config_payload.get('sha256')!=SOURCE_SHA256 or config_sha256(config_payload['config'])!=SOURCE_SHA256: raise ValueError('Semantic source config mismatch')
    expected_encoder=config_payload['config']['probability']['parents'][0]['visual_state_sha256']
    if p['classification_encoder_sha256']!=expected_encoder: raise ValueError('Original control encoder identity mismatch')
    if len({r['image_id'] for r in rows})!=n: raise ValueError('Duplicate cached identity')
    if any(r['split'] not in ('train','val') or r['label'] not in (0,1) or not r['patient_key'].startswith('CBIS_') for r in rows): raise ValueError('Disallowed cache population')
    if {r['patient_key'] for r in rows if r['split']=='train'} & {r['patient_key'] for r in rows if r['split']=='val'}: raise ValueError('Cached patient overlap')
    for arm in p['arms']:
        for suffix,shape in [('features',(n,16,1024)),('coordinates',(n,16,2)),('rectangles',(n,16,4)),('mask',(n,16))]:
            v=t[f'{arm}_{suffix}']
            if tuple(v.shape)!=shape or not torch.isfinite(v).all(): raise ValueError('Invalid cache tensor '+arm+'_'+suffix)
        if t[f'{arm}_mask'].dtype!=torch.bool: raise ValueError('Mask must be boolean')
        if not t[f'{arm}_mask'].all(): raise ValueError('Expected sixteen valid tokens')
        coords=t[f'{arm}_coordinates']
        if ((coords<0)|(coords>=1)).any(): raise ValueError('Invalid cached coordinates')
        for i,r in enumerate(rows):
            if not torch.equal(t[f'{arm}_rectangles'][i].long(),offset_rectangles(coords[i],r['image_size']).long()): raise ValueError('Crop rectangle mismatch')
    if tuple(t['global_features'].shape)!=(n,1024) or not torch.isfinite(t['global_features']).all(): raise ValueError('Invalid global feature')
    import pandas as pd
    if file_sha256(p['manifest_path'])!=p['manifest_sha256']: raise ValueError('Manifest changed')
    manifest=pd.read_csv(p['manifest_path']); byid=manifest.set_index('image_id')
    for r in rows:
        if r['image_id'] not in byid.index: raise ValueError('Unknown cached identity')
        original=byid.loc[r['image_id']]
        if any(r[k]!=original[k] for k in ('patient_key','split','label','file_name')): raise ValueError('Mislabeled cache row')
        if original['source']!='CBIS-DDSM': raise ValueError('Quarantined source')
    expected=8 if p['population']=='smoke8' else 469 if p['population']=='development469' else -1
    if n!=expected: raise ValueError('Population cache count mismatch')
    if p['population']=='development469':
        selected=manifest[(manifest.source=='CBIS-DDSM') & manifest.split.isin(['train','val']) & manifest.label.isin([0,1])]
        if {r['image_id'] for r in rows}!=set(selected.image_id): raise ValueError('Full development identity mismatch')
    if p['population']=='smoke8':
        selected=manifest[(manifest.source=='CBIS-DDSM') & manifest.split.isin(['train','val']) & manifest.label.isin([0,1])]
        ids=[int(i) for split in ('train','val') for label in (0,1) for i in selected[(selected.split==split)&(selected.label==label)].head(2).image_id]
        if [r['image_id'] for r in rows]!=ids: raise ValueError('Smoke must be metadata-first8')
    for dep in p['dependencies']:
        if file_sha256(dep['path'])!=dep['sha256']: raise ValueError('Cache runtime dependency changed')
    for path,sha in p['source_code_sha256'].items():
        if file_sha256(ROOT/path)!=sha: raise ValueError('Cache source code changed')
    for r in rows:
        if file_sha256(r['image_path'])!=r['image_sha256']: raise ValueError('Cached image changed')
    if file_sha256(p['source_config_path'])!=p['source_config_file_sha256']: raise ValueError('Cache config changed')
    return cache
