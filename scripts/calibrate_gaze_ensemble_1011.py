"""Exploratory exact-legacy16 frozen-distribution ensembles; no source CNN forwards."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aligned_records(caches):
    maps=[]
    for cache in caches:
        records=cache['records'];mapping={int(r['image_id']):r for r in records}
        if len(mapping)!=len(records):raise ValueError('Duplicate cached image')
        patients={split:{r['patient_key'] for r in records if r['split']==split} for split in ('train','val')}
        if patients['train']&patients['val']:raise ValueError('Train/val patient overlap')
        if any(r['split'] not in ('train','val') for r in records):raise ValueError('Nontrain/val cache records')
        maps.append(mapping)
    if any(set(mapping)!=set(maps[0]) for mapping in maps[1:]):raise ValueError('Mismatched image IDs')
    for image_id in maps[0]:
        reference=maps[0][image_id]
        for mapping in maps[1:]:
            record=mapping[image_id]
            if record['split']!=reference['split'] or record['patient_key']!=reference['patient_key']:raise ValueError('Mismatched split/patient')
            for key in ('coords','valid','loss_mask','target'):
                if not torch.equal(torch.as_tensor(reference[key]),torch.as_tensor(record[key])):raise ValueError(f'Mismatched {key} at image{image_id}')
        coords=torch.as_tensor(reference['coords']);valid=torch.as_tensor(reference['valid'],dtype=torch.bool);mask=torch.as_tensor(reference['loss_mask'],dtype=torch.bool);target=torch.as_tensor(reference['target'],dtype=torch.long)
        if not mask.any() or not valid[mask].all() or not torch.isfinite(coords[valid]).all() or ((coords[valid]<0)|(coords[valid]>=1)).any():raise ValueError('Malformed coordinates/masks')
        expected=(coords[:,1]*48).long()*24+(coords[:,0]*24).long()
        if not torch.equal(target[mask],expected[mask]):raise ValueError('Malformed target floor')
        for mapping in maps:
            logp=torch.as_tensor(mapping[image_id]['raw_logp'])
            if logp.shape!=(len(coords),1152) or not torch.isfinite(logp).all():raise ValueError('Malformed base distribution')
            if not torch.allclose(logp.logsumexp(-1),torch.zeros(len(coords)),atol=2e-5):raise ValueError('Unnormalized base distribution')
    return maps


def calibrated_logp(first,second,weight=.5,temperature=1.,background=None,edge=None,bg_coefficient=0.,edge_coefficient=0.):
    """No current target or observed coordinate enters mixture/calibration."""
    if not 0<=weight<=1 or temperature<=0:raise ValueError('Invalid mixture settings')
    a=(first/temperature).log_softmax(-1);b=(second/temperature).log_softmax(-1)
    if weight==1:mixture=a
    elif weight==0:mixture=b
    else:mixture=torch.logaddexp(a+np.log(weight),b+np.log1p(-weight))
    if bg_coefficient and background is None:raise ValueError('Background geometry missing')
    if edge_coefficient and edge is None:raise ValueError('Edge geometry missing')
    if background is not None:mixture=mixture-bg_coefficient*background[:,None]
    if edge is not None:mixture=mixture-edge_coefficient*edge[:,None]
    return mixture.log_softmax(-1)


def per_image_nll(logp,target,mask):
    if ((target[mask]<0)|(target[mask]>=logp.shape[-1])).any():raise ValueError('Malformed target')
    safe=torch.where(mask,target,torch.zeros_like(target));nll=-logp.gather(-1,safe[:,:,None]).squeeze(-1)
    return (nll*mask).sum(1)/mask.sum(1)


def load_geometry(path,records):
    geometry=[];invalid=[]
    for record in records:
        image_id=int(record['image_id']);cached=torch.load(Path(path)/f'{image_id}.pt',map_location='cpu',weights_only=False)
        if not torch.equal(cached['coords'],record['coords']) or not torch.equal(cached['mask'],record['loss_mask']):raise ValueError('Geometry cache target/prefix mismatch')
        g=cached['geometry'];background=torch.as_tensor(g['background'],dtype=torch.float32);edge=torch.as_tensor(g['edge'],dtype=torch.float32)
        if background.shape!=(1152,) or edge.shape!=(1152,) or not torch.isfinite(background).all() or not torch.isfinite(edge).all():raise ValueError('Invalid geometry vectors')
        if not g['geometry_valid']:invalid.append(image_id)
        geometry.append((background,edge))
    return torch.stack([g[0] for g in geometry]),torch.stack([g[1] for g in geometry]),invalid


def analyze(a):
    torch.set_num_threads(a.threads);started=time.monotonic();paths=[Path(a.control_cache),Path(a.improved_cache)];caches=[torch.load(path,map_location='cpu',weights_only=False) for path in paths];maps=aligned_records(caches)
    counts={split:sum(r['split']==split for r in maps[0].values()) for split in ('train','val')}
    if counts!={'train':412,'val':93}:raise ValueError('Requires fixed train412/val93 exact-legacy16 caches')
    records=[maps[0][i] for i in sorted(maps[0]) if maps[0][i]['split']=='val'];ids=[int(r['image_id']) for r in records]
    first=torch.stack([r['raw_logp'] for r in records]);second=torch.stack([maps[1][i]['raw_logp'] for i in ids]);target=torch.stack([r['target'] for r in records]);mask=torch.stack([r['loss_mask'] for r in records]).bool()
    background,edge,invalid=load_geometry(a.geometry_cache,records);rows=[];image_rows=[]
    for weight in (0.,.25,.5,.75,1.):
        for temperature in (.75,1.,1.25):
            for bg in (0.,1.,2.,3.):
                for ed in ((0.,.25,.5) if a.edge_sweep else (0.,)):
                    name=f'control{weight:g}_T{temperature:g}_bg{bg:g}_edge{ed:g}';logp=calibrated_logp(first,second,weight,temperature,background,edge,bg,ed);scores=per_image_nll(logp,target,mask)
                    rows.append(dict(recipe=name,control_weight=weight,improved_weight=1-weight,temperature=temperature,background_coefficient=bg,edge_coefficient=ed,image_macro_nll=float(scores.mean()),images=93))
                    image_rows.extend(dict(recipe=name,image_id=image_id,patient_key=records[j]['patient_key'],nll=float(scores[j])) for j,image_id in enumerate(ids))
    frame=pd.DataFrame(rows).sort_values('image_macro_nll');best=frame.iloc[0].to_dict();output=Path(a.output);output.mkdir(parents=True,exist_ok=False);frame.to_csv(output/'recipes.csv',index=False);pd.DataFrame(image_rows).to_csv(output/'per_image_nll.csv',index=False)
    sourceproof=[dict(role=role,cache=str(path.resolve()),cache_sha256=digest(path),base_checkpoint=cache.get('base_checkpoint'),base_checkpoint_sha256=cache['base_checkpoint_sha256'],inherited_selection_patient_keys=cache.get('selection_patient_keys',[])) for role,path,cache in zip(('control','improved'),paths,caches)]
    provenance=dict(protocol='Exploratory calibration on previously selected fixed val93, not independent holdout; train412 records aligned but not used to fit this grid. Frozen teacher logp each conditioned on exactly same legacy16 observed prefix/crops. Image-only geometry vectors; no GT boxes, no current target input except scoring. Original proposal coefficient is0 throughout; original head is never applied to human keys.',sourceproof=sourceproof,geometry_cache=str(Path(a.geometry_cache).resolve()),geometry_provenance_sha256=digest(Path(a.geometry_cache)/'provenance.json'),aligned_images=505,validation_images=93,geometry_invalid_ids=invalid,settings=dict(control_weights=[0,.25,.5,.75,1],temperatures=[.75,1,1.25],background_coefficients=[0,1,2,3],edge_coefficients=[0,.25,.5] if a.edge_sweep else [0]),runtime_recipe='Load both exact SHA parent models; forward each with actual past coordinates/crops, temperature-normalize each raw distribution, probability-mix with fixed control weight, add background/edge bias then normalize. Both encoders required; no raw cache is a deployed model.')
    summary=dict(status='complete',best_exploratory=best,meets_nll_5_3=bool(best['image_macro_nll']<=5.3),raw_control_nll=float(per_image_nll(first,target,mask).mean()),raw_improved_nll=float(per_image_nll(second,target,mask).mean()),seconds=time.monotonic()-started,provenance=provenance)
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in summary.items() if k!='provenance'},indent=2));return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control-cache',default='runs/dual_target_1011/human_control_raw_cache.pt');p.add_argument('--improved-cache',default='runs/dual_target_1011/human_improved_raw_cache.pt');p.add_argument('--geometry-cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--edge-sweep',action='store_true');a=p.parse_args();analyze(a)
