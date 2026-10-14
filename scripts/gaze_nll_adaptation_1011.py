"""Cache frozen visuals, calibrate, or adapt exact legacy16 human-gaze likelihood."""
import argparse,csv,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.data import load_gazes,whole_tensor,patch_tensor
from src.spatial import gaussian_targets
from src.trajectory_1011 import restore,segments,window_sample,check_provenance
from src.guided_trajectory_1011 import candidate_geometry,nonvisual_state
from src.gaze_nll_adaptation_1011 import initialize,hard_nll,prior_targets,restore_adaptation
from src.proposal_utility_1011 import state_sha256
from guided_trajectory_1011 import load_guided,base_path
from train_attention_1011 import digest


def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))
def load_base(path):
    c=torch.load(base_path(path),map_location='cpu',weights_only=False)
    if c['kind']=='gaze_nll_adaptation_1011':return load_adapted(path)
    return load_guided(path)[0] if c['kind']=='guided_trajectory_1011' else restore(c),c


def load_adapted(path,base_override=None):
    checkpoint=torch.load(base_path(path),map_location='cpu',weights_only=False)
    dependency=base_override or checkpoint['base_checkpoint']
    if digest(base_path(dependency))!=checkpoint['base_checkpoint_sha256']:raise ValueError('Adaptation dependency SHA mismatch')
    base,_=load_base(dependency)
    return restore_adaptation(checkpoint,base).eval(),checkpoint


def routing():
    with (ROOT/'manifests/trajectory.csv').open() as source:val_ids={r['image_id'] for r in csv.DictReader(source) if r['split']=='val'}
    with (ROOT/'manifests/all.csv').open() as source:return [{k:r[k] for k in ('image_id','file_name','patient_key','split')} for r in csv.DictReader(source) if r['split']=='train' or (r['split']=='val' and r['image_id'] in val_ids)]


def validate_records(records,smoke=False):
    ids=[r['image_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate cache image IDs')
    counts={split:sum(r['split']==split for r in records) for split in ('train','val')}
    if not smoke and counts!={'train':412,'val':93}:raise ValueError(f'Wrong full cache split counts: {counts}')
    return counts


def cache(a):
    output=Path(a.cache);output.mkdir(parents=True,exist_ok=False);model,checkpoint=load_base(a.base);model=model.to(a.device).eval();gazes=load_gazes();records=routing()
    if a.limit:records=[r for r in records if r['split']=='train'][:a.limit]+[r for r in records if r['split']=='val'][:a.limit]
    validate_records(records,smoke=bool(a.limit))
    allowed=routing();check_provenance(checkpoint,[r['patient_key'] for r in allowed if r['split']=='train'],[r['patient_key'] for r in allowed if r['split']=='val'],[r['patient_key'] for r in records if r['split']=='val'])
    provenance=dict(kind='gaze_nll_visual_cache_1011',base_checkpoint=str(base_path(a.base).resolve()),base_checkpoint_sha256=digest(base_path(a.base)),gaze_source=str(ROOT/'data/gaze_data/merged_gaze_data.csv'),gaze_sha256=digest(ROOT/'data/gaze_data/merged_gaze_data.csv'),manifest_sha256=digest(ROOT/'manifests/trajectory.csv'),all_manifest_sha256=digest(ROOT/'manifests/all.csv'),records=records,protocol='Frozen encoder whole pyramid and preceding observed224px crops, exactlegacy16 hardcell targets, no boxes',annotation_used=False,feature_dtype='float32',smoke=bool(a.limit),visual_state_sha256=state_sha256(model.visual.state_dict()))
    save(output/'provenance.json',provenance)
    for row in records:
        with Image.open(ROOT/'data/image'/row['file_name']) as source:image=source.convert('L')
        parts=[p/image.size for p in segments(gazes[row['file_name']],*image.size,32.)];sample=window_sample(parts,int(row['image_id']),42,0,'legacy16',16,8,.5);n=int(sample['valid'].sum());coords=sample['coords'][:16];mask=sample['loss_mask'][:16]
        with torch.no_grad():
            features=[f[0].cpu() for f in model.visual(whole_tensor(image,model.input_size)[None].to(a.device))]
            vectors=torch.zeros(16,model.patch.in_features)
            if model.foveal and n>1:
                crops=torch.stack([patch_tensor(image,p.numpy()*image.size) for p in coords[:n-1]])
                for start in range(0,len(crops),a.patch_batch):vectors[start+1:start+1+a.patch_batch]=model.visual.patch_vector(crops[start:start+a.patch_batch].to(a.device)).cpu()
            geometry=candidate_geometry(image,model.grid)
        torch.save(dict(features=features,vectors=vectors,coords=coords,mask=mask,geometry=geometry,image_size=image.size,image_sha256=digest(ROOT/'data/image'/row['file_name']),**row),output/f"{row['image_id']}.pt")
        print(json.dumps(dict(event='cached',image_id=row['image_id'],split=row['split'])),flush=True)


def cache_records(a):
    provenance=json.loads((Path(a.cache)/'provenance.json').read_text())
    if provenance.get('kind') not in ('gaze_nll_visual_cache_1011','gaze_nll_resolution_cache_1011'):raise ValueError('Unknown visual cache kind')
    if provenance.get('all_manifest_sha256')!=digest(ROOT/'manifests/all.csv'):raise ValueError('Cache train routing manifest mismatch')
    if provenance['manifest_sha256']!=digest(ROOT/'manifests/trajectory.csv'):raise ValueError('Cache routing manifest mismatch')
    base,checkpoint=load_base(a.base)
    if provenance['base_checkpoint_sha256']!=digest(base_path(a.base)):
        if provenance.get('visual_state_sha256')!=state_sha256(base.visual.state_dict()):raise ValueError('Cache visual encoder mismatch')
    allowed_rows=routing();check_provenance(checkpoint,[r['patient_key'] for r in allowed_rows if r['split']=='train'],[r['patient_key'] for r in allowed_rows if r['split']=='val'])
    if provenance['gaze_sha256']!=digest(ROOT/'data/gaze_data/merged_gaze_data.csv'):raise ValueError('Cache gaze source mismatch')
    records=[torch.load(Path(a.cache)/f"{r['image_id']}.pt",map_location='cpu',weights_only=False) for r in provenance['records']]
    validate_records(records,smoke=provenance.get('smoke',False))
    if provenance['kind']=='gaze_nll_resolution_cache_1011':
        status_path=Path(a.cache)/'status.json'
        if not status_path.exists() or json.loads(status_path.read_text()).get('state')!='complete':raise ValueError('Resolution cache extraction not complete')
        size=provenance.get('input_size');shapes=provenance.get('native_feature_shapes')
        if not size or len(size)!=2 or any(v<32 or v%32 for v in size):raise ValueError('Invalid whole input resolution')
        if provenance.get('grid')!=[24,48] or provenance.get('preceding_crop_size',provenance.get('crop_size'))!=224:raise ValueError('Resolution cache changed action protocol')
        if not shapes or len(shapes)!=3 or any(len(s)!=3 for s in shapes):raise ValueError('Missing native pyramid shapes')
        if any([list(f.shape) for f in r['features']]!=shapes for r in records):raise ValueError('Native pyramid shape mismatch')
        for shape,stride,channels in zip(shapes,(8,16,32),base.visual.channels):
            if shape!=[channels,size[1]//stride,size[0]//stride]:raise ValueError('Pyramid resolution metadata mismatch')
    allowed={(r['image_id'],r['patient_key'],r['split']) for r in routing()}
    if any((r['image_id'],r['patient_key'],r['split']) not in allowed for r in records):raise ValueError('Cache outside train/val routing')
    return records,provenance


def apply_cache_resolution(model,provenance):
    model.feature_memory_format=provenance.get('feature_memory_format')
    if model.feature_memory_format not in (None,'contiguous_nchw'):raise ValueError('Unsupported projection feature layout')
    if provenance.get('input_size'):
        model.input_size=tuple(provenance['input_size']);model.model_config['input_size']=list(model.input_size)
    return model


def batch(records,device):
    features=[torch.stack([r['features'][i] for r in records]).to(device) for i in range(3)]
    return features,torch.stack([r['coords'] for r in records]).to(device),torch.stack([r['mask'] for r in records]).to(device),torch.stack([r['vectors'] for r in records]).to(device)


@torch.no_grad()
def validate(model,records,a):
    model.eval();scores=[]
    for start in range(0,len(records),a.batch_size):
        group=records[start:start+a.batch_size];f,coords,mask,vectors=batch(group,a.device);memory=model.maps_features(f);logp=model.teacher_logits(memory,coords,vectors)
        for i,r in enumerate(group):scores.append(dict(image_id=int(r['image_id']),patient_key=r['patient_key'],nll=float(hard_nll(logp[i:i+1],coords[i:i+1],mask[i:i+1],model.grid))))
    return float(np.mean([r['nll'] for r in scores])),scores


def train(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);records,provenance=cache_records(a);tr=[r for r in records if r['split']=='train'];va=[r for r in records if r['split']=='val'];base,base_checkpoint=load_base(a.base);model=apply_cache_resolution(initialize(base),provenance).to(a.device)
    if a.train_scope=='history':
        for name,parameter in model.named_parameters():parameter.requires_grad_(name.startswith(('history.','history_head.','patch.','rnn.','initial.','motion.','behavior_gate.')) or name in ('bos','dynamic_scale','motion_scale','local_scale','return_scale','return_decay'))
    external_reference=None
    if a.reference_logp_cache:
        payload=torch.load(a.reference_logp_cache,map_location='cpu',weights_only=False)
        if payload.get('kind')!='gaze_raw_legacy16_cache_1011':raise ValueError('Wrong external reference cache kind')
        if digest(base_path(payload['base_checkpoint']))!=payload['base_checkpoint_sha256']:raise ValueError('External reference source SHA mismatch')
        save(output/'reference_provenance.json',dict(cache_sha256=digest(a.reference_logp_cache),base_checkpoint=payload['base_checkpoint'],base_checkpoint_sha256=payload['base_checkpoint_sha256'],protocol='Independent frozen observed-prefix KL teacher; cohort/coords/masks identical'))
        external_reference={int(r['image_id']):r for r in payload['records']}
        if len(external_reference)!=len(payload['records']) or set(external_reference)!={int(r['image_id']) for r in records}:raise ValueError('External reference cohort mismatch')
        for r in records:
            ref=external_reference[int(r['image_id'])]
            if ref['patient_key']!=r['patient_key'] or ref['split']!=r['split'] or not torch.equal(ref['coords'],r['coords']) or not torch.equal(ref['loss_mask'],r['mask']):raise ValueError('External reference observed target mismatch')
            if not torch.isfinite(ref['raw_logp']).all():raise ValueError('Nonfinite external reference')
    reference=apply_cache_resolution(initialize(base),provenance).to(a.device).eval() if a.kl_weight and external_reference is None else None
    if reference is not None:
        for parameter in reference.parameters():parameter.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=a.weight_decay);best=float('inf');history=[];started=time.monotonic();save(output/'config.json',vars(a));save(output/'cache_provenance.json',provenance);save(output/'initial_state.json',dict(seed=a.seed,nonvisual_state_sha256=state_sha256(nonvisual_state(model)),base_checkpoint_sha256=digest(base_path(a.base))))
    initial,perimage=validate(model,va,a);save(output/'initial_validation.json',dict(nll=initial,images=perimage));best=initial
    initial_checkpoint=dict(kind='gaze_nll_adaptation_1011',state=nonvisual_state(model),model_config=model.model_config,base_checkpoint=str(base_path(a.base).resolve()),base_checkpoint_sha256=digest(base_path(a.base)),epoch=0,feature_memory_format=getattr(model,'feature_memory_format',None),args=vars(a),training_patient_keys=sorted(set(base_checkpoint['training_patient_keys'])|{r['patient_key'] for r in tr}),selection_patient_keys=sorted(set(base_checkpoint['selection_patient_keys'])|{r['patient_key'] for r in va}),selection='Minimum exactlegacy16 human hardNLL including epoch0 on exploratoryval93',protocol='Frozenencoder adaptation; epoch0 eligible')
    torch.save(initial_checkpoint,output/'initial.pt');torch.save(initial_checkpoint,output/'best.pt');save(output/'best_validation.json',dict(nll=initial,epoch=0,images=perimage))
    for epoch in range(1,a.epochs+1):
        model.train();order=np.random.default_rng(a.seed+epoch).permutation(len(tr));losses=[]
        for start in range(0,len(tr),a.batch_size):
            group=[tr[i] for i in order[start:start+a.batch_size]];f,coords,mask,vectors=batch(group,a.device);optimizer.zero_grad();memory=model.maps_features(f);logp=model.teacher_logits(memory,coords,vectors);nll=hard_nll(logp,coords,mask,model.grid)
            prior=-(prior_targets(coords,mask,model.grid)*memory['prior'].log_softmax(-1)).sum(-1).mean();loss=nll+a.prior_weight*prior
            if a.soft_weight:
                targets=torch.stack([gaussian_targets(coords[i],model.grid,a.soft_sigma,r['image_size']) for i,r in enumerate(group)])
                soft=(-(targets*logp.log_softmax(-1)).sum(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
                loss=(1-a.soft_weight)*nll+a.soft_weight*soft.mean()+a.prior_weight*prior
            if a.kl_weight:
                if external_reference is not None:ref=torch.stack([external_reference[int(r['image_id'])]['raw_logp'] for r in group]).to(a.device).log_softmax(-1)
                else:
                    with torch.no_grad():ref=reference.teacher_logits(reference.maps_features(f),coords,vectors).log_softmax(-1)
                kl=(ref.exp()*(ref-logp.log_softmax(-1))).sum(-1)
                loss=loss+a.kl_weight*((kl*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(nll.detach()))
        score,perimage=validate(model,va,a);record=dict(epoch=epoch,train_nll=float(np.mean(losses)),validation_nll=score);history.append(record);save(output/'history.json',history);print(json.dumps(record),flush=True)
        c=dict(kind='gaze_nll_adaptation_1011',state=nonvisual_state(model),model_config=model.model_config,base_checkpoint=str(base_path(a.base).resolve()),base_checkpoint_sha256=digest(base_path(a.base)),epoch=epoch,feature_memory_format=getattr(model,'feature_memory_format',None),args=vars(a),training_patient_keys=sorted(set(base_checkpoint['training_patient_keys'])|{r['patient_key'] for r in tr}),selection_patient_keys=sorted(set(base_checkpoint['selection_patient_keys'])|{r['patient_key'] for r in va}),selection='Minimum exactlegacy16 human hardNLL on exploratoryval93',protocol='Frozen encoder; hard human-gaze NLL plus explicit hard-gaze prior; no ROI supervision')
        torch.save(c,output/'last.pt')
        if score<best:best=score;torch.save(c,output/'best.pt');save(output/'best_validation.json',dict(nll=score,epoch=epoch,images=perimage))
    save(output/'summary.json',dict(initial_validation_nll=initial,best_validation_nll=best,epochs=a.epochs,seconds=time.monotonic()-started,train_images=len(tr),validation_images=len(va),goal_nll=5.3,goal_met=best<=5.3,improved_over_initial=best<initial))


@torch.no_grad()
def calibrate(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);records,provenance=cache_records(a);base,_=load_base(a.base);base=base.to(a.device).eval();model=apply_cache_resolution(initialize(base),provenance).to(a.device).eval();scores={}
    for r in records:
        if r['split']!='val':continue
        f,coords,mask,vectors=batch([r],a.device);memory=model.maps_features(f);raw=model.teacher_logits(memory,coords,vectors);relevance=base.proposal(memory['keys']).sigmoid().squeeze(-1) if hasattr(base,'proposal') else torch.zeros_like(raw[:,0]);g=r['geometry'];geo=-3*g['background'].to(raw)-.5*g['edge'].to(raw)*(1-relevance)
        for temperature in (.5,.75,1.,1.25,1.5,2.):
            for coefficient in ((0.,1.,2.) if hasattr(base,'proposal') else (0.,)):
                # coefficient0 is raw temperature-only calibration; proposal variants retain geometry.
                logits=raw if coefficient==0 else raw+coefficient*relevance[:,None]+geo[:,None]
                key=f'T{temperature}_proposal{coefficient}';scores.setdefault(key,[]).append(float(hard_nll(logits/temperature,coords,mask,model.grid)))
    save(output/'calibration.json',dict(image_macro_nll={k:float(np.mean(v)) for k,v in scores.items()},images=len(next(iter(scores.values()))),base_checkpoint_sha256=digest(base_path(a.base)),exploratory_validation_tuned=True,no_training=True,protocol='Sameobservedlegacy16; coefficient0 rawwithoutgeometry, coefficient1/2 retain geometry'))


def parser():
    p=argparse.ArgumentParser();p.add_argument('command',choices=('cache','train','calibrate'));p.add_argument('--base',required=True);p.add_argument('--cache',required=True);p.add_argument('--output');p.add_argument('--device',default='cpu');p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=4);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--patch-batch',type=int,default=8);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-4);p.add_argument('--prior-weight',type=float,default=0.);p.add_argument('--limit',type=int,default=0);p.add_argument('--train-scope',choices=('all','history'),default='all');p.add_argument('--soft-weight',type=float,default=0.);p.add_argument('--soft-sigma',type=float,default=32.);p.add_argument('--kl-weight',type=float,default=0.);p.add_argument('--reference-logp-cache',help='Optional independently frozen observed-prefix teacher cache for KL');return p
if __name__=='__main__':
    a=parser().parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads)
    if a.command!='cache' and not a.output:raise ValueError('Output required')
    if not 0<=a.soft_weight<=1 or a.soft_sigma<=0 or a.kl_weight<0:raise ValueError('Invalid regularization parameters')
    globals()[a.command](a)
