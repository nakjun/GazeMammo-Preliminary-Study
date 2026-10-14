"""Matched frozen global controls and compact mean residual classification heads."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from .gaze_classifier_1011 import GazeDiagnosisHead
from .gaze_attention_1011 import GazeAttention1011, attention_loss, head_state
from .metrics import classification_metrics, cluster_bootstrap

ARMS = ('global', 'random', 'grid', 'gaze_baseline', 'gaze_improved')
SEEDS = (42,43,44)
ROOT = Path(__file__).resolve().parents[1]

def implementation_hashes():
    return {name: digest(ROOT/name) for name in ('src/official_cbis_classifier_1012.py', 'src/official_cbis_trajectory_1012.py', 'src/official_cbis_data_1012.py', 'src/gaze_classifier_1011.py', 'src/gaze_attention_1011.py', 'src/data.py', 'src/spatial.py', 'src/trajectory_1011.py')}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''): h.update(block)
    return h.hexdigest()

def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode()); h.update(str(value.dtype).encode()); h.update(str(tuple(value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def metrics(labels, probabilities):
    y=np.asarray(labels); probabilities=np.asarray(probabilities)
    if y.shape!=probabilities.shape or y.ndim!=1 or not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any(): raise ValueError('Invalid classification probabilities')
    if set(y.tolist()) != {0,1}: raise ValueError('Both binary classes required')
    result = classification_metrics(labels, probabilities, .5)
    result.update(accuracy=float(accuracy_score(labels, np.asarray(probabilities)>=.5)),
                  f1=float(f1_score(labels, np.asarray(probabilities)>=.5, zero_division=0)))
    result['confusion_matrix'] = [[result['tn'], result['fp']], [result['fn'], result['tp']]]
    return result

def forward(model, tensors, ids, arm, k=16):
    global_features = tensors['global_features'][ids]
    if arm == 'global':
        logits, _ = model(global_features)
        return {'logits': logits, 'global_logits': logits, 'residual': torch.zeros_like(logits)}
    return model(global_features, tensors[arm+'_features'][ids, :k],
                 tensors[arm+'_coordinates'][ids, :k], tensors[arm+'_mask'][ids, :k])

@torch.inference_mode()
def evaluate(model, tensors, ids, arm, k=16, batch=32):
    model.eval()
    chunks = [forward(model, tensors, part, arm, k) for part in ids.split(batch)]
    return {key: torch.cat([chunk[key].cpu() for chunk in chunks]) for key in chunks[0] if key != 'values'}

def restore(checkpoint, global_checkpoint=None):
    if checkpoint['kind'] != 'official_cbis_classifier_1012': raise ValueError('Invalid checkpoint kind')
    if checkpoint.get('implementation_sha256') != implementation_hashes(): raise ValueError('Classifier implementation hash mismatch')
    if checkpoint['arm'] == 'global':
        model = GazeDiagnosisHead(**checkpoint['model_config']); model.load_state_dict(checkpoint['state'])
        if state_digest(model.state_dict()) != checkpoint['global_state_sha256']: raise ValueError('Global state hash mismatch')
        return model.eval()
    if global_checkpoint is None or global_checkpoint['arm'] != 'global': raise ValueError('Matched global checkpoint required')
    for key in ('seed', 'global_state_sha256', 'source_provenance'):
        if checkpoint[key] != global_checkpoint[key]: raise ValueError('Global checkpoint provenance mismatch: '+key)
    baseline = restore(global_checkpoint)
    model = GazeAttention1011(baseline, **checkpoint['model_config'])
    missing, unexpected = model.load_state_dict(checkpoint['state'], strict=False)
    if unexpected or any(not name.startswith('global_model.') for name in missing): raise ValueError('Head state mismatch')
    return model.eval()

def train_head(tensors, frame, arm, seed, directory, provenance, epochs=60, k=16, global_checkpoint=None, patience=12):
    if arm not in ARMS or seed not in SEEDS: raise ValueError('Unregistered classifier arm/seed')
    torch.manual_seed(seed)
    if arm == 'global': model = GazeDiagnosisHead(input_dim=1024, dim=128, mode='global', dropout=.3)
    else: model = GazeAttention1011(restore(global_checkpoint), variant='mean_residual', input_dim=1024, dim=64, dropout=.2)
    train = torch.tensor(np.flatnonzero(frame.split.to_numpy()=='train'))
    val = torch.tensor(np.flatnonzero(frame.split.to_numpy()=='val'))
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.float32)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-4, weight_decay=.01)
    frozen = state_digest(model.global_model.state_dict()) if arm != 'global' else None
    best = -float('inf'); stale = 0; history = []
    directory.mkdir()
    # Patch epoch zero is the exact matched baseline candidate.
    for epoch in range(1 if arm=='global' else 0, epochs+1):
        losses=[]
        if epoch:
            model.train()
            order=train[torch.randperm(len(train), generator=torch.Generator().manual_seed(seed+epoch))]
            for ids in order.split(32):
                optimizer.zero_grad(); out=forward(model,tensors,ids,arm,k)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(out['logits'],labels[ids]) if arm=='global' else attention_loss(out,labels[ids])[0]
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); optimizer.step(); losses.append(float(loss.detach()))
        out=evaluate(model,tensors,val,arm,k); score=metrics(frame.iloc[val.tolist()].label,out['logits'].sigmoid().numpy())
        history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,validation=score))
        if score['auroc']>best:
            best=score['auroc'];stale=0
            state=copy.deepcopy(model.state_dict() if arm=='global' else head_state(model))
            checkpoint=dict(kind='official_cbis_classifier_1012',arm=arm,seed=seed,k=k,epoch=epoch,state=state,model_config=model.config,
                global_state_sha256=state_digest(state) if arm=='global' else frozen,source_provenance=provenance,implementation_sha256=implementation_hashes(),
                training_patient_keys=sorted(frame.iloc[train.tolist()].patient_key.unique()),selection_patient_keys=sorted(frame.iloc[val.tolist()].patient_key.unique()),threshold=.5,training_config=dict(max_epochs=epochs,patience=patience))
            torch.save(checkpoint,directory/'best.pt')
        elif epoch: stale+=1
        if frozen is not None and (state_digest(model.global_model.state_dict())!=frozen or any(p.grad is not None for p in model.global_model.parameters())): raise RuntimeError('Frozen baseline changed')
        print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,auroc=score['auroc'])),flush=True)
        if stale>=patience: break
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    model=restore(checkpoint,global_checkpoint);out=evaluate(model,tensors,val,arm,k)
    predictions=frame.iloc[val.tolist()][['image_id','patient_key','label']].reset_index(drop=True).copy()
    predictions['probability']=out['logits'].sigmoid().numpy(); predictions['global_logit']=out['global_logits'].numpy();predictions['residual_logit']=out['residual'].numpy()
    predictions.to_csv(directory/'validation_predictions.csv',index=False)
    details=[]
    for j,idx in enumerate(val.tolist()):
        record=dict(image_id=frame.iloc[idx].image_id,probability=float(predictions.iloc[j].probability),global_logit=float(out['global_logits'][j]),residual_logit=float(out['residual'][j]))
        if arm!='global': record.update(coordinates=tensors[arm+'_coordinates'][idx,:k].tolist(),rectangles=tensors[arm+'_rectangles'][idx,:k].tolist(),attention=out['attention'][j].tolist(),signed_contribution=out['signed_contribution'][j].tolist())
        details.append(record)
    (directory/'diagnostics.json').write_text(json.dumps(details,indent=2,default=lambda x:x.item(),allow_nan=False))
    (directory/'history.json').write_text(json.dumps(history,indent=2,allow_nan=False))
    return checkpoint, predictions, metrics(predictions.label,predictions.probability)

def paired_ci(candidate, reference, repetitions=5000):
    for frame in (candidate,reference):
        if frame.image_id.duplicated().any(): raise ValueError('Duplicate paired image identity')
    if set(candidate.image_id)!=set(reference.image_id): raise ValueError('Paired image population mismatch')
    aligned=reference.set_index('image_id').loc[candidate.image_id]
    for key in ('patient_key','label'):
        if candidate[key].tolist()!=aligned[key].tolist(): raise ValueError('Paired identity mismatch: '+key)
    result=cluster_bootstrap(candidate,reference,repetitions=repetitions,seed=42)
    result['delta']=metrics(candidate.label,candidate.probability)['auroc']-metrics(reference.label,reference.probability)['auroc']
    return result


def dependency(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=digest(path))


def check_dependencies(items):
    for item in items:
        if digest(item['path'])!=item['sha256']: raise ValueError('Dependency changed: '+item['path'])


def population(metadata):
    from .official_cbis_data_1012 import validate_protocol
    validate_protocol(metadata)
    import pandas as pd
    metadata=Path(metadata)
    train=pd.read_csv(metadata/'trainval.csv');test=pd.read_csv(metadata/'test.csv')
    required={'image_id','source_sn','patient_key','label','split','canonical_path'}
    for frame in (train,test):
        if not required<=set(frame) or frame.image_id.duplicated().any():raise ValueError('Invalid manifest identity')
        if not frame.label.isin([0,1]).all():raise ValueError('Unknown pathology label')
    if len(train)!=2393 or len(test)!=645:raise ValueError('Official population counts changed')
    if set(train.split)!={'train','val'} or set(test.split)!={'test'}:raise ValueError('Unexpected split')
    if set(train.patient_key)&set(test.patient_key) or (train.groupby('patient_key').split.nunique()>1).any():raise ValueError('Patient overlap')
    if set(train.image_id)&set(test.image_id):raise ValueError('Image overlap')
    return train,test


def trajectory_proof(path, train, allow_smoke=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    if ck.get('smoke',False) and not allow_smoke:raise ValueError('Smoke trajectory prohibited for full classifier')
    for key,allowed in [('training_patient_keys',set(train.loc[train.split=='train','patient_key'])),('selection_patient_keys',set(train.loc[train.split=='val','patient_key']))]:
        values=ck.get(key,ck.get('provenance',{}).get(key))
        if values is None or not set(values)<=allowed:raise ValueError('Trajectory patient proof missing/unsafe: '+key)
    if ck.get('kind')!='official_cbis_trajectory_1012':raise ValueError('Old trajectory checkpoint prohibited')
    if any(name.startswith('visual.') for name in ck.get('state',{})):raise ValueError('Compact trajectory must exclude visual weights')
    return ck


def verify_freeze(path):
    freeze=json.loads(Path(path).read_text())
    body={k:v for k,v in freeze.items() if k!='sha256'}
    if freeze.get('sha256')!=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest():raise ValueError('Freeze digest mismatch')
    if freeze['arms']!=list(ARMS) or freeze['head_seeds']!=list(SEEDS) or freeze['threshold']!=.5 or freeze['tta'] is not False or freeze['path_seed']!=42:raise ValueError('Unregistered frozen protocol')
    check_dependencies(freeze['dependencies'])
    if len(freeze['heads'])!=15 or len(freeze['trajectories'])!=2:raise ValueError('Incomplete model freeze')
    if {(h['arm'],h['seed']) for h in freeze['heads']}!={(a,s) for a in ARMS for s in SEEDS}:raise ValueError('Frozen head family identity mismatch')
    anchored={(d['path'],d['sha256']) for d in freeze['dependencies']}
    if any((d['path'],d['sha256']) not in anchored for d in freeze['heads']+freeze['trajectories']):raise ValueError('Unanchored frozen model')
    return freeze


@torch.inference_mode()
def extract_image(models, image, image_id):
    from .data import whole_tensor,patch_tensor,select_centers,seed_for
    from .official_cbis_trajectory_1012 import rollout
    if isinstance(image_id,bool) or not isinstance(image_id,(int,np.integer)):raise ValueError('Opaque numeric identity required')
    baseline,improved=models;device=next(baseline.parameters()).device
    encoder=baseline.visual
    image=image.convert('L')
    if image.size!=(1500,3000):raise ValueError('Expected canonical canvas')
    result={'global_features':encoder.patch_vector(whole_tensor(image,(256,512))[None].to(device))[0].float().cpu()}
    for arm in ARMS[1:]:
        if arm in ('random','grid'):
            centers=select_centers(image,arm,16,seed_for(42,int(image_id)))
            centers=np.resize(centers,(16,2));coords=centers/np.asarray(image.size)
        else:
            value=rollout(baseline if arm=='gaze_baseline' else improved,image,int(image_id),seed=42)
            coords=value[0] if isinstance(value,tuple) else value
        coords=torch.as_tensor(coords,dtype=torch.float32)
        if coords.shape!=(16,2) or not torch.isfinite(coords).all() or ((coords<0)|(coords>=1)).any():raise ValueError('Invalid image-only trajectory')
        centers=coords.double().numpy()*np.asarray(image.size)
        crops=torch.stack([patch_tensor(image,p,224) for p in centers]).to(device)
        # Smaller batches keep ConvNeXt-Base crop extraction within GPU0 memory.
        features=torch.cat([encoder.patch_vector(batch).float().cpu() for batch in crops.split(4)])
        rounded=np.round(centers).astype(int);rect=np.c_[rounded-112,rounded+112]
        result[arm+'_features']=features;result[arm+'_coordinates']=coords
        result[arm+'_rectangles']=torch.from_numpy(rect);result[arm+'_mask']=torch.ones(16,dtype=torch.bool)
    return result


def smoke_population(frame):
    import pandas as pd
    return pd.concat([frame[(frame.split==split)&(frame.label==label)].head(2) for split in ('train','val') for label in (0,1)],ignore_index=True)


def validate_cache(cache,metadata,expected_split):
    import pandas as pd
    if cache.get('kind')!='official_cbis_features_1012':raise ValueError('Wrong classifier cache kind')
    train,test=population(metadata);expected=train if expected_split=='trainval' else test
    p=cache['provenance'];rows=pd.DataFrame(cache['rows']);n=len(rows)
    if p.get('population')=='smoke8':
        if expected_split!='trainval':raise ValueError('TEST smoke population prohibited')
        expected=smoke_population(expected)
    elif p.get('population')!='full':raise ValueError('Unknown feature population')
    if p['split']!=expected_split or p['arms']!=list(ARMS) or p['path_seed']!=42 or p['crop_size']!=224 or p['feature_dim']!=1024 or p['gt_inputs']!=[]:raise ValueError('Feature protocol mismatch')
    check_dependencies(p['dependencies'])
    if p['implementation_sha256']!=implementation_hashes():raise ValueError('Classifier source changed')
    if expected_split=='test':
        freeze=verify_freeze(p['freeze_path'])
        if p['freeze_sha256']!=freeze['sha256']:raise ValueError('TEST cache freeze changed')
    if rows.image_id.tolist()!=expected.image_id.tolist():raise ValueError('Cache population/order mismatch')
    for key in ('image_id','source_sn','patient_key','split'):
        if rows[key].tolist()!=expected[key].tolist():raise ValueError('Cache row mismatch: '+key)
    # TEST labels are intentionally absent until metric calculation.
    if expected_split=='test' and 'label' in rows:raise ValueError('TEST labels entered feature cache')
    if expected_split=='trainval' and rows.label.tolist()!=expected.label.tolist():raise ValueError('Cache labels changed')
    for row in cache['rows']:
        if digest(row['canonical_path'])!=row['image_sha256']:raise ValueError('Canonical image changed')
    shapes={'global_features':(n,1024)}
    for arm in ARMS[1:]:
        shapes.update({arm+'_features':(n,16,1024),arm+'_coordinates':(n,16,2),arm+'_rectangles':(n,16,4),arm+'_mask':(n,16)})
    if set(cache['tensors'])!=set(shapes):raise ValueError('Unexpected cached tensor')
    for key,shape in shapes.items():
        value=cache['tensors'][key]
        if tuple(value.shape)!=shape or not torch.isfinite(value).all():raise ValueError('Invalid cached tensor: '+key)
        if key.endswith('_mask') and (value.dtype!=torch.bool or not value.all()):raise ValueError('Sixteen valid boolean tokens required')
        if key.endswith('_coordinates') and ((value<0)|(value>=1)).any():raise ValueError('Invalid coordinates')
    for arm in ARMS[1:]:
        centers=torch.round(cache['tensors'][arm+'_coordinates'].double()*torch.tensor([1500,3000])).long()
        if not torch.equal(cache['tensors'][arm+'_rectangles'].long(),torch.cat([centers-112,centers+112],-1)):raise ValueError('Crop rectangle provenance mismatch')
    return rows


def build_cache(metadata,baseline,improved,output,split='trainval',device='cuda:0',freeze_path=None,smoke=False):
    from PIL import Image
    from .official_cbis_trajectory_1012 import load_selected_trajectory
    train,test=population(metadata)
    if split=='test' and freeze_path is None:raise ValueError('TEST extraction requires final freeze')
    if split=='test' and smoke:raise ValueError('TEST smoke extraction prohibited')
    freeze=verify_freeze(freeze_path) if split=='test' else None
    for path in (baseline,improved):trajectory_proof(path,train,allow_smoke=smoke)
    if freeze and [digest(baseline),digest(improved)]!=[d['sha256'] for d in freeze['trajectories']]:raise ValueError('TEST source differs from frozen trajectory')
    models=(load_selected_trajectory(baseline,device),load_selected_trajectory(improved,device))
    if any(m.selected_protocol_sha256!=digest(Path(metadata)/'protocol.json') for m in models):raise ValueError('Trajectory cohort protocol differs')
    for path,arm in ((baseline,'baseline'),(improved,'improved')):
        if torch.load(path,map_location='cpu',weights_only=False)['arm']!=arm:raise ValueError('Trajectory arm order mismatch')
    if state_digest(models[0].visual.state_dict())!=state_digest(models[1].visual.state_dict()):raise ValueError('Visual encoders differ')
    for model in models:model.requires_grad_(False).eval()
    output=Path(output);parts=output.with_suffix('.rows');parts.mkdir(parents=True,exist_ok=True)
    deps=[dependency(models[0].original_pretrained_path)] if hasattr(models[0],'original_pretrained_path') else []
    deps += [dependency(Path(metadata)/name) for name in ('protocol.json','trainval.csv','test.csv')]+[dependency(torch.load(baseline,map_location='cpu',weights_only=False)['original_pretrained_path']),dependency(baseline),dependency(improved)]
    p=dict(population='smoke8' if smoke else 'full',split=split,arms=list(ARMS),path_seed=42,crop_size=224,feature_dim=1024,gt_inputs=[],dependencies=deps,implementation_sha256=implementation_hashes(),visual_state_sha256=state_digest(models[0].visual.state_dict()),training_patient_keys=sorted(train.loc[train.split=='train','patient_key'].unique()),selection_patient_keys=sorted(train.loc[train.split=='val','patient_key'].unique()))
    if freeze:p.update(freeze_path=str(Path(freeze_path).resolve()),freeze_sha256=freeze['sha256'])
    source=smoke_population(train) if smoke else train if split=='trainval' else test;rows=[];features=[]
    signature=hashlib.sha256(json.dumps(p,sort_keys=True).encode()).hexdigest()
    for record in source.to_dict('records'):
        row={key:record[key] for key in ('image_id','source_sn','patient_key','split','canonical_path')}
        if split=='trainval':row['label']=int(record['label'])
        row['canonical_path']=str(Path(row['canonical_path']).resolve());row['image_sha256']=digest(row['canonical_path'])
        item_path=parts/(str(row['image_id'])+'.pt')
        if item_path.exists():
            item=torch.load(item_path,map_location='cpu',weights_only=False)
            if item['row']!=row or item['signature']!=signature:raise ValueError('Partial cache provenance changed')
        else:
            with Image.open(row['canonical_path']) as image:feature=extract_image(models,image,int(row['image_id']))
            item=dict(row=row,tensors=feature,signature=signature)
            temporary=item_path.with_suffix('.tmp');torch.save(item,temporary);temporary.replace(item_path)
        rows.append(row);features.append(item['tensors'])
        print(json.dumps(dict(stage='classification_cache',split=split,completed=len(rows),total=len(source),image_id=int(row['image_id']))),flush=True)
    cache=dict(kind='official_cbis_features_1012',rows=rows,provenance=p,tensors={key:torch.stack([f[key] for f in features]) for key in features[0]})
    validate_cache(cache,metadata,split)
    temporary=output.with_suffix('.tmp');torch.save(cache,temporary);temporary.replace(output)
    return output


def train_all(cache_path,metadata,output,epochs=60,patience=12):
    cache=torch.load(cache_path,map_location='cpu',weights_only=False);frame=validate_cache(cache,metadata,'trainval')
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    provenance=dict(cache_sha256=digest(cache_path),cache_path=str(Path(cache_path).resolve()),**cache['provenance'])
    for seed in SEEDS:
        global_ck=None
        for arm in ARMS:
            target=directory/f'{arm}_seed{seed}'
            if (target/'best.pt').exists() and (target/'history.json').exists():
                ck=torch.load(target/'best.pt',map_location='cpu',weights_only=False)
                if ck['source_provenance']!=provenance or ck['seed']!=seed or ck['arm']!=arm:raise ValueError('Existing classifier provenance differs')
                restore(ck,global_ck)
            else:
                if target.exists():
                    # An interrupted head may resume by deterministic fresh training in the same destination.
                    import shutil
                    shutil.rmtree(target)
                ck,_,_=train_head(cache['tensors'],frame,arm,seed,target,provenance,epochs=epochs,global_checkpoint=global_ck,patience=patience)
            if arm=='global':global_ck=ck


def freeze_all(metadata,heads,baseline,improved,output):
    train,_=population(metadata);dependencies=[dependency(Path(metadata)/name) for name in ('protocol.json','trainval.csv','test.csv')]
    trajectories=[dependency(baseline),dependency(improved)]
    for item,arm in zip(trajectories,('baseline','improved')):
        ck=trajectory_proof(item['path'],train)
        if ck['arm']!=arm or ck['protocol_sha256']!=digest(Path(metadata)/'protocol.json'):raise ValueError('Trajectory freeze protocol/arm mismatch')
    models=[];shared=None
    for seed in SEEDS:
        global_ck=None
        for arm in ARMS:
            item=dependency(Path(heads)/f'{arm}_seed{seed}'/'best.pt');ck=torch.load(item['path'],map_location='cpu',weights_only=False)
            if ck.get('training_config')!={'max_epochs':60,'patience':12}:raise ValueError('Only full registered training can freeze TEST')
            if ck['arm']!=arm or ck['seed']!=seed or ck['threshold']!=.5:raise ValueError('Classifier freeze identity mismatch')
            if set(ck['training_patient_keys'])!=set(train.loc[train.split=='train','patient_key']) or set(ck['selection_patient_keys'])!=set(train.loc[train.split=='val','patient_key']):raise ValueError('Head population differs')
            restore(ck,global_ck)
            if shared is None:shared=ck['source_provenance']
            elif shared!=ck['source_provenance']:raise ValueError('Heads used different feature caches')
            if arm=='global':global_ck=ck
            item.update(arm=arm,seed=seed);models.append(item)
    cache=torch.load(shared['cache_path'],map_location='cpu',weights_only=False);validate_cache(cache,metadata,'trainval')
    if cache['provenance']['population']!='full':raise ValueError('Smoke cache cannot authorize TEST')
    if digest(shared['cache_path'])!=shared['cache_sha256']:raise ValueError('Training cache changed')
    if [digest(baseline),digest(improved)]!=[d['sha256'] for d in shared['dependencies'][-2:]]:raise ValueError('Head trajectory dependencies differ')
    dependencies+=shared['dependencies']+trajectories+models+[dependency(shared['cache_path'])]+[dependency(ROOT/name) for name in implementation_hashes()]
    body=dict(frozen=True,protocol_sha256=digest(Path(metadata)/'protocol.json'),kind='official_cbis_final_freeze_1012',arms=list(ARMS),head_seeds=list(SEEDS),path_seed=42,threshold=.5,tta=False,heads=models,trajectories=trajectories,dependencies=dependencies)
    body['sha256']=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
    output=Path(output)
    if output.exists() and json.loads(output.read_text())!=body:raise ValueError('Final freeze is immutable')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(body,indent=2));return body


def evaluate_test(cache_path,metadata,freeze_path,output,repetitions=5000):
    import pandas as pd
    freeze=verify_freeze(freeze_path);cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    frame=validate_cache(cache,metadata,'test');_,labels=population(metadata)
    if cache['provenance']['freeze_sha256']!=freeze['sha256']:raise ValueError('Evaluation freeze mismatch')
    output=Path(output)
    if output.exists():raise ValueError('Official TEST evaluation output already exists')
    output.mkdir(parents=True)
    frame['label']=labels.label.to_numpy();ids=torch.arange(len(frame));predictions={};scores={}
    for seed in SEEDS:
        global_ck=None
        for arm in ARMS:
            item=next(d for d in freeze['heads'] if d['arm']==arm and d['seed']==seed)
            ck=torch.load(item['path'],map_location='cpu',weights_only=False);model=restore(ck,global_ck)
            if ck['source_provenance']['visual_state_sha256']!=cache['provenance']['visual_state_sha256']:raise ValueError('Evaluation visual differs')
            out=evaluate(model,cache['tensors'],ids,arm)
            pred=frame[['image_id','patient_key','label']].copy();pred['probability']=out['logits'].sigmoid().numpy()
            pred.to_csv(output/f'{arm}_seed{seed}.csv',index=False);predictions[arm,seed]=pred;scores[arm,seed]=metrics(pred.label,pred.probability)
            if arm=='global':global_ck=ck
    report={'freeze_sha256':freeze['sha256'],'cache_sha256':digest(cache_path),'threshold':.5,'tta':False,'arms':{},'paired_patient_bootstrap':{'repetitions':repetitions,'seed':42,'comparisons':{}},'limitations':'TRAIN patient purge64; local missing TRAIN1; own canonical preprocessing; 3 classifier seeds; no TTA; some historical TEST gaze/ROI observations exist; no TEST model family selection.'}
    ensembles={}
    for arm in ARMS:
        pred=predictions[arm,42].copy();pred['probability']=np.mean([predictions[arm,seed].probability.to_numpy() for seed in SEEDS],axis=0)
        pred.to_csv(output/f'{arm}_ensemble.csv',index=False);ensembles[arm]=pred
        singles={str(seed):scores[arm,seed] for seed in SEEDS}
        summary={key:dict(mean=float(np.mean([scores[arm,s][key] for s in SEEDS])),sd=float(np.std([scores[arm,s][key] for s in SEEDS],ddof=1))) for key in ('auroc','auprc','accuracy','balanced_accuracy','sensitivity','specificity','f1','brier')}
        report['arms'][arm]=dict(single_heads=singles,single_head_mean_sd=summary,probability_ensemble=metrics(pred.label,pred.probability))
    # Mean-head AUROC differences and ensemble differences have distinct bootstrap estimands.
    from sklearn.metrics import roc_auc_score
    rng=np.random.default_rng(42);groups=frame.patient_key.unique();bygroup={g:np.flatnonzero(frame.patient_key.to_numpy()==g) for g in groups}
    for reference in ('gaze_baseline','global','grid'):
        values=[]
        for _ in range(repetitions):
            ix=np.concatenate([bygroup[g] for g in rng.choice(groups,len(groups),replace=True)]);y=frame.label.to_numpy()[ix]
            if len(np.unique(y))<2:continue
            values.append(np.mean([roc_auc_score(y,predictions['gaze_improved',s].probability.to_numpy()[ix])-roc_auc_score(y,predictions[reference,s].probability.to_numpy()[ix]) for s in SEEDS]))
        report['paired_patient_bootstrap']['comparisons'][reference]=dict(mean_head_auroc=dict(delta=float(np.mean([scores['gaze_improved',s]['auroc']-scores[reference,s]['auroc'] for s in SEEDS])),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),valid_replicates=len(values)),probability_ensemble_auroc=paired_ci(ensembles['gaze_improved'],ensembles[reference],repetitions))
    (output/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));return report
