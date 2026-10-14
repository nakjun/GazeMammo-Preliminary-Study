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

ARMS = ('random', 'grid', 'proposal', 'gaze', 'hybrid')
ROOT = Path(__file__).resolve().parents[1]

def implementation_hashes():
    return {name: digest(ROOT/name) for name in ('src/gaze_patch_classifier_1012.py', 'src/gaze_classifier_1011.py', 'src/gaze_attention_1011.py')}


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
    if checkpoint['kind'] != 'gaze_patch_classifier_1012': raise ValueError('Invalid checkpoint kind')
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
            checkpoint=dict(kind='gaze_patch_classifier_1012',arm=arm,seed=seed,k=k,epoch=epoch,state=state,model_config=model.config,
                global_state_sha256=state_digest(state) if arm=='global' else frozen,source_provenance=provenance,implementation_sha256=implementation_hashes(),
                training_patient_keys=sorted(frame.iloc[train.tolist()].patient_key.unique()),selection_patient_keys=sorted(frame.iloc[val.tolist()].patient_key.unique()),threshold=.5)
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
