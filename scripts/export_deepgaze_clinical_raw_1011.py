"""Terminal-only canonical505 clinical readout probability export, optional actual val93 parity."""
import argparse,json,sys,time,copy
from pathlib import Path
import numpy as np,pandas as pd,torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from train_deepgaze_clinical_1011 import load_clinical,load_native_cache,verified_centerbias,group_data
from gaze_nll_adaptation_1011 import cache_records
from evaluate_deepgaze3_1011 import cell_mass
from src.deepgaze_clinical_1011 import prepare_saliency,forward_saliency,history_batch
from src.deepgaze_context_1011 import native_features,dense_centerbias
from src.proposal_utility_1011 import state_sha256
from train_attention_1011 import digest

def terminal_trial(trial):
    trial=Path(trial);config=json.loads((trial/'config.json').read_text());summary_path=trial/'summary.json'
    if not summary_path.exists():raise ValueError('Clinical trial not terminal; no export permitted')
    summary=json.loads(summary_path.read_text());history=json.loads((trial/'history.json').read_text())
    if summary.get('epochs')!=config['epochs'] or len(history)!=config['epochs'] or history[-1]['epoch']!=config['epochs']:raise ValueError('Clinical trial incomplete')
    integrity=json.loads((trial/'frozen_feature_integrity.json').read_text())
    if integrity.get('unchanged') is not True:raise ValueError('Clinical frozen encoder proof failed')
    return config,summary

def canonical_targets(coords):return (coords[...,1]*48).long().clamp(0,47)*24+(coords[...,0]*24).long().clamp(0,23)

def alignment(reference,record):
    if record['patient_key']!=reference['patient_key'] or record['split']!=reference['split'] or not torch.equal(record['coords'],reference['coords']) or not torch.equal(record['loss_mask'],reference['loss_mask']) or not torch.equal(record['target'],reference['target']):raise ValueError('Canonical parent target/mask alignment mismatch')

@torch.no_grad()
def _predict_head(model,features,coords,bos,cb,size):
    saliency=prepare_saliency(model,features,size);raw=[bos]
    for t in range(1,16):
        hx,hy=history_batch(coords,t,size);raw.append(cell_mass(forward_saliency(model,saliency,size,cb,hx,hy)))
    raw=torch.stack(raw,1);raw=raw-raw.logsumexp(-1,keepdim=True)
    if not torch.isfinite(raw).all() or not torch.allclose(raw.exp().sum(-1),torch.ones_like(raw[:,:,0]),atol=1e-5):raise ValueError('Invalid normalized clinical probabilities')
    return raw

@torch.no_grad()
def predict(model,features,coords,bos,cb,size):
    with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):return _predict_head(model,features,coords,bos,cb,size)

@torch.no_grad()
def export(a):
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;output=Path(a.output)
    if output.exists():raise FileExistsError(output)
    config,summary=terminal_trial(a.trial);config.update(device=a.device,smoke_limit=0);args=argparse.Namespace(**config);checkpoint=Path(a.trial)/'best.pt';checkpoint_sha=digest(checkpoint);model,clinical=load_clinical(checkpoint,a.device);records,source=cache_records(args);native,header=load_native_cache(args,records,state_sha256(model.features.state_dict()));bias=verified_centerbias(args,records);size=clinical['input_size'];references=[]
    for path in ['human_control_raw_cache_for_ensemble.pt','human_improved_raw_cache.pt']:
        raw=torch.load(ROOT/'runs/dual_target_1011'/path,map_location='cpu',weights_only=False);references.append({int(r['image_id']):r for r in raw['records']})
    exported=[];images=[];parity=[];started=time.monotonic()
    for start in range(0,len(records),a.batch_size):
        group=records[start:start+a.batch_size];features,coords,mask,bos,cb=group_data(group,native,bias,size,a.device);raw=predict(model,features,coords,bos,cb,size);target=canonical_targets(coords)
        for i,r in enumerate(group):
            record=dict(image_id=int(r['image_id']),patient_key=r['patient_key'],split=r['split'],raw_logp=raw[i].cpu(),coords=coords[i].cpu(),valid=mask[i].cpu(),loss_mask=mask[i].cpu(),target=target[i].cpu())
            for reference in references:alignment(reference[record['image_id']],record)
            nll=float((-raw[i].gather(1,target[i,:,None]).squeeze(1))[mask[i]].mean());images.append(dict(image_id=record['image_id'],patient_key=r['patient_key'],split=r['split'],nll=nll));exported.append(record)
            if a.live_val_parity and r['split']=='val':
                path=ROOT/'data/image'/r['file_name']
                if digest(path)!=r['image_sha256']:raise ValueError('Actual val image SHA changed')
                with Image.open(path) as image:rgb=torch.from_numpy(np.asarray(image.convert('L').resize(tuple(size),Image.Resampling.BILINEAR)).copy()).float()[None,None].repeat(1,3,1,1).to(a.device)
                actual_features=native_features(model,rgb);live=predict(model,actual_features,coords[i:i+1],bos[i:i+1],cb[i:i+1],size);error=float((live[0]-raw[i]).abs().max());parity.append(dict(image_id=record['image_id'],max_logp_error=error))
                if error>1e-4:raise ValueError(f'Actual/cache updated-head parity exceeds1e-4: {error}')
        print(json.dumps(dict(completed=len(exported),total=505)),flush=True)
    if len(exported)!=505 or sum(r['split']=='train' for r in exported)!=412 or sum(r['split']=='val' for r in exported)!=93:raise ValueError('Export not exact505')
    if digest(checkpoint)!=checkpoint_sha:raise ValueError('Best checkpoint changed during export')
    payload=dict(kind='gaze_raw_legacy16_cache_1011',base_checkpoint=str(checkpoint.resolve()),base_checkpoint_sha256=checkpoint_sha,grid=[24,48],records=exported,clinical_source=clinical,cache_provenance=dict(native_cache_provenance_sha256=digest(Path(args.native_cache)/'provenance.json'),canonical_cache_provenance_sha256=digest(Path(args.cache)/'provenance.json'),centerbias_sha256=digest(args.centerbias),centerbias_provenance_sha256=digest(args.centerbias_provenance),input_size=size),training_patient_keys=clinical['training_patient_keys'],selection_patient_keys=clinical['selection_patient_keys'],inherited_bos_selection_patient_keys=sorted(set().union(*(set(torch.load(s['path'],map_location='cpu',weights_only=False)['selection_patient_keys']) for s in clinical['bos_sources']))),actual_val_parity=dict(requested=a.live_val_parity,images=len(parity),max_logp_error=max([r['max_logp_error'] for r in parity],default=None),tolerance=1e-4,records=parity),head_numerics=dict(cudnn_enabled=True,allow_tf32=False,visual_tf32_preserved=True),canonical_inference_batch_size=a.batch_size,parity_diagnostic_sha256=digest(ROOT/'runs/dual_target_1011/deepgaze_clinical_parity_diagnostic.json'),protocol='Normalizedclinicalbest canonicaltrain412+val93; t0cachedactualimageBOS; t>=1strictlast4observedprefix; nooracle/future/GTboxes; exploratoryval',seconds=time.monotonic()-started)
    temporary=output.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(output);pd.DataFrame(images).to_csv(output.with_suffix('.csv'),index=False);output.with_suffix('.json').write_text(json.dumps(dict(checkpoint_sha256=checkpoint_sha,train_nll=np.mean([r['nll'] for r in images if r['split']=='train']),val_nll=np.mean([r['nll'] for r in images if r['split']=='val']),actual_val_parity=payload['actual_val_parity'],seconds=payload['seconds']),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trial',default='runs/dual_target_1011/deepgaze_clinical_seed42');p.add_argument('--output',default='runs/dual_target_1011/deepgaze_clinical_best_raw_cache.pt');p.add_argument('--device',default='cuda:0');p.add_argument('--threads',type=int,default=4);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--live-val-parity',action='store_true');export(p.parse_args())
