"""Clinical DGIII readout training; frozen native CNN and truthful past-only history."""
import argparse,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.deepgaze_clinical_1011 import freeze_features,history_batch,prepare_saliency,forward_saliency,head_state,inventory
from src.deepgaze_context_1011 import dense_centerbias
from src.gaze_path_augmentation_1011 import augmented_sample
from src.gaze_mirror_augmentation_1011 import mirror_coordinates,flip_for_epoch
from src.proposal_utility_1011 import state_sha256
from gaze_nll_adaptation_1011 import cache_records,save
from train_gaze_path_augmentation_1011 import load_pools
from evaluate_deepgaze3_1011 import load_deepgaze,cell_mass
from train_attention_1011 import digest


def load_native_cache(a,records,feature_sha):
    root=Path(a.native_cache);header=json.loads((root/'provenance.json').read_text())
    if json.loads((root/'status.json').read_text()).get('state')!='complete' or header.get('kind')!='deepgaze_clinical_native_cache_1011':raise ValueError('Native DG cache incomplete')
    if header['checkpoint_sha256']!=digest(a.checkpoint) or header['feature_state_sha256']!=feature_sha:raise ValueError('Native DG dependency mismatch')
    if header['source_cache_provenance_sha256']!=digest(Path(a.cache)/'provenance.json'):raise ValueError('Native DG gaze source mismatch')
    source={int(r['image_id']):r for r in records};items={}
    expected={f'{i}_mirror' for i,r in source.items() if r['split']=='train'}|{str(i) for i in source}
    names=[m['name'] for m in header['records']]
    if len(names)!=len(set(names)) or not set(names)<=expected:raise ValueError('Native DG invalid/duplicate source variants')
    if not header.get('smoke') and set(names)!=expected:raise ValueError('Native DG full917 variants required')
    if header.get('smoke') and not a.smoke_limit:raise ValueError('Smoke native cache forbidden for full trial')
    w,h=header['input_size'];spatial=(h//64,w//64)
    for meta in header['records']:
        path=root/f"{meta['name']}.pt"
        if digest(path)!=meta['file_sha256']:raise ValueError('Native DG file SHA mismatch')
        item=torch.load(path,map_location='cpu',weights_only=False);r=source[int(item['image_id'])]
        expected_name=f"{int(item['image_id'])}_mirror" if item['flipped'] else str(int(item['image_id']))
        if meta['name']!=expected_name or int(meta['image_id'])!=int(item['image_id']) or bool(meta['flipped'])!=bool(item['flipped']) or meta['split']!=item['split']:raise ValueError('Native DG file/variant identity mismatch')
        expected_coords=mirror_coordinates(r['coords'],r['mask']) if item['flipped'] else r['coords']
        if not torch.equal(item['coords'],expected_coords) or not torch.equal(item['mask'],r['mask']):raise ValueError('Native DG canonical target/mask mismatch')
        if item['patient_key']!=r['patient_key'] or item['split']!=r['split'] or item['image_sha256']!=r['image_sha256'] or (item['split']=='val' and item['flipped']):raise ValueError('Native DG patient/variant mismatch')
        if [f.shape for f in item['features']]!=[torch.Size((c,*spatial)) for c in (1888,128,32)] or any(f.dtype!=torch.float32 or not torch.isfinite(f).all() for f in item['features']):raise ValueError('Native DG features must stay native float32')
        if item['bos_logp'].shape!=(1152,) or not torch.isfinite(item['bos_logp']).all() or abs(float(item['bos_logp'].exp().sum())-1)>1e-5:raise ValueError('Invalid actual-image BOS')
        items[meta['name']]=item
    return items,header


def group_data(group,native,bias,size,device):
    items=[native[f"{int(r['image_id'])}_mirror" if r.get('flipped') else str(int(r['image_id']))] for r in group];features=[torch.stack([i['features'][k] for i in items]).to(device) for k in range(3)];coords=torch.stack([r['coords'] for r in group]).to(device);mask=torch.stack([r['mask'] for r in group]).to(device);bos=torch.stack([i['bos_logp'] for i in items]).to(device);cb=dense_centerbias(bias,size,device).expand(len(group),-1,-1).clone()
    for i,r in enumerate(group):
        if r.get('flipped'):cb[i]=cb[i].flip(-1)
    return features,coords,mask,bos,cb


@torch.no_grad()
def evaluate(model,records,native,bias,size,a):
    model.eval();values=[]
    for start in range(0,len(records),a.batch_size):
        group=records[start:start+a.batch_size]
        if any(r.get('flipped') for r in group):raise ValueError('Validation flip forbidden')
        f,coords,mask,bos,cb=group_data(group,native,bias,size,a.device);saliency=prepare_saliency(model,f,size);predictions=[bos]
        for t in range(1,16):hx,hy=history_batch(coords,t,size);predictions.append(cell_mass(forward_saliency(model,saliency,size,cb,hx,hy)))
        q=torch.stack(predictions,1);target=(coords[...,1]*48).long().clamp(0,47)*24+(coords[...,0]*24).long().clamp(0,23);nll=-q.gather(-1,target[...,None]).squeeze(-1);scores=(nll*mask).sum(-1)/mask.sum(-1)
        for r,value in zip(group,scores):values.append(dict(image_id=int(r['image_id']),patient_key=r['patient_key'],nll=float(value)))
    return float(np.mean([r['nll'] for r in values])),values


def load_clinical(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    if checkpoint.get('kind')!='deepgaze_clinical_readout_1011' or digest(checkpoint['source_checkpoint'])!=checkpoint['source_checkpoint_sha256']:raise ValueError('Clinical DG dependency mismatch')
    model=freeze_features(load_deepgaze(checkpoint['source'],checkpoint['source_checkpoint'],device))
    if state_sha256(model.features.state_dict())!=checkpoint['feature_state_sha256']:raise ValueError('Clinical DG frozen feature restore mismatch')
    if set(checkpoint['state'])!=set(head_state(model)):raise ValueError('Clinical DG checkpoint must contain exact readout-only state')
    missing,unexpected=model.load_state_dict(checkpoint['state'],strict=False)
    if unexpected or any(not k.startswith('features.') for k in missing):raise ValueError('Clinical DG readout restore mismatch')
    if state_sha256(model.features.state_dict())!=checkpoint['feature_state_sha256']:raise ValueError('Clinical DG feature state changed during restore')
    return model.eval(),checkpoint


def verified_centerbias(a,records):
    provenance=json.loads(Path(a.centerbias_provenance).read_text());rawpath=ROOT/'runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt'
    if provenance['probability_tensor_sha256']!=digest(a.centerbias) or provenance['source_raw_cache_sha256']!=digest(rawpath) or provenance['all_manifest_sha256']!=digest(ROOT/'manifests/all.csv'):raise ValueError('Clinical centerbias source SHA mismatch')
    tr=[r for r in records if r['split']=='train']
    if set(provenance['training_image_ids'])!={int(r['image_id']) for r in tr} or len(tr)!=412 or provenance.get('validation_used') is not False:raise ValueError('Clinical centerbias train allowlist mismatch')
    bias=torch.load(a.centerbias,map_location='cpu',weights_only=False)
    if bias.shape!=(48,24) or not torch.isfinite(bias).all() or (bias<=0).any() or abs(float(bias.sum())-1)>1e-6:raise ValueError('Clinical centerbias must be positive normalized probability')
    return bias


def train(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);records,source=cache_records(a);base_sha=source['base_checkpoint_sha256'];pools,_=load_pools(a.pool_cache,records,base_sha,source['visual_state_sha256'],a.cache,False,a.pool_labels);model=freeze_features(load_deepgaze(a.source,a.checkpoint,a.device));frozen_before=state_sha256(model.features.state_dict());native,header=load_native_cache(a,records,frozen_before);size=header['input_size'];bias=verified_centerbias(a,records)
    tr=[r for r in records if r['split']=='train' and str(int(r['image_id'])) in native and f"{int(r['image_id'])}_mirror" in native];va=[r for r in records if r['split']=='val' and str(int(r['image_id'])) in native]
    if a.smoke_limit:tr=tr[:a.smoke_limit];va=va[:a.smoke_limit]
    if not tr or not va:raise ValueError('Need clinical train and canonical validation sources')
    if not a.smoke_limit:
        expected_val={int(r['image_id']) for r in records if r['split']=='val'}
        if len(tr)!=412 or len(va)!=93 or {int(r['image_id']) for r in va}!=expected_val:raise ValueError('Clinical full trial must use train412/fixedval93')
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4);initial,perimage=evaluate(model,va,native,bias,size,a);best=initial;started=time.monotonic();history=[]
    save(output/'config.json',vars(a));save(output/'provenance.json',dict(native_cache=header,centerbias_sha256=digest(a.centerbias),centerbias_provenance=json.loads(Path(a.centerbias_provenance).read_text()),frozen_feature_state_sha256=frozen_before,initial_readout_state_sha256=state_sha256(head_state(model)),inventory=inventory(model),protocol='Clinical readout fine-tuning of natural-image DGIII; frozenDenseNet; nativefeatures; image-macro hardcell NLL; t0actual frozen arithmetic; t>=1truthfulNaN last4past; originalcanonical93val; noROI/gaze targetchanges'))
    def checkpoint(epoch):return dict(kind='deepgaze_clinical_readout_1011',state=head_state(model),source_checkpoint=str(Path(a.checkpoint).resolve()),source_checkpoint_sha256=digest(a.checkpoint),source=a.source,feature_state_sha256=frozen_before,input_size=size,centerbias=bias,bos_sources=header['bos_sources'],bos_weights=header['bos_weights'],epoch=epoch,args=vars(a),training_patient_keys=sorted({r['patient_key'] for r in tr}),selection_patient_keys=sorted({r['patient_key'] for r in va}),selection='Minimumsamecanonical93hardNLL includingepoch0, exploratory')
    torch.save(checkpoint(0),output/'initial.pt');torch.save(checkpoint(0),output/'best.pt');save(output/'initial_validation.json',dict(nll=initial,images=perimage));save(output/'best_validation.json',dict(nll=best,epoch=0,images=perimage))
    for epoch in range(1,a.epochs+1):
        model.train();model.features.eval();order=np.random.default_rng(a.seed+epoch).permutation(len(tr));train_scores=[];selections=[]
        for start in range(0,len(tr),a.batch_size):
            group=[]
            for index in order[start:start+a.batch_size]:
                r=tr[index];sample=augmented_sample(pools[int(r['image_id'])],int(r['image_id']),epoch,a.seed);flipped=flip_for_epoch(int(r['image_id']),epoch,a.seed);coords=mirror_coordinates(sample['coords'],sample['mask']) if flipped else sample['coords'];group.append({**r,'coords':coords,'mask':sample['mask'],'flipped':flipped});selections.append(dict(image_id=int(r['image_id']),pool_indices=sample['pool_indices'],raw_indices=sample['raw_indices'],flipped=flipped))
            f,coords,mask,bos,cb=group_data(group,native,bias,size,a.device);optimizer.zero_grad();saliency=prepare_saliency(model,f,size);leaves=[s.detach().requires_grad_(True) for s in saliency];target=(coords[...,1]*48).long().clamp(0,47)*24+(coords[...,0]*24).long().clamp(0,23);counts=mask.sum(-1);scores=-bos.gather(1,target[:,:1]).squeeze(1)*mask[:,0]/counts
            for t in range(1,16):
                hx,hy=history_batch(coords,t,size);q=cell_mass(forward_saliency(model,leaves,size,cb,hx,hy));perimage=-q.gather(1,target[:,t:t+1]).squeeze(1)*mask[:,t]/counts;loss=perimage.mean()
                if not torch.isfinite(loss):raise ValueError('Nonfinite DG clinical loss')
                loss.backward();scores=scores+perimage.detach()
            torch.autograd.backward(saliency,[leaf.grad for leaf in leaves]);optimizer.step();train_scores.extend(scores.cpu().tolist())
        value,perimage=evaluate(model,va,native,bias,size,a);record=dict(epoch=epoch,train_nll=float(np.mean(train_scores)),validation_nll=value);history.append(record);save(output/'history.json',history);save(output/f'selections_epoch{epoch:02d}.json',dict(ordered_selections=selections));torch.save(checkpoint(epoch),output/'last.pt');print(json.dumps(record),flush=True)
        if value<best:best=value;torch.save(checkpoint(epoch),output/'best.pt');save(output/'best_validation.json',dict(nll=best,epoch=epoch,images=perimage))
    frozen_after=state_sha256(model.features.state_dict());save(output/'frozen_feature_integrity.json',dict(before=frozen_before,after=frozen_after,unchanged=frozen_before==frozen_after))
    if frozen_before!=frozen_after:raise ValueError('Frozen DG features changed')
    save(output/'summary.json',dict(initial_validation_nll=initial,best_validation_nll=best,epochs=a.epochs,train_images=len(tr),validation_images=len(va),seconds=time.monotonic()-started,goal_met=best<=5.3,input_size=size,roi_crop_size=224,whole_pixel_cost_relative256= size[0]*size[1]/(256*512)))


def parser():
    p=argparse.ArgumentParser();p.add_argument('--source',default='/tmp/deepgaze_feasibility_1011');p.add_argument('--checkpoint',default='runs/dual_target_1011/deepgaze_public/assets/deepgaze3.pth');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--pool-cache',default='runs/dual_target_1011/train_gaze_pool_cache');p.add_argument('--pool-labels',default='runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt');p.add_argument('--native-cache',required=True);p.add_argument('--centerbias',default='runs/dual_target_1011/deepgaze_public/kde256_smoke/train_centerbias.pt');p.add_argument('--centerbias-provenance',default='runs/dual_target_1011/deepgaze_public/train_centerbias_provenance.json');p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--smoke-limit',type=int,default=0);return p
if __name__=='__main__':
    a=parser().parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;train(a)
