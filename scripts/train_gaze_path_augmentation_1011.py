"""Matched canonical/augmented training paths; fixed canonical93 validation."""
import argparse,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.gaze_path_augmentation_1011 import augmented_sample,selected_indices
from src.gaze_nll_adaptation_1011 import initialize,hard_nll
from src.guided_trajectory_1011 import nonvisual_state
from src.spatial import gaussian_targets
from src.proposal_utility_1011 import state_sha256
from gaze_nll_adaptation_1011 import cache_records,load_base,batch,validate,save
from guided_trajectory_1011 import base_path
from train_attention_1011 import digest


def load_pools(path,records,parent_sha,visual_sha,source_cache,smoke_allowed=False,pool_labels=None):
    directory=Path(path);header=json.loads((directory/'provenance.json').read_text());status=json.loads((directory/'status.json').read_text())
    if status.get('state')!='complete':raise ValueError('Allpoint crop cache incomplete')
    if header.get('kind')!='gaze_stratum_pool_cache_1011':raise ValueError('Unknown allpoint pool cache kind')
    fresh=header.get('fresh_encoder_validation')
    if not header.get('smoke'):
        if not fresh or fresh.get('passed') is not True or fresh.get('tolerance')!=0 or fresh.get('production_batch_size')!=8:raise ValueError('Fresh encoder production parity proof missing')
        if digest(fresh['proof_path'])!=fresh['proof_sha256'] or json.loads(Path(fresh['proof_path']).read_text())!=fresh['cases']:raise ValueError('Fresh encoder parity proof SHA/content mismatch')
    proof=header.get('canonical_vector_validation',{})
    if proof.get('passed') is not True or proof.get('tolerance')!=0 or proof.get('max_abs_error')!=0:raise ValueError('Allpoint canonical vector proof missing or changed')
    if header.get('base_checkpoint_sha256')!=parent_sha or header.get('visual_state_sha256')!=visual_sha:raise ValueError('Allpoint crop parent/encoder mismatch')
    audited=None
    if pool_labels is not None:
        if header.get('pool_labels_sha256')!=digest(pool_labels):raise ValueError('Pool audit labels SHA mismatch')
        audit=torch.load(pool_labels,map_location='cpu',weights_only=False)
        if audit.get('kind')!='gaze_chosen_segment_pool_labels_1011' or header.get('pool_audit_provenance')!=audit['provenance']:raise ValueError('Pool audit provenance mismatch')
        if audit['provenance']['source_parent_sha256']!=parent_sha or audit['provenance']['sampler']['sampler_source_sha256']!=digest(ROOT/'src/gaze_path_augmentation_1011.py'):raise ValueError('Pool audit parent/sampler source mismatch')
        audited={int(r['image_id']):r for r in audit['records']}
    source_provenance=json.loads((Path(source_cache)/'provenance.json').read_text())
    if header.get('source_cache_provenance_sha256')!=digest(Path(source_cache)/'provenance.json'):raise ValueError('Allpoint canonical cache provenance mismatch')
    for key in ('gaze_sha256','manifest_sha256','all_manifest_sha256'):
        if header.get(key)!=source_provenance.get(key):raise ValueError(f'Allpoint source mismatch: {key}')
    if header.get('annotation_used') is not False:raise ValueError('Allpoint pool annotation input declaration missing')
    train=[r for r in records if r['split']=='train'];pools={};manifest_records=header['records']
    if header.get('smoke') and smoke_allowed:
        chosen={int(r['image_id']) for r in manifest_records};train=[r for r in train if int(r['image_id']) in chosen]
    elif header.get('smoke'):raise ValueError('Smoke allpoint cache cannot train full experiment')
    metadata={int(r['image_id']):r for r in manifest_records}
    if not train or len(metadata)!=len(manifest_records):raise ValueError('Empty or duplicate allpoint train cohort')
    if len(manifest_records)!=len(train) or {int(r['image_id']) for r in manifest_records}!={int(r['image_id']) for r in train}:raise ValueError('Allpoint crop train cohort mismatch')
    for row in train:
        image_id=int(row['image_id']);file=directory/f"{row['image_id']}.pt"
        if metadata[image_id].get('file_sha256')!=digest(file):raise ValueError('Allpoint pool file SHA mismatch')
        pool=torch.load(file,map_location='cpu',weights_only=False);n=len(pool['coords'])
        if pool.get('image_sha256')!=row['image_sha256']:raise ValueError('Allpoint source image SHA mismatch')
        if pool['patient_key']!=row['patient_key'] or pool.get('split','train')!='train' or pool['coords'].shape!=(n,2) or pool['patch_vectors'].shape!=(n,row['vectors'].shape[-1]):raise ValueError('Allpoint pool patient/shape mismatch')
        if not torch.isfinite(pool['coords']).all() or not torch.isfinite(pool['patch_vectors']).all() or (pool['coords']<0).any() or (pool['coords']>1).any():raise ValueError('Invalid allpoint cached coordinates/vectors')
        raw=torch.as_tensor(pool['raw_indices']);pool['raw_indices']=raw
        if audited is not None:
            auditrow=audited.get(image_id)
            if auditrow is None or auditrow['patient_key']!=row['patient_key'] or auditrow['split']!='train' or not torch.equal(auditrow['pool_coords'],pool['coords']) or not torch.equal(auditrow['pool_raw_indices'],raw):raise ValueError('Pool differs from audited chosen segment')
        if raw.shape!=(n,) or (raw[1:]<=raw[:-1]).any():raise ValueError('Allpoint raw indices must be increasing')
        canonical_ids=selected_indices(n,image_id,0,augmented=False)
        if pool.get('canonical_pool_indices') is None or not np.array_equal(np.asarray(pool['canonical_pool_indices']),canonical_ids):raise ValueError('Allpoint canonical pool indices mismatch')
        if metadata[image_id].get('canonical_raw_indices')!=raw[canonical_ids].tolist():raise ValueError('Allpoint canonical raw index source mismatch')
        canonical=augmented_sample(pool,image_id,0,augmented=False)
        if not torch.equal(canonical['vectors'],row['vectors']):raise ValueError('Copied canonical preceding crop vectors changed')
        if not torch.equal(canonical['coords'],row['coords']) or not torch.equal(canonical['mask'],row['mask']):raise ValueError('Chosen pool does not reproduce exact canonical segment')
        # Frozen encoder numeric consistency is checked by extraction; preserve original vectors in canonical control.
        pools[image_id]=pool
    return pools,header


def train(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);records,provenance=cache_records(a);base,base_checkpoint=load_base(a.base);parent_sha=digest(base_path(a.base));pools,header=load_pools(a.pool_cache,records,parent_sha,provenance['visual_state_sha256'],a.cache,bool(a.smoke_limit),a.pool_labels);tr=[r for r in records if r['split']=='train'];va=[r for r in records if r['split']=='val']
    if a.smoke_limit:tr=[r for r in tr if int(r['image_id']) in pools][:a.smoke_limit];va=va[:a.smoke_limit]
    model=initialize(base).to(a.device);reference=initialize(base).to(a.device).eval()
    for p in reference.parameters():p.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4);started=time.monotonic();history=[];initial,perimage=validate(model,va,a);best=initial
    save(output/'config.json',vars(a));save(output/'provenance.json',dict(base_checkpoint_sha256=parent_sha,visual_cache=provenance,pool_header=header,pool_header_sha256=digest(Path(a.pool_cache)/'provenance.json'),sampling='Exact endpoints0,M-1 plus one uniformpoint each of14 np.array_split interior indexstrata; same chosen min32 segment; seed_for(seed,imageID+epoch*100000); canonical validation unchanged',target_role='Observed training targets vary by epoch; additional points are correlated observations of same image/path, not independent cases'))
    def checkpoint(epoch):return dict(kind='gaze_nll_adaptation_1011',state=nonvisual_state(model),model_config=model.model_config,base_checkpoint=str(base_path(a.base).resolve()),base_checkpoint_sha256=parent_sha,epoch=epoch,args=vars(a),training_patient_keys=sorted(set(base_checkpoint['training_patient_keys'])|{r['patient_key'] for r in tr}),selection_patient_keys=sorted(set(base_checkpoint['selection_patient_keys'])|{r['patient_key'] for r in va}),selection='Minimum samecanonical93 hardNLL incl epoch0; exploratory',protocol='Frozen visual, causal same-prefix KL teacher, trainpath resampling only' if a.mode=='augmented' else 'Matched canonical16 control')
    save(output/'initial_validation.json',dict(nll=initial,images=perimage));save(output/'best_validation.json',dict(nll=best,epoch=0,images=perimage));torch.save(checkpoint(0),output/'initial.pt');torch.save(checkpoint(0),output/'best.pt')
    for epoch in range(1,a.epochs+1):
        model.train();order=np.random.default_rng(a.seed+epoch).permutation(len(tr));selections=[];losses=[]
        for start in range(0,len(tr),a.batch_size):
            originals=[tr[i] for i in order[start:start+a.batch_size]];group=[]
            for r in originals:
                if a.mode=='augmented':
                    sample=augmented_sample(pools[int(r['image_id'])],int(r['image_id']),epoch,a.seed);group.append({**r,**{k:sample[k] for k in ('coords','vectors','mask')}})
                else:
                    sample=augmented_sample(pools[int(r['image_id'])],int(r['image_id']),0,a.seed,False);group.append(r)
                selections.append(dict(image_id=int(r['image_id']),pool_indices=sample['pool_indices'],raw_indices=sample['raw_indices']))
            f,coords,mask,vectors=batch(group,a.device);optimizer.zero_grad();memory=model.maps_features(f);q=model.teacher_logits(memory,coords,vectors);nll=hard_nll(q,coords,mask,model.grid)
            softtargets=torch.stack([gaussian_targets(coords[i],model.grid,32,r['image_size']) for i,r in enumerate(group)]);soft=(-(softtargets*q.log_softmax(-1)).sum(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            with torch.no_grad():ref=reference.teacher_logits(reference.maps_features(f),coords,vectors).log_softmax(-1)
            kl=(ref.exp()*(ref-q.log_softmax(-1))).sum(-1);kl=((kl*mask).sum(-1)/mask.sum(-1).clamp_min(1)).mean();loss=nll+.5*soft.mean()+.5*kl
            if not torch.isfinite(loss):raise ValueError('Nonfinite augmented training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append((float(nll.detach()),len(group)))
        score,perimage=validate(model,va,a);record=dict(epoch=epoch,train_nll=float(sum(value*count for value,count in losses)/sum(count for _,count in losses)),validation_nll=score);history.append(record);save(output/'history.json',history);save(output/f'selections_epoch{epoch:02d}.json',dict(epoch=epoch,ordered_selections=selections));print(json.dumps(record),flush=True);torch.save(checkpoint(epoch),output/'last.pt')
        if score<best:best=score;torch.save(checkpoint(epoch),output/'best.pt');save(output/'best_validation.json',dict(nll=best,epoch=epoch,images=perimage))
    save(output/'summary.json',dict(mode=a.mode,initial_validation_nll=initial,best_validation_nll=best,train_images=len(tr),validation_images=len(va),epochs=a.epochs,seconds=time.monotonic()-started,goal_met=best<=5.3,smoke=bool(a.smoke_limit)))


def parser():
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--pool-cache',required=True);p.add_argument('--pool-labels',default='runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt');p.add_argument('--output',required=True);p.add_argument('--mode',choices=('canonical','augmented'),required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--smoke-limit',type=int,default=0);return p
if __name__=='__main__':
    a=parser().parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(a.threads);train(a)
