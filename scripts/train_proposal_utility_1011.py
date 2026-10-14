"""Train-only frozen feature cache and last-epoch head regression."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.data import whole_tensor
from src.proposal_utility_1011 import *
from guided_trajectory_1011 import load_guided,train_annotations,base_path
from trajectory_1011 import rows,save


def train(a):
    torch.set_num_threads(4)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);save(output/'config.json',vars(a));seed_everything(a.seed)
    base=base_path(a.base);base_hash=file_sha256(base);annotation=ROOT/'manifests/instances_clipped.json';annotation_hash=file_sha256(annotation)
    frame=rows('train');allowlist,boxes=train_annotations(frame,annotation);cache_path=Path(a.cache) if a.cache else output/'feature_cache.pt'
    if cache_path.exists():cache=torch.load(cache_path,map_location='cpu',weights_only=False)
    else:
        model,base_ck=load_guided(base,a.base_checkpoint_override);freeze_model(model);model.to(a.device);records=[]
        if tuple(model.grid)!=(24,48):raise ValueError('Expected 24x48 grid')
        for _,row in frame.iterrows():
            with Image.open(ROOT/'data/image'/row.file_name) as source:
                image=source.convert('L');targets=utility_targets(int(row.image_id),row.patient_key,boxes[int(row.image_id)],image.size,allowlist,model.grid)
                with torch.inference_mode():keys=model.maps(whole_tensor(image,model.input_size)[None].to(a.device))['keys'][0].cpu().clone()
            records.append(dict(image_id=int(row.image_id),patient_key=row.patient_key,keys=keys,known=targets['known'],utility=targets['utility'],occupancy=targets['occupancy']))
            if len(records)%20==0:print(json.dumps(dict(event='cache',images=len(records),total=len(frame))),flush=True)
        cache=dict(kind='proposal_utility_features_1011',base_checkpoint_sha256=base_hash,annotation_sha256=annotation_hash,grid=[24,48],crop_size=224,records=records)
        cache_path.parent.mkdir(parents=True,exist_ok=True);torch.save(cache,cache_path);del model
    records=validate_cache(cache,base_hash,allowlist,annotation_hash);known=[r for r in records if r['known']]
    if not known:raise ValueError('No box-supervised training images')
    seed_everything(a.seed);dim=known[0]['keys'].shape[-1];head=ProposalUtilityHead(dim).to(a.device);initial_hash=state_sha256(head.state_dict());optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=1e-4);history=[];started=time.monotonic()
    provenance=dict(training_image_ids=sorted(allowlist),training_patient_keys=sorted(set(allowlist.values())),box_supervised_image_ids=sorted(r['image_id'] for r in known),boxless_images_skipped=len(records)-len(known),annotation_sha256=annotation_hash,feature_cache=str(cache_path.resolve()),validation_read=False,test_read=False,selection='Last epoch chosen a priori; no validation model selection',score_interpretation='Marked-box crop capture utility, not calibrated lesion or malignancy probability',full_guided_model_frozen=True)
    save(output/'provenance.json',provenance)
    for epoch in range(1,a.epochs+1):
        sampler_seed=a.seed+epoch;order=np.random.default_rng(sampler_seed).permutation(len(known));losses=[];positives=[];negatives=[]
        for start in range(0,len(order),a.batch_size):
            batch=[known[int(i)] for i in order[start:start+a.batch_size]];keys=torch.stack([r['keys'] for r in batch]).to(a.device);targets=torch.stack([r[a.target] for r in batch]).to(a.device);optimizer.zero_grad();scores=head(keys);terms=[]
            for score,target in zip(scores,targets):
                loss,stats=balanced_mse(score,target);terms.append(loss)
                if stats['positive'] is not None:positives.append(stats['positive'])
                if stats['negative'] is not None:negatives.append(stats['negative'])
            loss=torch.stack(terms).mean()
            if not torch.isfinite(loss):raise ValueError("Nonfinite head loss")
            loss.backward();optimizer.step();losses.extend(float(t.detach()) for t in terms)
        record=dict(epoch=epoch,loss=float(np.mean(losses)),positive_mse=float(np.mean(positives)) if positives else None,negative_mse=float(np.mean(negatives)) if negatives else None,sampler_seed=sampler_seed,images=len(known));history.append(record);save(output/'history.json',history);print(json.dumps(record),flush=True)
        state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        checkpoint=dict(kind='proposal_utility_1011',head_state=state,head_state_sha256=state_sha256(state),initial_state_sha256=initial_hash,config=dict(dim=dim,architecture='sigmoid Linear(D,1)',grid=[24,48],crop_size=224),base_checkpoint=str(base.resolve()),base_checkpoint_sha256=base_hash,target=a.target,epoch=epoch,args=vars(a),**provenance)
        torch.save(checkpoint,output/'last.pt')
    save(output/'summary.json',dict(**history[-1],seconds=time.monotonic()-started,initial_state_sha256=initial_hash,selection=provenance['selection']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--base-checkpoint-override');p.add_argument('--output',required=True);p.add_argument('--cache');p.add_argument('--target',choices=['utility','occupancy'],default='utility');p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--device',default='cuda:0');p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
    if a.epochs<1 or a.batch_size<1: p.error('epochs and batch-size must be positive')
    train(a)
