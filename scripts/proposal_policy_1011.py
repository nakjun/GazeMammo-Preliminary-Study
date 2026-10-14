"""Save image-only fixed-checkpoint policies; annotation evaluation is separate."""
import argparse,csv,json,random,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.data import whole_tensor,seed_for
from src.guided_trajectory_1011 import candidate_geometry
from src.proposal_policy_1011 import POLICIES,rollout
from guided_trajectory_1011 import load_guided,base_path
from train_attention_1011 import digest


def image_records(a):
    if a.manifest:
        if a.split!='val':raise ValueError('This exploratory runner permits only validation manifest inference')
        with open(a.manifest,newline='') as source:
            # Only routing columns enter generation; annotation/gaze/label columns are discarded.
            records=[dict(image_id=int(r['image_id']),image=str(ROOT/'data/image'/r['file_name'])) for r in csv.DictReader(source) if r['split']=='val']
        a.requested_routing_count=len(records)
        with (ROOT/'manifests/trajectory.csv').open(newline='') as source:
            comparable_ids={int(r['image_id']) for r in csv.DictReader(source) if r['split']=='val'}
        records=[r for r in records if r['image_id'] in comparable_ids]
        a.comparable_routing_count=len(records)
        if a.image_ids:records=[r for r in records if r['image_id'] in a.image_ids]
        if a.image_ids and set(a.image_ids)!={r['image_id'] for r in records}:raise ValueError('Requested image outside validation manifest')
        return records
    if not a.image or a.image_id is None:raise ValueError('Provide image and image-id, or manifest')
    return [dict(image_id=a.image_id,image=a.image)]


def predict(a):
    seeds=a.seeds or [a.seed];random.seed(seeds[0]);np.random.seed(seeds[0]);torch.manual_seed(seeds[0]);torch.set_num_threads(a.threads)
    records=image_records(a);output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    (output/'config.json').write_text(json.dumps(vars(a),indent=2));started=time.monotonic();completed=0
    def status(state,error=None):
        (output/'status.json').write_text(json.dumps(dict(state=state,images=len(records),requested_routing_count=getattr(a,'requested_routing_count',len(records)),comparable_routing_count=getattr(a,'comparable_routing_count',len(records)),completed_paths=completed,seconds=time.monotonic()-started,error=error)))
    status('running')
    try:
        model,checkpoint=load_guided(a.checkpoint,a.base_checkpoint_override);model=model.to(a.device).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        checkpoint_hash=digest(base_path(a.checkpoint));utility=None;utility_hash=None
        if a.utility_checkpoint:
            from src.proposal_utility_1011 import load_utility_head,predict_utility
            utility,utility_checkpoint=load_utility_head(a.utility_checkpoint,checkpoint_hash,a.device)
            utility_hash=digest(base_path(a.utility_checkpoint))
        with (output/'traces.jsonl').open('w') as sink:
            for item in records:
                image_id=item['image_id']
                with Image.open(item['image']) as source:image=source.convert('L')
                geometry=candidate_geometry(image,model.grid)
                with torch.no_grad():
                    memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device))
                    utility_scores=predict_utility(utility,memory) if utility is not None else None
                static_geometry={**{k:geometry[k].tolist() for k in ('background','edge','foreground','inbounds')},'valid':geometry['geometry_valid'],'error':geometry['geometry_error']}
                for seed in seeds:
                    for policy in a.policies:
                        isutility=utility is not None and policy in ('P3','P4','P5')
                        head_target=utility_checkpoint['target'] if isutility else None
                        proposal_label=f'{head_target}_mse' if isutility else 'original_occupancy'
                        name=policy+(('_'+proposal_label) if isutility else '')+('_argmax' if a.strategy=='argmax' else '')
                        result=rollout(model,memory,image,geometry,policy,a.steps,seed_for(seed,image_id),a.alpha,a.tau,a.strategy,utility_scores if isutility else None)
                        record=dict(schema_version='proposal_policy_1011_v1',image_id=image_id,image=str(Path(item['image']).resolve()),image_size=list(image.size),grid=list(model.grid),crop_size=224,policy=name,policy_base=policy,strategy=a.strategy,seed=seed,sampling_seed=seed_for(seed,image_id),alpha=a.alpha,tau=a.tau,checkpoint=str(base_path(a.checkpoint).resolve()),checkpoint_sha256=checkpoint_hash,checkpoint_epoch=checkpoint['epoch'],rho=model.rho,ablation=model.ablation,geometry=static_geometry,inputs='image pixels and RNG image ID only',annotation_used=False,annotations_used=False,gaze_used=False,observed_gaze_used=False,labels_used=False,proposal_source=proposal_label,head_target=head_target,head_base_checkpoint_sha256=utility_checkpoint['base_checkpoint_sha256'] if isutility else None,utility_checkpoint_sha256=utility_hash if isutility else None,heuristic=policy=='P5',**result)
                        sink.write(json.dumps(record,allow_nan=False)+'\n');sink.flush()
                        summary={k:record[k] for k in ('image_id','policy','strategy','seed','coordinates','indices','proposal_source','checkpoint_sha256')}
                        (output/f'{name}_image{image_id}_seed{seed}.json').write_text(json.dumps(summary));completed+=1
                    status('running');print(json.dumps(dict(image_id=image_id,seed=seed,completed_paths=completed)),flush=True)
        status('complete')
    except Exception as exc:
        status('failed',str(exc));raise


def parser():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--base-checkpoint-override');p.add_argument('--utility-checkpoint');p.add_argument('--image');p.add_argument('--image-id',type=int);p.add_argument('--image-ids',type=int,nargs='+');p.add_argument('--manifest');p.add_argument('--split',default='val');p.add_argument('--output',required=True);p.add_argument('--policies',nargs='+',choices=POLICIES,default=list(POLICIES));p.add_argument('--seed',type=int,default=42);p.add_argument('--seeds',type=int,nargs='+');p.add_argument('--strategy',choices=('sample','argmax'),default='sample');p.add_argument('--steps',type=int,default=16);p.add_argument('--alpha',type=float,default=.125);p.add_argument('--tau',type=float,default=.5);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);return p
if __name__=='__main__':predict(parser().parse_args())
