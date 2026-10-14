"""Train capture50 heads on original frozen train-only keys; fixedlast15 selection."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.proposal_capture_1011 import CaptureHead,capture_targets,balanced_capture_bce,validate_cache
from src.proposal_utility_1011 import seed_everything,state_sha256,file_sha256,balanced_mse
from guided_trajectory_1011 import train_annotations,base_path
from trajectory_1011 import rows,save


def train(a):
    torch.set_num_threads(4);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);save(output/'config.json',vars(a));base=base_path(a.base);base_hash=file_sha256(base);annotation=ROOT/'manifests/instances_clipped.json';annotation_hash=file_sha256(annotation);frame=rows('train');allowlist,boxes=train_annotations(frame,annotation)
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);records=validate_cache(cache,base_hash,allowlist,annotation_hash);byid={int(r['image_id']):r for r in records};known=[];unattainable=[]
    for _,row in frame.iterrows():
        image_id=int(row.image_id)
        with Image.open(ROOT/'data/image'/row.file_name) as image:target=capture_targets(image_id,row.patient_key,boxes[image_id],image.size,allowlist)
        if target['known']!=bool(byid[image_id]['known']):raise ValueError('Cache known supervision mismatch')
        if target['known']:
            known.append(dict(image_id=image_id,keys=byid[image_id]['keys'],target=target[a.target]))
            if not target['attainable_capture50']:unattainable.append(image_id)
    if not known:raise ValueError('No known training boxes')
    seed_everything(a.seed);dim=known[0]['keys'].shape[-1];head=CaptureHead(dim,a.architecture).to(a.device);initial_hash=state_sha256(head.state_dict());optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=1e-4);history=[];started=time.monotonic()
    provenance=dict(training_image_ids=sorted(allowlist),training_patient_keys=sorted(set(allowlist.values())),box_supervised_image_ids=sorted(r['image_id'] for r in known),boxless_images_skipped=len(records)-len(known),known_unattainable_capture50_image_ids=unattainable,annotation_sha256=annotation_hash,feature_cache=str(Path(a.cache).resolve()),feature_cache_sha256=file_sha256(a.cache),training_manifest_sha256=file_sha256(ROOT/'manifests/all.csv'),validation_read=False,test_read=False,selection='Last epoch15 chosen a priori; no validation reads',score_interpretation=('Marked box >=50% physical224 crop capture score' if a.target=='capture50' else 'Maximum absolute marked box coverage fraction for physical224 crop')+', not calibrated medical lesion probability',known_zero_target_interpretation=('No candidate captures half a marked box; action-capture negative' if a.target=='capture50' else 'Crop has zero overlap with marked boxes')+', not clinically healthy',original_guided_model_frozen=True)
    save(output/'provenance.json',provenance)
    for epoch in range(1,a.epochs+1):
        sampler_seed=a.seed+epoch;order=np.random.default_rng(sampler_seed).permutation(len(known));losses=[];positive=[];negative=[]
        for start in range(0,len(order),a.batch_size):
            batch=[known[int(i)] for i in order[start:start+a.batch_size]];keys=torch.stack([r['keys'] for r in batch]).to(a.device);target=torch.stack([r['target'] for r in batch]).to(a.device);optimizer.zero_grad();logits=head.logits(keys);terms=[]
            for z,y in zip(logits,target):
                loss,stats=balanced_capture_bce(z,y) if a.target=='capture50' else balanced_mse(z.sigmoid(),y);terms.append(loss)
                if stats['positive'] is not None:positive.append(stats['positive'])
                if stats['negative'] is not None:negative.append(stats['negative'])
            loss=torch.stack(terms).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite capture loss')
            loss.backward();optimizer.step();losses.extend(float(t.detach()) for t in terms)
        record=dict(epoch=epoch,loss=float(np.mean(losses)),positive_loss=float(np.mean(positive)) if positive else None,negative_loss=float(np.mean(negative)) if negative else None,sampler_seed=sampler_seed,images=len(known));history.append(record);save(output/'history.json',history);print(json.dumps(record),flush=True)
        state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};checkpoint=dict(kind='proposal_capture_1011',head_state=state,head_state_sha256=state_sha256(state),initial_state_sha256=initial_hash,config=dict(dim=dim,architecture=a.architecture,grid=[24,48],crop_size=224),target=a.target,loss='balanced BCE' if a.target=='capture50' else 'balanced sigmoid MSE',base_checkpoint=str(base.resolve()),base_checkpoint_sha256=base_hash,epoch=epoch,args=vars(a),**provenance);torch.save(checkpoint,output/'last.pt')
    save(output/'summary.json',dict(**history[-1],seconds=time.monotonic()-started,initial_state_sha256=initial_hash,selection=provenance['selection']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--cache',default='runs/proposal_visit_1011/train_feature_cache.pt');p.add_argument('--output',required=True);p.add_argument('--target',choices=['capture50','absolute_utility'],default='capture50');p.add_argument('--architecture',choices=['linear','spatial'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    if a.epochs<1 or a.batch_size<1:p.error('epochs/batch-size must be positive')
    train(a)
