"""Fixedlast15 train-only center-offset regression from original frozen keys."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.proposal_center_offset_1011 import CenterOffsetHead,center_offset_targets,offset_loss
from src.proposal_utility_1011 import validate_cache,seed_everything,state_sha256,file_sha256
from guided_trajectory_1011 import train_annotations,base_path
from trajectory_1011 import rows,save


def train(a):
    torch.set_num_threads(4);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);save(output/'config.json',vars(a));base=base_path(a.base);base_hash=file_sha256(base);annotation=ROOT/'manifests/instances_clipped.json';annotation_hash=file_sha256(annotation);frame=rows('train');allowlist,boxes=train_annotations(frame,annotation);cache=torch.load(a.cache,map_location='cpu',weights_only=False);records=validate_cache(cache,base_hash,allowlist,annotation_hash);byid={int(r['image_id']):r for r in records};known=[];diagnostics=[]
    source_checkpoint=torch.load(base,map_location='cpu',weights_only=False)
    if source_checkpoint.get('kind')!='guided_trajectory_1011':raise ValueError('Original proposal must be guided checkpoint')
    proposal_state={k:source_checkpoint['state'][k].detach().clone() for k in ('proposal.weight','proposal.bias')};proposal_hash=state_sha256(proposal_state)
    if proposal_state['proposal.weight'].shape!=(1,records[0]['keys'].shape[-1]) or proposal_state['proposal.bias'].shape!=(1,) or not all(torch.isfinite(v).all() for v in proposal_state.values()):raise ValueError('Invalid frozen original proposal state')
    for _,row in frame.iterrows():
        image_id=int(row.image_id)
        with torch.no_grad():source_relevance=torch.nn.functional.linear(byid[image_id]['keys'],proposal_state['proposal.weight'],proposal_state['proposal.bias']).sigmoid().squeeze(-1)
        with Image.open(ROOT/'data/image'/row.file_name) as image:target=center_offset_targets(image_id,row.patient_key,boxes[image_id],image.size,allowlist,max_offset_crops=a.max_offset_crops,source_relevance=source_relevance,relevance_power=a.relevance_power)
        if target['known']!=bool(byid[image_id]['known']):raise ValueError('Cache known supervision mismatch')
        if target['known']:
            known.append(dict(image_id=image_id,keys=byid[image_id]['keys'],target=target['targets'],weights=target['weights']))
            diagnostics.append(dict(image_id=image_id,**{k:target[k] for k in ('positive_cells','box_count','assigned_boxes','unassigned_box_indices','clipped_component_fraction','weighted_clipped_component_fraction','relevance_fallback_box_indices')}))
    if not known:raise ValueError('No known training boxes')
    seed_everything(a.seed);dim=known[0]['keys'].shape[-1];config=dict(dim=dim,architecture=a.architecture,grid=[24,48],max_offset_crops=a.max_offset_crops);head=CenterOffsetHead(**config).to(a.device);initial_hash=state_sha256(head.state_dict());optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=1e-4);history=[];started=time.monotonic()
    provenance=dict(training_image_ids=sorted(allowlist),training_patient_keys=sorted(set(allowlist.values())),box_supervised_image_ids=sorted(r['image_id'] for r in known),boxless_images_skipped=len(records)-len(known),annotation_sha256=annotation_hash,feature_cache=str(Path(a.cache).resolve()),feature_cache_sha256=file_sha256(a.cache),training_manifest_sha256=file_sha256(ROOT/'manifests/all.csv'),validation_read=False,test_read=False,selection='Last epoch15 chosen a priori; no validation reads',original_guided_model_frozen=True,proposal_rank_source='Unchanged original proposal; this head predicts center offsets only',target_units='bbox center minus float32 normalized source-grid anchor center, divided by224px',bounds=dict(output='max_offset_crops*tanh(raw)',max_offset_crops=a.max_offset_crops,max_pixel_offset=224*a.max_offset_crops,target_clipped=True,inference_coords_clamped_inside_image=True),cell_assignment='Largest positive gridcell/box intersection area; stable first-box tie break',weights='Each assigned box has equal total cell weight; image total weight1; boxless images skipped',unassigned_box_count=sum(len(d['unassigned_box_indices']) for d in diagnostics),training_target_diagnostics=diagnostics,roi_coordinates='Continuous predicted coordinates; gaze likelihood grid remains24x48; physicalcrop224px',zero_initial_final_layer=True,relevance_power=a.relevance_power,original_proposal_checkpoint_sha256=base_hash,original_proposal_state_sha256=proposal_hash,original_proposal_requires_grad=False,within_box_weighting='Original frozen R^p; invalid/allzero scores fall back to uniform; p0 exactuniform',relevance_fallback_box_count=sum(len(d['relevance_fallback_box_indices']) for d in diagnostics))
    save(output/'provenance.json',provenance)
    for epoch in range(1,a.epochs+1):
        sampler_seed=a.seed+epoch;order=np.random.default_rng(sampler_seed).permutation(len(known));losses=[]
        for start in range(0,len(order),a.batch_size):
            batch=[known[int(i)] for i in order[start:start+a.batch_size]];keys=torch.stack([r['keys'] for r in batch]).to(a.device);target=torch.stack([r['target'] for r in batch]).to(a.device);weights=torch.stack([r['weights'] for r in batch]).to(a.device);optimizer.zero_grad();prediction=head(keys);terms=[offset_loss(p,y,w) for p,y,w in zip(prediction,target,weights)];loss=torch.stack(terms).mean()
            if not torch.isfinite(loss):raise ValueError('Nonfinite offset loss')
            loss.backward();optimizer.step();losses.extend(float(t.detach()) for t in terms)
        record=dict(epoch=epoch,smooth_l1=float(np.mean(losses)),sampler_seed=sampler_seed,images=len(known));history.append(record);save(output/'history.json',history);print(json.dumps(record),flush=True)
        state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};checkpoint=dict(kind='proposal_center_offset_1011',head_state=state,head_state_sha256=state_sha256(state),initial_state_sha256=initial_hash,config=config,target='marked_box_center_offset',crop_size=224,base_checkpoint=str(base.resolve()),base_checkpoint_sha256=base_hash,epoch=epoch,args=vars(a),**provenance);torch.save(checkpoint,output/'last.pt')
    if state_sha256(proposal_state)!=proposal_hash or any(v.requires_grad or v.grad is not None for v in proposal_state.values()):raise ValueError('Original proposal changed during offset training')
    save(output/'summary.json',dict(**history[-1],seconds=time.monotonic()-started,initial_state_sha256=initial_hash,unassigned_box_count=provenance['unassigned_box_count'],selection=provenance['selection']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',default='runs/guided_1011_combined_seed42/best.pt');p.add_argument('--cache',default='runs/proposal_visit_1011/train_feature_cache.pt');p.add_argument('--output',required=True);p.add_argument('--architecture',choices=['linear','spatial'],required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--epochs',type=int,default=15);p.add_argument('--lr',type=float,default=1e-3);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--relevance-power',type=int,choices=[0,1,2],default=0);p.add_argument('--max-offset-crops',type=float,default=1.);p.add_argument('--device',default='cuda:1');a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.max_offset_crops<=0:p.error('epochs/batch-size/offsetbounds must be positive')
    train(a)
