"""Separate post-generation observed-prefix likelihood diagnostic; no box inputs."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.data import load_gazes,whole_tensor,patch_tensor
from src.trajectory_1011 import segments,window_sample
from src.guided_trajectory_1011 import candidate_geometry
from src.proposal_policy_1011 import policy_distribution
from src.proposal_utility_1011 import load_utility_head,predict_utility
from guided_trajectory_1011 import load_guided,base_path
from train_attention_1011 import digest


@torch.no_grad()
def evaluate(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(a.threads)
    started=time.monotonic();model,checkpoint=load_guided(a.checkpoint,a.base_checkpoint_override);model=model.to(a.device).eval()
    checkpoint_hash=digest(base_path(a.checkpoint));utility=None
    if a.utility_checkpoint:utility,utility_checkpoint=load_utility_head(a.utility_checkpoint,checkpoint_hash,a.device)
    frame=pd.read_csv(a.manifest);frame=frame[(frame.split=='val')&(frame.label>=0)].sort_values('image_id')
    if a.image_ids:frame=frame[frame.image_id.isin(a.image_ids)]
    gazes=load_gazes();records=[]
    for _,row in frame.iterrows():
        with Image.open(ROOT/'data/image'/row.file_name) as source:image=source.convert('L')
        parts=[p/image.size for p in segments(gazes[row.file_name],*image.size,32.)]
        sample=window_sample(parts,int(row.image_id),42,0,'legacy16',16,8,.5)
        coords=sample['coords'][None].to(a.device);valid=sample['loss_mask'].to(a.device)
        geometry=candidate_geometry(image,model.grid);memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device))
        vectors=None;n=int(sample['valid'].sum())
        if model.foveal:
            vectors=torch.zeros(1,coords.shape[1],model.patch.in_features,device=a.device)
            if n>1:vectors[:,1:n]=model.visual.patch_vector(torch.stack([patch_tensor(image,p.numpy()*image.size) for p in sample['coords'][:n-1]]).to(a.device))
        proposal=model.proposal(memory['keys']).squeeze(-1);proposal_valid=bool(model.proposal_valid and torch.isfinite(proposal).all());r=proposal.sigmoid() if proposal_valid else torch.zeros_like(proposal)
        roi=predict_utility(utility,memory) if utility is not None else None
        hidden=None;scores={p:[] for p in a.policies};scores['raw']=[]
        gx,gy=model.grid;ids=(coords[0,:,1]*gy).long().clamp(0,gy-1)*gx+(coords[0,:,0]*gx).long().clamp(0,gx-1)
        for t in range(coords.shape[1]):
            # The decoder sees only past coordinates and the immediately previous observed crop.
            raw,hidden,_=model.step_history(memory,coords[:,:t],hidden,vectors[:,t] if vectors is not None else None)
            if not valid[t]:continue
            scores['raw'].append(float(-raw[0,ids[t]]))
            for policy in a.policies:
                dist=policy_distribution(raw,r,geometry,policy,a.alpha,a.tau,model.rho,model.ablation,proposal_valid,roi if policy in ('P3','P4') else None)
                value=-dist['q'][0,ids[t]].clamp_min(torch.finfo(raw.dtype).tiny).log()
                if not torch.isfinite(value):raise ValueError('Nonfinite diagnostic NLL')
                scores[policy].append(float(value))
        for policy,values in scores.items():
            proposal_label=a.proposal_label or (utility_checkpoint['target']+'_mse' if utility is not None else '')
            name=policy+(('_'+proposal_label) if utility is not None and policy in ('P3','P4') else '')
            records.append(dict(image_id=int(row.image_id),patient_key=row.patient_key,policy=name,nll=float(np.mean(values)),valid_steps=len(values)))
        print(json.dumps(dict(image_id=int(row.image_id),completed_images=len(records)//len(scores))),flush=True)
    table=pd.DataFrame(records);table.to_csv(output/'per_image.csv',index=False)
    summary=dict(images=len(frame),image_macro_nll=table.groupby('policy').nll.mean().to_dict(),seconds=time.monotonic()-started,checkpoint_sha256=checkpoint_hash,base_checkpoint_sha256=checkpoint['base_checkpoint_sha256'],utility_checkpoint_sha256=digest(a.utility_checkpoint) if a.utility_checkpoint else None,manifest_sha256=digest(a.manifest),gaze_source=str(ROOT/'data/gaze_data/merged_gaze_data.csv'),gaze_sha256=digest(ROOT/'data/gaze_data/merged_gaze_data.csv'),observed_gaze_used=True,annotations_used=False,labels_used_for_likelihood=False,protocol='Separate legacy16 observed-prefix diagnostic; no future coordinates or crops enter decoder; P5 heuristic NLL not applicable',training_patient_keys=checkpoint['training_patient_keys'],selection_patient_keys=checkpoint['selection_patient_keys'],validation_status='Exploratory previously selected validation; not independent holdout')
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));(output/'config.json').write_text(json.dumps(vars(a),indent=2));print(json.dumps(summary['image_macro_nll']),flush=True)


def parser():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default=str(ROOT/'runs/guided_1011_combined_seed42/best.pt'));p.add_argument('--base-checkpoint-override');p.add_argument('--utility-checkpoint');p.add_argument('--proposal-label',help='Defaults to loaded head target plus _mse');p.add_argument('--manifest',default=str(ROOT/'manifests/all.csv'));p.add_argument('--output',required=True);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=4);p.add_argument('--policies',nargs='+',choices=('P0','P1','P2','P3','P4'),default=['P0','P1','P2','P3','P4']);p.add_argument('--alpha',type=float,default=.125);p.add_argument('--tau',type=float,default=.5);p.add_argument('--image-ids',nargs='+',type=int);return p
if __name__=='__main__':evaluate(parser().parse_args())
