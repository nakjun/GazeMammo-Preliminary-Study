"""Actual-image fixed legacy16 rescore; references always use strict past-only prefixes."""
import argparse,gc,json,time,sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.data import whole_tensor,patch_tensor
from src.spatial import SpatialTrajectory
from src.reference_gaze import ReferenceGaze
from train_attention_1011 import digest

def strict(model,memory,coords,vectors):
    if isinstance(model,ReferenceGaze):return torch.stack([model.prefix_logits(memory,coords[:,:t]) for t in range(16)],1)
    hidden=None;out=[];zero=torch.zeros_like(coords[:,0])
    for t in range(16):
        previous=coords[:,t-1] if t else zero;older=coords[:,t-2] if t>1 else zero
        raw,hidden=model.step(memory,previous,older,vectors[:,t] if vectors is not None else None,hidden,t);out.append(raw)
    return torch.stack(out,1)

def run(a):
    torch.set_num_threads(4);audit=json.loads(Path(a.audit).read_text());candidates=[r for r in audit['candidates'] if r['eligible_provenance'] and not r['compatible_visual_caches']];out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    frame=pd.read_csv(ROOT/'manifests/trajectory.csv');frame=frame[frame.split=='val'];assert len(frame)==93 and frame.image_id.is_unique
    summaries=[];images=[];steps=[]
    for candidate in candidates:
        path=ROOT/candidate['path'];assert digest(path)==candidate['checkpoint_sha256'];checkpoint=torch.load(path,map_location='cpu',weights_only=False);config=checkpoint['model_config'];reference=checkpoint['kind'].startswith('reference')
        model=(ReferenceGaze(**config,pretrained=False) if reference else SpatialTrajectory(**config,pretrained=False));model.load_state_dict(checkpoint['state'],strict=True);model.to(a.device).eval();del checkpoint
        proofs=[];values=[];started=time.monotonic()
        with torch.no_grad():
            for _,row in frame.iloc[:a.limit or 93].iterrows():
                record=torch.load(Path(a.source_cache)/f'{int(row.image_id)}.pt',map_location='cpu',weights_only=False);assert record['split']=='val' and record['patient_key']==row.patient_key
                coords=record['coords'][None].to(a.device);mask=record['mask'].bool().to(a.device) if 'mask' in record else record['loss_mask'].bool().to(a.device)
                assert coords.shape==(1,16,2) and torch.isfinite(coords).all() and ((coords>=0)&(coords<=1)).all() and mask.any()
                assert digest(ROOT/'data/image'/record['file_name'])==record['image_sha256']
                with Image.open(ROOT/'data/image'/record['file_name']) as source:
                    image=source.convert('L');memory=model.maps(whole_tensor(image,model.input_size)[None].to(a.device));vectors=None
                    if model.foveal:
                        patches=torch.stack([patch_tensor(image,p.numpy()*image.size) for p in record['coords'][:15]]);encoded=torch.cat([model.visual.patch_vector(patches[s:s+8].to(a.device)) for s in range(0,15,8)]);vectors=encoded.new_zeros(1,16,encoded.shape[-1]);vectors[:,1:]=encoded
                raw=strict(model,memory,coords,vectors)
                if len(proofs)<2:
                    changed=coords.clone();changed[:,8:]=1-changed[:,8:];changed_vectors=None if vectors is None else vectors.clone()
                    if changed_vectors is not None:changed_vectors[:,9:]+=1
                    perturbed=strict(model,memory,changed,changed_vectors);causal=float((raw[:,:9]-perturbed[:,:9]).abs().max());assert causal<=1e-5
                    parallel=model.teacher_logits(memory,coords,vectors);parallel_changed=model.teacher_logits(memory,changed,changed_vectors)
                    proofs.append(dict(image_id=int(row.image_id),strict_current_future_invariance_max=causal,parallel_vs_strict_max=float((parallel-raw).abs().max()),parallel_current_future_invariance_max=float((parallel[:,:9]-parallel_changed[:,:9]).abs().max())))
                gx,gy=model.grid;target=(coords[0,:,1]*gy).long().clamp(0,gy-1)*gx+(coords[0,:,0]*gx).long().clamp(0,gx-1);nll=-raw[0].log_softmax(-1).gather(1,target[:,None]).squeeze(1);value=float(nll[mask].mean());values.append(value);images.append(dict(checkpoint=candidate['path'],image_id=int(row.image_id),patient_key=row.patient_key,nll=value));steps.extend(dict(checkpoint=candidate['path'],image_id=int(row.image_id),step=t,nll=float(nll[t])) for t in torch.nonzero(mask).flatten().tolist())
        elapsed=time.monotonic()-started;summaries.append(dict(checkpoint=candidate['path'],checkpoint_sha256=candidate['checkpoint_sha256'],images=len(values),nll=sum(values)/len(values),seconds=elapsed,estimated_val93_seconds=elapsed*93/len(values),causality_proofs=proofs,inference='strict prefix-only loop',complete=len(values)==93));(out/'metrics.json').write_text(json.dumps(dict(results=summaries,protocol='Fixed val93 canonical16/min32/grid24x48; exploratory checkpoint comparison; no training or protected labels.',source_cache_provenance_sha256=digest(Path(a.source_cache)/'provenance.json')),indent=2));pd.DataFrame(images).to_csv(out/'per_image.csv',index=False);pd.DataFrame(steps).to_csv(out/'per_step.csv',index=False);print(json.dumps(summaries[-1]),flush=True);del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit',default='runs/dual_target_1011/source_checkpoint_audit/metrics.json');p.add_argument('--source-cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int,default=0);run(p.parse_args())
