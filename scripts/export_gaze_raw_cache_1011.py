"""Export causal observed-prefix raw predictions from frozen visual caches."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from gaze_nll_adaptation_1011 import cache_records,load_base,batch,apply_cache_resolution
from src.gaze_nll_adaptation_1011 import initialize
from train_attention_1011 import digest
from guided_trajectory_1011 import base_path


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--hidden',action='store_true');a=p.parse_args();a.device='cpu';torch.set_num_threads(a.threads)
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise ValueError('Preserve existing raw cache')
    records,provenance=cache_records(a);base,checkpoint=load_base(a.base);model=apply_cache_resolution(initialize(base),provenance).eval();export=[];started=time.monotonic()
    for start in range(0,len(records),a.batch_size):
        group=records[start:start+a.batch_size];f,coords,mask,vectors=batch(group,'cpu');memory=model.maps_features(f);raw=model.teacher_logits(memory,coords,vectors).cpu();causal_hidden=None
        if a.hidden:
            hidden=None;states=[]
            for t in range(coords.shape[1]):
                _,hidden,_=model.step_history(memory,coords[:,:t],hidden,vectors[:,t] if model.foveal else None);states.append(hidden[-1].cpu())
            causal_hidden=torch.stack(states,1)
        gx,gy=model.grid;target=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1)
        if not torch.isfinite(raw).all():raise ValueError('Nonfinite raw logp')
        for i,r in enumerate(group):
            record=dict(image_id=int(r['image_id']),patient_key=r['patient_key'],split=r['split'],raw_logp=raw[i],coords=coords[i].cpu(),valid=mask[i].cpu(),loss_mask=mask[i].cpu(),target=target[i].cpu())
            if causal_hidden is not None:record['hidden']=causal_hidden[i]
            export.append(record)
        print(json.dumps(dict(completed=len(export),total=len(records))),flush=True)
    payload=dict(kind='gaze_raw_legacy16_cache_1011',base_checkpoint=str(base_path(a.base).resolve()),base_checkpoint_sha256=digest(base_path(a.base)),grid=list(model.grid),records=export,cache_provenance=provenance,training_patient_keys=checkpoint['training_patient_keys'],selection_patient_keys=checkpoint['selection_patient_keys'],protocol='Exactlegacy16 rawconditionalteacher logp; onlypreviousobservedcoords/crops; noannotations/test',seconds=time.monotonic()-started)
    temporary=output.with_suffix('.tmp');torch.save(payload,temporary);temporary.replace(output);print(json.dumps(dict(output=str(output),images=len(export),seconds=payload['seconds'])),flush=True)
if __name__=='__main__':main()
