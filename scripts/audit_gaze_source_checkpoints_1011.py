"""Metadata/lineage audit of preexisting gaze parents, and exact compatible val93 rescoring."""
import argparse,json,sys,time,gc
from pathlib import Path
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.proposal_utility_1011 import state_sha256
from src.spatial import SpatialTrajectory
from src.gaze_nll_adaptation_1011 import initialize,hard_nll
from gaze_nll_adaptation_1011 import load_base,batch
from train_attention_1011 import digest

CANDIDATES=['runs/gaze_focus_seed42/convnext_base_motion/best.pt','runs/gaze_focus_seed42/convnext_base_pretrain/best.pt','runs/gaze_focus_seed42/resnet50_motion/best.pt','runs/spatial_v2_seed42/pretrain_fpn/best.pt','runs/full_1011_seed42/control/best.pt','runs/full_1011_seed42/improved/best.pt','runs/gaze_reference_seed42/hat/best.pt','runs/gaze_reference_seed42/chestsearch/best.pt']


def audit(a):
    torch.set_num_threads(a.threads);frame=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','file_name','patient_key','split']);train=set(frame[frame.split=='train'].patient_key);protected=set(frame[frame.split.isin(['val','test','quarantine'])].patient_key);val=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','patient_key','split']);val=val[val.split=='val'];source_provenance={name:json.loads((Path(path)/'provenance.json').read_text()) for name,path in [('control',a.control_cache),('improved',a.improved_cache)]};rows=[];step_results=[];image_results=[]
    for relative in CANDIDATES:
        path=ROOT/relative
        if not path.exists():continue
        checkpoint=torch.load(path,map_location='cpu',weights_only=False);config=checkpoint.get('model_config',{});kind=checkpoint.get('kind');patients=set(checkpoint.get('training_patient_keys',[]));overlap=sorted(patients&protected);outside=sorted(patients-train);state=checkpoint.get('state',{});visual={k.removeprefix('visual.'):v for k,v in state.items() if k.startswith('visual.')};visual_sha=state_sha256(visual) if visual else None;compatible=[name for name,p in source_provenance.items() if visual_sha==p['visual_state_sha256']];eligible=bool(patients and not overlap and not outside and config.get('grid')==[24,48]);selection=checkpoint.get('selection_patient_keys',[]);row=dict(path=relative,checkpoint_sha256=digest(path),kind=kind,epoch=checkpoint.get('epoch'),model_config=config,training_patients=len(patients),selection_patients=len(selection),training_protected_overlap=overlap,training_outside_current_allowed=outside,eligible_provenance=eligible,visual_state_sha256=visual_sha,compatible_visual_caches=compatible,lineage_initializer=checkpoint.get('args',{}).get('init'),old_published_nll_not_compared=True,rescore_status='pending' if eligible and compatible else ('requires_new_encoder_features' if eligible else 'provenance_or_grid_ineligible'))
        if eligible and compatible and kind in ('spatial_v2','trajectory_1011'):
            cache_path=Path(a.control_cache if compatible[0]=='control' else a.improved_cache);records=[torch.load(cache_path/f'{int(r.image_id)}.pt',map_location='cpu',weights_only=False) for _,r in val.iterrows()]
            if any(r['split']!='val' or r['patient_key']!=str(val.set_index('image_id').loc[int(r['image_id'])].patient_key) for r in records):raise ValueError('Fixedval93 source patient mismatch')
            if kind=='spatial_v2':model=SpatialTrajectory(**config,pretrained=False);model.load_state_dict(state,strict=True)
            else:model,_=load_base(path)
            model.eval();adapter=initialize(model).eval();values=[];started=time.monotonic()
            with torch.no_grad():
                for start in range(0,93,a.batch_size):
                    group=records[start:start+a.batch_size];features,coords,mask,vectors=batch(group,'cpu');memory=adapter.maps_features(features);raw=model.teacher_logits(memory,coords,vectors if model.foveal else None);gx,gy=model.grid;targets=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1);nll=-raw.log_softmax(-1).gather(-1,targets[:,:,None]).squeeze(-1)
                    for i,record in enumerate(group):
                        image_nll=float(nll[i][mask[i]].mean());values.append(image_nll);image_results.append(dict(checkpoint=relative,image_id=int(record['image_id']),patient_key=record['patient_key'],nll=image_nll))
                        step_results.extend(dict(checkpoint=relative,image_id=int(record['image_id']),step=t,nll=float(nll[i,t])) for t in torch.nonzero(mask[i],as_tuple=False).flatten().tolist())
            row.update(rescore_status='complete',exact_legacy16_val93_nll=float(sum(values)/len(values)),rescore_seconds=time.monotonic()-started,cache_provenance_sha256=digest(cache_path/'provenance.json'));del model,adapter,records
        elif eligible and not compatible:row['runtime_estimate']='New val93 whole-image and canonical15 preceding-crop encoder features required (~1,488crop/whole forwards); actual runtime unknown, benchmark2images before extraction. No GPU extraction authorized by this metadata audit.'
        rows.append(row);print(json.dumps({k:v for k,v in row.items() if k in ('path','eligible_provenance','compatible_visual_caches','rescore_status','exact_legacy16_val93_nll')}),flush=True);del checkpoint,state,visual;gc.collect()
    completed=[r for r in rows if r['rescore_status']=='complete'];result=dict(protocol='Preexisting gaze-only parent metadata and exactfixedval93 hardNLL; originalcanonical16/min32/grid24x48/masks unchanged. All102val/test150/quarantine193 patients protected against TRAIN overlap; existing selection onval disclosed, not independent. Old published numbers on other targets/cohorts are not compared. No new training/CNN extraction/protected gaze payloads/boxes.',candidates=rows,best_completed=min(completed,key=lambda r:r['exact_legacy16_val93_nll']) if completed else None,cache_sources=source_provenance)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);(output/'metrics.json').write_text(json.dumps(result,indent=2));pd.DataFrame(image_results).to_csv(output/'per_image.csv',index=False);pd.DataFrame(step_results).to_csv(output/'per_step.csv',index=False);lines=['# Existing gaze parent checkpoint audit','',result['protocol'],'','| Parent | Eligible | Cache compatible | Exact val93 NLL |','|---|---|---|---:|']
    for r in rows:lines.append(f"| {r['path']} | {r['eligible_provenance']} | {','.join(r['compatible_visual_caches']) or 'new encoder needed'} | {r.get('exact_legacy16_val93_nll','not rescored')} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n');return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--control-cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--improved-cache',default='runs/gaze_nll_adaptation_trials_1011_v2/frozen_visual_cache');p.add_argument('--output',default='runs/dual_target_1011/source_checkpoint_audit');p.add_argument('--threads',type=int,default=4);p.add_argument('--batch-size',type=int,default=4);audit(p.parse_args())
