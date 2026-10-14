"""Exploratory fixedhead offset strength/NMS calibration; no training changes."""
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'center_offset_spatial_calibration_state.json';state=dict(status='running',stages=[],physical_crop_size=224,patch_budget=16,validation_status='Exploratory previously selected fixedval93; calibration selected on validation capture rate',head_selection='FrozenSpatial epoch15, no retraining or epoch selection',selection_rule='Prefer any qualifyingK<=12, highest image-macroactual50 then lowerK; otherwise highest qualifyingK16',forced_roi_path_nll_reported=False)
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    def physical_summary(path):
        import numpy as np
        import pandas as pd
        from evaluate_proposal_policy_1011 import rectangles,coverage
        with (ROOT/'manifests/trajectory.csv').open() as f:ids={int(r['image_id']) for r in csv.DictReader(f) if r['split']=='val'}
        boxes={i:[] for i in ids}
        for r in json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations']:
            if int(r['image_id']) in boxes:boxes[int(r['image_id'])].append(r['bbox'])
        records=[];settings={}
        with path.open() as f:
            for line in f:
                t=json.loads(line);image_id=int(t['image_id'])
                if image_id not in ids:raise ValueError('Refusing nonval calibration image')
                rect=rectangles(t['coordinates'],np.array(t['image_size']),224);value=float(np.mean([coverage(rect,b).max()>=.5 for b in boxes[image_id]]));records.append(dict(policy=t['policy'],image_id=image_id,seed=t['seed'],actual_recall50=value));settings[t['policy']]=dict(budget=t['requested_roi_budget'],strength=t['offset_strength'],nms=t['max_crop_iou'])
        table=pd.DataFrame(records).groupby(['policy','image_id']).actual_recall50.mean().groupby('policy').mean().sort_values(ascending=False)
        return table,settings
    sys.path.insert(0,str(ROOT/'scripts'))
    try:
        from src.proposal_utility_1011 import file_sha256
        head=root/'center_offset_spatial_seed42/last.pt';head_hash=file_sha256(head);base_hash=file_sha256(ROOT/'runs/guided_1011_combined_seed42/best.pt');new_output=root/'center_offset_spatial_calibration_seed42_new';combined=root/'center_offset_spatial_calibration_seed42'
        stage('center_offset_spatial_calibration_generation','proposal_center_offset_1011.py',['--output',new_output,'--device',a.device,'--offset-checkpoint',head,'--strengths',.25,.5,.75,'--nms-thresholds',.1,.3,.5,'--budgets',8,12,16,'--seeds',42,'--skip-half-nms03'])
        combined.mkdir(parents=True,exist_ok=False);reused=[];count=0
        with (combined/'traces.jsonl').open('w') as sink:
            with (new_output/'traces.jsonl').open() as src:
                for line in src:sink.write(line);count+=1
            source_path=root/'center_offset_spatial_pilot_seed42/traces.jsonl'
            with source_path.open() as src:
                for line in src:
                    t=json.loads(line)
                    if t['offset_strength'] not in (0,.5) or t['requested_roi_budget'] not in (8,12,16):continue
                    if t['offset_checkpoint_sha256']!=head_hash or t['checkpoint_sha256']!=base_hash or t['max_crop_iou']!=.3 or t['seed']!=42:raise ValueError('Reused source checkpoint/config proof mismatch')
                    old_policy=t['policy']
                    if t['offset_strength']==.5:t['policy']+='_nms0p3'
                    t.update(reused_from=str(source_path.resolve()),reused_source_policy=old_policy,gaze_component_used=bool(any(not f for f in t['reservation_flags'])),conditional_gaze_steps=sum(not f for f in t['reservation_flags']),forced_roi_path_nll_reported=False)
                    sink.write(json.dumps(t,allow_nan=False)+'\n');count+=1;reused.append(dict(image_id=t['image_id'],policy=t['policy'],original_policy=old_policy))
        reuse=dict(generated_paths=count-len(reused),reused_paths=len(reused),total_paths=count,new_configurations=24,reused_calibration_configurations=3,reused_zero_control_configurations=3,source_trace_sha256=file_sha256(source_path),head_checkpoint_sha256=head_hash,base_checkpoint_sha256=base_hash,source_proof=reused);(combined/'reuse_provenance.json').write_text(json.dumps(reuse,indent=2));state['reuse']=reuse;save()
        table,settings=physical_summary(combined/'traces.jsonl');table.to_csv(root/'center_offset_spatial_calibration_quick_physical_summary.csv');state['quick_physical50']=table.to_dict();save()
        qualifying=[(policy,float(value)) for policy,value in table.items() if value>=.85 and settings[policy]['strength']>0];eligible=[x for x in qualifying if settings[x[0]]['budget']<=12] or qualifying
        if eligible:
            selected=sorted(eligible,key=lambda x:(-x[1],settings[x[0]]['budget'],x[0]))[0][0];config=settings[selected];state['selected_confirmation']=dict(policy=selected,seed42_actual50=float(table[selected]),**config);save()
            confirm=root/'center_offset_spatial_confirmation';stage('center_offset_spatial_confirmation_generation','proposal_center_offset_1011.py',['--output',confirm,'--device',a.device,'--offset-checkpoint',head,'--strengths',0,config['strength'],'--max-iou',config['nms'],'--policy-nms-label','--budgets',config['budget'],'--seeds',42,43,44]);confirm_table,_=physical_summary(confirm/'traces.jsonl');confirm_table.to_csv(root/'center_offset_spatial_confirmation_quick_physical_summary.csv');state['confirmation_physical50']=confirm_table.to_dict();save()
            references={}
            with (combined/'traces.jsonl').open() as source:
                for line in source:
                    t=json.loads(line)
                    if t['policy']==selected:references[int(t['image_id'])]=(t['coordinates'],t['source_anchor_indices'])
            repeats=0;seed_paths={}
            with (confirm/'traces.jsonl').open() as source:
                for line in source:
                    t=json.loads(line)
                    if t['policy']!=selected:continue
                    image_id=int(t['image_id']);path=(t['coordinates'],t['source_anchor_indices']);seed_paths.setdefault(image_id,[]).append(path)
                    if t['seed']==42:
                        if path!=references[image_id]:raise ValueError('Selected confirmation seed42 coordinate/source-index repeat mismatch')
                        repeats+=1
            if repeats!=93:raise ValueError('Confirmation seed42 repeat cohort mismatch')
            all_reserved_identical=None
            if config['budget']==16:
                all_reserved_identical=all(len(paths)==3 and paths[0]==paths[1]==paths[2] for paths in seed_paths.values())
                if not all_reserved_identical:raise ValueError('Fully reserved K16 changed coordinates across generation seeds')
            state['confirmation_repeat_proof']=dict(seed42_exact_repeat_images=repeats,fully_reserved_K16_all_three_seeds_identical=all_reserved_identical,gaze_component_used=config['budget']<16);save()

        stage('center_offset_spatial_calibration_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',combined/'traces.jsonl','--output',root/'center_offset_spatial_calibration_seed42_posthoc','--baseline','O_spatial_s0p0_K8','--crop-size',224])
        if eligible:
            nms_name=str(config['nms']).replace('.','p');baseline=f'O_spatial_s0p0_K{config["budget"]}_nms{nms_name}';stage('center_offset_spatial_confirmation_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',confirm/'traces.jsonl','--output',root/'center_offset_spatial_confirmation_posthoc','--baseline',baseline,'--crop-size',224])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
