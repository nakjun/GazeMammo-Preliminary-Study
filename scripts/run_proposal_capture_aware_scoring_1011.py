"""Twelve fixed capture-aware ranking policies with exactalpha1 controls."""
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'capture_aware_scoring_state.json';state=dict(status='running',stages=[],physical_crop_size=224,patch_budget=16,offset_head='FixedSpatialp1 epoch15 strength.5',nms=.1,alphas=[.5,.75],validation_status='Exploratory fixedval93 policy calibration, no validation training',forced_roi_path_nll_reported=False)
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    def physical_summary(path):
        import numpy as np,pandas as pd
        from evaluate_proposal_policy_1011 import rectangles,coverage
        with (ROOT/'manifests/trajectory.csv').open() as f:ids={int(r['image_id']) for r in csv.DictReader(f) if r['split']=='val'}
        boxes={i:[] for i in ids}
        for r in json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations']:
            if int(r['image_id']) in boxes:boxes[int(r['image_id'])].append(r['bbox'])
        rows=[];settings={}
        with path.open() as source:
            for line in source:
                t=json.loads(line);i=int(t['image_id'])
                if i not in ids:raise ValueError('Nonval capture-aware record')
                actual=rectangles(t['coordinates'],np.array(t['image_size']),224);rows.append(dict(policy=t['policy'],image_id=i,seed=t['seed'],actual50=float(np.mean([coverage(actual,b).max()>=.5 for b in boxes[i]]))));settings[t['policy']]=dict(budget=t['requested_roi_budget'],recipe=t.get('score_recipe'))
        frame=pd.DataFrame(rows)
        for policy,group in frame.groupby('policy'):
            for seed,subgroup in group.groupby('seed'):
                if set(subgroup.image_id)!=ids or len(subgroup)!=93:raise ValueError('Policy/seed cohort incomplete or duplicated')
        table=frame.groupby(['policy','image_id']).actual50.mean().groupby('policy').mean().sort_values(ascending=False);return table,settings
    try:
        from src.proposal_utility_1011 import file_sha256
        smoke=root/'capture_aware_integration_smoke'
        stage('capture_aware_integration_smoke','proposal_capture_aware_offset_1011.py',['--output',smoke,'--device',a.device,'--image-ids',64,73,'--alphas',1,.5,.75])
        proof=[]
        for image_id in (64,73):
            for capture in ('absolute_utility_linear','capture50_linear'):
                for budget in (8,12,16):
                    new=json.loads((smoke/f'C_{capture}_a1p0_K{budget}_image{image_id}_seed42.json').read_text());old=json.loads((root/f'center_offset_spatial_p1_pilot_seed42/O_spatial_p1_s0p5_K{budget}_nms0p1_image{image_id}_seed42.json').read_text())
                    if new['coordinates']!=old['coordinates'] or new['source_anchor_indices']!=old['source_anchor_indices']:raise ValueError('Alpha1 exact originalR realpath control failed')
                    proof.append(dict(image_id=image_id,capture_source=capture,budget=budget,exact_coordinates_and_source_indices=True))
        state['alpha1_controls']=proof;save()
        generated=root/'capture_aware_scoring_seed42_new';stage('capture_aware_scoring_generation','proposal_capture_aware_offset_1011.py',['--output',generated,'--device',a.device,'--alphas',.5,.75]);combined=root/'capture_aware_scoring_seed42';combined.mkdir(parents=True,exist_ok=False);reference=root/'center_offset_spatial_p1_pilot_seed42/traces.jsonl';base_hash=file_sha256(ROOT/'runs/guided_1011_combined_seed42/best.pt');offset_hash=file_sha256(root/'center_offset_spatial_p1_seed42/last.pt');reused=[]
        with (combined/'traces.jsonl').open('w') as sink:
            with (generated/'traces.jsonl').open() as source:
                for line in source:sink.write(line)
            with reference.open() as source:
                for line in source:
                    t=json.loads(line)
                    if t['offset_strength']!=.5:continue
                    if t['checkpoint_sha256']!=base_hash or t['offset_checkpoint_sha256']!=offset_hash or t['max_crop_iou']!=.1 or t['seed']!=42:raise ValueError('MatchedR control checkpoint/source proof mismatch')
                    t['reused_from']=str(reference.resolve());sink.write(json.dumps(t,allow_nan=False)+'\n');reused.append(dict(image_id=t['image_id'],policy=t['policy']))
        (combined/'reuse_provenance.json').write_text(json.dumps(dict(new_configurations=12,reused_R_configurations=3,reused_paths=len(reused),source_trace_sha256=file_sha256(reference),source_head_sha256=offset_hash,proof=reused),indent=2));table,settings=physical_summary(combined/'traces.jsonl');table.to_csv(root/'capture_aware_scoring_quick_physical_summary.csv');state['quick_physical50']=table.to_dict();save();qualify=[(k,float(v)) for k,v in table.items() if settings[k]['recipe'] is not None and v>=.85];eligible=[x for x in qualify if settings[x[0]]['budget']<=12] or qualify
        if eligible:
            selected=sorted(eligible,key=lambda x:(-x[1],settings[x[0]]['budget'],x[0]))[0][0];config=settings[selected];state['selected_confirmation']=dict(policy=selected,seed42_actual50=float(table[selected]),**config);save();confirm=root/'capture_aware_scoring_confirmation';recipe=config['recipe']
            stage('capture_aware_scoring_confirmation_generation','proposal_capture_aware_offset_1011.py',['--output',confirm,'--device',a.device,'--capture-sources',recipe['capture_source'],'--alphas',1,recipe['alpha'],'--budgets',config['budget'],'--seeds',42,43,44]);confirmation,_=physical_summary(confirm/'traces.jsonl');state['confirmation_physical50']=confirmation.to_dict();confirmation.to_csv(root/'capture_aware_scoring_confirmation_quick_physical_summary.csv');save()
        stage('capture_aware_scoring_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',combined/'traces.jsonl','--output',root/'capture_aware_scoring_seed42_posthoc','--baseline','O_spatial_p1_s0p5_K12_nms0p1','--crop-size',224])
        if eligible:
            baseline=f'C_{recipe["capture_source"]}_a1p0_K{config["budget"]}';stage('capture_aware_scoring_confirmation_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',confirm/'traces.jsonl','--output',root/'capture_aware_scoring_confirmation_posthoc','--baseline',baseline,'--crop-size',224])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
