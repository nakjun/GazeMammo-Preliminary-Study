"""Twelve predeclared shifted-geometry NMS controls on fixedp1 Spatial offsets."""
import argparse,csv,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'center_offset_middle_nms_state.json';state=dict(status='running',stages=[],strengths=[.5,.6],budgets=[12,16],nms_thresholds=[.15,.2,.25],physical_crop_size=224,patch_budget=16,matched_control_nms=.1,head='Fixedp1Spatial epoch15',validation_status='Exploratory fixedval93 global parameter calibration',forced_roi_path_nll_reported=False)
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
                if i not in ids:raise ValueError('Nonval strength record')
                rect=rectangles(t['coordinates'],np.array(t['image_size']),224);rows.append(dict(policy=t['policy'],image_id=i,seed=t['seed'],actual50=float(np.mean([coverage(rect,b).max()>=.5 for b in boxes[i]]))));settings[t['policy']]=dict(budget=t['requested_roi_budget'],strength=t['offset_strength'],nms=t['max_crop_iou'])
        frame=pd.DataFrame(rows)
        for _,group in frame.groupby(['policy','seed']):
            if set(group.image_id)!=ids or len(group)!=93:raise ValueError('Incomplete/duplicated policy cohort')
        return frame.groupby(['policy','image_id']).actual50.mean().groupby('policy').mean().sort_values(ascending=False),settings
    try:
        from src.proposal_utility_1011 import file_sha256
        prerequisite=json.loads((root/'center_offset_fine_strength_state.json').read_text())
        if prerequisite['status']!='complete' or max(prerequisite['quick_physical50'].values())>=.85:raise ValueError('Middle NMS stage requires completed fine-strength strict failure')
        generated=root/'center_offset_middle_nms_seed42_new';head=root/'center_offset_spatial_p1_seed42/last.pt';head_hash=file_sha256(head);base_hash=file_sha256(ROOT/'runs/guided_1011_combined_seed42/best.pt');stage('center_offset_middle_nms_generation','proposal_center_offset_1011.py',['--output',generated,'--device',a.device,'--offset-checkpoint',head,'--strengths',.5,.6,'--nms-thresholds',.15,.2,.25,'--budgets',12,16,'--seeds',42]);combined=root/'center_offset_middle_nms_seed42';combined.mkdir(parents=True,exist_ok=False);reference=root/'center_offset_fine_strength_seed42/traces.jsonl';reused=[]
        with (combined/'traces.jsonl').open('w') as sink:
            with (generated/'traces.jsonl').open() as source:
                for line in source:sink.write(line)
            with reference.open() as source:
                for line in source:
                    t=json.loads(line)
                    if t['offset_strength'] not in (.5,.6) or t['requested_roi_budget'] not in (12,16):continue
                    if t['checkpoint_sha256']!=base_hash or t['offset_checkpoint_sha256']!=head_hash or t['max_crop_iou']!=.1 or t['seed']!=42:raise ValueError('Matched halfstrength source proof mismatch')
                    t['reused_from']=str(reference.resolve());sink.write(json.dumps(t,allow_nan=False)+'\n');reused.append(dict(image_id=t['image_id'],policy=t['policy']))
        (combined/'reuse_provenance.json').write_text(json.dumps(dict(new_configurations=12,reused_configurations=4,reused_paths=len(reused),source_trace_sha256=file_sha256(reference),offset_checkpoint_sha256=head_hash,proof=reused),indent=2));table,settings=physical_summary(combined/'traces.jsonl');table.to_csv(root/'center_offset_middle_nms_quick_physical_summary.csv');state['quick_physical50']=table.to_dict();save()
        stage('center_offset_middle_nms_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',combined/'traces.jsonl','--output',root/'center_offset_middle_nms_seed42_posthoc','--baseline','O_spatial_p1_s0p5_K12_nms0p1','--crop-size',224]);qualify=[(k,float(v)) for k,v in table.items() if v>=.85];eligible=[x for x in qualify if settings[x[0]]['budget']<=12] or qualify
        if eligible:
            selected=sorted(eligible,key=lambda x:(-x[1],settings[x[0]]['budget'],x[0]))[0][0];config=settings[selected];state['selected_confirmation']=dict(policy=selected,seed42_actual50=float(table[selected]),**config);save();confirm=root/'center_offset_middle_nms_confirmation';stage('center_offset_middle_nms_confirmation_generation','proposal_center_offset_1011.py',['--output',confirm,'--device',a.device,'--offset-checkpoint',head,'--strengths',0,config['strength'],'--nms-thresholds',config['nms'],'--budgets',config['budget'],'--seeds',42,43,44]);confirm_table,_=physical_summary(confirm/'traces.jsonl');state['confirmation_physical50']=confirm_table.to_dict();confirm_table.to_csv(root/'center_offset_middle_nms_confirmation_quick_physical_summary.csv');save();baseline=f'O_spatial_p1_s0p0_K{config["budget"]}_nms{str(config["nms"]).replace(".","p")}';stage('center_offset_middle_nms_confirmation_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',confirm/'traces.jsonl','--output',root/'center_offset_middle_nms_confirmation_posthoc','--baseline',baseline,'--crop-size',224])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
