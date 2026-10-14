"""TrainR-weighted fixedlast15 Spatial offsets then predefined nine-policy pilots."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'center_offset_weighted_state.json';state=dict(status='running',stages=[],powers=[1,2],physical_crop_size=224,patch_budget=16,validation_status='Exploratory fixedval93 policy selection; no validation data used for training',head_selection='SameSpatial architecture/initialseed42, fixedlast15',forced_roi_path_nll_reported=False)
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    def physical_summary(path):
        import csv,numpy as np,pandas as pd
        from evaluate_proposal_policy_1011 import rectangles,coverage
        with (ROOT/'manifests/trajectory.csv').open() as f:ids={int(r['image_id']) for r in csv.DictReader(f) if r['split']=='val'}
        boxes={i:[] for i in ids}
        for r in json.loads((ROOT/'manifests/instances_clipped.json').read_text())['annotations']:
            if int(r['image_id']) in boxes:boxes[int(r['image_id'])].append(r['bbox'])
        rows=[];settings={}
        with path.open() as f:
            for line in f:
                t=json.loads(line);i=int(t['image_id'])
                if i not in ids:raise ValueError('Nonval weighted pilot record')
                rect=rectangles(t['coordinates'],np.array(t['image_size']),224);rows.append(dict(policy=t['policy'],image_id=i,seed=t['seed'],actual50=float(np.mean([coverage(rect,b).max()>=.5 for b in boxes[i]]))));settings[t['policy']]=dict(power=t.get('relevance_power',0),budget=t['requested_roi_budget'],strength=t['offset_strength'],nms=t['max_crop_iou'])
        table=pd.DataFrame(rows).groupby(['policy','image_id']).actual50.mean().groupby('policy').mean().sort_values(ascending=False);return table,settings
    try:
        import torch
        from src.proposal_utility_1011 import file_sha256
        from src.proposal_center_offset_1011 import load_center_offset_head,predict_center_offsets
        base_hash=file_sha256(ROOT/'runs/guided_1011_combined_seed42/best.pt');p0=torch.load(root/'center_offset_spatial_seed42/last.pt',map_location='cpu',weights_only=False)
        for power in (1,2):stage(f'center_offset_spatial_p{power}_training','train_proposal_center_offset_1011.py',['--output',root/f'center_offset_spatial_p{power}_seed42','--architecture','spatial','--relevance-power',power,'--seed',42,'--epochs',15,'--device',a.device])
        cache=torch.load(ROOT/'runs/proposal_visit_1011/train_feature_cache.pt',map_location='cpu',weights_only=False);keys=cache['records'][0]['keys'][None];restore=[]
        for power in (1,2):
            path=root/f'center_offset_spatial_p{power}_seed42/last.pt'
            for device in ('cpu',a.device):
                head,ck=load_center_offset_head(path,base_hash,device)
                if ck['epoch']!=15 or ck['initial_state_sha256']!=p0['initial_state_sha256'] or ck['relevance_power']!=power or ck['original_proposal_requires_grad'] or ck['validation_read'] or ck['test_read']:raise ValueError('Weighted head initialization/provenance mismatch')
                with torch.no_grad():value=predict_center_offsets(head,keys.to(device))
                if value.shape!=(1,1152,2) or not torch.isfinite(value).all() or value.abs().max()>1:raise ValueError('Weighted checkpoint restore/bounds mismatch')
                restore.append(dict(power=power,device=device,head_state_sha256=ck['head_state_sha256'],original_proposal_state_sha256=ck['original_proposal_state_sha256']))
        state['restore_smoke']=restore;save();del head,cache,keys,value
        for power in (1,2):stage(f'center_offset_spatial_p{power}_pilot_generation','proposal_center_offset_1011.py',['--output',root/f'center_offset_spatial_p{power}_pilot_seed42','--offset-checkpoint',root/f'center_offset_spatial_p{power}_seed42/last.pt','--device',a.device,'--strengths',.5,.75,1,'--nms-thresholds',.1,'--budgets',8,12,16,'--seeds',42])
        combined=root/'center_offset_weighted_pilot_seed42';combined.mkdir(parents=True,exist_ok=False);count=0
        with (combined/'traces.jsonl').open('w') as sink:
            for power in (1,2):
                with (root/f'center_offset_spatial_p{power}_pilot_seed42/traces.jsonl').open() as source:
                    for line in source:sink.write(line);count+=1
            reference=root/'center_offset_spatial_calibration_seed42/traces.jsonl';reused=[]
            with reference.open() as source:
                for line in source:
                    t=json.loads(line)
                    if t['policy']!='O_spatial_s0p5_K12_nms0p1':continue
                    if t['checkpoint_sha256']!=base_hash or t['offset_checkpoint_sha256']!=file_sha256(root/'center_offset_spatial_seed42/last.pt'):raise ValueError('p0 reference source hash mismatch')
                    t['reused_from']=str(reference.resolve());sink.write(json.dumps(t,allow_nan=False)+'\n');count+=1;reused.append(int(t['image_id']))
        (combined/'reuse_provenance.json').write_text(json.dumps(dict(paths=count,reused_p0_reference_images=reused,reference_trace_sha256=file_sha256(reference)),indent=2));table,settings=physical_summary(combined/'traces.jsonl');table.to_csv(root/'center_offset_weighted_quick_physical_summary.csv');state['quick_physical50']=table.to_dict();save()
        qualify=[(k,float(v)) for k,v in table.items() if settings[k]['power']>0 and v>=.85];eligible=[x for x in qualify if settings[x[0]]['budget']<=12] or qualify
        if eligible:
            selected=sorted(eligible,key=lambda x:(-x[1],settings[x[0]]['budget'],x[0]))[0][0];config=settings[selected];state['selected_confirmation']=dict(policy=selected,seed42_actual50=float(table[selected]),**config);save();confirm=root/'center_offset_weighted_confirmation'
            stage('center_offset_weighted_confirmation_generation','proposal_center_offset_1011.py',['--output',confirm,'--offset-checkpoint',root/f'center_offset_spatial_p{config["power"]}_seed42/last.pt','--device',a.device,'--strengths',0,config['strength'],'--nms-thresholds',.1,'--budgets',config['budget'],'--seeds',42,43,44]);confirm_table,_=physical_summary(confirm/'traces.jsonl');state['confirmation_physical50']=confirm_table.to_dict();confirm_table.to_csv(root/'center_offset_weighted_confirmation_quick_physical_summary.csv');save()
        stage('center_offset_weighted_pilot_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',combined/'traces.jsonl','--output',root/'center_offset_weighted_pilot_seed42_posthoc','--baseline','O_spatial_s0p5_K12_nms0p1','--crop-size',224])
        if eligible:
            baseline=f'O_spatial_p{config["power"]}_s0p0_K{config["budget"]}_nms0p1';stage('center_offset_weighted_confirmation_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',confirm/'traces.jsonl','--output',root/'center_offset_weighted_confirmation_posthoc','--baseline',baseline,'--crop-size',224])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
