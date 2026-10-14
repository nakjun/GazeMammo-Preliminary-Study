"""Actual K14 ROI reservations plus two original conditional gaze actions."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);state_path=root/'center_offset_K14_confirmation_state.json';state=dict(status='running',stages=[],physical_crop_size=224,patch_budget=16,reserved_roi_steps=14,conditional_gaze_steps=2,offset_strength=.5,nms=.15,validation_status='Exploratory selected globalK14 from completedK16 prefixes; canonical human gaze NLL remains separate')
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    try:
        prior=json.loads((root/'center_offset_middle_nms_state.json').read_text());prefix=json.loads((root/'center_offset_selected_prefix_summary.json').read_text())
        if prior['status']!='complete' or prefix['actual50_image_macro']['14']<.85:raise ValueError('K14 requires completed qualifyingK16 confirmation andprefix')
        output=root/'center_offset_K14_confirmation';stage('center_offset_K14_confirmation_generation','proposal_center_offset_1011.py',['--output',output,'--device',a.device,'--offset-checkpoint',root/'center_offset_spatial_p1_seed42/last.pt','--strengths',0,.5,'--nms-thresholds',.15,'--budgets',14,'--seeds',42,43,44])
        references={}
        with (root/'center_offset_middle_nms_confirmation/traces.jsonl').open() as source:
            for line in source:
                t=json.loads(line)
                if t['offset_strength']==.5:references[(int(t['image_id']),int(t['seed']))]=(t['coordinates'][:14],t['source_anchor_indices'][:14],t['offset_checkpoint_sha256'],t['checkpoint_sha256'])
        proofs=[]
        with (output/'traces.jsonl').open() as source:
            for line in source:
                t=json.loads(line)
                if t['offset_strength']!=.5:continue
                key=(int(t['image_id']),int(t['seed']));actual=(t['coordinates'][:14],t['source_anchor_indices'][:14],t['offset_checkpoint_sha256'],t['checkpoint_sha256'])
                if key not in references or actual!=references[key]:raise ValueError('K14 first14 coordinate/source-index/hash prefix mismatch')
                if t['reservation_flags']!=[True]*14+[False]*2 or t['conditional_gaze_steps']!=2 or not t['gaze_component_used']:raise ValueError('K14 action role mismatch')
                proofs.append(dict(image_id=key[0],seed=key[1],first14_exact=True,reserved=14,conditional_gaze=2))
        if len(proofs)!=279:raise ValueError('K14 selected prefix proof cohort mismatch')
        state['prefix_proof']=dict(images=93,seeds=[42,43,44],records=len(proofs),all_first14_coordinates_source_indices_and_source_hashes_exact=True);(output/'prefix_proof.json').write_text(json.dumps(dict(**state['prefix_proof'],per_trace=proofs),indent=2));save()
        stage('center_offset_K14_confirmation_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',output/'traces.jsonl','--output',root/'center_offset_K14_confirmation_posthoc','--baseline','O_spatial_p1_s0p0_K14_nms0p15','--crop-size',224]);state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
