"""Durable continuous ROI-offset pilot with exact source-grid zero controls."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'center_offset_pilot_state.json';state=dict(status='running',stages=[],physical_crop_size=224,patch_budget=16,gaze_likelihood_grid=[24,48],coordinate_protocol='center_offset_continuous_roi_v1')
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    try:
        import torch
        from src.proposal_center_offset_1011 import load_center_offset_head,predict_center_offsets
        from src.proposal_utility_1011 import file_sha256
        base_hash=file_sha256(ROOT/'runs/guided_1011_combined_seed42/best.pt');cache=torch.load(ROOT/'runs/proposal_visit_1011/train_feature_cache.pt',map_location='cpu',weights_only=False);keys=cache['records'][0]['keys'][None];restore=[]
        for arch in ('linear','spatial'):
            for device in ('cpu',a.device):
                head,ck=load_center_offset_head(root/f'center_offset_{arch}_seed42/last.pt',base_hash,device)
                if ck['epoch']!=15 or ck['validation_read'] or ck['test_read']:raise ValueError('Offset checkpoint selection/provenance mismatch')
                with torch.no_grad():offsets=predict_center_offsets(head,keys.to(device))
                if offsets.shape!=(1,1152,2) or not torch.isfinite(offsets).all() or offsets.abs().max()>1:raise ValueError('Offset checkpoint finite/bounds smoke failed')
                restore.append(dict(architecture=arch,device=device,state_sha256=ck['head_state_sha256'],epoch=ck['epoch']))
        state['restore_smoke']=restore;save();del head,cache,keys,offsets
        smoke=root/'center_offset_integration_smoke'
        stage('center_offset_integration_smoke','proposal_center_offset_1011.py',['--output',smoke,'--device',a.device,'--offset-checkpoint',root/'center_offset_linear_seed42/last.pt','--image-ids',64,73,'--budgets',0,3,8,12,16,'--strengths',0,.5,1,'--seeds',42])
        controls=[]
        for image_id in (64,73):
            for budget in (0,3,8,12,16):
                new=json.loads((smoke/f'O_linear_s0p0_K{budget}_image{image_id}_seed42.json').read_text())
                if budget in (0,3):old_path=ROOT/f'runs/proposal_visit_1011/original_sampling/P{0 if budget==0 else 5}_image{image_id}_seed42.json'
                else:old_path=root/f'budget224_seed42/B{budget}_first_crop224_image{image_id}_seed42.json'
                old=json.loads(old_path.read_text())
                if new['coordinates']!=old['coordinates'] or new['source_anchor_indices']!=old['indices']:raise ValueError(f'Zero offset exact baseline mismatch {image_id}/{budget}')
                controls.append(dict(image_id=image_id,budget=budget,exact_coordinates_and_indices=True,reference=str(old_path)))
        state['zero_offset_controls']=controls;save()
        for arch in ('linear','spatial'):
            output=root/f'center_offset_{arch}_pilot_seed42'
            stage(f'center_offset_{arch}_pilot_seed42_generation','proposal_center_offset_1011.py',['--output',output,'--device',a.device,'--offset-checkpoint',root/f'center_offset_{arch}_seed42/last.pt','--budgets',8,12,16,'--strengths',0,.5,1,'--seeds',42])
        state['status']='generation_complete';save()
        evaluator=ROOT/'scripts/evaluate_proposal_center_offset_1011.py'
        if not evaluator.exists():state['posthoc_pending']='Separate continuous-coordinate evaluator not yet available';save();return
        for arch in ('linear','spatial'):
            name=f'center_offset_{arch}_pilot_seed42';output=root/name
            stage(name+'_posthoc',evaluator.name,['--traces',output/'traces.jsonl','--output',root/f'{name}_posthoc','--baseline',f'O_{arch}_s0p0_K8','--crop-size',224])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
