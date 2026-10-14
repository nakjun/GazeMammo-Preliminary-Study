"""Both capture50 head generation pilots precede CPU posthoc analysis."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');p.add_argument('--target',choices=['capture50','absolute_utility'],default='capture50');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);prefix='capture' if a.target=='capture50' else 'capture_absolute';state_path=root/f'{prefix}_pilot_state.json';state=dict(status='running',stages=[],crop_size=224,patch_budget=16,output_role='Capture ROI ranking only; original guided conditional gaze branch unchanged',selection='Fixedlast15 train-only heads; exploratory val93 policy pilot')
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    try:
        import torch
        from src.proposal_capture_1011 import load_capture_head,predict_capture
        from src.proposal_utility_1011 import file_sha256
        base_hash=file_sha256(ROOT/'runs/guided_1011_combined_seed42/best.pt');cache=torch.load(ROOT/'runs/proposal_visit_1011/train_feature_cache.pt',map_location='cpu',weights_only=False);keys=cache['records'][0]['keys'][None];restore=[]
        for arch in ('linear','spatial'):
            for device in ('cpu',a.device):
                head,ck=load_capture_head(root/f'{prefix}_{arch}_seed42/last.pt',base_hash,device)
                if ck['epoch']!=15 or ck['validation_read'] or ck['test_read']:raise ValueError('Capture head selection/provenance mismatch')
                with torch.no_grad():scores=predict_capture(head,keys.to(device))
                if scores.shape!=(1,1152) or not torch.isfinite(scores).all() or (scores<0).any() or (scores>1).any():raise ValueError('Capture head restore finite smoke failed')
                restore.append(dict(architecture=arch,device=device,state_sha256=ck['head_state_sha256'],epoch=ck['epoch']))
        state['restore_smoke']=restore;save();del head,cache,keys,scores
        for arch in ('linear','spatial'):
            name=f'{prefix}_{arch}_pilot_seed42';output=root/name
            stage(name+'_generation','proposal_budget_1011.py',['--output',output,'--device',a.device,'--capture-checkpoint',root/f'{prefix}_{arch}_seed42/last.pt','--budgets',0,3,6,8,12,16,'--layouts','first','--seeds',42,'--crop-size',224,'--max-iou',.3])
        for arch in ('linear','spatial'):
            name=f'{prefix}_{arch}_pilot_seed42';output=root/name
            stage(name+'_posthoc','evaluate_proposal_policy_1011.py',['--traces',output/'traces.jsonl','--output',root/f'{name}_posthoc','--baseline','B0_first_crop224','--crop-size',224])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
