"""Durable stages: occupancy MSE control, then utility MSE, no validation."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/proposal_visit_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'utility_training_state.json';state=dict(status='running',selection='last epoch15 a priori',stages=[])
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    save()
    try:
        for target in ('occupancy','utility'):
            output=root/f'{target}_head_seed42';cmd=[sys.executable,str(ROOT/'scripts/train_proposal_utility_1011.py'),'--base',str(ROOT/'runs/guided_1011_combined_seed42/best.pt'),'--output',str(output),'--cache',str(root/'train_feature_cache.pt'),'--target',target,'--seed','42','--epochs','15','--device',a.device]
            stage=dict(target=target,command=cmd,status='running',started=time.time());state['stages'].append(stage);save()
            with (root/f'{target}_training.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            stage.update(returncode=result.returncode,ended=time.time(),status='complete' if result.returncode==0 else 'failed');save()
            if result.returncode:raise RuntimeError(f'{target} training failed: {result.returncode}')
        import torch
        occupancy=torch.load(root/'occupancy_head_seed42/last.pt',map_location='cpu',weights_only=False)
        utility=torch.load(root/'utility_head_seed42/last.pt',map_location='cpu',weights_only=False)
        if occupancy['initial_state_sha256']!=utility['initial_state_sha256']:raise ValueError('Control initial state mismatch')
        state['initial_state_sha256']=utility['initial_state_sha256'];state['status']='training_complete';save()
    except Exception as exc:
        state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
