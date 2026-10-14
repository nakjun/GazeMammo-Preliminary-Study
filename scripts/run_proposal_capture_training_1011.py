"""Durable fixedlast15 capture50 head training, no validation selection."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');p.add_argument('--target',choices=['capture50','absolute_utility'],default='capture50');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);prefix='capture' if a.target=='capture50' else 'capture_absolute';state_path=root/f'{prefix}_training_state.json';state=dict(status='running',stages=[],crop_size=224,target=a.target,selection='Last epoch15 chosen a priori; no validation reads')
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    try:
        for arch in ('linear','spatial'):
            cmd=[sys.executable,str(ROOT/'scripts/train_proposal_capture_1011.py'),'--output',str(root/f'{prefix}_{arch}_seed42'),'--architecture',arch,'--target',a.target,'--seed','42','--epochs','15','--device',a.device]
            entry=dict(architecture=arch,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
            with (root/f'{prefix}_{arch}_training.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
            if result.returncode:raise RuntimeError(f'{arch} training failed: {result.returncode}')
        state['status']='training_complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
