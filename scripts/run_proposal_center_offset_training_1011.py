"""Durable train-only fixedlast15 center-offset heads."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'center_offset_training_state.json';state=dict(status='running',stages=[],physical_crop_size=224,offset_bound_pixels=224,selection='Last epoch15 train-only, chosen a priori')
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    try:
        for arch in ('linear','spatial'):
            cmd=[sys.executable,str(ROOT/'scripts/train_proposal_center_offset_1011.py'),'--output',str(root/f'center_offset_{arch}_seed42'),'--architecture',arch,'--seed','42','--epochs','15','--device',a.device]
            entry=dict(architecture=arch,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
            with (root/f'center_offset_{arch}_training.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
            if result.returncode:raise RuntimeError(f'{arch} training failed: {result.returncode}')
        state['status']='training_complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
