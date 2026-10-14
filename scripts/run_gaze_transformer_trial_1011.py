"""Durable single causal-transformer residual trial."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'scripts/train_gaze_transformer_residual_1011.py','--pool-cache','runs/dual_target_1011/train_gaze_pool_cache','--output',str(output/'trial'),'--device',a.device,'--mode','augmented','--epochs','15','--batch-size','8','--lr','0.0001','--seed','42']
    state=dict(state='running',command=command,started=time.time())
    def save():temporary=output/'status.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(output/'status.json')
    save()
    with (output/'train.log').open('w') as sink:result=subprocess.run(command,cwd=ROOT,stdout=sink,stderr=subprocess.STDOUT)
    state.update(return_code=result.returncode,seconds=time.time()-state['started'],state='complete' if result.returncode==0 else 'failed');save();return result.returncode
if __name__=='__main__':raise SystemExit(main())
