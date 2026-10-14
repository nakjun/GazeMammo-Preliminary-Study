"""Matched canonical/stratified observed-path trials with durable stage records."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool-cache',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--wait-for-pool',action='store_true');a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=False);state=dict(state='running',started=time.time(),stages=[])
    def save():
        temporary=output/'status.tmp';temporary.write_text(json.dumps(state,indent=2));temporary.replace(output/'status.json')
    if a.wait_for_pool:
        state['state']='waiting_for_pool';save()
        while True:
            status_path=Path(a.pool_cache)/'status.json'
            if status_path.exists():
                pool_status=json.loads(status_path.read_text());state['pool_status']=pool_status;save()
                if pool_status.get('state')=='complete':break
                if pool_status.get('state')=='failed':state['state']='failed';state['error']='Pool extraction failed';save();return 2
            time.sleep(10)
        state['state']='running';save()
    for mode in ('canonical','augmented'):
        command=[sys.executable,'scripts/train_gaze_path_augmentation_1011.py','--pool-cache',a.pool_cache,'--output',str(output/mode),'--mode',mode,'--device',a.device,'--epochs','15','--batch-size','8','--seed','42','--lr','0.00003']
        stage=dict(name=mode,command=command,state='running',started=time.time());state['stages'].append(stage);save()
        with (output/f'{mode}.log').open('w') as sink:result=subprocess.run(command,cwd=ROOT,stdout=sink,stderr=subprocess.STDOUT)
        stage.update(return_code=result.returncode,state='complete' if result.returncode==0 else 'failed',seconds=time.time()-stage['started']);save()
        if result.returncode:state['state']='failed';save();return result.returncode
    state.update(state='complete',seconds=time.time()-state['started']);save();return 0
if __name__=='__main__':raise SystemExit(main())
