"""Four controlled regularization trials, preserving epoch-zero eligibility."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    common=[sys.executable,'scripts/gaze_nll_adaptation_1011.py','train','--base','runs/full_1011_seed42/control/best.pt','--cache','runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache','--device',a.device,'--epochs','15','--batch-size','4','--seed','42','--prior-weight','0']
    recipes=[('history_soft24_kl05','history',1e-4,.5,24,.5),('all_lr3e5_soft24_kl05','all',3e-5,.5,24,.5),('all_lr1e5_hard_kl1','all',1e-5,0,24,1),('all_lr3e5_soft32_kl1','all',3e-5,.5,32,1)]
    state=dict(state='running',started=time.time(),stages=[])
    def save():(output/'status.json').write_text(json.dumps(state,indent=2))
    for name,scope,lr,soft,sigma,kl in recipes:
        command=common+['--output',str(output/name),'--train-scope',scope,'--lr',str(lr),'--soft-weight',str(soft),'--soft-sigma',str(sigma),'--kl-weight',str(kl)]
        stage=dict(name=name,command=command,state='running',started=time.time());state['stages'].append(stage);save()
        with (output/f'{name}.log').open('w') as sink:result=subprocess.run(command,cwd=ROOT,stdout=sink,stderr=subprocess.STDOUT)
        stage.update(return_code=result.returncode,state='complete' if result.returncode==0 else 'failed',seconds=time.time()-stage['started']);save()
        if result.returncode:state['state']='failed';save();return result.returncode
    state.update(state='complete',seconds=time.time()-state['started']);save();return 0
if __name__=='__main__':raise SystemExit(main())
