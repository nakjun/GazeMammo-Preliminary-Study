"""Two frozen-encoder high-resolution adaptation trials with exact legacy target."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    common=[sys.executable,'scripts/gaze_nll_adaptation_1011.py','train','--base','runs/full_1011_seed42/control/best.pt','--cache','runs/dual_target_1011/control_resolution512_cache_v2','--device',a.device,'--epochs','15','--batch-size','4','--seed','42','--train-scope','all','--soft-weight','.5']
    recipes=[('H1_soft32_lr3e5_kl05',3e-5,32,.5,0),('H2_soft64_lr1e4_prior01',1e-4,64,0,.1)];state=dict(state='running',started=time.time(),stages=[])
    def save():(output/'status.json').write_text(json.dumps(state,indent=2))
    for name,lr,sigma,kl,prior in recipes:
        command=common+['--output',str(output/name),'--lr',str(lr),'--soft-sigma',str(sigma),'--kl-weight',str(kl),'--prior-weight',str(prior)]
        stage=dict(name=name,command=command,state='running',started=time.time());state['stages'].append(stage);save()
        with (output/f'{name}.log').open('w') as sink:result=subprocess.run(command,cwd=ROOT,stdout=sink,stderr=subprocess.STDOUT)
        stage.update(return_code=result.returncode,state='complete' if result.returncode==0 else 'failed',seconds=time.time()-stage['started']);save()
        if result.returncode:state['state']='failed';save();return result.returncode
    state.update(state='complete',seconds=time.time()-state['started']);save();return 0
if __name__=='__main__':raise SystemExit(main())
