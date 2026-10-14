"""Durable cached human-likelihood trials; exploratory validation selection."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=15);a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    script=[sys.executable,'scripts/gaze_nll_adaptation_1011.py'];bases={'human_improved':'runs/full_1011_seed42/improved/best.pt','human_control':'runs/full_1011_seed42/control/best.pt','guided_raw':'runs/guided_1011_combined_seed42/best.pt'};stages=[]
    cache=str(output/'frozen_visual_cache')
    stages.append(('frozen_visual_cache',script+['cache','--base',bases['human_improved'],'--cache',cache,'--device',a.device]))
    control_cache=str(output/'control_visual_cache')
    stages.append(('control_visual_cache',script+['cache','--base',bases['human_control'],'--cache',control_cache,'--device',a.device]))
    for name,base in bases.items():
        selected_cache=control_cache if name=='human_control' else cache
        common=['--base',base,'--cache',selected_cache,'--device',a.device]
        stages.append((name+'_calibration',script+['calibrate']+common+['--output',str(output/f'{name}_calibration')]))
    for name in ('human_control','human_improved','guided_raw'):
        base=bases[name];selected_cache=control_cache if name=='human_control' else cache;common=['--base',base,'--cache',selected_cache,'--device',a.device]
        for prior in (0.,.1):stages.append((f'{name}_prior{prior}',script+['train']+common+['--output',str(output/f'{name}_prior{prior}'),'--prior-weight',str(prior),'--epochs',str(a.epochs),'--batch-size','4','--lr','0.0001','--seed','42']))
    state=dict(state='running',stages=[],started=time.time())
    def save():(output/'status.json').write_text(json.dumps(state,indent=2))
    for name,command in stages:
        stage=dict(name=name,command=command,state='running',started=time.time());state['stages'].append(stage);save()
        with (output/f'{name}.log').open('w') as sink:result=subprocess.run(command,cwd=ROOT,stdout=sink,stderr=subprocess.STDOUT)
        stage.update(return_code=result.returncode,seconds=time.time()-stage['started'],state='complete' if result.returncode==0 else 'failed');save()
        if result.returncode:state['state']='failed';save();return result.returncode
    state.update(state='complete',seconds=time.time()-state['started']);save();return 0
if __name__=='__main__':raise SystemExit(main())
