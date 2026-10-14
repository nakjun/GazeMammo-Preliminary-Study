"""Durable sequential experiment stages with explicit return codes."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'runs/proposal_visit_1011';MONITOR=ROOT/'runs/proposal_visit_1011_monitor'


def main():
    OUTPUT.mkdir(parents=True,exist_ok=False);MONITOR.mkdir(parents=True,exist_ok=True)
    python=sys.executable;common=[python,'scripts/proposal_policy_1011.py','--checkpoint','runs/guided_1011_combined_seed42/best.pt','--manifest','manifests/all.csv','--split','val','--device','cuda:0']
    stages=[('original_sampling',common+['--seeds','42','43','44','--output',str(OUTPUT/'original_sampling')]),('original_sampling_posthoc',[python,'scripts/evaluate_proposal_policy_1011.py','--traces',str(OUTPUT/'original_sampling/traces.jsonl'),'--output',str(OUTPUT/'original_sampling_posthoc'),'--baseline','P0']),('original_argmax',common+['--seeds','42','--strategy','argmax','--output',str(OUTPUT/'original_argmax')]),('original_argmax_posthoc',[python,'scripts/evaluate_proposal_policy_1011.py','--traces',str(OUTPUT/'original_argmax/traces.jsonl'),'--output',str(OUTPUT/'original_argmax_posthoc'),'--baseline','P0_argmax'])]
    state=dict(state='running',stages=[],started=time.time())
    def save():(MONITOR/'status.json').write_text(json.dumps(state,indent=2))
    save()
    for name,command in stages:
        stage=dict(name=name,command=command,state='running',started=time.time());state['stages'].append(stage);save()
        with (MONITOR/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        stage.update(return_code=result.returncode,seconds=time.time()-stage['started'],state='complete' if result.returncode==0 else 'failed');save()
        if result.returncode:
            state['state']='failed';save();return result.returncode
    state.update(state='complete',seconds=time.time()-state['started']);save();return 0
if __name__=='__main__':raise SystemExit(main())
