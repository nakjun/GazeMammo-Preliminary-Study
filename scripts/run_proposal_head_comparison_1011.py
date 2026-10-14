"""Durable exploratory head comparisons, with distinct generation and posthoc stages."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/proposal_visit_1011');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/'head_comparison_state.json';state=dict(status='running',stages=[],validation_status='Exploratory previously selected validation93',training_seeds=[42],cohort_note='First occupancy sampling began with all.csv val102; posthoc restricts predefined trajectory.csv labelled val93. Future policy subprocesses use routing-only predefined val93 intersection. All extra generated images remain validation.')
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    checkpoint=ROOT/'runs/guided_1011_combined_seed42/best.pt'
    def common(target):return ['--checkpoint',checkpoint,'--utility-checkpoint',root/f'{target}_head_seed42/last.pt','--manifest',ROOT/'manifests/all.csv','--device',a.device]
    save()
    try:
        stage('utility_head_integration_smoke','proposal_policy_1011.py',common('utility')+['--split','val','--image-ids',64,73,'--seeds',42,'--policies','P0','P3','P4','P5','--output',root/'utility_head_integration_smoke'])
        stage('utility_teacher_integration_smoke','evaluate_proposal_policy_nll_1011.py',common('utility')+['--image-ids',64,73,'--policies','P0','P3','P4','--output',root/'utility_teacher_integration_smoke'])
        smoke=json.loads((root/'utility_teacher_integration_smoke/summary.json').read_text())
        if not {'P0','P3_utility_mse','P4_utility_mse'}.issubset(smoke['image_macro_nll']):raise ValueError('Teacher head smoke labels mismatch')
        for target in ('occupancy','utility'):
            sampling=root/f'{target}_sampling';argmax=root/f'{target}_argmax'
            stage(f'{target}_sampling','proposal_policy_1011.py',common(target)+['--split','val','--seeds',42,43,44,'--policies','P0','P3','P4','P5','--output',sampling])
            stage(f'{target}_sampling_posthoc','evaluate_proposal_policy_1011.py',['--traces',sampling/'traces.jsonl','--output',root/f'{target}_sampling_posthoc','--baseline','P0'])
            stage(f'{target}_teacher_nll','evaluate_proposal_policy_nll_1011.py',common(target)+['--policies','P0','P3','P4','--output',root/f'{target}_teacher_nll'])
            stage(f'{target}_argmax','proposal_policy_1011.py',common(target)+['--split','val','--seeds',42,'--strategy','argmax','--policies','P0','P3','P4','P5','--output',argmax])
            stage(f'{target}_argmax_posthoc','evaluate_proposal_policy_1011.py',['--traces',argmax/'traces.jsonl','--output',root/f'{target}_argmax_posthoc','--baseline','P0_argmax'])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
