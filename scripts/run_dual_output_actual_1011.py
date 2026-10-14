"""Authoritative same-config actual-image NLL and ROI14+RAWq2 evaluation."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011/unified_runtime');p.add_argument('--device',default='cuda:1');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);config=root/'config_cudnn_ieee.json';payload=json.loads(config.read_text());state_path=root/'actual_state.json';state=dict(status='running',stages=[],combined_config_sha256=payload['sha256'],compute=payload['config']['compute'],numerics=payload['config']['probability']['numerics'],goals=dict(canonical_nll_max=5.3,physical_capture_min=.85),forced_path_nll_reported=False)
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    try:
        smoke=json.loads((root/'smoke_cudnn_ieee/status.json').read_text())
        if smoke['state']!='complete' or smoke['config_sha256']!=payload['sha256']:raise ValueError('Currentconfig smoke prerequisite failed')
        stage('actual_nll','dual_output_1011.py',['--phase','nll','--config',config,'--output',root/'actual_nll','--device',a.device]);stage('actual_generation','dual_output_1011.py',['--phase','generate','--config',config,'--output',root/'actual_generation','--device',a.device,'--seeds',42,43,44]);stage('actual_posthoc','evaluate_proposal_center_offset_1011.py',['--traces',root/'actual_generation/traces.jsonl','--output',root/'actual_posthoc','--baseline','DUAL_ROI14_RAW_ARITH075','--crop-size',224]);nll=json.loads((root/'actual_nll/summary.json').read_text());metrics=json.loads((root/'actual_posthoc/metrics.json').read_text());capture=metrics['summaries']['DUAL_ROI14_RAW_ARITH075']['actual_recall_05_step16'];state['outcome']=dict(canonical_actual_image_macro_nll=nll['canonical_legacy16_image_macro_nll'],physical_capture50=capture,nll_goal_met=nll['canonical_legacy16_image_macro_nll']<=5.3,roi_goal_met=capture>=.85,forced_path_nll='Not applicable');state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
