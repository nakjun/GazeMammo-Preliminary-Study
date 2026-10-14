"""Durable fixed224 ROI-budget sweep, separated from gaze NLL training."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/dual_target_1011');p.add_argument('--device',default='cuda:1');p.add_argument('--max-iou',type=float,default=.3);p.add_argument('--name',default='budget224_seed42');p.add_argument('--nms-sweep',action='store_true');a=p.parse_args();root=Path(a.root);root.mkdir(parents=True,exist_ok=True);state_path=root/f'{a.name}_state.json';state=dict(status='running',stages=[],crop_size=224,patch_budget=16,output_role='ROI search branch; forced ROI policy NLL not reported',validation_status='Exploratory fixed predefined val93',nms_crop_iou=a.max_iou)
    def save():
        temp=state_path.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2));temp.replace(state_path)
    def stage(name,script,args):
        cmd=[sys.executable,str(ROOT/'scripts'/script),*map(str,args)];entry=dict(name=name,command=cmd,status='running',started=time.time());state['stages'].append(entry);save()
        with (root/f'{a.name}_{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        entry.update(status='complete' if result.returncode==0 else 'failed',returncode=result.returncode,ended=time.time());save()
        if result.returncode:raise RuntimeError(f'{name} failed: {result.returncode}')
    try:
        settings=[('budget224_nms01_seed42',.1),('budget224_nms05_seed42',.5)] if a.nms_sweep else [(a.name,a.max_iou)]
        for name,threshold in settings:
            output=root/name
            stage(name+'_generation','proposal_budget_1011.py',['--output',output,'--device',a.device,'--budgets',0,1,2,3,4,6,8,12,16,'--layouts','first','interleaved','--seeds',42,'--crop-size',224,'--max-iou',threshold])
        for name,threshold in settings:
            output=root/name
            stage(name+'_posthoc','evaluate_proposal_policy_1011.py',['--traces',output/'traces.jsonl','--output',root/f'{name}_posthoc','--baseline','B0_first_crop224','--crop-size',224])
        state['status']='complete';save()
    except Exception as exc:state.update(status='failed',error=str(exc));save();raise
if __name__=='__main__':main()
