"""Persistent sequential full-cohort extraction then preregistered pilot."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'runs/gaze_multiscale_1012_round6';out.mkdir(exist_ok=True);manifest=out/'pipeline_manifest.json';m=dict(state='running',stage='extract',test_used=False,preregistered_head=dict(variant='mean_residual',input_dim=2048,dim=64,dropout=.2,seeds=[42],scales=[224,448],arms=['grid','gaze_baseline']),start_time=time.time())
def save():manifest.write_text(json.dumps(m,indent=2))
save()
for stage,cmd in [('extract',[sys.executable,'scripts/extract_gaze_multiscale_1012.py','--output',str(out/'features')]),('train',[sys.executable,'scripts/train_gaze_multiscale_head_1012.py','--features',str(out/'features'),'--output',str(out/'heads')])]:
 m['stage']=stage;save();result=subprocess.run(cmd,cwd=root)
 if result.returncode:m.update(state='failed',exit_code=result.returncode);save();sys.exit(result.returncode)
m.update(state='complete',stage='report_complete',finish_time=time.time());save()
