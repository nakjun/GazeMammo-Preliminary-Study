"""Persistent full cohort final-CN-boundary cache + matched adaptation pilot."""
import json,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from src.official_cbis_classifier_1012 import digest
out=root/'runs/gaze_boundary_adaptation_1012_round9';out.mkdir(exist_ok=True);mp=out/'pipeline_manifest.json';sources=['scripts/extract_gaze_boundary_1012.py','scripts/train_gaze_boundary_adaptation_1012.py','scripts/benchmark_convnext_lastblock_1012.py','src/data.py','src/gaze_backbones.py','scripts/train_gaze_multiview_1012.py','src/gaze_attention_1011.py','src/official_cbis_classifier_1012.py'];m=dict(state='running',stage='extract',start_time=time.time(),test_used=False,source_sha256={p:digest(root/p) for p in sources},preregistered=dict(arms=['grid','gaze_baseline'],seed=42,epochs=5,init='originalmatchedmeanresidualhead/globalfrozen',backbone='sameConvNeXtBasefinalCNblock+normonly',backbone_lr=1e-5,head_lr=3e-4,weight_decay=.05,bags_per_batch=4,patch_budget=16,control='same initializedhead/samebatch frozen originalfeature control5epochs'))
def save():mp.write_text(json.dumps(m,indent=2))
save()
for stage,cmd in [('extract',[sys.executable,'scripts/extract_gaze_boundary_1012.py','--output',str(out/'features')]),('train',[sys.executable,'scripts/train_gaze_boundary_adaptation_1012.py','--features',str(out/'features'),'--output',str(out/'heads')])]:
 assert all(digest(root/p)==h for p,h in m['source_sha256'].items());m['stage']=stage;save();result=subprocess.run(cmd,cwd=root)
 if result.returncode:m.update(state='failed',exit_code=result.returncode);save();sys.exit(result.returncode)
m.update(state='complete',stage='report_complete',finish_time=time.time());save()
