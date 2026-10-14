import sys,json,time,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.official_cbis_trajectory_1012 import build_fresh_model,teacher_inputs
from src.gaze_lesion_aux_1013 import objective

torch.set_num_threads(4);out=Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42');index=torch.load('runs/official_cbis_v2_experiment_1012/gaze_cache/index.pt',weights_only=False,map_location='cpu');records=[]
for p in index['records']:
 r=torch.load(p,weights_only=False,map_location='cpu')
 if r['split']=='train':records.append(r)
 if len(records)==4:break
boxes=json.load(open(out/'lesion_supervision_v2.json'))['boxes'];model=build_fresh_model();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01);inputs=[teacher_inputs(r) for r in records];coords,mask,vectors=[torch.stack([v[j] for v in inputs]) for j in range(3)];features=[torch.stack([r['features'][i] for r in records]) for i in range(3)];bs=[boxes.get(str(r['image_id']),[]) for r in records]
def step():
 opt.zero_grad();logits=model.teacher_logits(model.maps_features(features),coords,vectors);loss,terms=objective(model,logits,coords,mask,bs);loss.backward();grad=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.requires_grad and p.grad is not None);opt.step();return float(loss),grad
step();start=time.monotonic();loss,grad=step();seconds=time.monotonic()-start;assert grad>0;r=dict(batch=4,step_seconds=seconds,estimated420_epoch_minutes=420/4*seconds/60,loss=loss,gradient_abs_sum=grad,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2);(out/'train_benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
