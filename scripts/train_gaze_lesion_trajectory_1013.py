"""Fresh patientOOF causal gaze imitation/lesion auxiliary heads. TRAIN-only epoch choice."""
import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.official_cbis_trajectory_1012 import build_fresh_model,teacher_inputs,state_sha256
from src.gaze_lesion_aux_1013 import objective
from src.gaze_nll_adaptation_1011 import hard_nll
from src.official_cbis_classifier_1012 import digest

def atomic(payload,path):
 tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42');d=out/'trajectory_models';d.mkdir(exist_ok=False);indexpath='runs/official_cbis_v2_experiment_1012/gaze_cache/index.pt';index=torch.load(indexpath,weights_only=False,map_location='cpu');records=[torch.load(p,weights_only=False,map_location='cpu') for p in index['records']];records=[r for r in records if r['split']=='train'];assert len(records)==420;boxespath=out/'lesion_supervision_v2.json';boxes=json.load(open(boxespath))['boxes'];c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);tr=np.flatnonzero(f.split=='train');foldid=np.full(len(f),-1);outer=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(tr,f.label.values[tr],f.patient_key.values[tr]));cases=[]
 for fold,(fit,hold) in enumerate(outer):
  fp=set(f.patient_key.iloc[tr[fit]]);hp=set(f.patient_key.iloc[tr[hold]]);assert not fp&hp;foldid[tr[hold]]=fold;cases.append((f'fold{fold}',[r for r in records if r['patient_key'] in fp],fp,hp))
 cases.append(('fullTRAIN',records,set(f.patient_key.iloc[tr]),set()));atomic(dict(image_ids=f.image_id.tolist(),foldid=torch.tensor(foldid)),out/'patient_folds.pt');m=dict(state='running',stage='trajectory_training',seed=42,init='fresh random trajectory head/ImageNet only; never old420 supervised checkpoint',max_epochs=10,patience=3,lr=.0003,weight_decay=.01,batch=4,loss='imitation hardNLL + lambda*.2 first4proposal crop-overlap coverage +lambda*.05 sixteen validpoint near-duplicate expectation',aux_lambda={'imitation':0,'lesion_aux':1},selection='each outer TRAIN patient inner20percent objective; fresh outer/full retrain selected epoch; outer heldout gaze+bbox excluded',train_gaze=420,train_bbox_registered=318,VAL_GT_used=False,test_used=False,dependencies={str(p):digest(p) for p in [Path(__file__),Path('src/gaze_lesion_aux_1013.py'),boxespath,Path(indexpath)]},models=[])
 def save():
  p=out/'trajectory_manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save()
 def fit(rows,validation,arm,epochs,selection,folder):
  model=build_fresh_model();visualsha=state_sha256(model.visual.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);history=[];best=float('inf');stale=0;be=0;state=None
  def batches(rs,epoch,training):
   order=torch.randperm(len(rs),generator=torch.Generator().manual_seed(42+epoch)).tolist() if training else range(len(rs))
   for start in range(0,len(rs),4):
    batch=[rs[j] for j in list(order)[start:start+4]];inp=[teacher_inputs(r,epoch,False) for r in batch];coords,mask,vectors=[torch.stack([v[j] for v in inp]) for j in range(3)];features=[torch.stack([r['features'][i] for r in batch]) for i in range(3)];logits=model.teacher_logits(model.maps_features(features),coords,vectors)
    if arm=='lesion_aux':loss,terms=objective(model,logits,coords,mask,[boxes.get(str(r['image_id']),[]) for r in batch])
    else:loss=hard_nll(logits,coords,mask);_,terms=objective(model,logits,coords,mask,[boxes.get(str(r['image_id']),[]) for r in batch])
    yield loss,terms,len(batch)
  for ep in range(epochs+1):
   if ep:
    model.train()
    for loss,terms,n in batches(rows,ep,True):opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);opt.step()
   record=dict(epoch=ep)
   if selection:
    model.eval()
    with torch.inference_mode():vals=[(float(loss),{k:float(v) for k,v in terms.items()},n) for loss,terms,n in batches(validation,0,False)]
    total=sum(n for _,_,n in vals);score=sum(loss*n for loss,_,n in vals)/total;record.update(inner_objective=score,terms={k:sum(ts[k]*n for _,ts,n in vals)/total for k in vals[0][1]})
    if score<best:best=score;be=ep;stale=0;state={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('visual.')}
    else:stale+=1
   history.append(record);m.update(current_arm=arm,current_model=folder.name,current_phase='inner' if selection else 'refit',latest_epoch=ep);save();atomic(dict(state={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('visual.')},optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,history=history,best_state=state,selected_epoch=be),folder/('inner_latest.pt' if selection else 'refit_latest.pt'))
   if selection and stale>=3:break
  assert state_sha256(model.visual.state_dict())==visualsha
  return be,history,model
 for arm in ['imitation','lesion_aux']:
  for name,rows,fitpatients,heldoutpatients in cases:
   folder=d/f'{arm}_{name}';folder.mkdir();rf=pd.DataFrame([dict(patient=r['patient_key'],image=r['image_id'],bbox=bool(boxes.get(str(r['image_id'])))) for r in rows]);inner=list(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.arange(len(rows)),rf.bbox,rf.patient))[0];fitidx,holdidx=inner;fitrows=[rows[i] for i in fitidx];holdrows=[rows[i] for i in holdidx];assert not {r['patient_key'] for r in fitrows}&{r['patient_key'] for r in holdrows};assert not fitpatients&heldoutpatients;be,h,_=fit(fitrows,holdrows,arm,10,True,folder);_,rh,model=fit(rows,[],arm,be,False,folder);state={k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('visual.')};payload=dict(state=state,model_config=model.model_config,visual_sha256=state_sha256(model.visual.state_dict()),fit_patients=sorted(fitpatients),heldout_patients=sorted(heldoutpatients),fit_record_ids=[r['image_id'] for r in rows],epoch=be,arm=arm,source_sha256=digest(__file__));atomic(payload,folder/'selected.pt');(folder/'history.json').write_text(json.dumps(dict(inner=h,refit=rh),indent=2));m['models'].append(dict(arm=arm,model=name,selected_epoch=be,fit_records=len(rows),fit_bbox=sum(bool(boxes.get(str(r['image_id']))) for r in rows),checkpoint_sha256=digest(folder/'selected.pt'),inner_best=h[be]));save();print(json.dumps(dict(arm=arm,model=name,selected_epoch=be)),flush=True)
 assert all(digest(p)==sha for p,sha in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save()
if __name__=='__main__':main()
