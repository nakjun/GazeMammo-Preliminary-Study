"""Image-only patientOOF/fullTRAIN causal rollout, reuse current crop features for classifier."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.data import whole_tensor,patch_tensor,seed_for
from src.official_cbis_trajectory_1012 import build_fresh_model,state_sha256
from src.official_cbis_classifier_1012 import digest

def rollout_batch(model,rows):
 images=[Image.open(r['canonical_path']).convert('L') for r in rows];generators=[torch.Generator().manual_seed(seed_for(42,int(r['image_id']))) for r in rows];prefix=torch.empty(len(rows),0,2);hidden=None;vector=None;features=[]
 with torch.inference_mode():
  maps=model.maps(torch.stack([whole_tensor(image) for image in images]))
  for step in range(4):
   logp,hidden,_=model.step_history(maps,prefix,hidden,vector);cells=torch.stack([torch.multinomial(p.exp(),1,generator=g)[0] for p,g in zip(logp,generators)]);point=model.cell_xy[cells];prefix=torch.cat([prefix,point[:,None]],1);crops=torch.stack([patch_tensor(image,xy.double().numpy()*[1500,3000],224) for image,xy in zip(images,point)]);vector=model.visual.patch_vector(crops);features.append(vector)
 return prefix,torch.stack(features,1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42');c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');rows=c['rows'];folds=torch.load(out/'patient_folds.pt',weights_only=False,map_location='cpu');assert folds['image_ids']==[r['image_id'] for r in rows];mp=out/'prediction_manifest_k4.json';m=dict(state='running',stage='imageonly_OOF_rollout_crop_features',source_sha256=digest(__file__),test_used=False,oracle_used=False,cohort=2393,arms=['imitation','lesion_aux'],completed_records=0,records={},models={});start=time.time();records=out/'predicted_records_k4';records.mkdir(exist_ok=True)
 def save():
  tmp=mp.with_suffix('.tmp');tmp.write_text(json.dumps(m));tmp.replace(mp)
 if a.benchmark:
  model=build_fresh_model().eval().requires_grad_(False);rs=rows[:4];tick=time.monotonic();xy,v=rollout_batch(model,rs);seconds=time.monotonic()-tick;r=dict(batch=len(rs),seconds=seconds,full_twoarms_minutes=2393*2/len(rs)*seconds/60,features_shape=list(v.shape),imageonly=True,benchmark_only_fresh_head=True);(out/'rollout_benchmark_k4.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);return
 save()
 for arm in m['arms']:
  for name in ['fold0','fold1','fold2','fullTRAIN']:
   cp=out/'trajectory_models'/f'{arm}_{name}/selected.pt'
   if a.benchmark and not cp.exists():continue
   ck=torch.load(cp,weights_only=False,map_location='cpu');model=build_fresh_model();missing,unexpected=model.load_state_dict(ck['state'],strict=False);assert not unexpected and all(k.startswith('visual.') for k in missing);assert state_sha256(model.visual.state_dict())==ck['visual_sha256'];model.eval().requires_grad_(False);m['models'][str(cp)]=digest(cp);indices=[i for i,r in enumerate(rows) if (name=='fullTRAIN' and r['split']=='val') or (name!='fullTRAIN' and r['split']=='train' and int(folds['foldid'][i])==int(name[-1]))];assert all(rows[i]['patient_key'] not in ck['fit_patients'] for i in indices)
   if a.benchmark:
    rs=[rows[i] for i in indices[:4]];tick=time.monotonic();xy,v=rollout_batch(model,rs);seconds=time.monotonic()-tick;r=dict(batch=len(rs),seconds=seconds,full_twoarms_minutes=2393*2/len(rs)*seconds/60,features_shape=list(v.shape),imageonly=True);(out/'rollout_benchmark_k4.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);return
   pending=[]
   for i in indices:
    dest=records/f"{arm}_{rows[i]['image_id']}.pt"
    if dest.exists():
     old=torch.load(dest,weights_only=False,map_location='cpu');assert old['checkpoint_sha256']==m['models'][str(cp)] and old['image_id']==rows[i]['image_id'];m['records'][f'{arm}:{rows[i]["image_id"]}']=digest(dest);m['completed_records']=len(m['records'])
    else:pending.append(i)
   for startidx in range(0,len(pending),4):
    ids=pending[startidx:startidx+4];rs=[rows[i] for i in ids];xy,features=rollout_batch(model,rs);assert torch.isfinite(xy).all() and ((xy>=0)&(xy<1)).all() and features.shape==(len(rs),4,1024)
    for j,row in enumerate(rs):
     rounded=torch.round(xy[j].double()*torch.tensor([1500.,3000.])).long();rect=torch.cat([rounded-112,rounded+112],1);dest=records/f"{arm}_{row['image_id']}.pt";record=dict(image_id=row['image_id'],patient_key=row['patient_key'],split=row['split'],coordinates=xy[j],features=features[j],mask=torch.ones(4,dtype=torch.bool),rectangles=rect,checkpoint_sha256=m['models'][str(cp)],pixel_sha256=digest(row['canonical_path']),trajectory_model=name,inference_inputs='image_pixels_only;opaqueID deterministic seed',unique_centers=len(torch.unique(rounded,dim=0)));tmp=dest.with_suffix('.tmp');torch.save(record,tmp);tmp.replace(dest);m['records'][f'{arm}:{row["image_id"]}']=digest(dest);m['completed_records']=len(m['records'])
    m.update(current_arm=arm,current_model=name);save()
 assert m['completed_records']==4786;m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();print(json.dumps(dict(prediction='complete',records=m['completed_records'],minutes=m['elapsed_minutes'])),flush=True)
if __name__=='__main__':main()
