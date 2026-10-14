"""Image-only patientOOF/fullTRAIN causal rollout, reuse current crop features for classifier."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from src.data import whole_tensor,patch_tensor,seed_for
from src.official_cbis_trajectory_1012 import build_fresh_model,state_sha256
from src.official_cbis_classifier_1012 import digest

def rollout_batch(model,rows,whole_features=None):
 images=[Image.open(r['canonical_path']).convert('L') for r in rows];generators=[torch.Generator().manual_seed(seed_for(42,int(r['image_id']))) for r in rows];prefix=torch.empty(len(rows),0,2);hidden=None;vector=None;features=[]
 with torch.inference_mode():
  maps=model.maps_features(whole_features) if whole_features is not None else model.maps(torch.stack([whole_tensor(image) for image in images]))
  for step in range(16):
   logp,hidden,_=model.step_history(maps,prefix,hidden,vector);cells=torch.stack([torch.multinomial(p.exp(),1,generator=g)[0] for p,g in zip(logp,generators)]);point=model.cell_xy[cells];prefix=torch.cat([prefix,point[:,None]],1);crops=torch.stack([patch_tensor(image,xy.double().numpy()*[1500,3000],224) for image,xy in zip(images,point)]);vector=model.visual.patch_vector(crops);features.append(vector)
 return prefix,torch.stack(features,1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);base=Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42');out=base.parent/'round12_gaze_extent_pilot42';out.mkdir(exist_ok=True);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');rows=c['rows'];folds=torch.load(base/'patient_folds.pt',weights_only=False,map_location='cpu');assert folds['image_ids']==[r['image_id'] for r in rows];mp=out/'prediction_manifest_k16.json';m=dict(state='running',stage='imageonly_OOF_rollout_crop_features',source_sha256=digest(__file__),test_used=False,oracle_used=False,cohort=2393,arms=['imitation'],completed_records=0,records={},models={});start=time.time();records=out/'predicted_records_k16';records.mkdir(exist_ok=True)
 def save():
  tmp=mp.with_suffix('.tmp');tmp.write_text(json.dumps(m));tmp.replace(mp)
 if a.benchmark:
  model=build_fresh_model().eval().requires_grad_(False);ck=torch.load(base/'trajectory_models/imitation_fold0/selected.pt',weights_only=False,map_location='cpu');model.load_state_dict(ck['state'],strict=False);rs=[r for i,r in enumerate(rows) if r['split']=='train' and int(folds['foldid'][i])==0][:4];tick=time.monotonic()
  with torch.inference_mode():features=model.visual(torch.stack([whole_tensor(Image.open(r['canonical_path']).convert('L')) for r in rs]))
  xy,v=rollout_batch(model,rs,features);seconds=time.monotonic()-tick;old=[torch.load(base/'predicted_records_k4'/f"imitation_{r['image_id']}.pt",weights_only=False,map_location='cpu') for r in rs];assert torch.equal(xy[:,:4],torch.stack([r['coordinates'] for r in old]));err=float((v[:,:4]-torch.stack([r['features'] for r in old])).abs().max());assert err<.001;r=dict(prefix_coordinate_exact=True,prefix_feature_max_abs=err,batch=len(rs),two_arms=False,shared_wholeimage=True,seconds=seconds,full_minutes=2393/len(rs)*seconds/60,features_shape=list(v.shape),imageonly=True,benchmark_restored_OOF_head=True);(out/'rollout_benchmark_k16.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);return
 save()
 for name in ['fold0','fold1','fold2','fullTRAIN']:
  models={};checkpoints={}
  for arm in m['arms']:
   cp=base/'trajectory_models'/f'{arm}_{name}/selected.pt';ck=torch.load(cp,weights_only=False,map_location='cpu');model=build_fresh_model();missing,unexpected=model.load_state_dict(ck['state'],strict=False);assert not unexpected and all(k.startswith('visual.') for k in missing);assert state_sha256(model.visual.state_dict())==ck['visual_sha256'];model.eval().requires_grad_(False)
   if models:model.visual=next(iter(models.values())).visual
   models[arm]=model;checkpoints[arm]=ck;m['models'][str(cp)]=digest(cp)
  indices=[i for i,r in enumerate(rows) if (name=='fullTRAIN' and r['split']=='val') or (name!='fullTRAIN' and r['split']=='train' and int(folds['foldid'][i])==int(name[-1]))];assert all(rows[i]['patient_key'] not in ck['fit_patients'] for ck in checkpoints.values() for i in indices)
  for startidx in range(0,len(indices),4):
   ids=indices[startidx:startidx+4];rs=[rows[i] for i in ids]
   with torch.inference_mode():whole_features=next(iter(models.values())).visual(torch.stack([whole_tensor(Image.open(r['canonical_path']).convert('L')) for r in rs]))
   for arm,model in models.items():
    checkpoint_sha=m['models'][str(base/'trajectory_models'/f'{arm}_{name}/selected.pt')];pending=[]
    for j,row in enumerate(rs):
     dest=records/f"{arm}_{row['image_id']}.pt"
     if dest.exists():
      old=torch.load(dest,weights_only=False,map_location='cpu');assert old['checkpoint_sha256']==checkpoint_sha and old['image_id']==row['image_id'];m['records'][f'{arm}:{row["image_id"]}']=digest(dest)
     else:pending.append(j)
    if pending:
     prs=[rs[j] for j in pending];features_for_rows=[v[pending] for v in whole_features];xy,features=rollout_batch(model,prs,features_for_rows);assert torch.isfinite(xy).all() and ((xy>=0)&(xy<1)).all() and features.shape==(len(prs),16,1024)
     for j,row in enumerate(prs):
      rounded=torch.round(xy[j].double()*torch.tensor([1500.,3000.])).long();rect=torch.cat([rounded-112,rounded+112],1);dest=records/f"{arm}_{row['image_id']}.pt";record=dict(image_id=row['image_id'],patient_key=row['patient_key'],split=row['split'],coordinates=xy[j],features=features[j],mask=torch.ones(16,dtype=torch.bool),rectangles=rect,checkpoint_sha256=checkpoint_sha,pixel_sha256=digest(row['canonical_path']),trajectory_model=name,inference_inputs='image_pixels_only;opaqueID deterministic seed',unique_centers=len(torch.unique(rounded,dim=0)));tmp=dest.with_suffix('.tmp');torch.save(record,tmp);tmp.replace(dest);m['records'][f'{arm}:{row["image_id"]}']=digest(dest)
    m.update(current_arm=arm,current_model=name,completed_records=len(m['records']));save()
 assert m['completed_records']==2393;m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();print(json.dumps(dict(prediction='complete',records=m['completed_records'],minutes=m['elapsed_minutes'])),flush=True)
if __name__=='__main__':main()
