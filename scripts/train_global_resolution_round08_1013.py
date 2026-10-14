"""Matched two-resolution global/imitation4/random4 TRAIN-inner-selected small joint heads."""
import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.official_cbis_classifier_1012 import digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views
from scripts.train_gaze_focus_selected_1013 import local_summary

class Head(torch.nn.Module):
 def __init__(self):super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(4096,32),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(32,1))
 def forward(self,x):return self.net(x).flatten()
def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round08_global_resolution_pilot42');d=out/'heads';d.mkdir(exist_ok=True);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];globalp=out/'global_meanmax.pt';g=torch.load(globalp,weights_only=False,map_location='cpu');assert g['image_ids']==f.image_id.tolist();tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold].tolist()).issubset(set(hold.tolist()));local={'global_only':torch.zeros(2393,2048)};counts={}
 for source in ['imitation','random']:
  if source=='random':p=t['random_features'][:,:4];coords=t['random_coordinates'][:,:4];mask=t['random_mask'][:,:4].clone()
  else:
   rs=[torch.load(out.parent/'round06_lesion_trajectory_pilot42/predicted_records_k4'/f"imitation_{r['image_id']}.pt",weights_only=False,map_location='cpu') for r in c['rows']];assert [r['image_id'] for r in rs]==f.image_id.tolist();p=torch.stack([r['features'] for r in rs]);coords=torch.stack([r['coordinates'] for r in rs]);mask=torch.stack([r['mask'] for r in rs])
  for i in range(2393):
   seen=set()
   for j in range(4):
    key=tuple(torch.round(coords[i,j].double()*torch.tensor([1500.,3000.])).tolist())
    if key in seen:mask[i,j]=False
    else:seen.add(key)
  local[source]=local_summary(torch.zeros(2393,0),p,mask);counts[source]={split:float(mask[torch.tensor(np.flatnonzero(f.split==split))].sum(1).float().mean()) for split in ['train','val']}
 m=dict(state='running',seed=42,source_sha256=digest(__file__),dependencies={str(p):digest(p) for p in [Path(__file__),globalp,Path('scripts/train_gaze_multiview_1012.py'),Path('scripts/train_gaze_focus_selected_1013.py')]},feature_dim=4096,head_dim=32,dropout=.5,weight_decay=.1,lr=.0003,batch=64,max_epochs=40,patience=8,selection='TRAINpatientinner20percent pairedAUROC only; fullTRAIN fresh refit selected epoch then VAL once',normalization='TRAINfitmean/std floor.01 clamp10',paired_training='sameoppositeCCMLO normalizedfeatureaverage withpartnerdropout.5',test_used=False,oracle_used=False,valid_source_counts=counts,results=[])
 def save():
  p=d/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save()
 for resolution in ['low','high']:
  for source,l in local.items():
   arm=f'{resolution}_{source}';folder=d/arm;folder.mkdir(exist_ok=True);x=torch.cat([g[resolution],l],1);assert x.shape==(2393,4096)
   def run(ids,validation,epochs,select,tag):
    torch.manual_seed(42);model=Head();mean=x[ids].mean(0);std=x[ids].std(0).clamp_min(.01);z=((x-mean)/std).clamp(-10,10);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);best=-1;be=0;stale=0;state=None;hist=[];rp=folder/f'{tag}_recovery.pt';resume=torch.load(rp,weights_only=False,map_location='cpu') if rp.exists() else None
    if resume:model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);best=resume['best'];be=resume['be'];stale=resume['stale'];state=resume['best_state'];hist=resume['history']
    def forward(ii,training=False,single=False):
     use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
     if training:use&=torch.rand(len(ii))>=.5
     return model(torch.where(use[:,None],.5*(z[ii]+z[other[ii]]),z[ii]))
    epochstart=resume['epoch']+1 if resume else 0
    if select and stale>=8:epochstart=epochs+1
    for ep in range(epochstart,epochs+1):
     if ep:
      model.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
      for ii in order.split(64):opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(forward(ii,True),y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
     model.eval();r=dict(epoch=ep)
     if select:
      with torch.inference_mode():p=forward(validation).sigmoid().numpy()
      score=metrics(y[validation].numpy(),p)['auroc'];r['inner_auroc']=score
      if score>best:best=score;be=ep;stale=0;state=copy.deepcopy(model.state_dict())
      else:stale+=1
     hist.append(r);atomic(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,best_state=state,history=hist,mean=mean,std=std),rp);m.update(current_arm=arm,phase=tag,latest_epoch=ep);save()
     if select and stale>=8:break
    if select:model.load_state_dict(state);return be,hist
    atomic(dict(model=model.state_dict(),mean=mean,std=std,epoch=epochs),folder/'selected.pt')
    with torch.inference_mode():single=forward(validation,single=True).sigmoid().numpy();paired=forward(validation).sigmoid().numpy()
    return single,paired,hist
   selected,ih=run(fit,hold,40,True,'inner');single,paired,fh=run(tr,va,selected,False,'fullTRAIN');mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(arm=arm,resolution=resolution,source=source,selected_epoch=selected,inner_history=ih)
   for mode,p in [('singleview',single),('learned_paired_feature_average',paired),('fixed_probability_average',fixed)]:
    frame=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();frame['probability']=p;frame.to_csv(folder/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),p)
   (folder/'history.json').write_text(json.dumps(dict(inner=ih,fullTRAIN=fh),indent=2));m['results'].append(r);save();print(json.dumps(dict(arm=arm,selected_epoch=selected,paired=r['learned_paired_feature_average']['auroc'])),flush=True)
 assert all(digest(p)==sha for p,sha in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(d/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
