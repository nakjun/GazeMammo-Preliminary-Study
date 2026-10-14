"""Round06: source-only patientOOF rank→small global+mean/max joint classifier."""
import json,sys,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.official_cbis_classifier_1012 import digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

class LowJoint(torch.nn.Module):
 def __init__(self):
  super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(3072,64),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(64,1))
 def forward(self,x):return self.net(x).flatten()

def select_source(scores,coords,mask):
 xy=torch.round(coords.double()*torch.tensor([1500.,3000.]));seen=set();chosen=[]
 for j in sorted(range(16),key=lambda i:(-float(abs(scores[i])),i)):
  key=tuple(xy[j].tolist())
  if bool(mask[j]) and key not in seen:chosen.append(j);seen.add(key)
 return chosen

def local_summary(global_features,features,mask):
 mean=(features*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];maximum=features.masked_fill(~mask[...,None],float('-inf')).max(1).values;maximum=torch.where(torch.isfinite(maximum),maximum,torch.zeros_like(maximum));return torch.cat([global_features,mean,maximum],1)

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42/classifiers_k4');out.mkdir(parents=True,exist_ok=False);start=time.time();cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert len(f)==2393 and (f.split=='val').sum()==356 and set(f.split)=={'train','val'} and f.groupby('patient_key').split.nunique().max()==1;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);zero=torch.zeros_like(t['global_features']);summaries={'global_only':torch.cat([t['global_features'],zero,zero],1)};deps={p:digest(p) for p in [__file__,'scripts/train_gaze_multiview_1012.py']};counts={}
 for source in ['imitation','lesion_aux','random']:
  if source=='random':p=t['random_features'][:,:4];coords=t['random_coordinates'][:,:4];valid=t['random_mask'][:,:4].clone()
  else:
   records=[torch.load(out.parent/'predicted_records_k4'/f"{source}_{row['image_id']}.pt",map_location='cpu',weights_only=False) for row in c['rows']];assert [r['image_id'] for r in records]==f.image_id.tolist();p=torch.stack([r['features'] for r in records]);coords=torch.stack([r['coordinates'] for r in records]);valid=torch.stack([r['mask'] for r in records])
  for i in range(2393):
   seen=set()
   for j in range(4):
    pixel=tuple(torch.round(coords[i,j].double()*torch.tensor([1500.,3000.])).tolist())
    if pixel in seen:valid[i,j]=False
    else:seen.add(pixel)
  summaries[source+'_k4']=local_summary(t['global_features'],p,valid);counts[source+'_k4']={split:float(valid[torch.tensor(np.flatnonzero(f.split==split))].sum(1).float().mean()) for split in ['train','val']}
 deps[str(out.parent/'prediction_manifest_k4.json')]=digest(out.parent/'prediction_manifest_k4.json')
 m=dict(state='running',target=.88,seed42_pilot=True,test_used=False,oracle_used=False,trajectory_retrained=True,upstream_trajectory_seed42=True,selection='fresh patientOOF trajectories TRAIN/fullTRAIN trajectories VAL;first4chronological predictions/pixeldup mask;no expert input',cache_sha256=digest(cp),source_sha256=digest(__file__),dependencies=deps,head='global+maskedselectedmean/max3072→MLP64/dropout.5;globaltrainable',epochs=60,patience=12,lr=.0003,weight_decay=.05,batch=64,normalization='TRAINonlymean/std floor.01 clamp10',training='pairedsummaryfeatureaverage withpartnerdropout.5,bagBCE',selection_metric='learnedpairedVALAUROC',valid_source_counts=counts,results=[])
 def save():
  path=out/'manifest.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(m,indent=2));temp.replace(path);temp=out/'progress.tmp';pd.DataFrame(m['results']).to_csv(temp,index=False);temp.replace(out/'progress.csv')
 save()
 for arm,x in summaries.items():
  mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(.01);x=((x-mean)/std).clamp(-10,10);torch.manual_seed(42);model=LowJoint();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05);d=out/arm;d.mkdir();best=-1;stale=0;history=[]
  def forward(ids,mode):
   use=available[ids].clone() if mode!='single' else torch.zeros(len(ids),dtype=torch.bool)
   if mode=='train':use&=torch.rand(len(ids))>=.5
   mixed=torch.where(use[:,None],.5*(x[ids]+x[other[ids]]),x[ids]);return model(mixed)
  for epoch in range(61):
   losses=[]
   if epoch:
    model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(42+epoch))]
    for ids in order.split(64):
     opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(forward(ids,'train'),y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.inference_mode():p=forward(va,'paired').sigmoid().numpy()
   score=metrics(y[va].numpy(),p);history.append(dict(epoch=epoch,fitloss=float(np.mean(losses)) if losses else None,paired_validation=score))
   if score['auroc']>best:best=score['auroc'];stale=0;be=epoch;state=copy.deepcopy(model.state_dict());torch.save(dict(state=state,mean=mean,std=std,epoch=epoch,arm=arm,seed=42),d/'best.pt')
   else:stale+=1
   m['current_arm']=arm;m['latest_epoch']=epoch;path=out/'manifest.json';temp=path.with_suffix('.tmp');temp.write_text(json.dumps(m,indent=2));temp.replace(path)
   if stale>=12:break
  model.load_state_dict(state);model.eval()
  with torch.inference_mode():single=forward(va,'single').sigmoid().numpy();paired=forward(va,'paired').sigmoid().numpy();trainp=torch.cat([forward(ids,'paired').sigmoid() for ids in tr.split(64)]).numpy()
  mapping=dict(zip(f.iloc[va.tolist()].image_id,single));fixed=np.array([.5*(mapping[f.iloc[i].image_id]+mapping[f.iloc[partners[i][0]].image_id]) if partners[i] else mapping[f.iloc[i].image_id] for i in va.tolist()]);r=dict(arm=arm,best_epoch=be,stopped_epoch=epoch,train_paired_auroc=metrics(y[tr].numpy(),trainp)['auroc'])
  for mode,p in [('singleview',single),('learned_paired_feature_average',paired),('fixed_probability_average',fixed)]:
   pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=p;pred.to_csv(d/f'{mode}_predictions.csv',index=False);r[mode]=metrics(pred.label,pred.probability)
  (d/'history.json').write_text(json.dumps(history));m['results'].append(r);save();print(json.dumps(dict(arm=arm,single=r['singleview']['auroc'],paired=r['learned_paired_feature_average']['auroc'],fixedavg=r['fixed_probability_average']['auroc'],train=r['train_paired_auroc'])),flush=True)
 assert all(digest(p)==sha for p,sha in deps.items()) and digest(cp)==m['cache_sha256'];m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(out/'report.json').write_text(json.dumps(dict(results=m['results'],valid_source_counts=counts,limitations='Singlepilot42,not3seedmean. SameK4nominalvalidcounts may differ withpixeldup masks. Upstream freshpatientOOFtrajectory;318TRAINbbox auxiliarysupervision;globalfeaturepooledImageNetmaylimitsignal. Learnedpairedfeatureaverage is notfixedprobabilityaverage. NoTEST/oracle.'),indent=2));(out.parent/'progress.json').write_text(json.dumps(dict(active_goal='wholeimage+predictedgaze3seedmean>=.88',round=6,round_output=str(out),round_state='complete',pilot_results=m['results']),indent=2))
if __name__=='__main__':main()
