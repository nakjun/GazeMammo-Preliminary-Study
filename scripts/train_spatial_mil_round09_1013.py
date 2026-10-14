"""Cached high spatialMIL matched global/imitation/random, TRAIN-inner epoch choice."""
import sys,time,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.gaze_spatial_mil_round09_1013 import SpatialGazeMIL
from src.official_cbis_classifier_1012 import digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round09_spatial_mil_pilot42');out.mkdir(exist_ok=True);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold].tolist()).issubset(set(hold.tolist()));root=out.parent/'round08_global_resolution_pilot42/high_spatial_records';g=torch.stack([torch.load(root/f'{r.image_id}.pt',weights_only=False,map_location='cpu')['spatial'].flatten(1).T for r in f.itertuples()]);assert g.shape==(2393,512,1024);local={};counts={}
 for source in ['global_only','imitation','random']:
  if source=='global_only':l=torch.zeros(2393,4,1024);coords=torch.zeros(2393,4,2);mask=torch.zeros(2393,4,dtype=torch.bool)
  elif source=='random':l=t['random_features'][:,:4];coords=t['random_coordinates'][:,:4];mask=t['random_mask'][:,:4].clone()
  else:
   rs=[torch.load(out.parent/'round06_lesion_trajectory_pilot42/predicted_records_k4'/f"imitation_{r['image_id']}.pt",weights_only=False,map_location='cpu') for r in c['rows']];l=torch.stack([r['features'] for r in rs]);coords=torch.stack([r['coordinates'] for r in rs]);mask=torch.stack([r['mask'] for r in rs]);del rs
  for i in range(2393):
   seen=set()
   for j in range(4):
    key=tuple(torch.round(coords[i,j].double()*torch.tensor([1500.,3000.])).tolist())
    if key in seen:mask[i,j]=False
    else:seen.add(key)
  local[source]=(l,coords,mask);counts[source]=float(mask[va].sum(1).float().mean())
 def global_norm(ids):
  total=0;s=torch.zeros(1024,dtype=torch.float64);sq=s.clone()
  for ii in ids.split(16):z=g[ii].float();s+=z.sum((0,1)).double();sq+=z.square().sum((0,1)).double();total+=z.shape[0]*512
  mean=(s/total).float();std=((sq-s.square()/total)/(total-1)).clamp_min(0).sqrt().float().clamp_min(.01);return mean,std
 norms={'inner':global_norm(fit),'fullTRAIN':global_norm(tr)};m=dict(state='running',seed=42,source_sha256=digest(__file__),dependencies={p:digest(p) for p in [__file__,'src/gaze_spatial_mil_round09_1013.py','scripts/train_gaze_multiview_1012.py',str(out.parent/'round08_global_resolution_pilot42/extraction_manifest.json')]},CNN_extraction=False,backbone_frozen=True,global_tokens=512,dim=32,dropout=.4,context_dropout=.5,patch_dropout=.3,weight_decay=.1,lr=.0003,batch=16,max_epochs=40,patience=8,normalization='TRAINfit channelmean/std separately global/local;clamp5;global statsshared across3arms',selection='TRAINpatientinner20percent AUROC;freshfullTRAINrefit;VAL356once',paired_mode='learnedglobal/localcontextaverage;fixedprobavg separate',test_used=False,oracle_used=False,valid_local_counts=counts,results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save()
 for arm,(l,coords,mask) in local.items():
  folder=out/arm;folder.mkdir(exist_ok=True)
  def run(ids,validation,epochs,select,tag):
   torch.manual_seed(42);model=SpatialGazeMIL(local=arm!='global_only');gm,gs=norms[tag];valid=l[ids][mask[ids]];lm=valid.mean(0) if len(valid) else torch.zeros(1024);ls=valid.std(0).clamp_min(.01) if len(valid)>1 else torch.ones(1024);lp=((l-lm)/ls).clamp(-5,5);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);best=-1;be=0;stale=0;state=None;history=[];rp=folder/f'{tag}_recovery.pt';resume=torch.load(rp,weights_only=False,map_location='cpu') if rp.exists() else None
   if resume:model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);best=resume['best'];be=resume['be'];stale=resume['stale'];state=resume['best_state'];history=resume['history']
   def forward(ii,training=False,single=False):
    use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
    if training:use&=torch.rand(len(ii))>=.5
    a=((g[ii].float()-gm)/gs).clamp(-5,5);p=((g[other[ii]].float()-gm)/gs).clamp(-5,5);return model(a,lp[ii],coords[ii],mask[ii],p,lp[other[ii]],coords[other[ii]],mask[other[ii]],use)
   def evaluate(ii,single=False):
    model.eval()
    with torch.inference_mode():return torch.cat([forward(j,single=single).sigmoid() for j in ii.split(16)]).numpy()
   first=resume['epoch']+1 if resume else 0
   if select and stale>=8:first=epochs+1
   for ep in range(first,epochs+1):
    tick=time.time()
    if ep:
     model.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
     for ii in order.split(16):opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(forward(ii,True),y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    r=dict(epoch=ep,seconds=time.time()-tick)
    if select:
     score=metrics(y[validation].numpy(),evaluate(validation))['auroc'];r['inner_auroc']=score
     if score>best:best=score;be=ep;stale=0;state=copy.deepcopy(model.state_dict())
     else:stale+=1
    history.append(r);atomic(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,best_state=state,history=history,gm=gm,gs=gs,lm=lm,ls=ls),rp);m.update(current_arm=arm,phase=tag,latest_epoch=ep);save()
    if select and stale>=8:break
   if select:return be,history
   atomic(dict(model=model.state_dict(),gm=gm,gs=gs,lm=lm,ls=ls,epoch=epochs,arm=arm),folder/'selected.pt');return evaluate(validation,True),evaluate(validation),evaluate(tr),history
  selected,ih=run(fit,hold,40,True,'inner');single,paired,tp,fh=run(tr,va,selected,False,'fullTRAIN');mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(arm=arm,selected_epoch=selected,train_paired_auc=metrics(y[tr].numpy(),tp)['auroc'],inner_history=ih)
  for mode,p in [('singleview',single),('learned_paired_context_average',paired),('fixed_probability_average',fixed)]:
   frame=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();frame['probability']=p;frame.to_csv(folder/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),p)
  (folder/'history.json').write_text(json.dumps(dict(inner=ih,fullTRAIN=fh),indent=2));m['results'].append(r);save();print(json.dumps(dict(arm=arm,epoch=selected,paired=r['learned_paired_context_average']['auroc'],train=r['train_paired_auc'])),flush=True)
 assert all(digest(p)==sha for p,sha in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
