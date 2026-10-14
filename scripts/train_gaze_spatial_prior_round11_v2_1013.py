"""Immutable independently fitted global MIL, matched-count gaze/random residual pilot."""
import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.gaze_spatial_mil_round09_1013 import SpatialGazeMIL
from src.gaze_spatial_residual_round11_1013 import SpatialResidual
from src.official_cbis_classifier_1012 import digest,state_digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round11_gaze_spatial_prior_pilot42');out.mkdir(exist_ok=True);old=out.parent/'round09_spatial_mil_pilot42';cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold].tolist()).issubset(set(hold.tolist()));oldm=json.load(open(old/'manifest.json'));assert digest('scripts/train_spatial_mil_round09_1013.py')==oldm['source_sha256'];innerpath=old/'global_only/inner_recovery.pt';fullpath=old/'global_only/selected.pt';inner=torch.load(innerpath,weights_only=False,map_location='cpu');full=torch.load(fullpath,weights_only=False,map_location='cpu');assert inner['be']==full['epoch']==12 and inner['best_state'] is not None;deps={str(p):digest(p) for p in [Path(__file__),Path('src/gaze_spatial_residual_round11_1013.py'),Path('src/gaze_spatial_mil_round09_1013.py'),innerpath,fullpath,Path(cp)]};m=dict(state='running',stage='cache_frozen_global',source_sha256=digest(__file__),dependencies=deps,test_used=False,oracle_used=False,seed=42,selection='TRAINinnerholdout only using globalinnerbest trained fitpatients; fullTRAIN global checkpoint fixed for finalVAL',inner_fit_patients=sorted(set(f.patient_key.iloc[fit])),inner_hold_patients=sorted(set(f.patient_key.iloc[hold])),global_initializer='round09 global_only independently selected innerbest/fullTRAIN,never joint gaze checkpoint',residual_bound=.5,head_dim=16,zero_initial_output=True,local_input='gazeinformed globaltokens only,noactualcropfeatures',lr=.0003,weight_decay=.2,dropout=.5,patch_dropout=0,batch=64,max_epochs=30,patience=8,random_valid_count_matched_per_image=True,results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save();globalcache={};token_cache={};root=out.parent/'round08_global_resolution_pilot42/high_spatial_records'
 for tag,ck,state in [('inner',inner,inner['best_state']),('fullTRAIN',full,full['model'])]:
  path=out/f'global_{tag}_tokens.pt';contextpath=out.parent/'round10_frozen_global_residual_pilot42'/f'global_{tag}_contexts.pt';cache=torch.load(contextpath,weights_only=False,map_location='cpu');assert cache['image_ids']==f.image_id.tolist() and cache['source_checkpoint_sha256']==digest(innerpath if tag=='inner' else fullpath);globalcache[tag]=cache;model=SpatialGazeMIL(local=False).eval().requires_grad_(False);model.load_state_dict(state);sha=state_digest(model.state_dict())
  if path.exists():
   tokens=torch.load(path,weights_only=False,map_location='cpu');assert tokens['source_checkpoint_sha256']==cache['source_checkpoint_sha256'];token_cache[tag]=tokens
  else:
   projected=[];attention=[]
   with torch.inference_mode():
    for startidx in range(0,2393,16):
     rows=f.iloc[startidx:startidx+16];g=torch.stack([torch.load(root/f'{r.image_id}.pt',weights_only=False,map_location='cpu')['spatial'].flatten(1).T for r in rows.itertuples()]).float();g=((g-ck['gm'])/ck['gs']).clamp(-5,5);z=model.project(g)+.1*model.position(model.xy*2-1)[None];w=model.attention(z).squeeze(-1).softmax(1);projected.append(z);attention.append(w)
   tokens=dict(projected=torch.cat(projected),attention=torch.cat(attention),xy=model.xy,source_checkpoint_sha256=cache['source_checkpoint_sha256']);atomic(tokens,path);token_cache[tag]=tokens
  assert state_digest(model.state_dict())==sha
 # Exact zero-residual fullVAL prediction parity against immutable baseline CSV.
 cache=globalcache['fullTRAIN'];gl=cache['logits'];base=torch.where(available[va],.5*(gl[va]+gl[other[va]]),gl[va]).sigmoid().numpy();ref=pd.read_csv(old/'global_only/learned_paired_context_average_predictions.csv');assert ref.image_id.tolist()==f.iloc[va.tolist()].image_id.tolist();parity=float(np.abs(base-ref.probability).max());assert parity<1e-6;m['zero_residual_global_prediction_max_abs_error']=parity
 rs=[torch.load(out.parent/'round06_lesion_trajectory_pilot42/predicted_records_k4'/f"imitation_{r['image_id']}.pt",weights_only=False,map_location='cpu') for r in c['rows']];coords=torch.stack([r['coordinates'] for r in rs]);gmask=torch.stack([r['mask'] for r in rs]);del rs
 for i in range(2393):
  seen=set()
  for j in range(4):
   key=tuple(torch.round(coords[i,j].double()*torch.tensor([1500.,3000.])).tolist())
   if key in seen:gmask[i,j]=False
   else:seen.add(key)
 counts=gmask.sum(1);rmask=torch.arange(4)[None]<counts[:,None];assert torch.equal(rmask.sum(1),counts);sources={'imitation':(coords,gmask),'random_matched':(t['random_coordinates'][:,:4],rmask)};locals={};coverage={}
 for arm,(coordinates,mask) in sources.items():
  summaries={}
  for tag,ck,state in [('inner',inner,inner['best_state']),('fullTRAIN',full,full['model'])]:
   tok=token_cache[tag];xy=tok['xy']*torch.tensor([1500.,3000.]);centers=coordinates*torch.tensor([1500.,3000.]);distance=(xy[None,:,None]-centers[:,None]).square().sum(-1);prior=(torch.exp(-distance/(2*112.**2))*mask[:,None]).amax(-1);weights=prior*tok['attention'];weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8);pooled=(tok['projected']*weights[...,None]).sum(1);mean=tok['projected'].mean(1);model=SpatialGazeMIL(local=False).eval().requires_grad_(False);model.load_state_dict(state)
   with torch.inference_mode():summaries[tag]=model.context(torch.cat([pooled,mean],1))
   assert torch.isfinite(summaries[tag]).all()
  locals[arm]=summaries;coverage[arm]=float((prior[va]>=torch.exp(torch.tensor(-.5))).sum(1).float().mean())
 m.update(stage='spatial_residual_train',VALID_mean_effective_K=float(counts[va].float().mean()),TRAIN_mean_effective_K=float(counts[tr].float().mean()),prior='max-union Gaussian sigma112canonicalpixels, same224crop half-width;original globalattention multiplied,renormalized',VALID_mean_prior_tokens_within_sigma=coverage,actual_crop_vectors_used=False);save()
 # Reuse global-only baseline as a frozen reference, not retrained.
 baseline=dict(arm='global_only',epoch=0)
 for mode in ['singleview','learned_paired_context_average','fixed_probability_average']:
  ref=pd.read_csv(old/'global_only'/f'{mode}_predictions.csv');baseline[mode]=metrics(ref.label,ref.probability)
 m['results'].append(baseline);save()
 for arm,summaries in locals.items():
  folder=out/arm;folder.mkdir(exist_ok=True)
  def run(ids,validation,epochs,select,tag):
   torch.manual_seed(42);model=SpatialResidual();ctx=globalcache[tag]['context'];logit=globalcache[tag]['logits'];l=summaries[tag];lm=l[ids].mean(0);ls=l[ids].std(0).clamp_min(.01);lc=((l-lm)/ls).clamp(-5,5);cm=ctx[ids].mean(0);cs=ctx[ids].std(0).clamp_min(.01);xc=((ctx-cm)/cs).clamp(-5,5);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.2);best=-1;be=0;stale=0;state=None;hist=[];rp=folder/f'{tag}_recovery.pt';resume=torch.load(rp,weights_only=False,map_location='cpu') if rp.exists() else None
   if resume:model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);best=resume['best'];be=resume['be'];stale=resume['stale'];state=resume['best_state'];hist=resume['history']
   def forward(ii,training=False,single=False):
    use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
    if training:use&=torch.rand(len(ii))>=.5
    cctx=torch.where(use[:,None],.5*(xc[ii]+xc[other[ii]]),xc[ii]);gl=torch.where(use,.5*(logit[ii]+logit[other[ii]]),logit[ii]);return gl+model(cctx,lc[ii],lc[other[ii]],use)
   def evaluate(ii,single=False):
    model.eval()
    with torch.inference_mode():return torch.cat([forward(j,single=single).sigmoid() for j in ii.split(64)]).numpy()
   first=resume['epoch']+1 if resume else 0
   if select and stale>=8:first=epochs+1
   for ep in range(first,epochs+1):
    if ep:
     model.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
     for ii in order.split(64):opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(forward(ii,True),y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    r=dict(epoch=ep)
    if select:
     score=metrics(y[validation].numpy(),evaluate(validation))['auroc'];r['inner_auroc']=score
     if score>best:best=score;be=ep;stale=0;state=copy.deepcopy(model.state_dict())
     else:stale+=1
    hist.append(r);atomic(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,best_state=state,history=hist,lm=lm,ls=ls,cm=cm,cs=cs),rp);m.update(current_arm=arm,phase=tag,latest_epoch=ep);save()
    if select and stale>=8:break
   if select:return be,hist
   atomic(dict(model=model.state_dict(),lm=lm,ls=ls,cm=cm,cs=cs,epoch=epochs,global_cache_sha256=digest(out.parent/'round10_frozen_global_residual_pilot42'/f'global_{tag}_contexts.pt')),folder/'selected.pt');return evaluate(validation,True),evaluate(validation),evaluate(tr),hist
  selected,ih=run(fit,hold,30,True,'inner');single,paired,tp,fh=run(tr,va,selected,False,'fullTRAIN');mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(arm=arm,epoch=selected,train_paired_auc=metrics(y[tr].numpy(),tp)['auroc'],inner_history=ih)
  for mode,p in [('singleview',single),('learned_paired_context_average',paired),('fixed_probability_average',fixed)]:
   df=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();df['probability']=p;df.to_csv(folder/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),p)
  (folder/'history.json').write_text(json.dumps(dict(inner=ih,fullTRAIN=fh),indent=2));m['results'].append(r);save();print(json.dumps(dict(arm=arm,epoch=selected,paired=r['learned_paired_context_average']['auroc'])),flush=True)
 assert all(digest(p)==sha for p,sha in deps.items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
