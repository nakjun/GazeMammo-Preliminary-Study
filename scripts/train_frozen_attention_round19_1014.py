"""Immutable independently fitted global MIL, matched-count gaze/random residual pilot."""
import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.gaze_spatial_mil_round09_1013 import SpatialGazeMIL
from src.gaze_frozen_attention_round19_1014 import FrozenAttentionResidual
from src.official_cbis_classifier_1012 import digest,state_digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round19_frozen_attention_pilot42');out.mkdir(exist_ok=True);old=out.parent/'round09_spatial_mil_pilot42';cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold].tolist()).issubset(set(hold.tolist()));oldm=json.load(open(old/'manifest.json'));assert digest('scripts/train_spatial_mil_round09_1013.py')==oldm['source_sha256'];innerpath=old/'global_only/inner_recovery.pt';fullpath=old/'global_only/selected.pt';inner=torch.load(innerpath,weights_only=False,map_location='cpu');full=torch.load(fullpath,weights_only=False,map_location='cpu');assert inner['be']==full['epoch']==12 and inner['best_state'] is not None;deps={str(p):digest(p) for p in [Path(__file__),Path('src/gaze_frozen_attention_round19_1014.py'),Path('src/gaze_spatial_mil_round09_1013.py'),innerpath,fullpath,Path(cp)]};m=dict(state='running',stage='cache_frozen_global',source_sha256=digest(__file__),dependencies=deps,test_used=False,oracle_used=False,seed=42,selection='TRAINinnerholdout only using globalinnerbest trained fitpatients; fullTRAIN global checkpoint fixed for finalVAL',inner_fit_patients=sorted(set(f.patient_key.iloc[fit])),inner_hold_patients=sorted(set(f.patient_key.iloc[hold])),global_initializer='round09 global_only independently selected innerbest/fullTRAIN,never joint gaze checkpoint',residual_bound=None,normalization_candidates=['standard','standard_L2sqrt1024'],scale_candidates=[.25,1],head_dim=8,zero_initial_output=True,lr=.0003,weight_decay=.5,dropout=.5,patch_dropout=.5,batch=64,max_epochs=30,patience=8,random_valid_count_matched_per_image=True,results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save();globalcache={};root=out.parent/'round08_global_resolution_pilot42/high_spatial_records'
 for tag,ck,state in [('inner',inner,inner['best_state']),('fullTRAIN',full,full['model'])]:
  path=out.parent/'round10_frozen_global_residual_pilot42'/f'global_{tag}_contexts.pt';assert path.exists()
  if path.exists():
   cache=torch.load(path,weights_only=False,map_location='cpu');assert cache['source_checkpoint_sha256']==digest(innerpath if tag=='inner' else fullpath) and cache['image_ids']==f.image_id.tolist();globalcache[tag]=cache;continue
  model=SpatialGazeMIL(local=False).eval().requires_grad_(False);model.load_state_dict(state);sha=state_digest(model.state_dict());contexts=[]
  with torch.inference_mode():
   for startidx in range(0,2393,16):
    rows=f.iloc[startidx:startidx+16];g=torch.stack([torch.load(root/f'{r.image_id}.pt',weights_only=False,map_location='cpu')['spatial'].flatten(1).T for r in rows.itertuples()]).float();g=((g-ck['gm'])/ck['gs']).clamp(-5,5);z,_=model.encode(g,torch.zeros(len(rows),4,1024),torch.zeros(len(rows),4,2),torch.zeros(len(rows),4,dtype=torch.bool));contexts.append(z)
   context=torch.cat(contexts);logits=model.global_head(context).flatten()
  assert state_digest(model.state_dict())==sha;cache=dict(image_ids=f.image_id.tolist(),context=context,logits=logits,source_checkpoint_sha256=digest(innerpath if tag=='inner' else fullpath),frozen_global_state_sha256=sha);atomic(cache,path);globalcache[tag]=cache
 # Exact zero-residual fullVAL prediction parity against immutable baseline CSV.
 cache=globalcache['fullTRAIN'];gl=cache['logits'];base=torch.where(available[va],.5*(gl[va]+gl[other[va]]),gl[va]).sigmoid().numpy();ref=pd.read_csv(old/'global_only/learned_paired_context_average_predictions.csv');assert ref.image_id.tolist()==f.iloc[va.tolist()].image_id.tolist();parity=float(np.abs(base-ref.probability).max());assert parity<1e-6;m['zero_residual_global_prediction_max_abs_error']=parity
 selected=torch.load(out.parent/'round12_gaze_extent_pilot42/diverse_selections.pt',weights_only=False,map_location='cpu');assert selected['image_ids']==f.image_id.tolist();sel=selected['selection']['8'];mask=sel['mask'];rs=[torch.load(out.parent/'round12_gaze_extent_pilot42/predicted_records_k16'/f"imitation_{r['image_id']}.pt",weights_only=False,map_location='cpu') for r in c['rows']];v=torch.stack([r['features'] for r in rs]).gather(1,sel['indices'][:,:,None].expand(-1,-1,1024));counts=mask.sum(1);rmask=torch.arange(8)[None]<counts[:,None];assert torch.equal(rmask.sum(1),counts);locals={'gaze8':(v,mask),'random8':(t['random_features'][:,:8],rmask)};m.update(stage='attention_train',VALID_mean_effective_K=float(counts[va].float().mean()),TRAIN_mean_effective_K=float(counts[tr].float().mean()),selected_indices_sha256=digest(out.parent/'round12_gaze_extent_pilot42/diverse_selections.pt'));save()
 # Actual global cache is immutable; zero output and subsequent attention gradients are checked independently.
 torch.manual_seed(42);sm=FrozenAttentionResidual(1).eval();ct=torch.randn(3,32);vv=torch.randn(3,8,1024);mk=torch.ones(3,8,dtype=torch.bool);use=torch.ones(3,dtype=torch.bool);outzero=sm(ct,ct,vv,vv,mk,mk,use);assert torch.equal(outzero,torch.zeros_like(outzero));opt=torch.optim.AdamW(sm.parameters(),lr=.001);outzero.sum().backward();assert float(sm.output.weight.grad.abs().sum())>0;opt.step();opt.zero_grad();sm(ct,ct,vv,vv,mk,mk,use).sum().backward();assert float(sm.query.weight.grad.abs().sum())>0;m['zero_parity_attention_gradient_smoke']=True;save()
 # Reuse global-only baseline as a frozen reference, not retrained.
 baseline=dict(arm='global_only',epoch=0)
 for mode in ['singleview','learned_paired_context_average','fixed_probability_average']:
  ref=pd.read_csv(old/'global_only'/f'{mode}_predictions.csv');baseline[mode]=metrics(ref.label,ref.probability)
 m['results'].append(baseline);save()
 for arm,(l,mask) in locals.items():
  folder=out/arm;folder.mkdir(exist_ok=True);candidates=[];config_records=[]
  def run(ids,validation,epochs,select,tag,normalization,scale,trial):
   torch.manual_seed(42);model=FrozenAttentionResidual(scale);ctx=globalcache[tag]['context'];logit=globalcache[tag]['logits'];valid=l[ids][mask[ids]];lm=valid.mean(0);ls=valid.std(0).clamp_min(.01);lc=((l-lm)/ls).clamp(-5,5);lc=torch.nn.functional.normalize(lc,dim=-1)*32 if normalization=='standard_L2sqrt1024' else lc;cm=ctx[ids].mean(0);cs=ctx[ids].std(0).clamp_min(.01);xc=((ctx-cm)/cs).clamp(-5,5);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.5);best=-1;be=0;stale=0;state=None;hist=[];rp=trial/f'{tag}_recovery.pt';resume=torch.load(rp,weights_only=False,map_location='cpu') if rp.exists() else None
   if resume:model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);best=resume['best'];be=resume['be'];stale=resume['stale'];state=resume['best_state'];hist=resume['history']
   def forward(ii,training=False,single=False):
    use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
    if training:use&=torch.rand(len(ii))>=.5
    cctx=torch.where(use[:,None],.5*(xc[ii]+xc[other[ii]]),xc[ii]);gl=torch.where(use,.5*(logit[ii]+logit[other[ii]]),logit[ii]);return gl+model(xc[ii],xc[other[ii]],lc[ii],lc[other[ii]],mask[ii],mask[other[ii]],use)
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
   if select:return be,hist,best
   atomic(dict(model=model.state_dict(),lm=lm,ls=ls,cm=cm,cs=cs,epoch=epochs,global_cache_sha256=digest(out.parent/'round10_frozen_global_residual_pilot42'/f'global_{tag}_contexts.pt')),folder/'selected.pt');return evaluate(validation,True),evaluate(validation),evaluate(tr),hist
  for normalization in m['normalization_candidates']:
   for scale in m['scale_candidates']:
    trial=folder/f'{normalization}_scale{scale}';trial.mkdir(exist_ok=True);epoch,history,score=run(fit,hold,30,True,'inner',normalization,scale,trial);candidates.append((score,epoch,normalization,scale,history,trial));config_records.append(dict(normalization=normalization,scale=scale,epoch=epoch,inner_auc=score))
  _,selected,normalization,scale,ih,trial=max(candidates,key=lambda x:x[0]);single,paired,tp,fh=run(tr,va,selected,False,'fullTRAIN',normalization,scale,trial);mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(arm=arm,epoch=selected,selected_normalization=normalization,selected_scale=scale,config_candidates=config_records,train_paired_auc=metrics(y[tr].numpy(),tp)['auroc'],inner_history=ih)
  for mode,p in [('singleview',single),('learned_paired_context_average',paired),('fixed_probability_average',fixed)]:
   df=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();df['probability']=p;df.to_csv(folder/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),p)
  (folder/'history.json').write_text(json.dumps(dict(inner=ih,fullTRAIN=fh),indent=2));m['results'].append(r);save();print(json.dumps(dict(arm=arm,epoch=selected,paired=r['learned_paired_context_average']['auroc'])),flush=True)
 assert all(digest(p)==sha for p,sha in deps.items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
