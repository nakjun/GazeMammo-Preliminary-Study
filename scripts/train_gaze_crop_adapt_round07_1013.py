"""Matched frozen-head continuation vs true final-CN-block local adaptation; TRAIN inner choice."""
import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views
from scripts.train_gaze_focus_selected_1013 import LowJoint,local_summary

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round07_crop_block_adapt_pilot42');heads=out/'heads';heads.mkdir(exist_ok=True);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold].tolist()).issubset(set(hold.tolist()));m=dict(state='running',stage='train',seed=42,warmup_max_epochs=30,warmup_patience=8,continuation_max_epochs=2,continuation_patience=2,backbone_lr=1e-5,head_lr=.0003,weight_decay=.05,paired_training=True,partner_dropout=.5,adapt_batch=4,head='same3072→64/dropout.5 meanmax jointhead',normalization='TRAINfit only cached pretrained pooled mean/std floor.01 clamp10; fixed during adaptation',selection='TRAINpatientinner20percent only; selectedepochs fresh fullTRAINrefit then VAL once',test_used=False,oracle_used=False,TRAIN_lesion_supervision_upstream=318,dependencies={p:digest(p) for p in [__file__,'scripts/extract_gaze_crop_boundary_round07_1013.py','scripts/train_gaze_focus_selected_1013.py','scripts/train_gaze_multiview_1012.py']},results=[])
 def save():
  p=heads/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save()
 for arm in ['global_only','imitation','lesion_aux','random']:
  folder=heads/arm;folder.mkdir(exist_ok=True);b=None;mask=None
  if arm=='global_only':x=torch.cat([t['global_features'],torch.zeros(2393,2048)],1)
  else:
   rs=[torch.load(out/'boundary_records'/f'{r.image_id}.pt',weights_only=False,map_location='cpu') for r in f.itertuples()];v=torch.stack([r['pooled'][arm] for r in rs]);mask=torch.stack([r['mask'][arm] for r in rs]);b=torch.stack([r['features'][arm] for r in rs]);del rs;x=local_summary(t['global_features'],v,mask)
  def warmup(ids,validation,epochs,select,tag):
   torch.manual_seed(42);head=LowJoint();mean=x[ids].mean(0);std=x[ids].std(0).clamp_min(.01);z=((x-mean)/std).clamp(-10,10);opt=torch.optim.AdamW(head.parameters(),lr=.0003,weight_decay=.05);best=-1;be=0;stale=0;state=None;history=[];p=folder/f'{tag}_warmup_recovery.pt';resume=torch.load(p,weights_only=False,map_location='cpu') if p.exists() else None
   if resume:head.load_state_dict(resume['head']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);best=resume['best'];be=resume['be'];stale=resume['stale'];state=resume['best_state'];history=resume['history']
   def logits(ii,train=False,single=False):
    use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
    if train:use&=torch.rand(len(ii))>=.5
    return head(torch.where(use[:,None],.5*(z[ii]+z[other[ii]]),z[ii]))
   for ep in range(resume['epoch']+1 if resume else 0,epochs+1):
    if ep:
     head.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
     for ii in order.split(64):opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(logits(ii,True),y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1);opt.step()
    head.eval();record=dict(epoch=ep)
    if select:
     with torch.inference_mode():prob=logits(validation).sigmoid().numpy()
     score=metrics(y[validation].numpy(),prob)['auroc'];record['inner_auroc']=score
     if score>best:best=score;be=ep;stale=0;state=copy.deepcopy(head.state_dict())
     else:stale+=1
    history.append(record);atomic(dict(head=head.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,best_state=state,history=history),p)
    if select and stale>=8:break
   if select:head.load_state_dict(state)
   return head,mean,std,be,history
  wh,imean,istd,we,history=warmup(fit,hold,30,True,'inner');fullhead,fmean,fstd,_,fullhistory=warmup(tr,va,we,False,'fullTRAIN');initial=copy.deepcopy(wh.state_dict());fullinitial=copy.deepcopy(fullhead.state_dict());variants=['frozen_head_continue'] if arm=='global_only' else ['frozen_head_continue','lastblock_adapt']
  for variant in variants:
   def stage(ids,validation,init,mean,std,epochs,select,tag):
    torch.manual_seed(42);head=LowJoint();head.load_state_dict(init);encoder=None;params=[dict(params=head.parameters(),lr=.0003)]
    if variant=='lastblock_adapt':encoder=GazePyramid('convnext_base').eval().requires_grad_(False);assert state_digest(encoder.state_dict())==c['provenance']['visual_state_sha256'];encoder.features[7][-1].requires_grad_(True);encoder.norm.requires_grad_(True);params.append(dict(params=list(encoder.features[7][-1].parameters())+list(encoder.norm.parameters()),lr=1e-5))
    opt=torch.optim.AdamW(params,weight_decay=.05);fixed=((x-mean)/std).clamp(-10,10);best=-1;be=0;stale=0;state=None;hist=[];p=folder/f'{variant}_{tag}_recovery.pt';resume=torch.load(p,weights_only=False,map_location='cpu') if p.exists() else None
    def states():return dict(head=copy.deepcopy(head.state_dict()),block=copy.deepcopy(encoder.features[7][-1].state_dict()) if encoder else None,norm=copy.deepcopy(encoder.norm.state_dict()) if encoder else None)
    def load(s):
     head.load_state_dict(s['head'])
     if encoder:encoder.features[7][-1].load_state_dict(s['block']);encoder.norm.load_state_dict(s['norm'])
    if resume:load(resume['model']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);best=resume['best'];be=resume['be'];stale=resume['stale'];state=resume['best_state'];hist=resume['history']
    def vectors(ii):
     if encoder:
      z=encoder.features[7][-1](b[ii].flatten(0,1).float());v=encoder.norm(z.mean((-1,-2),keepdim=True)).flatten(1).reshape(len(ii),4,1024);return ((local_summary(t['global_features'][ii],v,mask[ii])-mean)/std).clamp(-10,10)
     return fixed[ii]
    def forward(ii,training=False,single=False,seed=None):
     z=vectors(ii);use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
     if training:use&=torch.rand(len(ii),generator=torch.Generator().manual_seed(seed))>=.5
     if not single:z=torch.where(use[:,None],.5*(z+vectors(other[ii])),z)
     if training:torch.manual_seed(seed+1)
     return head(z)
    def evaluate(ii,single=False):
     head.eval()
     if encoder:encoder.eval()
     with torch.inference_mode():return torch.cat([forward(j,single=single).sigmoid() for j in ii.split(8)]).numpy()
    for ep in range(resume['epoch']+1 if resume else 0,epochs+1):
     tick=time.time()
     if ep:
      head.train()
      if encoder:encoder.features[7][-1].train();encoder.norm.train()
      order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
      for bi,ii in enumerate(order.split(4)):
       opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(forward(ii,True,seed=42+ep*10000+bi),y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_([p for group in params for p in group['params']],1);opt.step()
     record=dict(epoch=ep,seconds=time.time()-tick)
     if select:
      score=metrics(y[validation].numpy(),evaluate(validation))['auroc'];record['inner_auroc']=score
      if score>best:best=score;be=ep;stale=0;state=states()
      else:stale+=1
     hist.append(record);atomic(dict(model=states(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,best_state=state,history=hist),p);m.update(current_arm=arm,current_variant=variant,current_phase=tag,latest_epoch=ep);save()
     if select and stale>=2:break
    if select:load(state);return be,hist,None
    return head,hist,(evaluate(validation,True),evaluate(validation),states())
   selected,inner,_=stage(fit,hold,initial,imean,istd,2,True,'inner');head,rhist,result=stage(tr,va,fullinitial,fmean,fstd,selected,False,'fullTRAIN');single,paired,state=result;mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);dest=folder/variant;dest.mkdir(exist_ok=True);atomic(dict(state=state,mean=fmean,std=fstd,selected_epoch=selected,warmup_epoch=we,arm=arm,variant=variant),dest/'best.pt');r=dict(arm=arm,variant=variant,warmup_epoch=we,selected_epoch=selected,inner_history=inner)
   for mode,prob in [('singleview',single),('learned_paired_feature_average',paired),('fixed_probability_average',fixed)]:
    df=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();df['probability']=prob;df.to_csv(dest/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),prob)
   (dest/'history.json').write_text(json.dumps(dict(warmup_inner=history,warmup_fullTRAIN=fullhistory,continuation_inner=inner,continuation_fullTRAIN=rhist),indent=2));m['results'].append(r);save();print(json.dumps(dict(arm=arm,variant=variant,selected_epoch=selected,paired=r['learned_paired_feature_average']['auroc'])),flush=True)
 assert all(digest(p)==sha for p,sha in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(heads/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
