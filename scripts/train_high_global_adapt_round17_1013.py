"""Strong highresolution spatial MIL continuation vs true last-CN-block adaptation."""
import sys,json,time,copy,resource
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.gaze_backbones import GazePyramid
from src.gaze_spatial_mil_round09_1013 import SpatialGazeMIL
from src.gaze_frozen_residual_round10_1013 import LocalResidual
from src.official_cbis_classifier_1012 import digest,state_digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
class Head(torch.nn.Module):
 def __init__(self,state,local):
  super().__init__();self.global_model=SpatialGazeMIL(local=False);self.global_model.load_state_dict(state);self.local_model=LocalResidual();self.local=local
  for name in ['query','local_head','gate']:getattr(self.global_model,name).requires_grad_(False)
 def forward(self,g,pg,v,pv,mask,pmask,use):
  coords=torch.zeros(len(g),4,2);z,_=self.global_model.encode(g,v,coords,mask);pz,_=self.global_model.encode(pg,pv,coords,pmask);z=torch.where(use[:,None],.5*(z+pz),z);logit=self.global_model.global_head(z).flatten()
  if self.local:logit=logit+self.local_model(z,v,mask,pv,pmask,use)
  return logit

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);tick=time.time();root=Path('runs/gaze_focus_auroc88_1013');out=root/'round17_high_global_adapt_pilot42';heads=out/'heads';heads.mkdir(exist_ok=True);cp=Path('runs/official_cbis_v2_experiment_1012/development.pt');old=root/'round09_spatial_mil_pilot42';innerp=old/'global_only/inner_recovery.pt';fullp=old/'global_only/selected.pt';extract=out/'extraction_manifest.json';assert all(p.exists() for p in [cp,innerp,fullp,extract]);em=json.load(open(extract));assert em['state']=='complete' and em['completed']==2393;c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);y=torch.tensor(f.label.values,dtype=torch.float32);tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;fi,hi=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fi];hold=tr[hi];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);assert set(other[hold].tolist())<=set(hold.tolist());ck={'inner':torch.load(innerp,weights_only=False,map_location='cpu'),'fullTRAIN':torch.load(fullp,weights_only=False,map_location='cpu')};assert {'best_state','gm','gs','be'}<=ck['inner'].keys() and ck['inner']['be']==ck['fullTRAIN']['epoch']==12;assert digest('scripts/train_spatial_mil_round09_1013.py')==json.load(open(old/'manifest.json'))['source_sha256'];encoder=GazePyramid('convnext_base').eval().requires_grad_(False);sha=state_digest(encoder.state_dict());assert sha==c['provenance']['visual_state_sha256'];initial=copy.deepcopy(encoder.features[7][-1].state_dict());boundary=[];local=[];gmask=[]
 for row in f.itertuples():
  p=out/'boundary_records'/f'{row.image_id}.pt';r=torch.load(p,weights_only=False,map_location='cpu');assert r['image_id']==row.image_id and r['encoder_sha256']==sha and r['pixel_sha256']==row.image_sha256 and digest(p)==em['records'][str(row.image_id)];boundary.append(r['spatial']);rec=torch.load(root/'round07_crop_block_adapt_pilot42/boundary_records'/f'{row.image_id}.pt',weights_only=False,map_location='cpu');assert rec['visual_sha256']==sha;local.append(rec['pooled']['imitation']);gmask.append(rec['mask']['imitation'])
 b=torch.stack(boundary);del boundary;l=torch.stack(local);mask=torch.stack(gmask);del local;rmask=torch.arange(4)[None]<mask.sum(1)[:,None];assert torch.equal(mask.sum(1),rmask.sum(1));rl=c['tensors']['random_features'][:,:4];m=dict(state='running',stage='restore_and_batch_benchmark',seed=42,backbone_architecture='original ConvNeXtBase; frozen prefix, original lastCNblock only adaptation',original_norm='Classifier norm outside raw-map spatialMIL path remains unchanged; no normalization layer inserted',max_epochs=10,patience=3,epoch0_included=True,head_lr=3e-5,backbone_lr=1e-6,weight_decay=.05,gradient_clip=1,block_stochastic_depth='eval for all arms; weights differentiable, dropout RNG matched',paired='global contexts averaged, partner dropout.5',local_head='zero-init dim16 boundedmeanmax residual; frozen crop encoder',VALID_effective_K=float(mask[va].sum(1).float().mean()),test_used=False,oracle_used=False,inner_fit_patients=sorted(set(f.patient_key.iloc[fit])),inner_hold_patients=sorted(set(f.patient_key.iloc[hold])),dependencies={str(p):digest(p) for p in [Path(__file__),Path('src/gaze_spatial_mil_round09_1013.py'),Path('src/gaze_frozen_residual_round10_1013.py'),Path('scripts/train_gaze_multiview_1012.py'),innerp,fullp,extract]},results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save();fp=out/'restored_frozen_maps.pt'
 if fp.exists():fm=torch.load(fp,weights_only=False,map_location='cpu');assert fm['image_ids']==f.image_id.tolist();frozen=fm['maps']
 else:
  parts=[]
  with torch.inference_mode():
   for ii in torch.arange(2393).split(4):parts.append(encoder.features[7][-1](b[ii].float()).half())
  frozen=torch.cat(parts);atomic(dict(image_ids=f.image_id.tolist(),maps=frozen,visual_sha256=sha),fp)
 oldsample=torch.stack([torch.load(root/'round08_global_resolution_pilot42/high_spatial_records'/f'{i}.pt',weights_only=False,map_location='cpu')['spatial'] for i in f.image_id.tolist()[:4]]).double();newsample=frozen[:4].double();cos=float(torch.nn.functional.cosine_similarity(newsample.flatten(1),oldsample.flatten(1)).min());rel=float((newsample-oldsample).norm()/oldsample.norm());assert cos>.99999 and rel<.001;m.update(restored_map_min_cos_float64=cos,restored_map_relative_L2=rel);save()
 torch.manual_seed(42);sm=Head(ck['inner']['best_state'],True).eval();base=Head(ck['inner']['best_state'],False).eval();ii=fit[:2];sg=((frozen[ii].float().flatten(2).transpose(1,2)-ck['inner']['gm'])/ck['inner']['gs']).clamp(-5,5);zeros=torch.zeros(2,4,1024);false=torch.zeros(2,4,dtype=torch.bool);use=torch.ones(2,dtype=torch.bool);a=sm(sg,sg,zeros,zeros,false,false,use);ref=base(sg,sg,zeros,zeros,false,false,use);assert torch.equal(a,ref);a.sum().backward();assert float(sm.local_model.net[-1].weight.grad.abs().sum())>0;m['zero_local_parity_gradient_smoke']=True;save()
 # Benchmark batch2/4 without changing any training initializer.
 bench=[]
 for batch in [2,4]:
  torch.manual_seed(42);block=copy.deepcopy(encoder.features[7][-1]).requires_grad_(True).eval();head=Head(ck['inner']['best_state'],True);opt=torch.optim.AdamW([dict(params=block.parameters(),lr=1e-6),dict(params=[p for p in head.parameters() if p.requires_grad],lr=3e-5)],weight_decay=.05);ii=fit[:batch];vi=l[ii];mi=mask[ii];gm=ck['inner']['gm'];gs=ck['inner']['gs']
  def step():
   opt.zero_grad();z=block(torch.cat([b[ii],b[other[ii]]]).float()).half().float().flatten(2).transpose(1,2);z=((z-gm)/gs).clamp(-5,5);logit=head(z[:batch],z[batch:],vi,l[other[ii]],mi,mask[other[ii]],available[ii]);loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,y[ii]);loss.backward();opt.step()
  step();t=time.time();step();seconds=time.time()-t;bench.append(dict(batch=batch,seconds=seconds,seconds_per_image=seconds/batch))
 batch=min(bench,key=lambda r:r['seconds_per_image'])['batch'];m.update(batch=batch,batch_benchmark=bench,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,stage='train');save();print(json.dumps(dict(stage='train',batch=batch,benchmark=bench)),flush=True)
 arms={'frozen_head_continue':(False,False,l,torch.zeros_like(mask)),'adapted_global_only':(True,False,l,torch.zeros_like(mask)),'adapted_gaze4':(True,True,l,mask),'adapted_random4':(True,True,rl,rmask)}
 for arm,(adapt,haslocal,v,vmask) in arms.items():
  folder=heads/arm;folder.mkdir(exist_ok=True)
  def run(ids,validation,epochs,select,tag):
   torch.manual_seed(42);base=ck[tag];state=base['best_state'] if tag=='inner' else base['model'];head=Head(state,haslocal);block=copy.deepcopy(encoder.features[7][-1]).eval().requires_grad_(adapt);gm=base['gm'];gs=base['gs'];valid=v[ids][vmask[ids]];lm=valid.mean(0) if len(valid) else torch.zeros(1024);ls=valid.std(0).clamp_min(.01) if len(valid)>1 else torch.ones(1024);vl=((v-lm)/ls).clamp(-5,5);params=[dict(params=[p for p in head.parameters() if p.requires_grad],lr=3e-5)]
   if adapt:params.append(dict(params=list(block.parameters()),lr=1e-6))
   opt=torch.optim.AdamW(params,weight_decay=.05);rp=folder/f'{tag}_resume.pt';best=-1;be=0;stale=0;hist=[];first=0
   if rp.exists():r=torch.load(rp,weights_only=False,map_location='cpu');head.load_state_dict(r['head']);block.load_state_dict(r['block']);opt.load_state_dict(r['optimizer']);torch.set_rng_state(r['rng']);best=r['best'];be=r['be'];stale=r['stale'];hist=r['history'];first=r['epoch']+1
   def forward(ii,training=False,single=False):
    use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
    if training:use&=torch.rand(len(ii))>=.5
    jj=torch.cat([ii,other[ii]]);z=(block(b[jj].float()).half().float() if adapt else frozen[jj].float()).flatten(2).transpose(1,2);z=((z-gm)/gs).clamp(-5,5);return head(z[:len(ii)],z[len(ii):],vl[ii],vl[other[ii]],vmask[ii],vmask[other[ii]],use)
   def evaluate(ii,single=False):
    head.eval()
    with torch.inference_mode():return torch.cat([forward(j,single=single).sigmoid() for j in ii.split(batch)]).numpy()
   if select and stale>=3:first=epochs+1
   for ep in range(first,epochs+1):
    t=time.time()
    if ep:
     head.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
     for bi,ii in enumerate(order.split(batch)):torch.manual_seed(42+ep*10000+bi);opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(forward(ii,True),y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_([p for group in params for p in group['params']],1);opt.step()
    row=dict(epoch=ep,train_seconds=time.time()-t)
    if select:
     score=metrics(y[validation].numpy(),evaluate(validation))['auroc'];row['inner_auroc']=score
     if score>best:best=score;be=ep;stale=0
     else:stale+=1
    hist.append(row);atomic(dict(head=head.state_dict(),block=block.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,history=hist,gm=gm,gs=gs,lm=lm,ls=ls),rp);m.update(current_arm=arm,current_phase=tag,epoch=ep);save()
    if select and stale>=3:break
   if select:return be,hist
   atomic(dict(head=head.state_dict(),block=block.state_dict(),epoch=epochs,gm=gm,gs=gs,lm=lm,ls=ls,adapt=adapt,haslocal=haslocal,initial_global_checkpoint_sha256=digest(innerp if tag=='inner' else fullp)),folder/'selected.pt');return evaluate(validation,True),evaluate(validation),evaluate(tr),hist
  selected,ih=run(fit,hold,10,True,'inner');single,paired,tp,fh=run(tr,va,selected,False,'fullTRAIN');mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(arm=arm,epoch=selected,inner_history=ih,train_paired_auc=metrics(y[tr].numpy(),tp)['auroc'])
  for mode,pr in [('singleview',single),('learned_paired_context_average',paired),('fixed_probability_average',fixed)]:df=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();df['probability']=pr;df.to_csv(folder/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),pr)
  (folder/'history.json').write_text(json.dumps(dict(inner=ih,fullTRAIN=fh),indent=2));m['results'].append(r);save();print(json.dumps(dict(arm=arm,epoch=selected,paired=r['learned_paired_context_average']['auroc'])),flush=True)
 assert state_digest(encoder.state_dict())==sha and all(digest(p)==h for p,h in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-tick)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
