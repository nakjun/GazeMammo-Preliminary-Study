"""Cached high spatialMIL matched global/imitation/random, TRAIN-inner epoch choice."""
import sys,time,json,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.gaze_supervised_spatial_round14_1013 import GazeSupervisedMIL
import hashlib
from src.official_cbis_classifier_1012 import digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();out=Path('runs/gaze_focus_auroc88_1013/round14_expert_attention_pilot42');out.mkdir(exist_ok=True);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold].tolist()).issubset(set(hold.tolist()));root=out.parent/'round08_global_resolution_pilot42/high_spatial_records';g=torch.stack([torch.load(root/f'{r.image_id}.pt',weights_only=False,map_location='cpu')['spatial'].flatten(1).T for r in f.itertuples()]);assert g.shape==(2393,512,1024);local={};counts={}
 # All local inputs remain deployable image-only fresh patientOOF trajectories.
 rs=[torch.load(out.parent/'round12_gaze_extent_pilot42/predicted_records_k16'/f"imitation_{r['image_id']}.pt",weights_only=False,map_location='cpu') for r in c['rows']];l=torch.stack([r['features'] for r in rs]);coords=torch.stack([r['coordinates'] for r in rs]);mask=torch.ones(2393,16,dtype=torch.bool);del rs
 for i in range(2393):
  seen=set()
  for j in range(16):
   key=tuple(torch.round(coords[i,j].double()*torch.tensor([1500.,3000.])).tolist())
   if key in seen:mask[i,j]=False
   else:seen.add(key)
 rmask=torch.arange(16)[None]<mask.sum(1)[:,None];assert torch.equal(rmask.sum(1),mask.sum(1));empty=(torch.zeros(2393,16,1024),torch.zeros(2393,16,2),torch.zeros(2393,16,dtype=torch.bool));local={'global_no_aux':empty,'global_gaze_aux':empty,'global_gaze_aux_imitation16':(l,coords,mask),'global_gaze_aux_random16':(t['random_features'],t['random_coordinates'],rmask)};counts={arm:float(v[2][va].sum(1).float().mean()) for arm,v in local.items()}
 indexpath=Path('runs/official_cbis_v2_experiment_1012/gaze_cache/index.pt');idx=torch.load(indexpath,weights_only=False,map_location='cpu');gmpath=Path(idx['protocol_path']).parent/'gaze.csv';assert digest(gmpath)==idx['gaze_manifest_sha256'];gf=pd.read_csv(gmpath);gf=gf[gf.split=='train'];assert len(gf)==420;experts={int(row.image_id):(row,Path(row.gaze_canonical_path+'.json')) for row in gf.itertuples()};recordpaths={int(Path(p).stem):Path(p) for p in idx['records']};imageindices={int(r.image_id):i for i,r in enumerate(f.itertuples())}
 def expert_targets(ids,tag):
  allowed=set(ids.tolist());target=torch.zeros(2393,512);valid=torch.zeros(2393,dtype=torch.bool);accessed=[];yy,xx=torch.meshgrid((torch.arange(32)+.5)/32,(torch.arange(16)+.5)/16,indexing='ij');xy=torch.stack([xx,yy],-1).reshape(512,2)*torch.tensor([1500.,3000.])
  for image_id,(row,affine) in experts.items():
   i=imageindices[image_id]
   if i not in allowed:continue
   assert f.iloc[i].split=='train' and f.iloc[i].patient_key==row.patient_key;assert tag!='inner' or f.iloc[i].patient_key not in set(f.patient_key.iloc[hold]);p=recordpaths[image_id];assert digest(p)==idx['record_hashes'][str(p)];assert affine.exists() and digest(row.gaze_canonical_path)==f.iloc[i].image_sha256;metadata=json.load(open(affine));assert 'composed_affine' in metadata or 'affine' in metadata
   rec=torch.load(p,weights_only=False,map_location='cpu');assert rec['split']=='train' and rec['patient_key']==row.patient_key;rowhash=digest(row.gaze_canonical_path)+digest(affine)+hashlib.sha256(json.dumps(row._asdict(),sort_keys=True).encode()).hexdigest();assert rec['row_hash']==rowhash;points=torch.cat(rec['parts']);assert torch.isfinite(points).all() and ((points>=0)&(points<=1)).all();points=points*torch.tensor([1500.,3000.]);density=torch.exp(-torch.cdist(xy,points).square()/(2*112**2)).mean(1);density=density/density.sum().clamp_min(1e-12);assert abs(float(density.sum())-1)<1e-5;target[i]=density;valid[i]=True;accessed.append(dict(image_id=image_id,patient=row.patient_key,record_sha256=digest(p),affine_sha256=digest(affine)))
  assert not valid[va].any() and (tag!='inner' or not valid[hold].any());return target,valid,accessed
 def global_norm(ids):
  total=0;s=torch.zeros(1024,dtype=torch.float64);sq=s.clone()
  for ii in ids.split(16):z=g[ii].float();s+=z.sum((0,1)).double();sq+=z.square().sum((0,1)).double();total+=z.shape[0]*512
  mean=(s/total).float();std=((sq-s.square()/total)/(total-1)).clamp_min(0).sqrt().float().clamp_min(.01);return mean,std
 norms={'inner':global_norm(fit),'fullTRAIN':global_norm(tr)};m=dict(state='running',seed=42,source_sha256=digest(__file__),dependencies={p:digest(p) for p in [__file__,'src/gaze_supervised_spatial_round14_1013.py','scripts/train_gaze_multiview_1012.py',str(out.parent/'round08_global_resolution_pilot42/extraction_manifest.json')]},CNN_extraction=False,backbone_frozen=True,global_tokens=512,dim=32,dropout=.4,context_dropout=.5,patch_dropout=.3,weight_decay=.1,lr=.0003,batch=16,max_epochs=40,patience=8,normalization='TRAINfit channelmean/std separately global/local;clamp5;global statsshared across3arms',selection='TRAINpatientinner20percent AUROC;freshfullTRAINrefit;VAL356once',aux_lambda=.05,aux_warmup_epochs=3,aux_sigma_canonical=112,paired_mode='learnedglobal/localcontextaverage;fixedprobavg separate',test_used=False,oracle_used=False,valid_local_counts=counts,results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save()
 torch.manual_seed(42);smoke=GazeSupervisedMIL();sx=torch.randn(2,512,1024);_,_,sw=smoke.encode(sx,torch.zeros(2,16,1024),torch.zeros(2,16,dtype=torch.bool));target=torch.zeros_like(sw);target[:,0]=1;loss=-(target*sw.clamp_min(1e-12).log()).sum(1).mean();loss.backward();assert sum(float(p.grad.abs().sum()) for p in smoke.attention.parameters() if p.grad is not None)>0;m['attention_gradient_smoke']=True;save()
 for arm,(l,coords,mask) in local.items():
  folder=out/arm;folder.mkdir(exist_ok=True)
  def run(ids,validation,epochs,select,tag):
   torch.manual_seed(42);model=GazeSupervisedMIL(local=arm.endswith('16'));targets,targetmask,accessed=expert_targets(ids,tag) if arm!='global_no_aux' else (torch.zeros(2393,512),torch.zeros(2393,dtype=torch.bool),[]);m.setdefault('expert_target_access',{})[arm+'_'+tag]=accessed;save();gm,gs=norms[tag];valid=l[ids][mask[ids]];lm=valid.mean(0) if len(valid) else torch.zeros(1024);ls=valid.std(0).clamp_min(.01) if len(valid)>1 else torch.ones(1024);lp=((l-lm)/ls).clamp(-5,5);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);best=-1;be=0;stale=0;state=None;history=[];rp=folder/f'{tag}_recovery.pt';resume=torch.load(rp,weights_only=False,map_location='cpu') if rp.exists() else None
   if resume:model.load_state_dict(resume['model']);opt.load_state_dict(resume['optimizer']);torch.set_rng_state(resume['rng']);best=resume['best'];be=resume['be'];stale=resume['stale'];state=resume['best_state'];history=resume['history']
   def forward(ii,training=False,single=False):
    use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
    if training:use&=torch.rand(len(ii))>=.5
    a=((g[ii].float()-gm)/gs).clamp(-5,5);p=((g[other[ii]].float()-gm)/gs).clamp(-5,5);logits,w,pw=model(a,lp[ii],mask[ii],p,lp[other[ii]],mask[other[ii]],use)
    if not training:return logits
    good=targetmask[ii];pgood=targetmask[other[ii]]&use;terms=[]
    if good.any():terms.append((targets[ii][good]*(targets[ii][good].clamp_min(1e-12).log()-w[good].clamp_min(1e-12).log())).sum(1).mean())
    if pgood.any():terms.append((targets[other[ii]][pgood]*(targets[other[ii]][pgood].clamp_min(1e-12).log()-pw[pgood].clamp_min(1e-12).log())).sum(1).mean())
    aux=torch.stack(terms).mean() if terms else logits.sum()*0
    return logits,aux
   def evaluate(ii,single=False):
    model.eval()
    with torch.inference_mode():return torch.cat([forward(j,single=single).sigmoid() for j in ii.split(16)]).numpy()
   first=resume['epoch']+1 if resume else 0
   if select and stale>=8:first=epochs+1
   for ep in range(first,epochs+1):
    tick=time.time()
    if ep:
     model.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
     for ii in order.split(16):opt.zero_grad();logits,aux=forward(ii,True);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ii])+(0 if ep<=3 else .05)*aux;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
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
