"""Same ConvNeXt global final-CN-block diagnosis adaptation with TRAIN inner selection."""
import sys,json,time,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views
from scripts.train_gaze_focus_selected_1013 import local_summary

class Joint(torch.nn.Module):
 def __init__(self,encoder):
  super().__init__();self.block=encoder.features[7][-1];self.norm=encoder.norm;self.head=torch.nn.Sequential(torch.nn.LayerNorm(6144),torch.nn.Linear(6144,32),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(32,1))
 def global_vector(self,b):return self.norm(torch.nn.functional.adaptive_avg_pool2d(self.block(b.float()),(2,2))).flatten(1)
 def forward(self,b,local):return self.head(torch.cat([self.global_vector(b),local],1)).flatten()

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round05_global_block_adapt_pilot42');droot=out/'heads_v2';droot.mkdir(exist_ok=False);start=time.time();cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);b=torch.stack([torch.load(out/'boundary_records'/f'{r.image_id}.pt',weights_only=False,map_location='cpu')['spatial'] for r in f.itertuples()]);assert b.shape==(2393,1024,16,8);sp=torch.load(out.parent/'round03_spatial_pilot42/spatial_summary.pt',weights_only=False,map_location='cpu');assert sp['image_ids']==f.image_id.tolist();frozen_g=sp['features'];local={'global_only':torch.zeros(2393,2048)};deps={p:digest(p) for p in [__file__,cp,'scripts/extract_global_boundary_focus_1013.py','scripts/train_gaze_multiview_1012.py']}
 for source in ['gaze_baseline','random']:
  path=out.parent/'round02_selected_meanmax_pilot42'/f'{source}_selection.pt';r=torch.load(path,weights_only=False,map_location='cpu');assert r['image_ids']==f.image_id.tolist();deps[str(path)]=digest(path);z=t[source+'_features'][torch.arange(2393)[:,None],r['indices'][:,:4]];local[source]=local_summary(torch.zeros(2393,0),z,r['mask'][:,:4])
 fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);assert set(other[hold].tolist()).issubset(set(hold.tolist()))
 m=dict(state='running',stage='train',test_used=False,oracle_used=False,seed=42,source_sha256=digest(__file__),dependencies=deps,adapt_max_epochs=3,patience=2,head_warmup_epochs=10,head_lr=.0003,backbone_lr=1e-5,weight_decay=.05,adapt_batch=4,warmup_batch=64,selection='TRAIN patient20percent inner AUROC only; selectedepoch fullTRAIN fresh retrain then VAL once',arms=list(local),backbone='original sameConvNeXt final CNblock+norm only; prefix frozen and unused after exact boundary',results=[])
 def save():
  p=droot/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save()
 for arm,l in local.items():
  d=droot/arm;d.mkdir();hist=[]
  def run(trainids,evalids,epochs,select):
   torch.manual_seed(42);encoder=GazePyramid('convnext_base').eval().requires_grad_(False);assert state_digest(encoder.state_dict())==c['provenance']['visual_state_sha256'];model=Joint(encoder);model.head.requires_grad_(True);initialblock=state_digest(model.block.state_dict());x=torch.cat([frozen_g,l],1)
   def mixed(ids,training=False,global_current=None):
    use=available[ids].clone()
    if training:use&=torch.rand(len(ids))>=.5
    if global_current is None:return torch.where(use[:,None],.5*(x[ids]+x[other[ids]]),x[ids])
    return use
   opt=torch.optim.AdamW(model.head.parameters(),lr=.0003,weight_decay=.05)
   for ep in range(10):
    model.head.train();order=trainids[torch.randperm(len(trainids))]
    for ids in order.split(64):
     opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model.head(mixed(ids,True)).flatten(),y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.head.parameters(),1);opt.step()
   assert state_digest(model.block.state_dict())==initialblock;model.block.requires_grad_(True);model.norm.requires_grad_(True);opt=torch.optim.AdamW([dict(params=list(model.block.parameters())+list(model.norm.parameters()),lr=1e-5),dict(params=model.head.parameters(),lr=.0003)],weight_decay=.05);best=-1;stale=0;state=None;be=0
   def infer(ids,mode='paired'):
    model.eval();pred=[]
    with torch.inference_mode():
     for ix in ids.split(4):
      g=model.global_vector(b[ix]);z=torch.cat([g,l[ix]],1)
      if mode=='paired':
       pg=model.global_vector(b[other[ix]]);pz=torch.cat([pg,l[other[ix]]],1);z=torch.where(available[ix,None],.5*(z+pz),z)
      pred.append(model.head(z).flatten().sigmoid())
    return torch.cat(pred).numpy()
   if select:
    p=infer(evalids);best=metrics(y[evalids].numpy(),p)['auroc'];state=copy.deepcopy(model.state_dict());hist.append(dict(stage='inner',epoch=0,auroc=best))
   for ep in range(1,epochs+1):
    model.train();order=trainids[torch.randperm(len(trainids))];losses=[];stepstart=time.time()
    for ids in order.split(4):
     opt.zero_grad();g=model.global_vector(b[ids]);pg=model.global_vector(b[other[ids]]);z=torch.cat([g,l[ids]],1);pz=torch.cat([pg,l[other[ids]]],1);use=mixed(ids,True,g);logit=model.head(torch.where(use[:,None],.5*(z+pz),z)).flatten();loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    record=dict(stage='inner' if select else 'fullTRAIN',epoch=ep,fit_loss=float(np.mean(losses)),seconds=time.time()-stepstart)
    if select:
     score=metrics(y[evalids].numpy(),infer(evalids))['auroc'];record['inner_auroc']=score
     if score>best:best=score;be=ep;stale=0;state=copy.deepcopy(model.state_dict())
     else:stale+=1
    hist.append(record);(d/'history_partial.json').write_text(json.dumps(hist,indent=2));torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),epoch=ep,stage=record['stage']),d/'latest_recovery.pt');m.update(current_arm=arm,current_stage=record['stage'],latest_epoch=ep);save()
    if select and stale>=2:break
   if select:model.load_state_dict(state);return be
   return model,infer
  chosen=run(fit,hold,3,True);m['selected_epoch']=chosen;save();model,infer=run(tr,va,chosen,False);torch.save(model.state_dict(),d/'best.pt');(d/'history.json').write_text(json.dumps(hist,indent=2));single=infer(va,'single');paired=infer(va);mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);record=dict(arm=arm,selected_epoch=chosen)
  for mode,p in [('singleview',single),('learned_paired_feature_average',paired),('fixed_probability_average',fixed)]:
   pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=p;pred.to_csv(d/f'{mode}_predictions.csv',index=False);record[mode]=metrics(y[va].numpy(),p)
  torch.save(model.state_dict(),d/'best.pt');(d/'history.json').write_text(json.dumps(hist,indent=2));m['results'].append(record);save();print(json.dumps(record),flush=True)
 assert all(digest(p)==sha for p,sha in deps.items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(droot/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
