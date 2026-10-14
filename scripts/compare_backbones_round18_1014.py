"""Same-protocol frozen ImageNet ConvNeXt/DenseNet/ResNet comparison, fixed gaze trajectories."""
import sys,json,time,copy,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from src.data import whole_tensor,patch_tensor
from src.official_cbis_classifier_1012 import digest,state_digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
def model_for(name,path):
 from torchvision import models
 model=getattr(models,name)(weights=None);state=torch.load(path,weights_only=False,map_location='cpu')
 if name=='densenet121':
  pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
  for key in list(state):
   match=pattern.match(key)
   if match:state[match.group(1)+match.group(2)]=state.pop(key)
 model.load_state_dict(state);return model.eval().requires_grad_(False)
def spatial(model,name,x):
 if name=='densenet121':return torch.nn.functional.relu(model.features(x))
 z=model.maxpool(model.relu(model.bn1(model.conv1(x))))
 for layer in [model.layer1,model.layer2,model.layer3,model.layer4]:z=layer(z)
 return z
class Joint(torch.nn.Module):
 def __init__(self,dim):
  super().__init__();self.g=torch.nn.Sequential(torch.nn.Linear(dim*2,64),torch.nn.GELU(),torch.nn.Dropout(.5));self.l=torch.nn.Sequential(torch.nn.Linear(dim*2,64),torch.nn.GELU(),torch.nn.Dropout(.5));self.head=torch.nn.Sequential(torch.nn.Linear(128,16),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(16,1))
 def forward(self,x):return self.head(torch.cat([self.g(x[:,:x.shape[1]//2]),self.l(x[:,x.shape[1]//2:])],1)).flatten()
def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();root=Path('runs/gaze_focus_auroc88_1013');out=root/'round18_backbone_comparison_pilot42';plan=json.load(open(out/'preregistration.json'));cpath=Path('runs/official_cbis_v2_experiment_1012/development.pt');c=torch.load(cpath,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);t=c['tensors'];tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];assert not set(f.patient_key.iloc[fit])&set(f.patient_key.iloc[hold]);partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);assert set(other[fit].tolist())<=set(fit.tolist()) and set(other[hold].tolist())<=set(hold.tolist());selection=torch.load(root/'round12_gaze_extent_pilot42/diverse_selections.pt',weights_only=False,map_location='cpu');assert selection['image_ids']==f.image_id.tolist();sel=selection['selection']['8'];gxy=sel['coordinates'];gmask=sel['mask'];assert gmask.all();ri=torch.empty(2393,8,dtype=torch.long)
 for i in range(2393):
  chosen=[];seen=set()
  for j in range(16):
   key=tuple(torch.round(t['random_coordinates'][i,j].double()*torch.tensor([1500.,3000.])).tolist())
   if key not in seen:chosen.append(j);seen.add(key)
   if len(chosen)==8:break
  assert len(chosen)==8;ri[i]=torch.tensor(chosen)
 rxy=t['random_coordinates'].gather(1,ri[:,:,None].expand(-1,-1,2));rc=torch.round(rxy.double()*torch.tensor([1500.,3000.])).long();assert torch.equal(t['random_rectangles'].gather(1,ri[:,:,None].expand(-1,-1,4)).long(),torch.cat([rc-112,rc+112],-1));atomic(dict(image_ids=f.image_id.tolist(),gaze_coordinates=gxy,random_coordinates=rxy,gaze_indices=sel['indices'],random_indices=ri,valid_mask=gmask,selection='gaze farthest8;random first8 unique from fixed16 metadata'),out/'fixed_coordinates.pt');m=dict(state='running',stage='preflight_benchmark',source_sha256=digest(__file__),protocol=plan,dependencies={str(p):digest(p) for p in [Path(__file__),out/'preregistration.json',cpath,root/'round12_gaze_extent_pilot42/diverse_selections.pt',Path('scripts/train_gaze_multiview_1012.py')]},weights={name:dict(path=p,sha256=digest(p)) for name,p in plan['standard_pretrained_files'].items()},results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 save();rows=c['rows'];features={};bench={}
 # ConvNeXt cache reuse smoke uses identical whole/crop pixels and original visual state.
 from src.gaze_backbones import GazePyramid
 model=GazePyramid('convnext_base').eval().requires_grad_(False);assert state_digest(model.state_dict())==c['provenance']['visual_state_sha256'];image=Image.open(rows[0]['canonical_path']).convert('L');tick=time.time()
 with torch.inference_mode():
  z=model(whole_tensor(image,(512,1024))[None])[-1].half().float();glob=torch.cat([model.norm(z.mean((-1,-2),keepdim=True)).flatten(1),model.norm(z.amax((-1,-2),keepdim=True)).flatten(1)],1)[0];patches=torch.stack([patch_tensor(image,p.double().numpy()*[1500,3000],224) for p in torch.cat([gxy[0],rxy[0]])]);v=torch.cat([model.patch_vector(pp) for pp in patches.split(4)])
 oldg=torch.load(root/'round08_global_resolution_pilot42/global_meanmax.pt',weights_only=False,map_location='cpu')['high'][0];rec=torch.load(root/'round12_gaze_extent_pilot42/predicted_records_k16'/f"imitation_{rows[0]['image_id']}.pt",weights_only=False,map_location='cpu');oldv=torch.cat([rec['features'][sel['indices'][0]],t['random_features'][0,ri[0]]]);rel=float((v-oldv).norm()/oldv.norm());grel=float((glob-oldg).norm()/oldg.norm());cos=float(torch.nn.functional.cosine_similarity(v.double(),oldv.double()).min());assert rel<.001 and grel<.001 and cos>.99999;bench['convnext_base']=dict(seconds_per_image16crops=time.time()-tick,full_minutes=0,cache_reuse=True,global_relative_L2=grel,patch_relative_L2=rel,patch_min_cos_float64=cos);del model
 # All weights exist locally; no download, no overlap with previous training.
 for name in ['densenet121','resnet50']:
  model=model_for(name,m['weights'][name]['path']);image=Image.open(rows[0]['canonical_path']).convert('L');x=whole_tensor(image,(512,1024))[None];crops=torch.stack([patch_tensor(image,p.double().numpy()*[1500,3000],224) for p in torch.cat([gxy[0],rxy[0]])]);tick=time.time()
  with torch.inference_mode():
   z=spatial(model,name,x);v=torch.cat([spatial(model,name,p).mean((-1,-2)) for p in crops.split(4)]);assert z.shape[-2:]==(32,16) and v.shape[0]==16 and torch.isfinite(z).all();spatial(model,name,x);[spatial(model,name,p) for p in crops.split(4)]
  seconds=(time.time()-tick)/2;bench[name]=dict(seconds_per_image16crops=seconds,full_minutes=seconds*2393/60,feature_dim=z.shape[1],model_state_sha256=state_digest(model.state_dict()));del model
 m['benchmark']=bench;save();(out/'benchmark.json').write_text(json.dumps(bench,indent=2));print(json.dumps(dict(stage='benchmark_complete',benchmark=bench)),flush=True)
 for name in plan['backbones']:
  m.update(stage='feature_extract',current_backbone=name);save();dest=out/name;dest.mkdir(exist_ok=True);fp=dest/'features.pt'
  if fp.exists():record=torch.load(fp,weights_only=False,map_location='cpu');assert record['image_ids']==f.image_id.tolist() and record['weights_sha256']==m['weights'][name]['sha256'];features[name]=record;continue
  if name=='convnext_base':
   globalcache=torch.load(root/'round08_global_resolution_pilot42/global_meanmax.pt',weights_only=False,map_location='cpu');assert globalcache['image_ids']==f.image_id.tolist() and globalcache['visual_sha256']==c['provenance']['visual_state_sha256'];rs=[torch.load(root/'round12_gaze_extent_pilot42/predicted_records_k16'/f"imitation_{r['image_id']}.pt",weights_only=False,map_location='cpu') for r in rows];gv=torch.stack([r['features'] for r in rs]).gather(1,sel['indices'][:,:,None].expand(-1,-1,1024));rv=t['random_features'].gather(1,ri[:,:,None].expand(-1,-1,1024));record=dict(image_ids=f.image_id.tolist(),global_summary=globalcache['high'],gaze_vectors=gv,random_vectors=rv,weights_sha256=m['weights'][name]['sha256'],reuse=True,original_visual_sha256=c['provenance']['visual_state_sha256']);atomic(record,fp);features[name]=record;continue
  model=model_for(name,m['weights'][name]['path']);statehash=state_digest(model.state_dict());records=dest/'records';records.mkdir(exist_ok=True);gg=[];gv=[];rv=[];m['completed_images']=0
  for i,row in enumerate(rows):
   p=records/f"{row['image_id']}.pt"
   if p.exists():r=torch.load(p,weights_only=False,map_location='cpu');assert r['image_id']==row['image_id'] and r['weights_sha256']==m['weights'][name]['sha256'] and r['pixel_sha256']==row['image_sha256']
   else:
    assert digest(row['canonical_path'])==row['image_sha256'];image=Image.open(row['canonical_path']).convert('L');x=whole_tensor(image,(512,1024))[None];patches=torch.stack([patch_tensor(image,xy.double().numpy()*[1500,3000],224) for xy in torch.cat([gxy[i],rxy[i]])])
    with torch.inference_mode():z=spatial(model,name,x).half().float();global_summary=torch.cat([z.mean((-1,-2)),z.amax((-1,-2))],1)[0];vectors=torch.cat([spatial(model,name,p).mean((-1,-2)) for p in patches.split(4)])
    r=dict(image_id=row['image_id'],global_summary=global_summary,gaze_vectors=vectors[:8],random_vectors=vectors[8:],weights_sha256=m['weights'][name]['sha256'],pixel_sha256=row['image_sha256'],source_sha256=m['source_sha256']);atomic(r,p)
   gg.append(r['global_summary']);gv.append(r['gaze_vectors']);rv.append(r['random_vectors']);m['completed_images']=i+1;save()
  assert state_digest(model.state_dict())==statehash;del model;record=dict(image_ids=f.image_id.tolist(),global_summary=torch.stack(gg),gaze_vectors=torch.stack(gv),random_vectors=torch.stack(rv),weights_sha256=m['weights'][name]['sha256'],reuse=False);atomic(record,fp);features[name]=record;print(json.dumps(dict(stage='feature_complete',backbone=name,images=2393)),flush=True)
 for name in plan['backbones']:
  record=features[name];dim=record['gaze_vectors'].shape[-1];gs=record['global_summary']
  for arm in plan['arms']:
   folder=out/name/arm;folder.mkdir(exist_ok=True)
   if arm=='global_only':ls=torch.zeros_like(gs)
   else:v=record['gaze_vectors'] if arm=='gaze8' else record['random_vectors'];ls=torch.cat([v.mean(1),v.max(1).values],1)
   x=torch.cat([gs,ls],1)
   def run(ids,validation,epochs,select,tag):
    torch.manual_seed(42);model=Joint(dim);mean=x[ids].mean(0);std=x[ids].std(0).clamp_min(.01);z=((x-mean)/std).clamp(-5,5);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);best=-1;be=0;stale=0;hist=[];rp=folder/f'{tag}_resume.pt';first=0
    if rp.exists():r=torch.load(rp,weights_only=False,map_location='cpu');model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);torch.set_rng_state(r['rng']);best=r['best'];be=r['be'];stale=r['stale'];hist=r['history'];first=r['epoch']+1
    def forward(ii,train=False,single=False):
     use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
     if train:use&=torch.rand(len(ii))>=.5
     return model(torch.where(use[:,None],.5*(z[ii]+z[other[ii]]),z[ii]))
    def evaluate(ii,single=False):
     model.eval()
     with torch.inference_mode():return torch.cat([forward(j,single=single).sigmoid() for j in ii.split(64)]).numpy()
    if select and stale>=8:first=epochs+1
    for ep in range(first,epochs+1):
     if ep:
      model.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
      for bi,ii in enumerate(order.split(64)):torch.manual_seed(42+ep*10000+bi);opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(forward(ii,True),y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
     r=dict(epoch=ep)
     if select:
      score=metrics(y[validation].numpy(),evaluate(validation))['auroc'];r['inner_auroc']=score
      if score>best:best=score;be=ep;stale=0
      else:stale+=1
     hist.append(r);atomic(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,history=hist,mean=mean,std=std),rp)
     if select and stale>=8:break
    if select:return be,hist
    atomic(dict(model=model.state_dict(),mean=mean,std=std,epoch=epochs,weights_sha256=record['weights_sha256']),folder/'selected.pt');return evaluate(validation,True),evaluate(validation),evaluate(tr),hist
   epoch,ih=run(fit,hold,40,True,'inner');single,paired,tp,fh=run(tr,va,epoch,False,'fullTRAIN');mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(backbone=name,arm=arm,epoch=epoch,train_paired_auroc=metrics(y[tr].numpy(),tp)['auroc'],inner_history=ih)
   for mode,prob in [('singleview',single),('paired_feature_mean',paired),('fixed_probability_average',fixed)]:df=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();df['probability']=prob;df.to_csv(folder/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),prob)
   m['results'].append(r);save();print(json.dumps(dict(backbone=name,arm=arm,paired=r['paired_feature_mean']['auroc'],epoch=epoch)),flush=True)
 assert all(digest(p)==sha for p,sha in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
