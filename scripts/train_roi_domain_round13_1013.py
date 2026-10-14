"""TRAIN-only ROI final-block adaptation; frozen global plus independent local MIL logits."""
import sys,json,time,copy,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from src.data import patch_tensor
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,state_digest,metrics
from scripts.benchmark_convnext_lastblock_1012 import boundary_features
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(v,p):tmp=p.with_suffix('.tmp');torch.save(v,tmp);tmp.replace(p)
class LocalMIL(torch.nn.Module):
 def __init__(self):
  super().__init__();self.project=torch.nn.Sequential(torch.nn.Linear(1024,16),torch.nn.GELU(),torch.nn.Dropout(.5));self.head=torch.nn.Linear(32,1)
 def forward(self,v,mask):
  z=self.project(v);mean=(z*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None];maximum=z.masked_fill(~mask[:,:,None],-1e9).max(1).values;maximum=torch.where(mask.any(1)[:,None],maximum,torch.zeros_like(maximum));return self.head(torch.cat([mean,maximum],1)).flatten()
def main():
 p=argparse.ArgumentParser();p.add_argument('--benchmark',action='store_true');a=p.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);start=time.time();root=Path('runs/gaze_focus_auroc88_1013');out=root/'round13_roi_domain_pilot42';out.mkdir(exist_ok=True);roiout=out/'roi_boundaries';roiout.mkdir(exist_ok=True)
 cp=Path('runs/official_cbis_v2_experiment_1012/development.pt');sup=root/'round06_lesion_trajectory_pilot42/lesion_supervision_v2.json';ctxroot=root/'round10_frozen_global_residual_pilot42';boundary=root/'round07_crop_block_adapt_pilot42/boundary_records';assert all(p.exists() for p in [cp,sup,ctxroot/'global_inner_contexts.pt',ctxroot/'global_fullTRAIN_contexts.pt',boundary]);c=torch.load(cp,weights_only=False,map_location='cpu');f=pd.DataFrame(c['rows']);assert {'rows','tensors','provenance'}<=c.keys();tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);fitidx,holdidx=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(tr,y[tr],f.patient_key.values[tr]));fit=tr[fitidx];hold=tr[holdidx];fitpatients=set(f.patient_key.iloc[fit]);holdpatients=set(f.patient_key.iloc[hold]);assert not fitpatients&holdpatients;partners,_,_=pair_views(f);other=torch.tensor([ps[0] if ps else i for i,ps in enumerate(partners)]);available=torch.tensor([bool(ps) for ps in partners]);assert set(other[hold].tolist())<=set(hold.tolist());imageidx={int(r.image_id):i for i,r in enumerate(f.itertuples())};s=json.load(open(sup));assert s['TRAIN_only'];encoder=GazePyramid('convnext_base').eval().requires_grad_(False);originalsha=state_digest(encoder.state_dict());assert originalsha==c['provenance']['visual_state_sha256'];initialblock=copy.deepcopy(encoder.features[7][-1].state_dict());initialnorm=copy.deepcopy(encoder.norm.state_dict());roi=[];conflicts=[]
 for j,r in enumerate(s['accepted']):
  i=imageidx[int(r['image_id'])];assert f.iloc[i].split=='train';source_label=0 if 'BENIGN' in r['source_file'] else 1 if 'MALIGNANT' in r['source_file'] else None
  if source_label is not None and source_label!=int(y[i]):conflicts.append(dict(image_id=int(r['image_id']),source=r['source_file'],label=int(y[i])));continue
  roi.append(dict(key=j,index=i,image_id=int(r['image_id']),patient=f.iloc[i].patient_key,box=r['canonical_box'],label=int(y[i])))
 assert roi and len({r['image_id'] for r in roi})<=318;contexts={tag:torch.load(ctxroot/f'global_{tag}_contexts.pt',weights_only=False,map_location='cpu') for tag in ['inner','fullTRAIN']};assert all(v['image_ids']==f.image_id.tolist() and {'context','logits','frozen_global_state_sha256'}<=v.keys() for v in contexts.values());m=dict(state='running',stage='ROI_boundary_extract',source_sha256=digest(__file__),seed=42,test_used=False,oracle_used=False,ROI_margin=1.5,ROI_crop='square preserving bbox context, pad black, resize224',ROI_images=len({r['image_id'] for r in roi}),ROI_boxes=len(roi),conflicts_excluded=conflicts,fit_patients=sorted(fitpatients),hold_patients=sorted(holdpatients),ROI_max_epochs=20,ROI_patience=5,encoder_lr=1e-5,head_lr=.0003,weight_decay=.05,alphas=[0,.25,.5,1],local_dim=16,local_max_epochs=30,local_patience=8,source_prefix='round06 image-only OOF imitation first4, not diverse K16',dependencies={str(q):digest(q) for q in [Path(__file__),cp,sup,ctxroot/'global_inner_contexts.pt',ctxroot/'global_fullTRAIN_contexts.pt']},results=[])
 def save():
  p=out/'manifest.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(m,indent=2));tmp.replace(p)
 def roi_tensor(r):
  row=f.iloc[r['index']];image=Image.open(row.canonical_path).convert('L');box=np.array(r['box']);center=(box[:2]+box[2:])/2;size=max(box[2:]-box[:2])*1.5;rect=np.r_[np.floor(center-size/2),np.ceil(center+size/2)].astype(int);image=image.crop(tuple(rect)).resize((224,224),Image.Resampling.BILINEAR);return patch_tensor(image,[112,112],224),rect
 if a.benchmark:
  sample=roi[:8];x=torch.stack([roi_tensor(r)[0] for r in sample]);tick=time.time()
  with torch.inference_mode():b=boundary_features(encoder,x).half();direct=encoder.patch_vector(x);restored=encoder.norm(encoder.features[7][-1](b.float()).mean((-1,-2),keepdim=True)).flatten(1)
  seconds=time.time()-tick;cos=float(torch.nn.functional.cosine_similarity(direct,restored).min());rel=float((direct-restored).norm()/direct.norm());assert cos>.99999 and rel<.001;block=copy.deepcopy(encoder.features[7][-1]).requires_grad_(True);norm=copy.deepcopy(encoder.norm).requires_grad_(True);head=torch.nn.Linear(1024,1);tick=time.time();loss=torch.nn.functional.binary_cross_entropy_with_logits(head(norm(block(b.float()).mean((-1,-2),keepdim=True)).flatten(1)).flatten(),torch.tensor([r['label'] for r in sample],dtype=torch.float32));loss.backward();backward=time.time()-tick;assert sum(float(p.grad.abs().sum()) for p in block.parameters() if p.grad is not None)>0;r=dict(ROI_boxes=len(roi),ROI_images=m['ROI_images'],conflicts_excluded=len(conflicts),batch=8,extract_seconds=seconds,extract_minutes=seconds/8*len(roi)/60,training_step_seconds=backward,epoch_seconds=backward*len(roi)/8,prefix_restore_min_cos=cos,relative_L2=rel);(out/'benchmark.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);return
 save()
 for n,r in enumerate(roi):
  dest=roiout/f"{r['key']}.pt"
  if dest.exists():record=torch.load(dest,weights_only=False,map_location='cpu');assert record['image_id']==r['image_id'] and record['visual_sha256']==originalsha
  else:
   x,rect=roi_tensor(r)
   with torch.inference_mode():b=boundary_features(encoder,x[None]).half()[0]
   record=dict(**r,boundary=b,rectangle=rect,visual_sha256=originalsha,pixel_sha256=digest(f.iloc[r['index']].canonical_path));atomic(record,dest)
  m['completed_ROI']=n+1;save()
 b=torch.stack([torch.load(roiout/f"{r['key']}.pt",weights_only=False,map_location='cpu')['boundary'] for r in roi]);ry=torch.tensor([r['label'] for r in roi],dtype=torch.float32);rf=torch.tensor([i for i,r in enumerate(roi) if r['patient'] in fitpatients]);rh=torch.tensor([i for i,r in enumerate(roi) if r['patient'] in holdpatients]);assert len(rf) and len(rh);m.update(stage='ROI_encoder_train',ROI_fit=len(rf),ROI_hold=len(rh));save()
 def pretrain(ids,validation,epochs,select,tag):
  torch.manual_seed(42);block=copy.deepcopy(encoder.features[7][-1]).requires_grad_(True);norm=copy.deepcopy(encoder.norm).requires_grad_(True);head=torch.nn.Linear(1024,1);opt=torch.optim.AdamW([dict(params=list(block.parameters())+list(norm.parameters()),lr=1e-5),dict(params=head.parameters(),lr=.0003)],weight_decay=.05);rp=out/f'ROI_{tag}_resume.pt';best=-1;be=0;stale=0;beststate=None;hist=[];first=0
  def state():return dict(block=copy.deepcopy(block.state_dict()),norm=copy.deepcopy(norm.state_dict()),head=copy.deepcopy(head.state_dict()))
  if rp.exists():v=torch.load(rp,weights_only=False,map_location='cpu');block.load_state_dict(v['state']['block']);norm.load_state_dict(v['state']['norm']);head.load_state_dict(v['state']['head']);opt.load_state_dict(v['optimizer']);torch.set_rng_state(v['rng']);best=v['best'];be=v['be'];stale=v['stale'];beststate=v['beststate'];hist=v['history'];first=v['epoch']+1
  for ep in range(first,epochs+1):
   if ep:
    block.train();norm.train();head.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
    for ii in order.split(8):opt.zero_grad();z=norm(block(b[ii].float()).mean((-1,-2),keepdim=True)).flatten(1);loss=torch.nn.functional.binary_cross_entropy_with_logits(head(z).flatten(),ry[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(list(block.parameters())+list(norm.parameters())+list(head.parameters()),1);opt.step()
   row=dict(epoch=ep)
   if select:
    block.eval();norm.eval();head.eval()
    with torch.inference_mode():pr=torch.cat([head(norm(block(b[ii].float()).mean((-1,-2),keepdim=True)).flatten(1)).flatten().sigmoid() for ii in validation.split(16)]).numpy()
    score=metrics(ry[validation].numpy(),pr)['auroc'];row['inner_ROI_auroc']=score
    if score>best:best=score;be=ep;stale=0;beststate=state()
    else:stale+=1
   hist.append(row);atomic(dict(state=state(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,stale=stale,beststate=beststate,history=hist,fit_patients=sorted({roi[i]['patient'] for i in ids.tolist()})),rp);m.update(phase=tag,epoch=ep);save()
   if select and stale>=5:break
  chosen=beststate if select else state();atomic(dict(state=chosen,epochs=be if select else epochs,history=hist,fit_patients=sorted({roi[i]['patient'] for i in ids.tolist()})),out/f'ROI_{tag}_selected.pt');return chosen,be,hist
 innerstate,ep,ih=pretrain(rf,rh,20,True,'inner');fullstate,_,fh=pretrain(torch.arange(len(roi)),None,ep,False,'fullTRAIN');m.update(ROI_selected_epoch=ep,ROI_inner_history=ih,stage='adapted_K4_vector_cache');save()
 # Reuse exact pre-lastblock K4 source pixels/boundaries; original encoder prefix is immutable.
 source_masks=[];frozen={arm:[] for arm in ['imitation','random']};adapted={tag:{arm:[] for arm in frozen} for tag in ['inner','fullTRAIN']}
 for tag,state in [('inner',innerstate),('fullTRAIN',fullstate)]:
  cachepath=out/f'local_vectors_{tag}.pt'
  if cachepath.exists():v=torch.load(cachepath,weights_only=False,map_location='cpu');assert v['image_ids']==f.image_id.tolist();adapted[tag]=v['adapted'];frozen=v['frozen'];source_masks=v['mask'];continue
  block=copy.deepcopy(encoder.features[7][-1]);norm=copy.deepcopy(encoder.norm);block.load_state_dict(state['block']);norm.load_state_dict(state['norm']);block.eval();norm.eval();vv={arm:[] for arm in frozen};ff={arm:[] for arm in frozen};mm=[]
  for row in f.itertuples():
   rec=torch.load(boundary/f'{row.image_id}.pt',weights_only=False,map_location='cpu');assert rec['image_id']==row.image_id and rec['visual_sha256']==originalsha and rec['pixel_sha256']==row.image_sha256;mask=rec['mask']['imitation'];mm.append(mask)
   with torch.inference_mode():
    for arm in frozen:bb=rec['features'][arm].float();vv[arm].append(norm(block(bb).mean((-1,-2),keepdim=True)).flatten(1));ff[arm].append(rec['pooled'][arm])
  v=dict(image_ids=f.image_id.tolist(),adapted={arm:torch.stack(vals) for arm,vals in vv.items()},frozen={arm:torch.stack(vals) for arm,vals in ff.items()},mask=torch.stack(mm),ROI_encoder_sha256=digest(out/f'ROI_{tag}_selected.pt'));atomic(v,cachepath);adapted[tag]=v['adapted'];frozen=v['frozen'];source_masks=v['mask']
 counts=source_masks.sum(1);rmask=torch.arange(4)[None]<counts[:,None];m.update(stage='local_MIL_train',effective_VAL_K=float(counts[va].float().mean()));save()
 baseline=dict(arm='global_only',epoch=0)
 for mode in ['singleview','learned_paired_context_average','fixed_probability_average']:
  ref=pd.read_csv(root/'round09_spatial_mil_pilot42/global_only'/f'{mode}_predictions.csv');baseline[mode]=metrics(ref.label,ref.probability)
 m['results'].append(baseline);save()
 for variant in ['frozen','ROI_adapted']:
  for arm in ['imitation','random']:
   mask=source_masks if arm=='imitation' else rmask;name=f'{variant}_{arm}';folder=out/name;folder.mkdir(exist_ok=True)
   def train(ids,validation,epochs,select,tag,alpha=0):
    torch.manual_seed(42);model=LocalMIL();v=frozen[arm] if variant=='frozen' else adapted[tag][arm];valid=v[ids][mask[ids]];mean=valid.mean(0);std=valid.std(0).clamp_min(.01);z=((v-mean)/std).clamp(-5,5);gl=contexts[tag]['logits'];opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.05);best=-1;be=0;ba=0;stale=0;hist=[];rp=folder/f'{tag}_resume.pt';first=0
    if rp.exists():q=torch.load(rp,weights_only=False,map_location='cpu');model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);torch.set_rng_state(q['rng']);best=q['best'];be=q['be'];ba=q['ba'];stale=q['stale'];hist=q['history'];first=q['epoch']+1
    def logits(ii,training=False,single=False):
     use=available[ii].clone() if not single else torch.zeros(len(ii),dtype=torch.bool)
     if training:use&=torch.rand(len(ii))>=.5
     local=model(z[ii],mask[ii]);partner=model(z[other[ii]],mask[other[ii]]);local=torch.where(use,.5*(local+partner),local);global_logit=torch.where(use,.5*(gl[ii]+gl[other[ii]]),gl[ii]);return local,global_logit
    def evaluate(ii,a,single=False):
     model.eval()
     with torch.inference_mode():return torch.cat([(lambda l,g:(g+a*l).sigmoid())(*logits(j,single=single)) for j in ii.split(64)]).numpy()
    for ep in range(first,epochs+1):
     if ep:
      model.train();order=ids[torch.randperm(len(ids),generator=torch.Generator().manual_seed(42+ep))]
      for ii in order.split(64):opt.zero_grad();l,g=logits(ii,True);loss=torch.nn.functional.binary_cross_entropy_with_logits(l,y[ii]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
     record=dict(epoch=ep)
     if select:
      scores=[metrics(y[validation].numpy(),evaluate(validation,a))['auroc'] for a in m['alphas']];j=int(np.argmax(scores));score=scores[j];record.update(inner_auroc=score,alpha=m['alphas'][j])
      if score>best:best=score;be=ep;ba=m['alphas'][j];stale=0
      else:stale+=1
     hist.append(record);atomic(dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=torch.get_rng_state(),epoch=ep,best=best,be=be,ba=ba,stale=stale,history=hist,mean=mean,std=std),rp)
     if select and stale>=8:break
    if select:return be,ba,hist
    atomic(dict(model=model.state_dict(),mean=mean,std=std,epoch=epochs,alpha=alpha),folder/'selected.pt');return evaluate(validation,alpha,True),evaluate(validation,alpha),evaluate(tr,alpha),hist
   selected,alpha,ih=train(fit,hold,30,True,'inner');single,paired,tp,fh=train(tr,va,selected,False,'fullTRAIN',alpha);mapping=dict(zip(va.tolist(),single));fixed=np.array([.5*(mapping[int(i)]+mapping[int(other[i])]) for i in va]);r=dict(arm=name,epoch=selected,alpha=alpha,train_paired_auc=metrics(y[tr].numpy(),tp)['auroc'],inner_history=ih)
   for mode,pr in [('singleview',single),('learned_paired_context_average',paired),('fixed_probability_average',fixed)]:df=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();df['probability']=pr;df.to_csv(folder/f'{mode}_predictions.csv',index=False);r[mode]=metrics(y[va].numpy(),pr)
   m['results'].append(r);save();print(json.dumps(dict(arm=name,epoch=selected,alpha=alpha,paired=r['learned_paired_context_average']['auroc'])),flush=True)
 assert state_digest(encoder.state_dict())==originalsha and all(digest(p)==sha for p,sha in m['dependencies'].items());m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();(out/'report.json').write_text(json.dumps(m,indent=2))
if __name__=='__main__':main()
