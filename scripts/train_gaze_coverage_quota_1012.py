"""Round12 fixed14grid+2source coverage quota, immutable Round10 OOF scores."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.official_cbis_classifier_1012 import digest,restore,metrics,paired_ci,state_digest
from src.gaze_attention_1011 import attention_loss,head_state
from scripts.train_gaze_diagnosis_selector_1012 import build_candidates
from scripts.train_gaze_multiview_1012 import pair_views

def grid_coverage(coords,count=14):
 xy=coords.double()*torch.tensor([1500.,3000.]);chosen=[int(((xy-xy.mean(0))**2).sum(1).argmin())]
 while len(chosen)<count:
  distances=torch.cdist(xy,xy[chosen]).min(1).values;distances[chosen]=-1;chosen.append(int(distances.argmax()))
 return chosen

def quota_indices(scores,coords,mask):
 """No labels:14grid coverage,2 rankedsource, relaxation then omittedgrid."""
 chosen=grid_coverage(coords[:16]);xy=torch.round(coords.double()*torch.tensor([1500.,3000.]));seen={tuple(xy[i].tolist()) for i in chosen};order=sorted(range(16,32),key=lambda j:(-float(abs(scores[j])),j));source=[];relaxed=0
 for distance in [96,0]:
  for j in order:
   key=tuple(xy[j].tolist())
   if not bool(mask[j]) or key in seen:continue
   if distance and any(float((xy[j]-xy[k]).norm())<distance for k in chosen+source):continue
   source.append(j);seen.add(key);relaxed+=int(distance==0)
   if len(source)==2:break
  if len(source)==2:break
 selected=chosen+source;gridfallback=0
 if len(selected)<16:
  for j in grid_coverage(coords[:16],16):
   if j not in selected:selected.append(j);gridfallback+=1
   if len(selected)==16:break
 assert len(selected)==16 and len({tuple(xy[j].tolist()) for j in selected})==16
 return torch.tensor(selected),dict(source_count=len(source),distance_relaxed_count=relaxed,grid_fallback_count=gridfallback)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);origin=Path('runs/gaze_diagnosis_selector_1012_round10_pilot42');control=Path('runs/gaze_selector_extension_1012_round11_pilot42');old=json.loads((origin/'manifest.json').read_text());ctrlm=json.loads((control/'manifest.json').read_text());assert old['state']==ctrlm['state']=='complete';cp='runs/official_cbis_v2_experiment_1012/development.pt';assert digest(cp)==old['cache_sha256']==ctrlm['cache_sha256'];assert digest('scripts/train_gaze_diagnosis_selector_1012.py')==old['source_sha256'];c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert len(f)==2393 and (f.split=='val').sum()==356 and set(f.split)=={'train','val'} and f.groupby('patient_key').split.nunique().max()==1;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f);deps={p:digest(p) for p in ['scripts/train_gaze_diagnosis_selector_1012.py','scripts/train_gaze_multiview_1012.py','src/gaze_attention_1011.py','src/official_cbis_classifier_1012.py']};deps.update({str(origin/'manifest.json'):digest(origin/'manifest.json'),str(control/'manifest.json'):digest(control/'manifest.json')})
 m=dict(state='running',stage='fixed_quota_bags',source_sha256=digest(__file__),cache_sha256=digest(cp),dependencies=deps,test_used=False,oracle_used=False,seed=42,quota=[14,2],grid_rule='centroidneareststart→physicalfarthest-first,stablecachedties',source_rule='absOOFscore,96pxanchor/source distance,pixeldupremove→relaxdistance→omittedgridfallback',candidate_budget=32,classifier_budget=16,selector_retrained=False,epochs=60,patience=12,lr=.0003,weight_decay=.01,batch=32,init='originalseed42gridhead/frozenglobal,notRound11besthead',puregrid_control='verifiedRound11sameinit/config/epochs60/patience12;predictionsreused',results=[],oof_audit={},selection_audit={})
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();bags={}
 for source in ['gaze_baseline','random']:
  path=origin/f'selector_{source}/selected_indices.pt';record=torch.load(path,map_location='cpu',weights_only=False);assert record['image_ids']==f.image_id.tolist() and record['candidate_source']==source and record['kind']=='patientOOFTRAIN_fullTRAINVAL_hardselection' and record['scores'].shape==(2393,32);m['dependencies'][str(path)]=digest(path)
  for fold,expected in enumerate(old['folds']):
   p=origin/f'selector_{source}/fold{fold}.pt';ck=torch.load(p,map_location='cpu',weights_only=False);assert ck['fit_patients']==expected['fit_patients'] and ck['heldout_patients']==expected['heldout_patients'] and not set(ck['fit_patients'])&set(ck['heldout_patients']);m['dependencies'][str(p)]=digest(p)
  p=origin/f'selector_{source}/fulltrain.pt';ck=torch.load(p,map_location='cpu',weights_only=False);assert set(ck['fit_patients'])==set(f.iloc[tr].patient_key);m['dependencies'][str(p)]=digest(p);m['oof_audit'][source]='freshfoldfitpatientsdisjoint;fullfitTRAINonly;scores/ids/orderpinned'
  candidate=build_candidates(t,source);selections=[quota_indices(record['scores'][i],candidate[1][i],candidate[2][i]) for i in range(len(f))];indices=torch.stack([s[0] for s in selections]);rows=torch.arange(len(f))[:,None];arm='grid14_'+source+'2';bags[arm]=tuple(x[rows,indices] for x in candidate);torch.save(dict(image_ids=f.image_id.tolist(),indices=indices,source=source,rule='fixed14/2'),out/f'{arm}_indices.pt');per=pd.DataFrame([s[1] for s in selections]);m['selection_audit'][arm]={}
  for split in ['train','val']:
   ids=np.flatnonzero(f.split==split);xy=bags[arm][1][ids]*torch.tensor([1500.,3000.]);d=torch.cdist(xy,xy);d.diagonal(dim1=1,dim2=2).fill_(float('inf'));s=per.iloc[ids];m['selection_audit'][arm][split]=dict(n=len(ids),mean_source_count=float(s.source_count.mean()),zero_source_images=int((s.source_count==0).sum()),one_source_images=int((s.source_count==1).sum()),distance_relaxed_points=int(s.distance_relaxed_count.sum()),grid_fallback_points=int(s.grid_fallback_count.sum()),nearest_distance_px=float(d.min(-1).values.mean()),unique16=True)
  del candidate,record
 m['bags_sha256']={a:state_digest(dict(features=b[0],coordinates=b[1],mask=b[2])) for a,b in bags.items()};m['stage']='classifier_training';save();preds={}
 for mode,name in [('singleview','validation_predictions.csv'),('multiview','multiview_validation_predictions.csv')]:
  path=control/'grid_seed42'/name;p=pd.read_csv(path);assert p.image_id.tolist()==f.iloc[va].image_id.tolist() and p.label.tolist()==f.iloc[va].label.tolist();preds[('grid',mode)]=p;m['dependencies'][str(path)]=digest(path)
 assert ctrlm['epochs']==60 and ctrlm['patience']==12 and ctrlm['lr']==.0003 and ctrlm['weight_decay']==.01 and ctrlm['batch']==32
 heads=Path('runs/official_cbis_v2_experiment_1012/heads');gp=heads/'global_seed42/best.pt';hp=heads/'grid_seed42/best.pt';m['dependencies'].update({str(p):digest(p) for p in [gp,hp]});assert ctrlm['dependencies'][str(gp)]==digest(gp) and ctrlm['dependencies'][str(hp)]==digest(hp);gc=torch.load(gp,map_location='cpu',weights_only=False);hc=torch.load(hp,map_location='cpu',weights_only=False)
 for arm,bag in bags.items():
  torch.manual_seed(42);model=restore(hc,gc);globalsha=state_digest(model.global_model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);d=out/f'{arm}_seed42';d.mkdir();best=-1;stale=0;history=[]
  def forward(ids):return model(t['global_features'][ids],bag[0][ids],bag[1][ids],bag[2][ids])
  for epoch in range(61):
   losses=[]
   if epoch:
    model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(42+epoch))]
    for ids in order.split(32):
     opt.zero_grad();loss=attention_loss(forward(ids),y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.inference_mode():prob=torch.cat([forward(ids)['logits'].sigmoid() for ids in va.split(32)]).numpy()
   score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,fitloss=float(np.mean(losses)) if losses else None,validation=score));assert state_digest(model.global_model.state_dict())==globalsha
   if score['auroc']>best:best=score['auroc'];stale=0;bp=prob.copy();be=epoch;torch.save(dict(state=copy.deepcopy(head_state(model)),epoch=epoch,arm=arm,seed=42,frozen_global_sha256=globalsha),d/'best.pt')
   else:stale+=1
   if epoch%10==0 or stale>=12:print(json.dumps(dict(arm=arm,epoch=epoch,auroc=score['auroc'],best=best,stale=stale)),flush=True)
   if stale>=12:break
  pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);preds[(arm,'singleview')]=pred;pmap=dict(zip(pred.image_id,bp));mv=pred.copy();mv['probability']=[.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()];mv.to_csv(d/'multiview_validation_predictions.csv',index=False);preds[(arm,'multiview')]=mv;(d/'history.json').write_text(json.dumps(history))
  for mode,p in [('singleview',pred),('multiview',mv)]:r=dict(arm=arm,mode=mode,epoch=be,stopped_epoch=epoch,metrics=metrics(p.label,p.probability));m['results'].append(r);save();print(json.dumps(r),flush=True)
 report=dict(results=m['results'],puregrid={mode:metrics(preds[('grid',mode)].label,preds[('grid',mode)].probability) for mode in ['singleview','multiview']},selection_audit=m['selection_audit'],paired={})
 for mode in ['singleview','multiview']:
  for arm in bags:report['paired'][arm+'_'+mode+'_minus_grid']=paired_ci(preds[(arm,mode)],preds[('grid',mode)],1000)
  report['paired']['gaze_'+mode+'_minus_matchedrandom']=paired_ci(preds[('grid14_gaze_baseline2',mode)],preds[('grid14_random2',mode)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(__file__)==m['source_sha256'] and digest(cp)==m['cache_sha256'] and all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';m['stage']='report_complete';save()
if __name__=='__main__':main()
