"""Round11: frozen Round10 OOF bags, same classifier training extended to60."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.official_cbis_classifier_1012 import digest,restore,metrics,paired_ci,state_digest
from src.gaze_attention_1011 import attention_loss,head_state
from scripts.train_gaze_diagnosis_selector_1012 import build_candidates
from scripts.train_gaze_multiview_1012 import pair_views

def fixed_bags(cache,origin):
 t=cache['tensors'];expected=[r['image_id'] for r in cache['rows']];bags={'grid':(t['grid_features'],t['grid_coordinates'],t['grid_mask'])};hashes={}
 for source in ['gaze_baseline','random']:
  path=Path(origin)/f'selector_{source}/selected_indices.pt';record=torch.load(path,map_location='cpu',weights_only=False)
  if record['image_ids']!=expected:raise ValueError('Selection image identity/order mismatch')
  assert record['kind']=='patientOOFTRAIN_fullTRAINVAL_hardselection' and record['candidate_source']==source;indices=record['indices'];assert indices.shape==(len(expected),16) and ((indices>=0)&(indices<32)).all();assert all(len(torch.unique(i))==16 for i in indices);rows=torch.arange(len(expected))[:,None];bags['selected_'+source]=tuple(x[rows,indices] for x in build_candidates(t,source));hashes[str(path)]=digest(path)
 return bags,hashes

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);origin=Path('runs/gaze_diagnosis_selector_1012_round10_pilot42');oldm=json.loads((origin/'manifest.json').read_text());assert oldm['state']=='complete';cp='runs/official_cbis_v2_experiment_1012/development.pt';assert digest(cp)==oldm['cache_sha256'];assert digest('scripts/train_gaze_diagnosis_selector_1012.py')==oldm['source_sha256'];c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert len(f)==2393 and set(f.split)=={'train','val'} and (f.split=='val').sum()==356 and f.groupby('patient_key').split.nunique().max()==1;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f);bags,selection_hashes=fixed_bags(c,origin);deps={str(origin/'manifest.json'):digest(origin/'manifest.json'),'scripts/train_gaze_diagnosis_selector_1012.py':digest('scripts/train_gaze_diagnosis_selector_1012.py'),'scripts/train_gaze_multiview_1012.py':digest('scripts/train_gaze_multiview_1012.py'),**selection_hashes}
 for arm in bags:
  for name in ['history.json','validation_predictions.csv','best.pt']:
   p=origin/f'{arm}_seed42'/name;deps[str(p)]=digest(p)
 m=dict(state='running',stage='classifier_training',source_sha256=digest(__file__),cache_sha256=digest(cp),dependencies=deps,selector_retrained=False,selection_fixed_from=str(origin),bags_sha256={a:state_digest(dict(features=b[0],coordinates=b[1],mask=b[2])) for a,b in bags.items()},test_used=False,oracle_used=False,seed=42,epochs=60,patience=12,patience_applied_after_epoch20_reproduction=True,lr=.0003,weight_decay=.01,batch=32,patch_budget=16,candidate_budget=32,init='Sameoriginalgridhead/frozenglobal/RNGasRound10',results=[],reproduction={})
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();heads=Path('runs/official_cbis_v2_experiment_1012/heads');gp=heads/'global_seed42/best.pt';hp=heads/'grid_seed42/best.pt';m['dependencies'].update({str(p):digest(p) for p in [gp,hp]});gc=torch.load(gp,map_location='cpu',weights_only=False);hc=torch.load(hp,map_location='cpu',weights_only=False);preds={}
 for arm,bag in bags.items():
  torch.manual_seed(42);model=restore(hc,gc);globalsha=state_digest(model.global_model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);d=out/f'{arm}_seed42';d.mkdir();best=-1;stale=0;history=[];oldhistory=json.loads((origin/f'{arm}_seed42/history.json').read_text());oldprob=pd.read_csv(origin/f'{arm}_seed42/validation_predictions.csv').set_index('image_id').loc[f.iloc[va.tolist()].image_id].probability.to_numpy(dtype=np.float32);oldck=torch.load(origin/f'{arm}_seed42/best.pt',map_location='cpu',weights_only=False)
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
   if epoch<=20 and history[-1]!=oldhistory[epoch]:raise RuntimeError(f'Round10historyexactreproductionfailed: {arm}epoch{epoch}')
   if score['auroc']>best:best=score['auroc'];bp=prob.copy();be=epoch;stale=0;torch.save(dict(state=copy.deepcopy(head_state(model)),epoch=epoch,arm=arm,seed=42,frozen_global_sha256=globalsha),d/'best.pt')
   else:stale+=1
   if epoch==20:
    assert oldck['epoch']==20 and np.array_equal(prob,oldprob);assert state_digest(head_state(model))==state_digest(oldck['state']);m['reproduction'][arm]=dict(first21history_rows_exact=True,epoch20_probabilities_float32_exact=True,epoch20_head_state_sha256_exact=True,bag_sha256=m['bags_sha256'][arm]);save();print(json.dumps(dict(arm=arm,epoch20_reproduction='passed')),flush=True)
   if epoch>=20 and (epoch%10==0 or stale>=12):print(json.dumps(dict(arm=arm,epoch=epoch,auroc=score['auroc'],best=best,stale=stale)),flush=True)
   if epoch>=20 and stale>=12:break
  pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);preds[(arm,'singleview')]=pred;pmap=dict(zip(pred.image_id,bp));mv=pred.copy();mv['probability']=[.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()];mv.to_csv(d/'multiview_validation_predictions.csv',index=False);preds[(arm,'multiview')]=mv;(d/'history.json').write_text(json.dumps(history))
  for mode,p in [('singleview',pred),('multiview',mv)]:r=dict(arm=arm,mode=mode,epoch=be,stopped_epoch=epoch,metrics=metrics(p.label,p.probability));m['results'].append(r);save();print(json.dumps(r),flush=True)
 report=dict(results=m['results'],reproduction=m['reproduction'],paired={})
 for mode in ['singleview','multiview']:
  for arm in ['selected_gaze_baseline','selected_random']:report['paired'][arm+'_'+mode+'_minus_grid']=paired_ci(preds[(arm,mode)],preds[('grid',mode)],1000)
  report['paired']['gaze_'+mode+'_minus_matchedrandom']=paired_ci(preds[('selected_gaze_baseline',mode)],preds[('selected_random',mode)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(__file__)==m['source_sha256'] and digest(cp)==m['cache_sha256'] and all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';m['stage']='report_complete';save()
if __name__=='__main__':main()
