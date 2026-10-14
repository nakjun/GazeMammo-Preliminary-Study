"""Small seed42 pilot: fixed predictedgaze16 + label-freecoveragegridK."""
import json,sys,copy,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,pandas as pd
from src.gaze16_grid_addition_1013 import build_bags
from src.official_cbis_classifier_1012 import digest,restore,metrics,state_digest
from src.gaze_attention_1011 import attention_loss,head_state
from scripts.train_gaze_multiview_1012 import pair_views

def atomic(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze16_grid_addition_pilot_1013');out.mkdir(exist_ok=False);start=time.time();cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert set(f.split)=={'train','val'} and len(f)==2393 and (f.split=='val').sum()==356;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);bags,selection=build_bags(t);partners,_,_=pair_views(f);torch.save(selection,out/'grid_additions.pt');heads=Path('runs/official_cbis_v2_experiment_1012/heads');gp=heads/'global_seed42/best.pt';hp=heads/'grid_seed42/best.pt';gc=torch.load(gp,map_location='cpu',weights_only=False);hc=torch.load(hp,map_location='cpu',weights_only=False);deps={p:digest(p) for p in [__file__,'src/gaze16_grid_addition_1013.py','src/gaze_attention_1011.py',str(gp),str(hp)]};m=dict(state='running',seed=42,pilot_only=True,ks=[0,1,2,4],gaze16_exactfixed=True,gaze_source='predictedgaze_baseline,notexpert',grid_selection='physicalfarthestfromallgazeanchors;pixelduplicatesremoved;nestedprefix;maskpadifneeded',epochs=120,patience=12,lr=.0003,weight_decay=.01,batch=32,head='originalmatchedgridmean_residualdim64dropout.2/frozenglobal',selection_metric='singleviewVALAUROC;multiviewfixedmetadataonlyprobavg',test_used=False,oracle_used=False,selector_trained=False,raw_features_extracted=False,cache_sha256=digest(cp),dependencies=deps,results=[])
 def save():
  atomic(out/'manifest.json',m);temp=out/'progress.tmp';pd.DataFrame(m['results']).to_csv(temp,index=False);temp.replace(out/'progress.csv')
 save()
 for k,bag in bags.items():
  torch.manual_seed(42);model=restore(hc,gc);frozen=state_digest(model.global_model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);d=out/f'grid{k}';d.mkdir();best=-1;stale=0;history=[]
  def forward(ids):return model(t['global_features'][ids],bag[0][ids],bag[1][ids],bag[2][ids])
  for epoch in range(121):
   losses=[]
   if epoch:
    model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(42+epoch))]
    for ids in order.split(32):
     opt.zero_grad();loss=attention_loss(forward(ids),y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.inference_mode():p=torch.cat([forward(ids)['logits'].sigmoid() for ids in va.split(32)]).numpy()
   score=metrics(y[va].numpy(),p);history.append(dict(epoch=epoch,fitloss=float(np.mean(losses)) if losses else None,validation=score));assert state_digest(model.global_model.state_dict())==frozen
   if score['auroc']>best:best=score['auroc'];stale=0;bp=p.copy();be=epoch;torch.save(dict(state=copy.deepcopy(head_state(model)),epoch=epoch,k=k,seed=42,frozen_global_sha256=frozen),d/'best.pt')
   else:stale+=1
   m['current_k']=k;m['latest_epoch']=epoch;atomic(out/'manifest.json',m)
   if stale>=12:break
  pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);mapping=dict(zip(pred.image_id,bp));mv=pred.copy();mv['probability']=[.5*(mapping[f.iloc[i].image_id]+mapping[f.iloc[partners[i][0]].image_id]) if partners[i] else mapping[f.iloc[i].image_id] for i in va.tolist()];mv.to_csv(d/'multiview_validation_predictions.csv',index=False);atomic(d/'history.json',history);single=metrics(pred.label,pred.probability);multi=metrics(mv.label,mv.probability);r=dict(k=k,nominal_patch_budget=16+k,best_epoch=be,stopped_epoch=epoch,singleview_auroc=single['auroc'],multiview_auroc=multi['auroc'],singleview_auprc=single['auprc'],multiview_auprc=multi['auprc'],valid_added_grid_val_mean=float(bag[2][va,16:].sum(1).float().mean()));m['results'].append(r);save();print(json.dumps(r),flush=True)
 assert all(digest(p)==sha for p,sha in deps.items()) and digest(cp)==m['cache_sha256'];m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();atomic(out/'report.json',dict(results=m['results'],limitations='Singleclassifierseed42pilot,not3seedgoalmetric. Gaze16alwaysretainedincludingduplicateswithoriginalduplicatecorrection. Addedgridbudget16+K differsfrompriorGrid14+gaze2/4. Noaddedrandomcomparisonyet.'))
if __name__=='__main__':main()
