"""AUROC88 round1: trainable paired global-only/gaze16/random16, seed42 pilot."""
import json,copy,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.gaze_focus_joint_1013 import JointGazeFocus
from src.official_cbis_classifier_1012 import digest,metrics
from scripts.train_gaze_multiview_1012 import pair_views

def write(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/gaze_focus_auroc88_1013/round01_joint_paired_pilot42');out.mkdir(exist_ok=False);start=time.time();cp='runs/official_cbis_v2_experiment_1012/development.pt';cache=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors'];assert len(f)==2393 and set(f.split)=={'train','val'} and (f.split=='val').sum()==356 and f.groupby('patient_key').split.nunique().max()==1;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f);other=torch.tensor([p[0] if p else i for i,p in enumerate(partners)]);available=torch.tensor([bool(p) for p in partners]);deps={p:digest(p) for p in [__file__,'src/gaze_focus_joint_1013.py','scripts/train_gaze_multiview_1012.py']};m=dict(state='running',stage='joint_training',target=.88,pilot_seed=42,classifier_seeds_confirmed=False,backbone='FixedImageNetConvNeXtBasecached1024features;architectureunchanged',new_hypothesis='Trainableglobal + global-conditionedlocalattention + learnedCC/MLOcontext/differencefusion; previous frozenweakscalar global removed',arms=['global_only','global_gaze16','global_random16'],capacity='Sameglobalencoder/classifier/pairedtraining;unusedpatchmodulespresentinglobal-only',lr=.0003,weight_decay=.01,epochs=80,patience=12,batch=32,partner_dropout=.5,local_patches_per_view=16,pair_rule='samepatient/sideoppositeview deterministicminid;missingcurrentfallback',selection='learnedpairedVALAUROCepoch0..80;singleview/fixedprobavg reported separately',test_used=False,oracle_used=False,hard_selector_trained=False,source_sha256=digest(__file__),cache_sha256=digest(cp),dependencies=deps,results=[])
 def save():
  write(out/'manifest.json',m);temp=out/'progress.tmp';pd.DataFrame(m['results']).to_csv(temp,index=False);temp.replace(out/'progress.csv')
 save()
 def batch(ids,source,mode):
  paired=available[ids].clone() if mode!='singleview' else torch.zeros(len(ids),dtype=torch.bool)
  if mode=='training':paired&=torch.rand(len(ids))>=.5
  both=torch.stack([ids,other[ids]],1);name='gaze_baseline' if source=='global_only' else source;return t['global_features'][both],t[name+'_features'][both],t[name+'_coordinates'][both],t[name+'_mask'][both],paired
 for arm,source in [('global_only','global_only'),('global_gaze16','gaze_baseline'),('global_random16','random')]:
  torch.manual_seed(42);model=JointGazeFocus(local=source!='global_only');opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);d=out/arm;d.mkdir();best=-1;stale=0;history=[]
  for epoch in range(81):
   losses=[]
   if epoch:
    model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(42+epoch))]
    for ids in order.split(32):
     opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(*batch(ids,source,'training')),y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.inference_mode():p=torch.cat([model(*batch(ids,source,'paired')).sigmoid() for ids in va.split(32)]).numpy()
   score=metrics(y[va].numpy(),p);history.append(dict(epoch=epoch,fitloss=float(np.mean(losses)) if losses else None,paired_validation=score))
   if score['auroc']>best:best=score['auroc'];stale=0;be=epoch;state=copy.deepcopy(model.state_dict());torch.save(dict(state=state,arm=arm,epoch=be,seed=42),d/'best.pt')
   else:stale+=1
   m['current_arm']=arm;m['latest_epoch']=epoch;write(out/'manifest.json',m)
   if stale>=12:break
  model.load_state_dict(state);model.eval();atomic=[]
  with torch.inference_mode():
   paired=torch.cat([model(*batch(ids,source,'paired')).sigmoid() for ids in va.split(32)]).numpy();single=torch.cat([model(*batch(ids,source,'singleview')).sigmoid() for ids in va.split(32)]).numpy()
  mapping=dict(zip(f.iloc[va.tolist()].image_id,single));fixed=np.array([.5*(mapping[f.iloc[i].image_id]+mapping[f.iloc[partners[i][0]].image_id]) if partners[i] else mapping[f.iloc[i].image_id] for i in va.tolist()])
  for mode,p in [('singleview',single),('learned_paired',paired),('fixed_probability_average',fixed)]:
   pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=p;pred.to_csv(d/f'{mode}_validation_predictions.csv',index=False);r=dict(arm=arm,mode=mode,best_epoch=be,stopped_epoch=epoch,metrics=metrics(pred.label,pred.probability));m['results'].append(r)
  write(d/'history.json',history);save();print(json.dumps(dict(arm=arm,epoch=be,single_auroc=m['results'][-3]['metrics']['auroc'],learned_paired_auroc=m['results'][-2]['metrics']['auroc'],fixed_average_auroc=m['results'][-1]['metrics']['auroc'])),flush=True)
 assert all(digest(p)==sha for p,sha in deps.items()) and digest(cp)==m['cache_sha256'];m.update(state='complete',elapsed_minutes=(time.time()-start)/60);save();write(out/'report.json',dict(results=m['results'],limitations='Singlepilot42not3seedmean;frozenpooledImageNetglobalfeatureslosefine/spatialdetail;learnedpairedinputdiffersfromfixedprobabilityaverage;VALreusedexploratory;attentioniswithinclassifiernotseparatelabelmemorizingROIselector;noTEST/oracle'))
 root=out.parent;write(root/'progress.json',dict(active_goal='wholeimage+predictedgaze,3classifierseedmeanAUROC>=.88',round=1,round_state='complete',round_output=str(out),pilot_results=m['results'],target_met=False,next_step='rootreviewbefore3seedexpansion'))
if __name__=='__main__':main()
