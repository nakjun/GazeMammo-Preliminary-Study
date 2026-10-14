"""Label-free same-breast CC/MLO cached multiview development."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.official_cbis_classifier_1012 import digest,metrics,paired_ci

def pair_views(f):
 side=f.source_sn.str.extract('(LEFT|RIGHT)')[0];view=f.source_sn.str.rsplit('_',n=1).str[-1];partners=[]
 for i,row in f.iterrows():
  candidates=f[(f.patient_key==row.patient_key)&(side==side.iloc[i])&(view!=view.iloc[i])&view.isin(['CC','MLO'])]
  candidates=candidates.sort_values('image_id').drop_duplicates('image_sha256')
  ids=candidates.index.tolist()[:1];assert all(f.iloc[j].split==row.split for j in ids);partners.append(ids)
 return partners,side,view

def pooled_summary(g,p,partners):
 own=torch.cat((g,p.mean(1),p.max(1).values),1);other=torch.stack([own[ids].mean(0) if ids else own[i] for i,ids in enumerate(partners)]);mask=torch.tensor([bool(x) for x in partners],dtype=torch.float32).unsqueeze(1)
 return torch.cat((own,other,mask),1)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[42]);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False)
 c=torch.load(a.cache,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert set(f.split)=={'train','val'} and len(f)==2393 and (f.split=='val').sum()==356 and f.groupby('patient_key').split.nunique().max()==1
 partners,side,view=pair_views(f);tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32)
 diag=dict(rule='same patient, same side, opposite CC/MLO; deduplicate image_sha256 then smallest image_id opposite-view partner; no labels used',splits={})
 for split in ['train','val']:
  ids=np.flatnonzero(f.split==split);diag['splits'][split]=dict(n=len(ids),eligible=sum(bool(partners[i]) for i in ids),missing=sum(not partners[i] for i in ids),multiple_partner=sum(len(partners[i])>1 for i in ids),label_disagreement_targets=sum(any(f.iloc[j].label!=f.iloc[i].label for j in partners[i]) for i in ids))
 diag['duplicate_hash_rows']=int(f.image_sha256.duplicated().sum());(out/'pairing_diagnostics.json').write_text(json.dumps(diag,indent=2));(out/'pairing.json').write_text(json.dumps({str(f.iloc[i].image_id):f.iloc[ids].image_id.tolist() for i,ids in enumerate(partners)},indent=2));print(json.dumps(diag),flush=True)
 m=dict(state='running',stage='pilot42' if a.seeds==[42] else 'three_seed_confirmation',cache_sha256=digest(a.cache),source_sha256=digest(__file__),protocol='extra-image multiview; not same-information single-view comparison',pairing=diag,patch_budget=32,per_view_patch_budget=16,test_used=False,seeds=a.seeds,arms=['grid','gaze_baseline','gaze_improved','random'],architecture='current global+patchmean+patchmax, partner pooled same summary, missing mask; linear64 GELU dropout.5 linear1',epochs=60,patience=12,lr=.0003,weight_decay=.01,normalization='train-only per-dimension mean/std floor .01 clamp10; preserve availability mask',selection='validation AUROC early stopping; preregistered single architecture',results=[],fixed_probability_average=[])
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();preds={}
 # Existing head probabilities for val only: every partner belongs to the same held-out split.
 for seed in a.seeds:
  for arm in m['arms']:
   path=Path('runs/official_cbis_v2_experiment_1012/heads')/f'{arm}_seed{seed}/validation_predictions.csv';base=pd.read_csv(path);base=base.set_index('image_id').loc[f.iloc[va.tolist()].image_id].reset_index();pmap=dict(zip(base.image_id,base.probability));prob=np.array([.5*(pmap[f.iloc[i].image_id]+np.mean([pmap[f.iloc[j].image_id] for j in partners[i]])) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()]);pred=base[['image_id','patient_key','label']].copy();pred['probability']=prob;pred.to_csv(out/f'{arm}_fixedavg_seed{seed}_validation_predictions.csv',index=False);m['fixed_probability_average'].append(dict(arm=arm,seed=seed,metrics=metrics(pred.label,pred.probability)));preds[(arm,'fixedavg',seed)]=pred;save()
 for arm in m['arms']:
  x=pooled_summary(t['global_features'],t[arm+'_features'],partners);mean=x[tr].mean(0);std=x[tr].std(0).clamp_min(.01);mask=x[:,-1].clone();x=((x-mean)/std).clamp(-10,10);x[:,-1]=mask
  for seed in a.seeds:
   torch.manual_seed(seed);d=out/f'{arm}_joint_seed{seed}';d.mkdir();model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(64,1));opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);best=-1;stale=0;history=[]
   for epoch in range(61):
    losses=[]
    if epoch:
     model.train();order=tr[torch.randperm(len(tr))]
     for ids in order.split(64):
      opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[ids]).flatten(),y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():prob=model(x[va]).flatten().sigmoid().numpy()
    score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,fit_loss=float(np.mean(losses)) if losses else None,validation=score))
    if score['auroc']>best:best=score['auroc'];stale=0;bp=prob.copy();be=epoch;bs=copy.deepcopy(model.state_dict())
    else:stale+=1
    if stale>=12:break
   pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);torch.save(dict(state=bs,mean=mean,std=std,epoch=be,arm=arm,seed=seed),d/'best.pt');(d/'history.json').write_text(json.dumps(history));preds[(arm,'joint',seed)]=pred;r=dict(arm=arm,seed=seed,epoch=be,metrics=metrics(pred.label,pred.probability));m['results'].append(r);save();print(json.dumps(r),flush=True)
 report=dict(results=m['results'],fixed_probability_average=m['fixed_probability_average'],paired={})
 for seed in a.seeds:
  for mode in ['fixedavg','joint']:
   for arm in m['arms']:
    if arm!='grid':report['paired'][f'{arm}_{mode}_{seed}_minus_multiview_grid']=paired_ci(preds[(arm,mode,seed)],preds[('grid',mode,seed)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(a.cache)==m['cache_sha256'];assert digest(__file__)==m['source_sha256'];m['state']='complete';save()
if __name__=='__main__':main()
