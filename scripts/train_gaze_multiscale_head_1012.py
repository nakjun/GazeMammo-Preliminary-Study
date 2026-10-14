"""Preregistered original mean-residual head on concatenated 224+448 features."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.official_cbis_classifier_1012 import digest,restore,metrics,paired_ci,state_digest
from src.gaze_attention_1011 import GazeAttention1011,attention_loss,head_state
from scripts.train_gaze_multiview_1012 import pair_views

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',nargs='+',type=int,default=[42]);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];audit=json.loads((Path(a.features).parent/'preregistration_audit.json').read_text());assert audit['training_allowed'] and all(digest(p)==h for p,h in audit['head_dependencies_sha256'].items());fm=json.loads((Path(a.features)/'manifest.json').read_text());assert fm['state']=='complete' and fm['completed_records']==2393;parts={arm:[] for arm in ['grid','gaze_baseline']}
 for row in c['rows']:
  path=Path(a.features)/'records'/f"{row['image_id']}.pt";assert digest(path)==fm['records'][str(row['image_id'])]['record_sha256'];r=torch.load(path,map_location='cpu',weights_only=False);assert r['pixel_sha256']==row['image_sha256'] and r['split']==row['split']
  for arm in parts:parts[arm].append(r['features'][arm])
 combined={arm:torch.cat([t[arm+'_features'],torch.stack(p)],-1) for arm,p in parts.items()};partners,_,_=pair_views(f);tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32)
 m=dict(state='running',stage='pilot42' if a.seeds==[42] else 'three_seed_confirmation',test_used=False,seeds=a.seeds,arms=list(combined),input_dim=2048,hidden_dim=64,variant='mean_residual',patch_budget=32,bag_tokens=16,scales=[224,448],global_head='original matched frozen global224 head',epochs=60,patience=12,lr=.0003,weight_decay=.01,dropout=.2,source_sha256=digest(__file__),cache_sha256=digest(cp),feature_manifest_sha256=digest(Path(a.features)/'manifest.json'),selection='validation AUROC, epoch0 included; single preregistered architecture',results=[],dependencies=dict(audit['head_dependencies_sha256']))
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();preds={}
 for seed in a.seeds:
  gp=Path('runs/official_cbis_v2_experiment_1012/heads')/f'global_seed{seed}/best.pt';m['dependencies'][str(gp)]=digest(gp);gc=torch.load(gp,map_location='cpu',weights_only=False)
  for arm in combined:
   torch.manual_seed(seed);model=GazeAttention1011(restore(gc),variant='mean_residual',input_dim=2048,dim=64,dropout=.2);globalhash=state_digest(model.global_model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);d=out/f'{arm}_seed{seed}';d.mkdir();history=[];best=-1;stale=0
   def forward(ids):return model(t['global_features'][ids],combined[arm][ids],t[arm+'_coordinates'][ids],t[arm+'_mask'][ids])
   for epoch in range(61):
    losses=[]
    if epoch:
     model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))]
     for ids in order.split(32):
      opt.zero_grad();loss=attention_loss(forward(ids),y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():prob=torch.cat([forward(ids)['logits'].sigmoid() for ids in va.split(32)]).numpy()
    score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,validation=score))
    if score['auroc']>best:best=score['auroc'];stale=0;bp=prob.copy();be=epoch;bs=copy.deepcopy(head_state(model))
    else:stale+=1
    assert state_digest(model.global_model.state_dict())==globalhash
    if stale>=12:break
   pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);preds[(arm,'singleview',seed)]=pred;torch.save(dict(kind='multiscale_mean_residual_1012',state=bs,model_config=model.config,global_sha256=globalhash,arm=arm,seed=seed,epoch=be),d/'best.pt');(d/'history.json').write_text(json.dumps(history));pmap=dict(zip(pred.image_id,bp));p=np.array([.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()]);mv=pred.copy();mv['probability']=p;mv.to_csv(d/'multiview_validation_predictions.csv',index=False);preds[(arm,'multiview',seed)]=mv
   for mode,prediction in [('singleview',pred),('multiview',mv)]:
    r=dict(arm=arm,mode=mode,seed=seed,epoch=be,metrics=metrics(prediction.label,prediction.probability));m['results'].append(r);save();print(json.dumps(r),flush=True)
 report=dict(results=m['results'],paired={})
 for seed in a.seeds:
  for arm in combined:
   baseline=pd.read_csv(Path('runs/official_cbis_v2_experiment_1012/heads')/f'{arm}_seed{seed}/validation_predictions.csv');report['paired'][f'{arm}_{seed}_minus_224_samearm']=paired_ci(preds[(arm,'singleview',seed)],baseline,1000)
  for mode in ['singleview','multiview']:report['paired'][f'gaze_{mode}_{seed}_minus_grid']=paired_ci(preds[('gaze_baseline',mode,seed)],preds[('grid',mode,seed)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(__file__)==m['source_sha256'];assert digest(cp)==m['cache_sha256'];assert all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';save()
if __name__=='__main__':main()
