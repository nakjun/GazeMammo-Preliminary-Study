"""Existing matched mean-residual head + copied last CN block,5 epochs."""
import argparse,copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.gaze_backbones import GazePyramid
from src.official_cbis_classifier_1012 import digest,restore,metrics,paired_ci,state_digest
from src.gaze_attention_1011 import attention_loss,head_state
from scripts.train_gaze_multiview_1012 import pair_views

def prefix_state(encoder):return {k:v for k,v in encoder.state_dict().items() if not k.startswith('features.7.2.') and not k.startswith('norm.')}
def boundary_vectors(encoder,boundary):
 z=encoder.features[7][-1](boundary.float());return encoder.norm(z.mean((-1,-2),keepdim=True)).flatten(1)

class Adapted(torch.nn.Module):
 def __init__(self,encoder,head):
  super().__init__();self.encoder=encoder.requires_grad_(False);self.encoder.features[7][-1].requires_grad_(True);self.encoder.norm.requires_grad_(True);self.head=head
 def forward(self,g,b,c,m):
  shape=b.shape[:2];v=boundary_vectors(self.encoder,b.flatten(0,1)).reshape(*shape,1024);return self.head(g,v,c,m)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';cache=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors'];fm=json.loads((Path(a.features)/'manifest.json').read_text());assert fm['state']=='complete' and fm['completed_records']==2393;tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));assert len(va)==356 and f.groupby('patient_key').split.nunique().max()==1;y=torch.tensor(f.label.values,dtype=torch.float32);partners,_,_=pair_views(f)
 sources=['scripts/train_gaze_multiview_1012.py','src/gaze_attention_1011.py','src/gaze_backbones.py','src/official_cbis_classifier_1012.py'];m=dict(state='running',stage='train',source_sha256=digest(__file__),cache_sha256=digest(cp),feature_manifest_sha256=digest(Path(a.features)/'manifest.json'),dependencies={p:digest(p) for p in sources},test_used=False,seeds=[42],arms=['grid','gaze_baseline'],variants=['frozen_head_control','lastblock_adaptation'],epochs=5,backbone_lr=1e-5,head_lr=3e-4,weight_decay=.05,bags_per_batch=4,patch_budget=16,init='original matched global+mean_residual head; epoch0 retains previous predictions within measured CPU/fp16 numeric error',selection='Best VAL AUROC among epochs0..5, ties first; multiview fixedavg after selection',oracle_used=False,results=[],epoch0_parity={})
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();preds={};heads=Path('runs/official_cbis_v2_experiment_1012/heads');seed=42;gp=heads/'global_seed42/best.pt';m['dependencies'][str(gp)]=digest(gp);gc=torch.load(gp,map_location='cpu',weights_only=False)
 def load_boundaries(ids,arm):
  rows=[]
  for i in ids.tolist():
   row=f.iloc[i];path=Path(a.features)/'records'/f"{row.image_id}.pt";r=torch.load(path,map_location='cpu',weights_only=False);assert r['pixel_sha256']==row.image_sha256 and r['split']==row.split;rows.append(r['features'][arm])
  return torch.stack(rows)
 # Once verify all artifacts before any learned updates.
 for row in cache['rows']:
  path=Path(a.features)/'records'/f"{row['image_id']}.pt";assert digest(path)==fm['records'][str(row['image_id'])]['record_sha256']
 for arm in m['arms']:
  hp=heads/f'{arm}_seed42/best.pt';m['dependencies'][str(hp)]=digest(hp);hc=torch.load(hp,map_location='cpu',weights_only=False)
  for variant in m['variants']:
   torch.manual_seed(seed);head=restore(hc,gc);encoder=None
   if variant=='lastblock_adaptation':
    encoder=GazePyramid('convnext_base',pretrained=True).eval();assert state_digest(encoder.state_dict())==fm['visual_state_sha256'];model=Adapted(encoder,head);frozen_prefix=state_digest(prefix_state(encoder));opt=torch.optim.AdamW([dict(params=list(encoder.features[7][-1].parameters())+list(encoder.norm.parameters()),lr=1e-5),dict(params=[p for p in head.parameters() if p.requires_grad],lr=3e-4)],weight_decay=.05)
   else:model=head;opt=torch.optim.AdamW([p for p in head.parameters() if p.requires_grad],lr=3e-4,weight_decay=.05)
   global_sha=state_digest(head.global_model.state_dict());d=out/f'{arm}_{variant}_seed42';d.mkdir();history=[];best=-1
   def forward(ids):
    if encoder is None:return model(t['global_features'][ids],t[arm+'_features'][ids],t[arm+'_coordinates'][ids],t[arm+'_mask'][ids])
    return model(t['global_features'][ids],load_boundaries(ids,arm),t[arm+'_coordinates'][ids],t[arm+'_mask'][ids])
   for epoch in range(6):
    losses=[]
    if epoch:
     model.train();order=tr[torch.randperm(len(tr),generator=torch.Generator().manual_seed(seed+epoch))]
     for ids in order.split(4):
      opt.zero_grad();loss=attention_loss(forward(ids),y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
    model.eval()
    with torch.inference_mode():prob=torch.cat([forward(ids)['logits'].sigmoid() for ids in va.split(4)]).numpy()
    score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,fit_loss=float(np.mean(losses)) if losses else None,validation=score));assert state_digest(head.global_model.state_dict())==global_sha
    if encoder is not None:assert state_digest(prefix_state(encoder))==frozen_prefix
    if epoch==0:
     ref=pd.read_csv(heads/f'{arm}_seed42/validation_predictions.csv').set_index('image_id').loc[f.iloc[va.tolist()].image_id].probability.values;m['epoch0_parity'][arm+'_'+variant]=dict(max_abs_probability=float(np.max(np.abs(prob-ref))),auroc=score['auroc'],frozen_global_sha256=global_sha,frozen_prefix_sha256=frozen_prefix if encoder is not None else None);save()
    if score['auroc']>best:
     best=score['auroc'];bp=prob.copy();be=epoch;state=dict(head=copy.deepcopy(head_state(head)),block=copy.deepcopy(encoder.features[7][-1].state_dict()) if encoder is not None else None,norm=copy.deepcopy(encoder.norm.state_dict()) if encoder is not None else None);torch.save(dict(state=state,epoch=epoch,arm=arm,variant=variant,seed=seed,global_sha256=global_sha),d/'best.pt')
    print(json.dumps(dict(arm=arm,variant=variant,epoch=epoch,auroc=score['auroc'])),flush=True)
   pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);preds[(arm,variant,'singleview')]=pred;(d/'history.json').write_text(json.dumps(history));pmap=dict(zip(pred.image_id,bp));mv=pred.copy();mv['probability']=[.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va.tolist()];mv.to_csv(d/'multiview_validation_predictions.csv',index=False);preds[(arm,variant,'multiview')]=mv
   for mode,p in [('singleview',pred),('multiview',mv)]:m['results'].append(dict(arm=arm,variant=variant,mode=mode,seed=seed,epoch=be,metrics=metrics(p.label,p.probability)))
   save();del model,head,opt,encoder
 report=dict(results=m['results'],epoch0_parity=m['epoch0_parity'],paired={})
 for arm in m['arms']:
  for mode in ['singleview','multiview']:report['paired'][f'{arm}_{mode}_adapt_minus_frozen']=paired_ci(preds[(arm,'lastblock_adaptation',mode)],preds[(arm,'frozen_head_control',mode)],1000)
 for mode in ['singleview','multiview']:report['paired'][f'gaze_{mode}_adapt_minus_grid']=paired_ci(preds[('gaze_baseline','lastblock_adaptation',mode)],preds[('grid','lastblock_adaptation',mode)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(cp)==m['cache_sha256'] and digest(__file__)==m['source_sha256'] and all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';m['stage']='report_complete';save()
if __name__=='__main__':main()
