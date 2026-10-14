"""Fresh patient-OOF diagnosis scorer→hard16 selection→matched classifier."""
import argparse,copy,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.official_cbis_classifier_1012 import digest,restore,metrics,paired_ci,state_digest
from src.gaze_attention_1011 import attention_loss,head_state
from scripts.train_gaze_oof_fusion_1012 import patient_folds
from scripts.train_gaze_multiview_1012 import pair_views

class Selector(torch.nn.Module):
 def __init__(self,dim=1024,hidden=32):
  super().__init__();self.patch=torch.nn.Sequential(torch.nn.LayerNorm(dim),torch.nn.Linear(dim,hidden),torch.nn.GELU(),torch.nn.Dropout(.2),torch.nn.Linear(hidden,1));self.global_head=torch.nn.Sequential(torch.nn.LayerNorm(dim),torch.nn.Linear(dim,1))
 def forward(self,g,p):
  s=self.patch(p).squeeze(-1);positive=.5*torch.logsumexp(s/.5,1);negative=.5*torch.logsumexp(-s/.5,1);return self.global_head(g).flatten()+.5*(positive-negative),s

def fit_selector(g,p,y,fit,seed,epochs=20,hidden=32):
 """Reads only fit labels; no fold-holdout or VAL epoch selection."""
 torch.manual_seed(seed);model=Selector(g.shape[-1],hidden);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);fit=torch.as_tensor(fit);history=[]
 for epoch in range(epochs):
  model.train();order=fit[torch.randperm(len(fit))];losses=[]
  for ids in order.split(32):
   opt.zero_grad();logits,s=model(g[ids],p[ids]);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ids])+.01*s.square().mean();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
  history.append(dict(epoch=epoch+1,fit_loss=float(np.mean(losses))))
 return model.eval(),history

@torch.inference_mode()
def patch_scores(model,g,p,ids):return torch.cat([model(g[x],p[x])[1] for x in torch.as_tensor(ids).split(32)])

def select_hard(scores,coords,valid,minimum_distance=96):
 """Label-free absolute evidence rank, pixel deduplication, distance then fill."""
 xy=torch.round(coords.double()*torch.tensor([1500.,3000.],dtype=torch.float64));order=sorted(range(len(scores)),key=lambda j:(-float(abs(scores[j])),j));unique=[];seen=set()
 for j in order:
  key=tuple(xy[j].tolist())
  if bool(valid[j]) and key not in seen:unique.append(j);seen.add(key)
 assert len(unique)>=16
 chosen=[]
 for j in unique:
  if all(float((xy[j]-xy[k]).norm())>=minimum_distance for k in chosen):chosen.append(j)
  if len(chosen)==16:break
 if len(chosen)<16:
  for j in unique:
   if j not in chosen:chosen.append(j)
   if len(chosen)==16:break
 return torch.tensor(chosen)

def build_candidates(t,source):return tuple(torch.cat([t['grid_'+key],t[source+'_'+key]],1) for key in ['features','coordinates','mask'])

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args();torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';c=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(c['rows']);t=c['tensors'];assert len(f)==2393 and (f.split=='val').sum()==356 and set(f.split)=={'train','val'};tr=np.flatnonzero(f.split=='train');va=np.flatnonzero(f.split=='val');y=torch.tensor(f.label.values,dtype=torch.float32);folds=patient_folds(f,42);partners,_,_=pair_views(f)
 deps=['scripts/train_gaze_oof_fusion_1012.py','scripts/train_gaze_multiview_1012.py','src/gaze_attention_1011.py','src/official_cbis_classifier_1012.py'];m=dict(state='running',stage='oof_selector_training',seed=42,test_used=False,oracle_used=False,source_sha256=digest(__file__),cache_sha256=digest(cp),dependencies={p:digest(p) for p in deps},selector=dict(init='fresh random; no fullTRAIN supervised classifier state',architecture='per-patchLN→Linear32 GELU dropout.2 scalar + freshglobalLNLinear',loss='bagBCE on global + half(positiveLSE-negativeLSE) + .01 patchscoresquared',epochs=20,lr=.0003,weight_decay=.01,normalization='LayerNorm per feature vector only; no cross-row statistics',training_distribution='Eacharm:32candidatepatches=grid16+source16; identicalconfig,seeds,folds,andbudget for gaze/random'),selection='absolute signedscore, stable cached-indexties, deduplicate roundedsourcepixelcenter, fixed96px spatialdistance then remainingrank fill16',candidate_budget=32,selected_budget=16,folds=[dict(fit_patients=sorted(f.iloc[a].patient_key.unique()),heldout_patients=sorted(f.iloc[b].patient_key.unique())) for a,b in folds],downstream=dict(init='original fullTRAIN trained stronggrid mean_residualhead; classifier not OOF, onlyselector is patientOOF',frozen_global=True,epochs=20,lr=.0003,weight_decay=.01,batch=32,selection='VALAUROC epoch0..20, tiesfirst; multiviewfixedprobavg'),limitations='Weak imagebag labels do not establish lesion localization. Bothscorers use separate candidate-arm distributions with identical architecture/normalization. VAL reused exploratorily.',results=[])
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();selected={'grid':(t['grid_features'],t['grid_coordinates'],t['grid_mask'])};audits={}
 for source in ['gaze_baseline','random']:
  candidate=build_candidates(t,source);scores=torch.full((len(f),32),float('nan'));record=out/f'selector_{source}';record.mkdir()
  for fold,(fit,hold) in enumerate(folds):
   assert not set(f.iloc[fit].patient_key)&set(f.iloc[hold].patient_key);model,history=fit_selector(t['global_features'],candidate[0],y,fit,142+fold);scores[hold]=patch_scores(model,t['global_features'],candidate[0],hold);torch.save(dict(state=model.state_dict(),fit_patients=sorted(f.iloc[fit].patient_key.unique()),heldout_patients=sorted(f.iloc[hold].patient_key.unique()),history=history,epochs=20,seed=142+fold),record/f'fold{fold}.pt');print(json.dumps(dict(stage='selector',source=source,fold=fold,state='complete')),flush=True)
  model,history=fit_selector(t['global_features'],candidate[0],y,tr,42);scores[va]=patch_scores(model,t['global_features'],candidate[0],va);torch.save(dict(state=model.state_dict(),fit_patients=sorted(f.iloc[tr].patient_key.unique()),history=history,epochs=20,seed=42),record/'fulltrain.pt');assert torch.isfinite(scores).all()
  indices=torch.stack([select_hard(scores[i],candidate[1][i],candidate[2][i]) for i in range(len(f))]);rows=torch.arange(len(f))[:,None];selected['selected_'+source]=tuple(x[rows,indices] for x in candidate);torch.save(dict(image_ids=f.image_id.tolist(),indices=indices,scores=scores,kind='patientOOFTRAIN_fullTRAINVAL_hardselection',candidate_source=source),record/'selected_indices.pt');audits[source]={}
  for split in ['train','val']:
   ids=np.flatnonzero(f.split==split);xy=selected['selected_'+source][1][ids]*torch.tensor([1500.,3000.]);d=torch.cdist(xy,xy);d.diagonal(dim1=1,dim2=2).fill_(float('inf'));audits[source][split]=dict(n=len(ids),source_patch_fraction=float((indices[ids]>=16).float().mean()),nearest_distance_px=float(d.min(-1).values.mean()),unique_rounded_centers_mean=float(np.mean([len(torch.unique(torch.round(x),dim=0)) for x in xy])))
  del model,candidate,scores
 (out/'selection_audit.json').write_text(json.dumps(audits,indent=2));m['stage']='downstream_classifier';save();heads=Path('runs/official_cbis_v2_experiment_1012/heads');gp=heads/'global_seed42/best.pt';hp=heads/'grid_seed42/best.pt';m['dependencies'].update({str(p):digest(p) for p in [gp,hp]});gc=torch.load(gp,map_location='cpu',weights_only=False);hc=torch.load(hp,map_location='cpu',weights_only=False);preds={}
 for arm,bag in selected.items():
  torch.manual_seed(42);model=restore(hc,gc);globalsha=state_digest(model.global_model.state_dict());opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.01);d=out/f'{arm}_seed42';d.mkdir();best=-1;history=[]
  def forward(ids):return model(t['global_features'][ids],bag[0][ids],bag[1][ids],bag[2][ids])
  for epoch in range(21):
   losses=[]
   if epoch:
    model.train();order=torch.tensor(tr)[torch.randperm(len(tr),generator=torch.Generator().manual_seed(42+epoch))]
    for ids in order.split(32):
     opt.zero_grad();loss=attention_loss(forward(ids),y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
   model.eval()
   with torch.inference_mode():prob=torch.cat([forward(ids)['logits'].sigmoid() for ids in torch.tensor(va).split(32)]).numpy()
   score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,fitloss=float(np.mean(losses)) if losses else None,validation=score));assert state_digest(model.global_model.state_dict())==globalsha
   if score['auroc']>best:best=score['auroc'];bp=prob.copy();be=epoch;torch.save(dict(state=copy.deepcopy(head_state(model)),epoch=epoch,arm=arm,seed=42,frozen_global_sha256=globalsha),d/'best.pt')
  pred=f.iloc[va][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);preds[(arm,'singleview')]=pred;pmap=dict(zip(pred.image_id,bp));mv=pred.copy();mv['probability']=[.5*(pmap[f.iloc[i].image_id]+pmap[f.iloc[partners[i][0]].image_id]) if partners[i] else pmap[f.iloc[i].image_id] for i in va];mv.to_csv(d/'multiview_validation_predictions.csv',index=False);preds[(arm,'multiview')]=mv;(d/'history.json').write_text(json.dumps(history))
  for mode,p in [('singleview',pred),('multiview',mv)]:r=dict(arm=arm,mode=mode,epoch=be,metrics=metrics(p.label,p.probability));m['results'].append(r);save();print(json.dumps(r),flush=True)
 report=dict(results=m['results'],selection_audit=audits,paired={})
 for mode in ['singleview','multiview']:
  for arm in ['selected_gaze_baseline','selected_random']:report['paired'][arm+'_'+mode+'_minus_grid']=paired_ci(preds[(arm,mode)],preds[('grid',mode)],1000)
  report['paired']['gaze_'+mode+'_minus_matchedrandom']=paired_ci(preds[('selected_gaze_baseline',mode)],preds[('selected_random',mode)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(__file__)==m['source_sha256'] and digest(cp)==m['cache_sha256'] and all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';m['stage']='report_complete';save()
if __name__=='__main__':main()
