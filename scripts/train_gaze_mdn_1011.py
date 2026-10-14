"""Single standalone canonical16 MDN experiment, fixed validation targets."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from src.gaze_mdn_1011 import GazeMDN
from gaze_nll_adaptation_1011 import cache_records
from train_attention_1011 import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--output',default='runs/dual_target_1011/gaze_mdn_canonical_seed42');p.add_argument('--device',default='cpu');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42);out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 out.mkdir(parents=True);records,prov=cache_records(a)
 if len(records)!=505 or prov.get('smoke'):raise ValueError('Requires full505')
 for r in records:r['embedding']=r['features'][-1].mean((-2,-1));r['image_id']=int(r['image_id'])
 train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='val'];model=GazeMDN().to(a.device);optimizer=torch.optim.Adam(model.parameters(),lr=.001);config=dict(**vars(a),seed=42,epochs=15,batch_size=32,lr=.001,recipe='standalone canonical16 hardcellNLL; no augmentation/mirror/ensemble teacher; eight diagonalGaussian components; exact float64 CDF cell integrals; image-domain normalization; scale1pixel..image; past4 coordinates/mask/step only',cache_sha256=digest(Path(a.cache)/'provenance.json'),base_sha256=digest(a.base));(out/'config.json').write_text(json.dumps(config,indent=2));history=[];best=float('inf');started=time.monotonic()
 def batch(group):
  image=torch.stack([r['embedding'] for r in group]).to(a.device);coords=torch.stack([r['coords'] for r in group]).to(a.device);mask=torch.stack([r['mask'] for r in group]).bool().to(a.device);size=torch.tensor([r['image_size'] for r in group],device=a.device);target=(coords[:,:,1]*48).long().clamp(0,47)*24+(coords[:,:,0]*24).long().clamp(0,23);return image,coords,mask,size,target
 def epoch(rows,training):
  total=0.;model.train(training)
  for start in range(0,len(rows),32):
   group=rows[start:start+32];image,coords,mask,size,target=batch(group)
   with torch.set_grad_enabled(training):
    raw=model.teacher_logits(image,coords,size);losses=(-raw.gather(-1,target[...,None]).squeeze(-1)*mask).sum(1)/mask.sum(1);loss=losses.mean()
    if training:optimizer.zero_grad();loss.backward();optimizer.step()
   total+=float(losses.detach().sum())
  return total/len(rows)
 # Displacement summaries are diagnostics only; no selected hyperparameters.
 diagnostics={}
 for name,rows in [('train',train),('val',val)]:
  distances=[]
  for r in rows:
   valid=r['mask'][1:] & r['mask'][:-1];distances.append(((r['coords'][1:]-r['coords'][:-1])*torch.tensor(r['image_size'])).norm(dim=-1)[valid])
  diagnostics[name]=torch.quantile(torch.cat(distances),torch.tensor([.1,.25,.5,.75,.9])).tolist()
 (out/'displacement_quantiles.json').write_text(json.dumps(dict(probabilities=[.1,.25,.5,.75,.9],pixels=diagnostics),indent=2))
 for ep in range(16):
  train_nll=epoch([train[i] for i in torch.randperm(len(train)).tolist()],True) if ep else epoch(train,False);val_nll=epoch(val,False);entry=dict(epoch=ep,train_nll=train_nll,validation_nll=val_nll);history.append(entry);(out/'history.json').write_text(json.dumps(history,indent=2))
  ck=dict(kind='gaze_mdn_1011',state=model.state_dict(),config=config,epoch=ep,training_patient_keys=sorted({r['patient_key'] for r in train}),selection_patient_keys=sorted({r['patient_key'] for r in val}))
  if val_nll<best:best=val_nll;torch.save(ck,out/'best.pt');(out/'best_validation.json').write_text(json.dumps(entry,indent=2))
  torch.save(ck,out/'last.pt');print(json.dumps(entry),flush=True)
 (out/'summary.json').write_text(json.dumps(dict(best_nll=best,epochs=15,seconds=time.monotonic()-started),indent=2))
if __name__=='__main__':main()
