"""Restore product correction, verify real-image pooled features, export canonical505."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np
from PIL import Image
from src.gaze_mdn_correction_1011 import GazeMDNCorrection
from gaze_nll_adaptation_1011 import cache_records,load_base
from train_attention_1011 import digest
from guided_trajectory_1011 import whole_tensor

def main():
 p=argparse.ArgumentParser();p.add_argument('--trial',default='runs/dual_target_1011/gaze_mdn_product_seed42');p.add_argument('--device',default='cpu');p.add_argument('--proof-only',action='store_true');a=p.parse_args();torch.set_num_threads(4);trial=Path(a.trial);path=trial/'best.pt';sha=digest(path);ck=torch.load(path,weights_only=False);config=ck['config'];args=argparse.Namespace(**config);records,prov=cache_records(args);model=GazeMDNCorrection().to(a.device);model.load_state_dict(ck['state']);model.eval();parents=[]
 for source,expected in config['parent_cache_sha256'].items():
  if digest(source)!=expected:raise ValueError('Parent cache changed')
  payload=torch.load(source,weights_only=False);parents.append({int(r['image_id']):r for r in payload['records']})
 if digest(ROOT/'runs/dual_target_1011/gaze_mdn_canonical_seed42/best.pt')!=config['mdn_initializer_sha256']:raise ValueError('MDN source changed')
 rows=[];export=[];proof=[];base,_=load_base(config['base']);base=base.to(a.device).eval()
 with torch.no_grad():
  for r in records:
   identity=int(r['image_id']);refs=[d[identity] for d in parents];coords=r['coords'][None].to(a.device);size=torch.tensor([r['image_size']],device=a.device);embedding=r['features'][-1].mean((-2,-1))[None].to(a.device);q0=.75*(.75*refs[0]['raw_logp'].exp()+.25*refs[1]['raw_logp'].exp())+.25*refs[2]['raw_logp'].exp();parent=(q0.log()-q0.log().logsumexp(-1,keepdim=True))[None].to(a.device)
   target=(r['coords'][:,1]*48).long().clamp(0,47)*24+(r['coords'][:,0]*24).long().clamp(0,23);mask=r['mask'].bool()
   for ref in refs:
    if ref['patient_key']!=r['patient_key'] or not torch.equal(ref['coords'],r['coords']) or not torch.equal(ref['loss_mask'],mask) or not torch.equal(ref['target'],target):raise ValueError('Canonical alignment changed')
   raw=model.teacher_logits(embedding,coords,size,parent)[0].cpu()
   if identity in [64,73]:
    imagepath=ROOT/'data/image'/r['file_name']
    if digest(imagepath)!=r['image_sha256']:raise ValueError('Image changed')
    with Image.open(imagepath) as image:fresh=base.visual(whole_tensor(image.convert('L'),base.input_size)[None].to(a.device))[-1].mean((-2,-1))
    actual=model.teacher_logits(fresh,coords,size,parent)[0].cpu();err=float((actual-raw).abs().max());proof.append(dict(image_id=identity,embedding_max_error=float((fresh-embedding).abs().max()),corrected_logp_error=err))
    if err>1e-4:raise ValueError('Actual pooled-feature correction parity failed')
   if not a.proof_only:
    baseline=float(-parent[0].cpu().gather(1,target[:,None]).squeeze(1)[mask].mean());score=float(-raw.gather(1,target[:,None]).squeeze(1)[mask].mean());rows.append(dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],baseline=baseline,nll=score));export.append(dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],raw_logp=raw,coords=r['coords'],loss_mask=mask,valid=mask,target=target))
 if digest(path)!=sha:raise ValueError('Checkpoint changed')
 (trial/f'actual_embedding_proof_{a.device.replace(":","_")}.json').write_text(json.dumps(dict(best_sha256=sha,proof=proof,tolerance=1e-4),indent=2))
 if a.proof_only:print(json.dumps(proof));return
 output=trial/'corrected_raw_cache.pt'
 if output.exists():raise FileExistsError(output)
 torch.save(dict(kind='gaze_raw_legacy16_cache_1011',records=export,grid=[24,48],base_checkpoint=str(path.resolve()),base_checkpoint_sha256=sha,config=config,visual_cache_provenance=prov,actual_embedding_proof=proof,training_patient_keys=ck['training_patient_keys'],selection_patient_keys=ck['selection_patient_keys']),output)
 val=[r for r in rows if r['split']=='val'];patients=sorted({r['patient_key'] for r in val});groups={p:[r for r in val if r['patient_key']==p] for p in patients};rng=np.random.default_rng(42);boot=[]
 for _ in range(5000):
  chosen=rng.choice(patients,len(patients),replace=True);sample=[r for p in chosen for r in groups[p]];boot.append(np.mean([r['nll']-r['baseline'] for r in sample]))
 report=dict(best_epoch=ck['epoch'],val_nll=np.mean([r['nll'] for r in val]),baseline_nll=np.mean([r['baseline'] for r in val]),difference=np.mean([r['nll']-r['baseline'] for r in val]),patient_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),alphas=model.alpha.tanh().cpu().tolist(),best_sha256=sha,proof=proof,exploratory_selection=True);(trial/'export_summary.json').write_text(json.dumps(report,indent=2));(trial/'export_per_image.json').write_text(json.dumps(rows,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
