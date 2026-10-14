"""Terminal-only505 medicalgaze probabilities, actual93 whole/past parity, fixed mixtures."""
import argparse,json,sys,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np,torch
from PIL import Image
from src.gaze_mammo_clip_1011 import restore_mammo_gaze
from src.data import patch_tensor
from src.proposal_utility_1011 import file_sha256,state_sha256
from src.dual_output_1011 import config_sha256
from gaze_nll_adaptation_1011 import routing

def main():
 p=argparse.ArgumentParser();p.add_argument('--trial',default='runs/dual_target_1011/mammo_clip_public/gaze_seed42');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;trial=Path(a.trial);summary=json.loads((trial/'summary.json').read_text());history=json.loads((trial/'history.json').read_text())
 if summary['state']!='complete' or summary['epochs']!=15 or len(history)!=16 or history[-1]['epoch']!=15:raise ValueError('Medicaltrial notterminal15')
 path=trial/'best.pt';sha=file_sha256(path);ck=torch.load(path,map_location='cpu',weights_only=False);model=restore_mammo_gaze(ck,'cuda:0');config=ck['config'];native=Path(config['source_cache']);header=json.loads((native/'provenance.json').read_text());whole_sha=state_sha256(model.visual.whole.state_dict());past_sha=state_sha256(model.visual.past.state_dict())
 if file_sha256(native/'provenance.json')!=config['cache_sha256']:raise ValueError('Medicalnative dependency changed')
 baseline_config_path=ROOT/'runs/dual_target_1011/crossview_unified_runtime/config.json';baseline_config=json.loads(baseline_config_path.read_text())
 if baseline_config['sha256']!=config_sha256(baseline_config['config']):raise ValueError('Current crossview logicalSHA mismatch')
 rawpaths=[ROOT/'runs/dual_target_1011/gaze_mdn_product_seed42/corrected_raw_cache.pt',ROOT/'runs/dual_target_1011/dino/gaze_augmented_teacherkl_seed42/best_raw_cache.pt',ROOT/'runs/dual_target_1011/gaze_crossview_seed42/fixed_export/best_raw_cache.pt'];raw_hashes={str(f):file_sha256(f) for f in rawpaths};parents=[{int(r['image_id']):r for r in torch.load(f,weights_only=False)['records']} for f in rawpaths];allowed={int(r['image_id']):r for r in routing()};ids=[int(r['image_id']) for r in header['records']]
 if len(set(ids))!=505 or set(ids)!=set(allowed) or any(set(p)!=set(allowed) for p in parents):raise ValueError('Exact505 baseline/native mismatch')
 control_cache=ROOT/'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache';records=[];proof=[];rows=[];started=time.monotonic()
 with torch.no_grad():
  for start in range(0,505,4):
   group=[]
   for meta in header['records'][start:start+4]:
    file=native/f"{meta['image_id']}.pt"
    if file_sha256(file)!=meta['sha256']:raise ValueError('Medicalfile changed')
    r=torch.load(file,weights_only=False);identity=int(r['image_id']);route=allowed[identity];refs=[p[identity] for p in parents];control=torch.load(control_cache/f'{identity}.pt',weights_only=False);r['vectors']=control['vectors']
    if r['image_id']!=int(meta['image_id']) or r['patient_key']!=meta['patient_key'] or r['split']!=meta['split'] or r['patient_key']!=route['patient_key'] or r['split']!=route['split'] or r['file_name']!=route['file_name'] or r['image_sha256']!=control['image_sha256'] or not torch.equal(r['coords'],control['coords']) or not torch.equal(r['mask'],control['mask']):raise ValueError('Medicalmetadata source mismatch')
    target=(r['coords'][:,1]*48).long().clamp(0,47)*24+(r['coords'][:,0]*24).long().clamp(0,23)
    for ref in refs:
     if ref['patient_key']!=r['patient_key'] or ref['split']!=r['split'] or not torch.equal(ref['coords'],r['coords']) or not torch.equal(ref['loss_mask'],r['mask']) or not torch.equal(ref['target'],target):raise ValueError('Currentbaseline canonical alignment changed')
    r['target']=target;group.append(r)
   f=torch.stack([r['feature'] for r in group]).cuda();coords=torch.stack([r['coords'] for r in group]).cuda();vectors=torch.stack([r['vectors'] for r in group]).cuda();raw=model.teacher_logits(model.maps_features([f]*3),coords,vectors).cpu()
   if not torch.isfinite(raw).all() or raw.logsumexp(-1).abs().max()>1e-5:raise ValueError('Medicalq normalization failure')
   for i,r in enumerate(group):
    identity=r['image_id'];q=raw[i];mask=r['mask'].bool();target=r['target'];base=.75*(.75*parents[0][identity]['raw_logp'].exp()+.25*parents[1][identity]['raw_logp'].exp())+.25*parents[2][identity]['raw_logp'].exp();baselog=base.log()-base.log().logsumexp(-1,keepdim=True)
    if r['split']=='val':
     imagepath=ROOT/'data/image'/r['file_name']
     if file_sha256(imagepath)!=r['image_sha256']:raise ValueError('Actual image changed')
     with Image.open(imagepath) as source:
      image=source.convert('L');actualmemory=model.maps(image);crops=torch.stack([patch_tensor(image,c.numpy()*image.size) for c in r['coords'][:-1]]);actual_vectors=torch.zeros_like(vectors[i:i+1])
      for begin in range(0,15,8):actual_vectors[:,begin+1:begin+9]=model.visual.patch_vector(crops[begin:begin+8].cuda())[None]
     actual=model.teacher_logits(actualmemory,coords[i:i+1],actual_vectors)[0].cpu();error=float((actual-q).abs().max());proof.append(dict(image_id=identity,max_logp_error=error,fresh_control_past_vector_error=float((actual_vectors-vectors[i:i+1]).abs().max()),actual_nll=float(-actual.gather(1,target[:,None]).squeeze(1)[mask].mean())))
     if error>1e-4:raise ValueError(f'Actualmedical whole/past parity failed {identity}/{error}')
    records.append(dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],raw_logp=q,coords=r['coords'],valid=mask,loss_mask=mask,target=target));row=dict(image_id=identity,patient_key=r['patient_key'],split=r['split'])
    for name,pred in [('mammo',q),('baseline',baselog)]+[(f'mix_{w}',torch.logaddexp(baselog+math.log(1-w),q+math.log(w))) for w in [.25,.5,.75]]:row[name]=float(-pred.gather(1,target[:,None]).squeeze(1)[mask].mean())
    rows.append(row)
   if len(records)%100==0:print(json.dumps(dict(completed=len(records),total=505)),flush=True)
 if file_sha256(path)!=sha or state_sha256(model.visual.whole.state_dict())!=whole_sha or state_sha256(model.visual.past.state_dict())!=past_sha or any(file_sha256(f)!=expected for f,expected in raw_hashes.items()):raise ValueError('Medical export dependencies changed')
 output=trial/'best_raw_cache.pt'
 if output.exists():raise FileExistsError(output)
 torch.save(dict(kind='gaze_raw_legacy16_cache_1011',records=records,grid=[24,48],base_checkpoint=str(path.resolve()),base_checkpoint_sha256=sha,config=config,training_patient_keys=ck['training_patient_keys'],selection_patient_keys=ck['selection_patient_keys'],actual_val_parity=dict(images=93,max_logp_error=max(r['max_logp_error'] for r in proof),tolerance=1e-4,records=proof)),output)
 result={split:{name:float(np.mean([r[name] for r in rows if r['split']==split])) for name in ['mammo','baseline','mix_0.25','mix_0.5','mix_0.75']} for split in ['train','val']};result.update(actual93_nll=float(np.mean([r['actual_nll'] for r in proof])),actual93_max_error=max(r['max_logp_error'] for r in proof),best_epoch=ck['epoch'],best_sha256=sha,baseline_config_sha256=baseline_config['sha256'],baseline_recipe='.75(.75MDN+.25KLDINO)+.25crossview',baseline_raw_sha256=raw_hashes,seconds=time.monotonic()-started,selection='fixedthree weights only; exploratoryval93');(trial/'export_summary.json').write_text(json.dumps(result,indent=2));(trial/'actual93_proof.json').write_text(json.dumps(proof,indent=2));(trial/'export_per_image.json').write_text(json.dumps(rows,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
