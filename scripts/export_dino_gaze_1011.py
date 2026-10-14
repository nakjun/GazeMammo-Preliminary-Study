"""Best frozen-DINO actual93 proof and canonical505 normalized export/fixed mixtures."""
import json,sys,math,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.data import whole_tensor,patch_tensor
from src.gaze_dino_1011 import restore_dino
from train_attention_1011 import digest

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
parser=argparse.ArgumentParser();parser.add_argument('--trial',default='runs/dual_target_1011/dino/gaze_seed42');args=parser.parse_args();trial=Path(args.trial);path=trial/'best.pt';sha=digest(path);ck=torch.load(path,weights_only=False);model=restore_dino(ck,'cuda:0');cache=Path(ck['config']['source_cache']);header=json.loads((cache/'provenance.json').read_text())
if digest(cache/'provenance.json')!=ck['config']['cache_sha256']:raise ValueError('DINO cache changed')
basepath=ROOT/'runs/dual_target_1011/gaze_mdn_product_seed42/corrected_raw_cache.pt';baseline_sha=digest(basepath);baseline=torch.load(basepath,weights_only=False);reference={int(r['image_id']):r for r in baseline['records']};ids=[int(r['image_id']) for r in header['records']]
if len(set(ids))!=505 or set(ids)!=set(reference):raise ValueError('Exact505 IDset mismatch')
records=[];proof=[];rows=[];started=time.monotonic()
with torch.no_grad():
 for index,meta in enumerate(header['records']):
  file=cache/f"{meta['image_id']}.pt"
  if digest(file)!=meta['sha256']:raise ValueError('Native image file changed')
  r=torch.load(file,weights_only=False);ref=reference[int(r['image_id'])];coords=r['coords'][None].cuda();pv=r['vectors'][None].cuda();f=r['feature'][None].cuda();memory=model.maps_features([f]*3);raw=model.teacher_logits(memory,coords,pv)[0].cpu();target=(r['coords'][:,1]*48).long().clamp(0,47)*24+(r['coords'][:,0]*24).long().clamp(0,23);mask=r['mask'].bool()
  if r['patient_key']!=ref['patient_key'] or r['split']!=ref['split'] or not torch.equal(r['coords'],ref['coords']) or not torch.equal(mask,ref['loss_mask']) or not torch.equal(target,ref['target']):raise ValueError('Baseline canonical alignment failed')
  if r['split']=='val':
   imagepath=ROOT/'data/image'/r['file_name']
   if digest(imagepath)!=r['image_sha256']:raise ValueError('Actual image changed')
   with Image.open(imagepath) as source:
    image=source.convert('L');whole=whole_tensor(image,model.input_size)[None].cuda();actual_memory=model.maps(whole);crops=torch.stack([patch_tensor(image,c.numpy()*image.size) for c in r['coords'][:-1]]);actual_vectors=torch.zeros_like(pv)
    for start in range(0,15,8):actual_vectors[:,start+1:start+9]=model.visual.patch_vector(crops[start:start+8].cuda())[None]
    actual=model.teacher_logits(actual_memory,coords,actual_vectors)[0].cpu()
   error=float((actual-raw).abs().max());proof.append(dict(image_id=int(r['image_id']),max_logp_error=error,fresh_crop_vector_error=float((actual_vectors-pv).abs().max()),actual_nll=float(-actual.gather(1,target[:,None]).squeeze(1)[mask].mean())))
   if error>1e-4:raise ValueError(f'Actual93 DINO cache parity failed {r["image_id"]}: {error}')
  if not torch.isfinite(raw).all() or not torch.allclose(raw.exp().sum(-1),torch.ones(16),atol=1e-5):raise ValueError('Unnormalized DINO distribution')
  record=dict(image_id=int(r['image_id']),patient_key=r['patient_key'],split=r['split'],raw_logp=raw,coords=r['coords'],loss_mask=mask,valid=mask,target=target);records.append(record);row=dict(image_id=record['image_id'],patient_key=r['patient_key'],split=r['split'])
  for name,prob in [('dino',raw),('baseline',ref['raw_logp'])]+[(f'mix_{w}',torch.logaddexp(ref['raw_logp']+math.log(1-w),raw+math.log(w))) for w in [.25,.5,.75]]:row[name]=float(-prob.gather(1,target[:,None]).squeeze(1)[mask].mean())
  rows.append(row)
  if (index+1)%50==0:print(json.dumps(dict(completed=index+1,total=505)),flush=True)
if digest(path)!=sha or digest(basepath)!=baseline_sha:raise ValueError('Dependencies changed during export')
output=trial/'best_raw_cache.pt'
if output.exists():raise FileExistsError(output)
torch.save(dict(kind='gaze_raw_legacy16_cache_1011',records=records,grid=[24,48],base_checkpoint=str(path),base_checkpoint_sha256=sha,config=ck['config'],training_patient_keys=ck['training_patient_keys'],selection_patient_keys=ck['selection_patient_keys'],actual_val_parity=dict(images=len(proof),max_logp_error=max(r['max_logp_error'] for r in proof),tolerance=1e-4,records=proof),numerical_policy=dict(matmulTF32=False,cudnnTF32=True)),output)
summary={split:{k:sum(r[k] for r in rows if r['split']==split)/sum(r['split']==split for r in rows) for k in ['dino','baseline','mix_0.25','mix_0.5','mix_0.75']} for split in ['train','val']};summary.update(actual93_nll=sum(r['actual_nll'] for r in proof)/93,actual93_max_error=max(r['max_logp_error'] for r in proof),best_epoch=ck['epoch'],best_sha256=sha,baseline_raw_sha256=baseline_sha,seconds=time.monotonic()-started,selection='fixedthree mixtures only; exploratoryval93');(trial/'export_summary.json').write_text(json.dumps(summary,indent=2));(trial/'export_per_image.json').write_text(json.dumps(rows,indent=2));(trial/'actual93_proof.json').write_text(json.dumps(proof,indent=2));print(json.dumps(summary))
