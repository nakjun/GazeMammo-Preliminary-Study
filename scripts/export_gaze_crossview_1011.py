"""Terminal cross-view best505 export; fresh93 parity; only3 fixed mixtures."""
import sys,json,time,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch,numpy as np,pandas as pd
from PIL import Image
from src.data import whole_tensor,patch_tensor
from src.gaze_crossview_1011 import restore_crossview
from src.proposal_utility_1011 import file_sha256,state_sha256
from gaze_nll_adaptation_1011 import cache_records,load_base

@torch.no_grad()
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;trial=ROOT/'runs/dual_target_1011/gaze_crossview_seed42';summary=json.loads((trial/'summary.json').read_text());assert summary['state']=='complete' and summary['epochs']==15;ck=torch.load(trial/'best.pt',map_location='cpu',weights_only=False);cfg=ck['config'];base,_=load_base(cfg['source_checkpoint']);model=restore_crossview(ck,base).cuda(1).eval();model.requires_grad_(False);fullsha=state_sha256(model.state_dict());checkpointsha=file_sha256(trial/'best.pt')
 for path,h in cfg['source_code_sha256'].items():assert file_sha256(ROOT/path)==h
 class Args:cache='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache';base=cfg['source_checkpoint']
 original,_=cache_records(Args());native=Path(cfg['prefix_cache']);header=json.loads((native/'provenance.json').read_text());assert file_sha256(native/'provenance.json')==cfg['prefix_cache_provenance_sha256'];meta={int(r['image_id']):r for r in header['records']};references=[]
 current_config=json.loads((ROOT/'runs/dual_target_1011/dino_mdn_unified_runtime/config.json').read_text())
 for d in current_config['config']['reference_caches']:
  assert file_sha256(d['raw_cache'])==d['raw_cache_sha256'];references.append({int(r['image_id']):r for r in torch.load(d['raw_cache'],map_location='cpu',weights_only=False)['records']})
 assert len(original)==505 and set(meta)=={int(r['image_id']) for r in original};out=trial/'fixed_export';out.mkdir(exist_ok=False);records=[];rows=[];parity=[];start=time.monotonic()
 for begin in range(0,505,8):
  group=original[begin:begin+8];prefixes=[]
  for r in group:
   identity=int(r['image_id']);file=native/f'{identity}.pt';assert file_sha256(file)==meta[identity]['sha256'];item=torch.load(file,map_location='cpu',weights_only=False);assert item['image_id']==identity and item['patient_key']==r['patient_key'] and item['split']==r['split'];prefixes.append(item['prefix5_23'])
  f3=torch.stack([r['features'][0] for r in group]).cuda(1);prefix=torch.stack(prefixes).cuda(1);coords=torch.stack([r['coords'] for r in group]).cuda(1);vectors=torch.stack([r['vectors'] for r in group]).cuda(1);memory,f5=model.maps_prefix(f3,prefix,True);cached=model.teacher_logits(memory,coords,vectors)
  for j,r in enumerate(group):
   identity=int(r['image_id']);mask=r['mask'].bool();target=(r['coords'][:,1]*48).long().clamp(0,47)*24+(r['coords'][:,0]*24).long().clamp(0,23);q=cached[j].cpu();assert torch.isfinite(q).all() and q.logsumexp(-1).abs().max()<=1e-5
   for ref in references:
    s=ref[identity];assert s['patient_key']==r['patient_key'] and s['split']==r['split'] and torch.equal(s['coords'],r['coords']) and torch.equal(s['loss_mask'],mask) and torch.equal(s['target'],target)
   parent=torch.logaddexp(references[0][identity]['raw_logp']+math.log(.75),references[1][identity]['raw_logp']+math.log(.25));parent=parent-parent.logsumexp(-1,keepdim=True);row=dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],candidate_nll=float(-q[mask,target[mask]].mean()),current_nll=float(-parent[mask,target[mask]].mean()))
   for weight in [.25,.5,.75]:mix=torch.logaddexp(parent+math.log(1-weight),q+math.log(weight));mix=mix-mix.logsumexp(-1,keepdim=True);row[f'weight_{weight}']=float(-mix[mask,target[mask]].mean())
   if r['split']=='val':
    path=ROOT/'data/image'/r['file_name'];assert file_sha256(path)==r['image_sha256']
    with Image.open(path) as source:image=source.convert('L')
    whole=whole_tensor(image,(256,512))[None].cuda(1);fresh_f3,fresh_prefix=model.visual.prefix_features(whole);actualmemory,actualf5=model.maps_prefix(fresh_f3,fresh_prefix,True);freshvectors=torch.zeros_like(vectors[j:j+1]);crops=torch.stack([patch_tensor(image,c.numpy()*image.size,224) for c in r['coords'][:-1]])
    for k in range(0,15,8):freshvectors[:,k+1:k+1+len(crops[k:k+8])]=model.visual.patch_vector(crops[k:k+8].cuda(1))[None]
    actual=model.teacher_logits(actualmemory,coords[j:j+1],freshvectors)[0].cpu();error=float((actual-q).abs().max());diag=dict(image_id=identity,max_logp_error=error,feature3_error=float((fresh_f3[0]-f3[j]).abs().max()),prefix5_23_error=float((fresh_prefix[0]-prefix[j]).abs().max()),updated_feature5_error=float((actualf5[0]-f5[j]).abs().max()),past_vector_error=float((freshvectors-vectors[j:j+1]).abs().max()))
    if error>1e-4:(out/'failed_parity_diagnostic.json').write_text(json.dumps(diag,indent=2));raise ValueError(f'Updatedactual/cache qerror {error}>1e-4')
    row['actual_candidate_nll']=float(-actual[mask,target[mask]].mean());parity.append(diag)
   rows.append(row);records.append(dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],coords=r['coords'],loss_mask=mask,valid=mask,target=target,raw_logp=q))
  (out/'state.json').write_text(json.dumps(dict(state='running',completed=len(records),actual_val_completed=len(parity))));print(json.dumps(dict(completed=len(records),actual_val_completed=len(parity))),flush=True)
 assert len(records)==505 and len(parity)==93 and state_sha256(model.state_dict())==fullsha and file_sha256(trial/'best.pt')==checkpointsha
 result=dict(state='complete',checkpoint_sha256=checkpointsha,best_epoch=ck['epoch'],splits={split:{key:float(np.mean([r[key] for r in rows if r['split']==split])) for key in ['candidate_nll','current_nll','weight_0.25','weight_0.5','weight_0.75']} for split in ['train','val']},actual93_nll=float(np.mean([r['actual_candidate_nll'] for r in rows if r['split']=='val'])),actual93_max_logp_error=max(r['max_logp_error'] for r in parity),tolerance=1e-4,prespecified_weights=[.25,.5,.75],current_parent_config_sha256=current_config['sha256'],frozen_and_selected_state_unchanged=True,seconds=time.monotonic()-start,ROI_integration=False)
 result['any_fixed_weight_gain']=min(result['splits']['val'][f'weight_{w}'] for w in [.25,.5,.75])<result['splits']['val']['current_nll'];torch.save(dict(kind='gaze_raw_legacy16_cache_1011',records=records,grid=[24,48],base_checkpoint=str(trial/'best.pt'),base_checkpoint_sha256=checkpointsha,config=cfg,actual_val_parity=dict(images=93,records=parity,max_logp_error=result['actual93_max_logp_error'],tolerance=1e-4),training_patient_keys=cfg['training_patient_keys'],selection_patient_keys=cfg['selection_patient_keys']),out/'best_raw_cache.pt');pd.DataFrame(rows).to_csv(out/'per_image.csv',index=False);(out/'summary.json').write_text(json.dumps(result,indent=2));(out/'actual93_parity.json').write_text(json.dumps(dict(records=parity),indent=2));(out/'state.json').write_text(json.dumps(dict(state='complete',completed=505,actual_val_completed=93)));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
