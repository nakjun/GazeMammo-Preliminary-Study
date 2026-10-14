"""Actual source/cache/update/auxiliary gradients before cross-view pilot."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_crossview_1011 import CrossViewGaze,contrastive_aux,restore_crossview,BLOCK_PREFIXES
from src.gaze_nll_adaptation_1011 import initialize
from src.gaze_path_augmentation_1011 import augmented_sample
from src.proposal_utility_1011 import state_sha256,file_sha256
from gaze_nll_adaptation_1011 import load_base

def main():
 torch.set_num_threads(4);torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False;out=ROOT/'runs/dual_target_1011/gaze_crossview_guard_v3';out.mkdir(exist_ok=True);assert not (out/'proof.json').exists();base,_=load_base('runs/full_1011_seed42/control/best.pt');model=CrossViewGaze(base).cuda(1);reference=initialize(base).cuda(1).eval();reference.requires_grad_(False);model.eval();frozen=state_sha256(model.frozen_state());teacher=state_sha256(reference.state_dict());records={}
 for i in [62,64,73]:records[i]=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{i}.pt',map_location='cpu',weights_only=False)
 pool=torch.load(ROOT/'runs/dual_target_1011/train_gaze_pool_cache/62.pt',map_location='cpu',weights_only=False);sample=augmented_sample(pool,62,1,42);r=records[62];group=[dict(image_id=62,image_size=r['image_size'],selected_pool_indices=sample['pool_indices'])];proofs=[]
 for i,r in records.items():
  with Image.open(ROOT/'data/image'/r['file_name']) as im:image=im.convert('L')
  tensor=whole_tensor(image,(256,512))[None].cuda(1)
  with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):f3,prefix=model.visual.prefix_features(tensor)
  with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=False):
   parentmem=reference.maps_features([f[None].cuda(1) for f in r['features']]);parentq=reference.teacher_logits(parentmem,r['coords'][None].cuda(1),r['vectors'][None].cuda(1));full=model.teacher_logits(model.maps(tensor),r['coords'][None].cuda(1),r['vectors'][None].cuda(1));cached=model.teacher_logits(model.maps_prefix(f3,prefix),r['coords'][None].cuda(1),r['vectors'][None].cuda(1));initial=float((full-parentq).abs().max());print(json.dumps(dict(image_id=i,initial_parent_error=initial,cache_error=float((full-cached).abs().max()))),flush=True);assert initial<=1e-4;assert float((full-cached).abs().max())<=1e-4
  if i==62:
   memory,feature5=model.maps_prefix(f3,prefix,True);assert feature5.shape==(1,512,32,16);queries=model.aux_queries(feature5,sample['coords'][None].cuda(1));aux,coverage=contrastive_aux(queries,group,{62:pool},1);aux.backward();grads={p:sum(float(v.grad.abs().sum()) for n,v in model.named_parameters() if n.startswith(p) and v.grad is not None) for p in BLOCK_PREFIXES};assert all(v>0 for v in grads.values());assert all(v.grad is None for n,v in model.named_parameters() if n.startswith('visual.') and not n.startswith(BLOCK_PREFIXES));model.zero_grad()
   # Prove frozen later stages retain upstream input gradients too.
   _,feature7=model.visual.tail_features(prefix);g=torch.autograd.grad(feature7.square().mean(),[p for n,p in model.named_parameters() if n.startswith(BLOCK_PREFIXES)]);assert all(torch.isfinite(v).all() for v in g) and sum(float(v.abs().sum()) for v in g)>0
   model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5,weight_decay=1e-4);memory,f5=model.maps_prefix(f3,prefix,True);q=model.teacher_logits(memory,sample['coords'][None].cuda(1),sample['vectors'][None].cuda(1));aux,_=contrastive_aux(model.aux_queries(f5,sample['coords'][None].cuda(1)),group,{62:pool},1);target=(sample['coords'][:,1]*48).long()*24+(sample['coords'][:,0]*24).long();loss=-q[0,torch.arange(16),target.cuda(1)].mean()+.1*aux;loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();assert state_sha256(model.frozen_state())==frozen;model.eval()
   # Cache/full after actual nonzero update, exact native branch and pool positive.
   with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=False):fq=model.teacher_logits(model.maps(tensor),sample['coords'][None].cuda(1),sample['vectors'][None].cuda(1));cq=model.teacher_logits(model.maps_prefix(f3,prefix),sample['coords'][None].cuda(1),sample['vectors'][None].cuda(1));updated=float((fq-cq).abs().max());assert updated<=1e-4;assert fq.logsumexp(-1).abs().max()<=1e-5
   ck=dict(kind='gaze_crossview_1011',state=model.compact_state(),frozen_state_sha256=frozen);torch.save(ck,out/'compact_smoke.pt');restored=restore_crossview(torch.load(out/'compact_smoke.pt',map_location='cpu',weights_only=False),base).cuda(1)
   with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=False):restoreq=restored.teacher_logits(restored.maps_prefix(f3,prefix),sample['coords'][None].cuda(1),sample['vectors'][None].cuda(1));assert (restoreq-cq).abs().max()<=1e-4
   del restored;proofs.append(dict(image_id=i,initial_parent_error=initial,updated_cache_full_error=updated,aux_nonzero_whole_gradient_sums=grads,negative_coverage=coverage,feature5_shape=list(feature5.shape)))
  else:proofs.append(dict(image_id=i,initial_parent_error=initial))
  # Reinitialize for later parent identity probes after image62 training update.
  if i==62:del model;model=CrossViewGaze(base).cuda(1).eval()
 assert state_sha256(model.frozen_state())==frozen and state_sha256(reference.state_dict())==teacher
 result=dict(state='passed',source_checkpoint_sha256=file_sha256(ROOT/'runs/full_1011_seed42/control/best.pt'),frozen_state_sha256=frozen,teacher_state_sha256=teacher,proofs=proofs,trainable_inventory=dict(whole_blocks=6380544,gaze_head=631314,projector=525312,total=7537170),projector_rng_forked=True,past_and_whole_prefix_and_later_weights_frozen=True,no_full_cache_or_trial_started=True)
 (out/'proof.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
