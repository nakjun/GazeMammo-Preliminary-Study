"""Actual source1520x912/crop224 Mammo-CLIP feature proof only; no clinical head."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from src.mammo_clip_visual_1011 import MammoClipVisual,preprocess,SOURCE
from src.proposal_utility_1011 import state_sha256,file_sha256

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;model=MammoClipVisual(ROOT/'runs/dual_target_1011/mammo_clip_public/assets/b5-model-best-epoch-7.tar').cuda().eval();before=state_sha256(model.state_dict());proof=[]
with torch.no_grad():
 for identity in [64,73]:
  ref=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{identity}.pt',weights_only=False);path=ROOT/'data/image'/ref['file_name']
  if file_sha256(path)!=ref['image_sha256']:raise ValueError('Image SHA mismatch')
  with Image.open(path) as source:
   image=source.convert('L');tensor,meta=preprocess(image);tensor=tensor[None].cuda();torch.cuda.synchronize();start=time.monotonic();native=model(tensor);torch.cuda.synchronize();seconds=time.monotonic()-start;pooled,author_native=model.encoder({'image':tensor,'breast_clip_train_mode':True});error=float((native-author_native).abs().max());poolerror=float((native.mean((-2,-1))-pooled).abs().max());center=np.round(ref['coords'][0].numpy()*image.size).astype(int);crop=image.crop((int(center[0])-112,int(center[1])-112,int(center[0])+112,int(center[1])+112));patch,patch_meta=preprocess(crop,size=(224,224));patch=patch[None].cuda();vector=model.patch_vector(patch);author_vector,author_patch=model.encoder({'image':patch,'breast_clip_train_mode':True});patcherror=float((vector-author_vector).abs().max());grid=torch.nn.functional.interpolate(native,size=(48,24),mode='bilinear',align_corners=False)
  if max(error,poolerror,patcherror)>1e-5 or not torch.isfinite(native).all() or not torch.isfinite(vector).all():raise ValueError('Source/wrapper feature proof failed')
  proof.append(dict(image_id=identity,source_full_vs_wrapper_error=error,source_pool_vs_mean_error=poolerror,source224vector_vs_wrapper_error=patcherror,native_shape=list(native.shape),fixed_grid_spatial_shape=list(grid.shape),patch_shape=list(author_patch.shape),vector_shape=list(vector.shape),whole_seconds=seconds,whole_preprocess=meta,crop_preprocess=patch_meta))
if before!=state_sha256(model.state_dict()) or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Frozen genuine source changed')
report=dict(checkpoint_sha256=model.checkpoint_sha256,strict_full_image_encoder_keys=model.loaded_keys,source_encoder_parameters=sum(p.numel() for p in model.parameters()),epoch=model.source_epoch,source_commit='0d68563352c06784400080e140ae3d447e779283',pretrain_variant=model.config['data_train'],source_mean=model.config['base']['mean'],source_std=model.config['base']['std'],no_new_clinical_parameters=True,no_clinical_training_or_cache=True,source_state_unchanged=True,proof=proof,source_file_sha256={str(f):file_sha256(f) for f in SOURCE.glob('*.py')},wrapper_sha256=file_sha256(ROOT/'src/mammo_clip_visual_1011.py'));out=ROOT/'runs/dual_target_1011/mammo_clip_public/actual64_73_feature_proof.json';out.write_text(json.dumps(report,indent=2));print(json.dumps(report))
