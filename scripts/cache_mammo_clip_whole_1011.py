"""Frozen UPMC B5 native whole maps only; exact train412+canonicalval93."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.mammo_clip_visual_1011 import MammoClipVisual,preprocess,SOURCE
from src.proposal_utility_1011 import state_sha256,file_sha256
from gaze_nll_adaptation_1011 import cache_records

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--medical-checkpoint',default='runs/dual_target_1011/mammo_clip_public/assets/b5-model-best-epoch-7.tar');p.add_argument('--output',default='runs/dual_target_1011/mammo_clip_public/whole_cache');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;model=MammoClipVisual(a.medical_checkpoint).cuda().eval();records,prov=cache_records(a);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);metadata=[];proof=[];before=state_sha256(model.state_dict());started=time.monotonic()
 if prov.get('smoke') or len(records)!=505:raise ValueError('Exact505 source required')
 with torch.no_grad():
  for index,r in enumerate(records):
   identity=int(r['image_id']);path=ROOT/'data/image'/r['file_name']
   if file_sha256(path)!=r['image_sha256']:raise ValueError('Medical image changed')
   with Image.open(path) as source:tensor,preprocess_meta=preprocess(source.convert('L'))
   with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):feature=model(tensor[None].cuda())[0].cpu().contiguous()
   if feature.shape!=(2048,48,29) or feature.dtype!=torch.float32 or not torch.isfinite(feature).all():raise ValueError('Native medical map invalid')
   entry=dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],file_name=r['file_name'],image_sha256=r['image_sha256'],image_size=r['image_size'],coords=r['coords'],mask=r['mask'],feature=feature,preprocess_metadata=preprocess_meta);torch.save(entry,out/f'{identity}.pt');metadata.append(dict(image_id=identity,patient_key=r['patient_key'],split=r['split'],sha256=file_sha256(out/f'{identity}.pt')))
   if identity in [64,73]:proof.append(dict(image_id=identity,shape=list(feature.shape),source_min=preprocess_meta['source_min'],source_range=preprocess_meta['source_range']))
   if (index+1)%50==0:print(json.dumps(dict(completed=index+1,total=505)),flush=True)
 if before!=state_sha256(model.state_dict()):raise ValueError('Medical encoder changed')
 if len({r['image_id'] for r in metadata})!=505 or sum(r['split']=='train' for r in metadata)!=412 or sum(r['split']=='val' for r in metadata)!=93:raise ValueError('Medical cohort mismatch')
 header=dict(kind='mammo_clip_whole_cache_1011',records=metadata,medical_checkpoint=str(Path(a.medical_checkpoint).resolve()),medical_checkpoint_sha256=model.checkpoint_sha256,strict_image_encoder_keys=model.loaded_keys,whole_state_sha256=before,source_cache_provenance_sha256=file_sha256(Path(a.cache)/'provenance.json'),source_past_visual_sha256=prov['visual_state_sha256'],source_cfg=model.config,source_code_sha256={str(f):file_sha256(f) for f in SOURCE.glob('*.py')},input_size=[912,1520],native_shape=[2048,48,29],grid=[24,48],crop_size=224,preprocessing='sourcecv2resize/minmax/scalarstats; existingPNGbordertrim; noextraROIcrop',proof=proof,seconds=time.monotonic()-started);(out/'provenance.json').write_text(json.dumps(header,indent=2));(out/'status.json').write_text(json.dumps(dict(state='complete',images=505),indent=2));print(json.dumps(dict(images=505,seconds=header['seconds'],proof=proof)))
if __name__=='__main__':main()
