"""Metadata-only recovery of completed505 native features after JSON config failure."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from omegaconf import OmegaConf
from src.mammo_clip_visual_1011 import MammoClipVisual,preprocess,SOURCE
from src.proposal_utility_1011 import state_sha256,file_sha256
from gaze_nll_adaptation_1011 import cache_records

def safe(value):
 if OmegaConf.is_config(value):return safe(OmegaConf.to_container(value,resolve=True))
 if isinstance(value,dict):return {str(k):safe(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [safe(v) for v in value]
 if isinstance(value,(str,int,float,bool)) or value is None:return value
 raise TypeError(f'Unsupported config metadata {type(value)}')

def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--medical-checkpoint',default='runs/dual_target_1011/mammo_clip_public/assets/b5-model-best-epoch-7.tar');p.add_argument('--output',default='runs/dual_target_1011/mammo_clip_public/whole_cache');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;out=Path(a.output);started=time.monotonic();records,prov=cache_records(a);reference={int(r['image_id']):r for r in records};files={int(f.stem):f for f in out.glob('*.pt')}
 if len(files)!=505 or set(files)!=set(reference):raise ValueError('Recovery requires exact existing505 featurefiles')
 failed=out/'provenance.json';original_script=ROOT/'scripts/cache_mammo_clip_whole_1011.py';failure_log=ROOT/'runs/dual_target_1011/mammo_clip_public/whole_cache.log';failure_info=dict(extractor_sha256=file_sha256(original_script),failure_log_sha256=file_sha256(failure_log),partial_header_sha256=file_sha256(failed) if failed.exists() else None)
 if failed.exists():failed.replace(out/'provenance_failed_partial.json')
 model=MammoClipVisual(a.medical_checkpoint).cuda().eval();whole_sha=state_sha256(model.state_dict());metadata=[];proof=[]
 with torch.no_grad():
  for identity,ref in reference.items():
   file=files[identity];sha=file_sha256(file);row=torch.load(file,weights_only=False)
   if row['image_id']!=identity or row['patient_key']!=ref['patient_key'] or row['split']!=ref['split'] or row['file_name']!=ref['file_name'] or row['image_sha256']!=ref['image_sha256'] or row['image_size']!=ref['image_size'] or not torch.equal(row['coords'],ref['coords']) or not torch.equal(row['mask'],ref['mask']) or row['feature'].shape!=(2048,48,29) or row['feature'].dtype!=torch.float32 or not torch.isfinite(row['feature']).all():raise ValueError('Existing nativefeature source/tensor mismatch')
   imagepath=ROOT/'data/image'/ref['file_name']
   if file_sha256(imagepath)!=row['image_sha256']:raise ValueError('Existing source image changed')
   pre=row['preprocess_metadata']
   if pre['size']!=[912,1520] or pre['source_range']<=0 or pre['mean']!=model.config['base']['mean'] or pre['std']!=model.config['base']['std'] or pre['additional_cropping'] is not False:raise ValueError('Existing source preprocessing mismatch')
   if identity in [64,73]:
    with Image.open(imagepath) as image:tensor,fresh_meta=preprocess(image.convert('L'))
    with torch.backends.cudnn.flags(enabled=True,allow_tf32=True):fresh=model(tensor[None].cuda())[0].cpu()
    error=float((fresh-row['feature']).abs().max());proof.append(dict(image_id=identity,fresh_saved_feature_error=error))
    if error!=0:raise ValueError('Recovery fresh64/73 source feature notexact')
   if file_sha256(file)!=sha:raise ValueError('Featurefile changed during recovery')
   metadata.append(dict(image_id=identity,patient_key=ref['patient_key'],split=ref['split'],sha256=sha))
 if state_sha256(model.state_dict())!=whole_sha:raise ValueError('Frozen source changed')
 header=dict(kind='mammo_clip_whole_cache_1011',records=metadata,medical_checkpoint=str(Path(a.medical_checkpoint).resolve()),medical_checkpoint_sha256=model.checkpoint_sha256,strict_image_encoder_keys=model.loaded_keys,whole_state_sha256=whole_sha,source_cache_provenance_sha256=file_sha256(Path(a.cache)/'provenance.json'),source_past_visual_sha256=prov['visual_state_sha256'],source_cfg=safe(model.config),source_code_sha256={str(f):file_sha256(f) for f in SOURCE.glob('*.py')},input_size=[912,1520],native_shape=[2048,48,29],grid=[24,48],crop_size=224,preprocessing='sourcecv2resize/minmax/scalarstats; existingPNGbordertrim; noextraROIcrop',proof=proof,original_extraction_seconds=None,metadata_recovery_seconds=time.monotonic()-started,recovery=dict(metadata_only=True,featurefiles_unchanged=True,failure=failure_info,json_conversion='recursiveOmegaConf.to_container(resolveTrue),no defaultstr'))
 temp=out/'provenance.json.tmp';temp.write_text(json.dumps(header,indent=2));temp.replace(out/'provenance.json');status=dict(state='complete',images=505,train=412,val=93,metadata_recovered=True,featurefiles_unchanged=True);temp=out/'status.json.tmp';temp.write_text(json.dumps(status,indent=2));temp.replace(out/'status.json');(out/'metadata_recovery_proof.json').write_text(json.dumps(dict(proof=proof,seconds=header['metadata_recovery_seconds'],failure=failure_info,files=505,featurefiles_unchanged=True),indent=2));print(json.dumps(dict(status=status,proof=proof,recovery_seconds=header['metadata_recovery_seconds'])))
if __name__=='__main__':main()
