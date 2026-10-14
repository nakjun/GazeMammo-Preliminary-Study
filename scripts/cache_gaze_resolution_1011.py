"""Re-extract frozen whole-image pyramids at higher resolution; reuse exact crop history."""
import argparse,json,sys,time
from pathlib import Path
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from gaze_nll_adaptation_1011 import cache_records,load_base,validate_records,routing,apply_cache_resolution
from src.gaze_nll_adaptation_1011 import initialize
from src.proposal_utility_1011 import state_sha256
from src.data import whole_tensor
from train_attention_1011 import digest
from guided_trajectory_1011 import base_path


def validate_source(provenance,visual_sha,input_size):
    if provenance.get('visual_state_sha256')!=visual_sha:raise ValueError('Source crop vectors belong to a different visual encoder')
    source_size=tuple(provenance.get('input_size',(256,512)))
    if tuple(input_size)==source_size:raise ValueError('Resolution cache must not alias unchanged whole-image input size')
    if provenance.get('annotation_used') is not False or provenance.get('smoke'):raise ValueError('Requires full original no-annotation source cache')
    return source_size


def validate_native_features(features,input_size,channels=(256,512,1024)):
    width,height=input_size;expected=[(channel,height//stride,width//stride) for channel,stride in zip(channels,(8,16,32))]
    if len(features)!=3 or [tuple(feature.shape) for feature in features]!=expected:raise ValueError(f'Native pyramid mismatch; expected{expected}')
    if any(feature.dtype!=torch.float32 or not torch.isfinite(feature).all() for feature in features):raise ValueError('Invalid native feature dtype/values')
    return expected


def replace_whole_features(source_record,features,input_size,visual_sha):
    """Only whole features/metadata change. Prefix and crop vectors are copied exactly."""
    if not torch.isfinite(source_record['vectors']).all():raise ValueError('Invalid reused preceding crop vectors')
    record={key:value for key,value in source_record.items() if key!='features'}
    record.update(features=features,input_size=list(input_size),whole_input_size=list(input_size),visual_state_sha256=visual_sha,reused_foveal_vectors=True,source_feature_shapes=[list(feature.shape) for feature in source_record['features']])
    for key in ('coords','mask','vectors'):
        if not torch.equal(record[key],source_record[key]):raise ValueError(f'Altered reused{key}')
    return record


@torch.no_grad()
def verify_live_cached_maps(model,input_tensor,features,coords=None,vectors=None):
    """Checks the actual model maps path against native cached pyramid maps."""
    live=model.maps(input_tensor)
    cached=model.maps_features([feature[None].to(input_tensor.device) for feature in features]);differences={}
    for key in ('keys','prior','global_context'):
        differences[key]=float((live[key]-cached[key]).abs().max())
        if not torch.allclose(live[key],cached[key],rtol=1e-5,atol=2e-5):raise ValueError(f'Cached/live{key} mismatch')
    if coords is not None:
        live_raw=model.teacher_logits(live,coords,vectors);cached_raw=model.teacher_logits(cached,coords,vectors)
        differences['raw_teacher_logp']=float((live_raw-cached_raw).abs().max())
        if not torch.allclose(live_raw,cached_raw,rtol=1e-5,atol=2e-5):raise ValueError('Cached/live raw teacher logp mismatch')
    return differences


def extract(a):
    torch.set_num_threads(a.threads)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    source_args=argparse.Namespace(cache=a.source_cache,base=a.base);records,source_provenance=cache_records(source_args);base,checkpoint=load_base(a.base)
    visual_sha=state_sha256(base.visual.state_dict());source_size=validate_source(source_provenance,visual_sha,a.input_size)
    if base.backbone_name!='convnext_base' or tuple(base.grid)!=(24,48):raise ValueError('This pilot requires original ConvNeXt-base24x48grid')
    input_size=tuple(a.input_size);base.input_size=input_size;base.model_config['input_size']=list(input_size);base.to(a.device).eval()
    for parameter in base.parameters():parameter.requires_grad_(False)
    original_count=len(records)
    if a.limit:
        records=[r for r in records if r['split']=='train'][:a.limit]+[r for r in records if r['split']=='val'][:a.limit]
    validate_records(records,smoke=bool(a.limit))
    if not a.limit and len(records)!=505:raise ValueError('Requires full fixed train412/val93')
    provenance=dict(source_provenance,kind='gaze_nll_resolution_cache_1011',base_checkpoint=str(base_path(a.base).resolve()),base_checkpoint_sha256=digest(base_path(a.base)),input_size=list(input_size),input_size_order='width,height (PIL)',source_input_size=list(source_size),whole_pixel_ratio=(input_size[0]*input_size[1])/(source_size[0]*source_size[1]),crop_size=224,preceding_crop_size=224,grid=[24,48],native_feature_shapes=[[channel,input_size[1]//stride,input_size[0]//stride] for channel,stride in zip((256,512,1024),(8,16,32))],source_cache=str(Path(a.source_cache).resolve()),source_cache_provenance_sha256=digest(Path(a.source_cache)/'provenance.json'),visual_state_sha256=visual_sha,reused_foveal_vectors=True,feature_memory_format='contiguous_nchw',source_foveal_vector_precision='Inherited original float32 crop vectors, copied exactly without CNN re-extraction',records=[{key:r[key] for key in ('image_id','file_name','patient_key','split')} for r in records],smoke=bool(a.limit),protocol='Frozen whole-image encoder at explicit higher input resolution, native3level pyramid; exact old224px preceding crop vectors/coords/masks reused. Exactlegacy16 targetgrid24x48 unchanged; no annotations/test/new data.')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2));(output/'config.json').write_text(json.dumps(vars(a),indent=2));started=time.monotonic();completed=0;checks={}
    verifier=initialize(base).to(a.device).eval() if a.verify_images else None
    if verifier is not None:apply_cache_resolution(verifier,provenance)
    def status(state,error=None):
        temporary_status=output/'status.tmp'
        temporary_status.write_text(json.dumps(dict(state=state,images=len(records),completed=completed,seconds=time.monotonic()-started,error=error)))
        temporary_status.replace(output/'status.json')
    status('running')
    try:
        for record in records:
            image_path=ROOT/'data/image'/record['file_name']
            if digest(image_path)!=record['image_sha256']:raise ValueError('Source image changed since foveal vectors were cached')
            with Image.open(image_path) as source:image=source.convert('L')
            tensor=whole_tensor(image,input_size)[None].to(a.device)
            with torch.no_grad():features=[feature[0].detach().cpu().contiguous() for feature in base.visual(tensor)]
            shapes=validate_native_features(features,input_size)
            cached=replace_whole_features(record,features,input_size,visual_sha);temporary=output/f"{record['image_id']}.tmp";torch.save(cached,temporary);temporary.replace(output/f"{record['image_id']}.pt")
            if verifier is not None and completed<a.verify_images:
                reloaded=torch.load(output/f"{record['image_id']}.pt",map_location='cpu',weights_only=False)
                checks[str(record['image_id'])]=verify_live_cached_maps(verifier,tensor,reloaded['features'],record['coords'][None].to(a.device),record['vectors'][None].to(a.device))
                for key in ('vectors','coords','mask'):
                    if not torch.equal(reloaded[key],record[key]):raise ValueError('Serialized reused history changed')
            completed+=1;status('running');print(json.dumps(dict(image_id=int(record['image_id']),completed=completed,native_shapes=shapes)),flush=True)
        summary=dict(state='complete',images=completed,source_images=original_count,input_size=list(input_size),source_input_size=list(source_size),native_feature_shapes=shapes,cached_live_map_max_errors=checks,preceding_crop_vectors_reused_exact=True,feature_memory_format=provenance['feature_memory_format'],visual_state_sha256=visual_sha,whole_pixel_ratio=provenance['whole_pixel_ratio'],seconds=time.monotonic()-started,source_provenance_sha256=provenance['source_cache_provenance_sha256'])
        (output/'summary.json').write_text(json.dumps(summary,indent=2));status('complete');print(json.dumps(summary),flush=True);return summary
    except Exception as error:status('failed',str(error));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');parser.add_argument('--source-cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');parser.add_argument('--output',required=True);parser.add_argument('--input-size',nargs=2,type=int,default=[512,1024]);parser.add_argument('--device',default='cuda:0');parser.add_argument('--threads',type=int,default=4);parser.add_argument('--limit',type=int,default=0);parser.add_argument('--verify-images',type=int,default=2);args=parser.parse_args()
    if tuple(args.input_size) not in ((384,768),(512,1024)):raise ValueError('Only predeclared resolution smoke384x768/full512x1024')
    if not args.limit and tuple(args.input_size)!=(512,1024):raise ValueError('Full pilot resolution fixed512x1024')
    extract(args)
