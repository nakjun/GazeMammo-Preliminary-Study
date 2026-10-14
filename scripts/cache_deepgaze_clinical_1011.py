"""Cache native frozen DGIII whole-image features plus actual frozen arithmetic BOS."""
import argparse,json,sys,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageOps
from src.data import whole_tensor
from src.gaze_mirror_augmentation_1011 import mirror_coordinates
from src.deepgaze_context_1011 import native_features
from src.deepgaze_clinical_1011 import freeze_features,inventory
from src.dual_output_1011 import ProbabilityComponent
from src.gaze_nll_adaptation_1011 import initialize
from src.proposal_utility_1011 import state_sha256
from gaze_nll_adaptation_1011 import cache_records,load_base,save
from evaluate_deepgaze3_1011 import load_deepgaze
from train_attention_1011 import digest


@torch.no_grad()
def extract(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;records,provenance=cache_records(a);model=freeze_features(load_deepgaze(a.source,a.checkpoint,a.device));parents=[];bos_sources=[]
    for path in ('runs/full_1011_seed42/control/best.pt','runs/full_1011_seed42/improved/best.pt'):
        parent,_=load_base(path);parents.append(initialize(parent).to(a.device).eval());bos_sources.append(dict(path=str(Path(path).resolve()),sha256=digest(path)))
    bos=ProbabilityComponent(parents,[.75,.25]).to(a.device).eval();output=Path(a.output);output.mkdir(parents=True,exist_ok=False);headers=[];started=time.monotonic();source_sha=digest(a.checkpoint)
    if a.image_ids:records=[r for r in records if int(r['image_id']) in a.image_ids]
    header=dict(kind='deepgaze_clinical_native_cache_1011',checkpoint_sha256=source_sha,checkpoint=str(Path(a.checkpoint).resolve()),source_commit='c7db17e2d1d7ea6468ffdee2cfaddf141095dcff',input_size=a.input_size,feature_state_sha256=state_sha256(model.features.state_dict()),source_cache_provenance_sha256=digest(Path(a.cache)/'provenance.json'),gaze_sha256=provenance['gaze_sha256'],manifest_sha256=provenance['manifest_sha256'],all_manifest_sha256=provenance['all_manifest_sha256'],bos_sources=bos_sources,bos_weights=[.75,.25],inventory=inventory(model),dtype='float32',grid=[24,48],roi_crop_size=224,validation_mirrored=False,smoke=bool(a.image_ids),records=headers)
    def status(state,error=None):save(output/'status.json',dict(state=state,completed=len(headers),seconds=time.monotonic()-started,error=error))
    save(output/'provenance.json',header);status('running')
    try:
        for r in records:
            image_id=int(r['image_id']);path=ROOT/'data/image'/r['file_name']
            if digest(path)!=r['image_sha256']:raise ValueError('Original image changed')
            with Image.open(path) as source:original=source.convert('L')
            for flipped in ((False,True) if r['split']=='train' else (False,)):
                image=ImageOps.mirror(original) if flipped else original;rgb=torch.from_numpy(np.asarray(image.resize(tuple(a.input_size),Image.Resampling.BILINEAR)).copy()).float()[None,None].repeat(1,3,1,1).to(a.device);features=[f[0].cpu() for f in native_features(model,rgb)]
                memory=bos.maps(whole_tensor(image,(256,512))[None].to(a.device));first,_,_=bos.step_history(memory,torch.zeros(1,0,2,device=a.device));name=f'{image_id}_mirror' if flipped else str(image_id)
                if sum(f.shape[0] for f in features)!=2048 or any(f.dtype!=torch.float32 for f in features):raise ValueError('Unexpected native DG features')
                item=dict(image_id=image_id,patient_key=r['patient_key'],split=r['split'],flipped=flipped,features=features,bos_logp=first[0].cpu(),image_sha256=r['image_sha256'],pixels_sha256=hashlib.sha256(np.asarray(image).tobytes()).hexdigest(),coords=mirror_coordinates(r['coords'],r['mask']) if flipped else r['coords'],mask=r['mask'])
                temporary=output/f'{name}.tmp';torch.save(item,temporary);temporary.replace(output/f'{name}.pt');headers.append(dict(name=name,image_id=image_id,patient_key=r['patient_key'],split=r['split'],flipped=flipped,file_sha256=digest(output/f'{name}.pt'),feature_shapes=[list(f.shape) for f in features]));save(output/'provenance.json',header);status('running');print(json.dumps(dict(name=name,completed=len(headers))),flush=True)
        save(output/'summary.json',dict(records=len(headers),bytes=sum(p.stat().st_size for p in output.glob('*.pt')),seconds=time.monotonic()-started));status('complete')
    except Exception as error:status('failed',str(error));raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='/tmp/deepgaze_feasibility_1011');p.add_argument('--checkpoint',default='runs/dual_target_1011/deepgaze_public/assets/deepgaze3.pth');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--output',required=True);p.add_argument('--input-size',nargs=2,type=int,default=[512,1024]);p.add_argument('--device',default='cuda:0');p.add_argument('--image-ids',nargs='+',type=int);extract(p.parse_args())
