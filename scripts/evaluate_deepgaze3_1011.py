"""Public pretrained DeepGazeIII with explicit frozen arithmetic image-only BOS."""
import argparse,json,sys,time,types,importlib,gc
from pathlib import Path
import numpy as np,pandas as pd,torch,torchvision
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.deepgaze_context_1011 import train_centerbias,dense_centerbias,native_features,forward_native
from src.data import whole_tensor
from src.dual_output_1011 import ProbabilityComponent
from src.gaze_nll_adaptation_1011 import initialize
from gaze_nll_adaptation_1011 import load_base,apply_cache_resolution
from train_attention_1011 import digest

def cell_mass(logdensity,grid=(24,48)):
    probability=logdensity.exp().flatten(2);h,w=logdensity.shape[-2:];gx,gy=grid
    yy=((torch.arange(h,device=logdensity.device)+.5)/h*gy).long();xx=((torch.arange(w,device=logdensity.device)+.5)/w*gx).long();index=(yy[:,None]*gx+xx[None]).flatten();mass=probability.new_zeros(logdensity.shape[0],gx*gy);mass.scatter_add_(1,index[None].expand(logdensity.shape[0],-1),probability[:,0]);mass=mass/mass.sum(-1,keepdim=True);return mass.clamp_min(1e-30).log()

def load_deepgaze(source,checkpoint,device):
    package=types.ModuleType('deepgaze_pytorch');package.__path__=[str(Path(source)/'deepgaze_pytorch')];sys.modules['deepgaze_pytorch']=package
    module=importlib.import_module('deepgaze_pytorch.deepgaze3');original=torch.hub.load
    def local_backbone(repo,name,**kwargs):
        if repo!='pytorch/vision:v0.6.0' or name!='densenet201':raise ValueError('Unexpected hub dependency')
        return torchvision.models.densenet201(weights=None)
    torch.hub.load=local_backbone
    try:model=module.DeepGazeIII(pretrained=False)
    finally:torch.hub.load=original
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True);return model.to(device).eval()

def histories(coords,t,size):
    result=coords.new_full((1,4,2),float('nan'))
    for i in range(min(t,4)):result[:,i]=coords[:,t-1-i]*coords.new_tensor(size)
    return result[:,:,0],result[:,:,1]

def run(a):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;out=Path(a.output);out.mkdir(parents=True,exist_ok=False);model=load_deepgaze(a.source,a.checkpoint,a.device);parents=[];sources=[];bias_probability=None
    for cachefile in ['human_control_raw_cache_for_ensemble.pt','human_improved_raw_cache.pt']:
        raw=torch.load(ROOT/'runs/dual_target_1011'/cachefile,map_location='cpu',weights_only=False);assert digest(raw['base_checkpoint'])==raw['base_checkpoint_sha256'];parent,_=load_base(raw['base_checkpoint']);parents.append(apply_cache_resolution(initialize(parent),raw['cache_provenance']).to(a.device).eval());sources.append(dict(path=raw['base_checkpoint'],sha256=raw['base_checkpoint_sha256']));del raw
    if a.centerbias=='train':
        bias_records=torch.load(ROOT/'runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt',map_location='cpu',weights_only=False)['records'];bias_probability=train_centerbias(bias_records,ROOT/'manifests/all.csv');torch.save(bias_probability,out/'train_centerbias.pt')
    bos=ProbabilityComponent(parents,[.75,.25]).to(a.device).eval();frame=pd.read_csv(ROOT/'manifests/trajectory.csv');frame=frame[frame.split=='val'];assert len(frame)==93 and frame.image_id.is_unique;values=[];rows=[];proofs=[];start=time.monotonic()
    with torch.no_grad():
        for _,row in frame.iloc[:a.limit or 93].iterrows():
            record=torch.load(Path(a.cache)/f'{int(row.image_id)}.pt',map_location='cpu',weights_only=False);assert record['split']=='val' and record['patient_key']==row.patient_key;coords=record['coords'][None].to(a.device);mask=record['mask'].bool().to(a.device);path=ROOT/'data/image'/record['file_name'];assert digest(path)==record['image_sha256'];assert coords.shape==(1,16,2) and torch.isfinite(coords).all() and ((coords>=0)&(coords<=1)).all()
            with Image.open(path) as image:
                image=image.convert('L');rgb=torch.from_numpy(np.asarray(image.resize(tuple(a.input_size),Image.Resampling.BILINEAR)).copy()).float()[None,None].repeat(1,3,1,1).to(a.device);memory=bos.maps(whole_tensor(image,(256,512))[None].to(a.device));first,_,_=bos.step_history(memory,coords[:,:0])
            width,height=a.input_size;centerbias=dense_centerbias(bias_probability,a.input_size,a.device) if bias_probability is not None else rgb.new_full((1,height,width),-np.log(height*width));features=native_features(model,rgb);predictions=[first]
            for t in range(1,16):
                hx,hy=histories(coords,t,a.input_size);density=forward_native(model,features,a.input_size,centerbias,hx,hy);q=cell_mass(density);assert torch.isfinite(q).all() and abs(float(q.exp().sum())-1)<1e-5;predictions.append(q)
            raw=torch.stack(predictions,1)
            if len(proofs)<2:
                modified=coords.clone();modified[:,8:]=1-modified[:,8:];hx,hy=histories(coords,8,a.input_size);cx,cy=histories(modified,8,a.input_size);difference=float((cell_mass(model(rgb,centerbias,hx,hy))-cell_mass(model(rgb,centerbias,cx,cy))).abs().max());print(json.dumps(dict(probe_future_difference=difference,history_equal=torch.equal(hx,cx) and torch.equal(hy,cy))),flush=True);assert difference<=1e-5
                empty_failure=None
                try:
                    nan=rgb.new_full((1,4),float('nan'));model(rgb,centerbias,nan,nan)
                except Exception as error:empty_failure=type(error).__name__+': '+str(error)
                cached_live_max=float((forward_native(model,features,a.input_size,centerbias,hx,hy)-model(rgb,centerbias,hx,hy)).abs().max());assert cached_live_max<=1e-5
                proofs.append(dict(native_feature_shapes=[list(x.shape) for x in features],cached_live_max=cached_live_max,image_id=int(row.image_id),future_invariance_max=difference,probability_sum=float(raw.exp().sum(-1).min()),empty_history_upstream_failure=empty_failure,t0='Actual frozen image-only arithmetic .75control+.25improved'))
            target=(coords[0,:,1]*48).long().clamp(0,47)*24+(coords[0,:,0]*24).long().clamp(0,23);nll=-raw[0].gather(1,target[:,None]).squeeze(1);value=float(nll[mask].mean());values.append(value);rows.extend(dict(image_id=int(row.image_id),patient_key=row.patient_key,step=t,nll=float(nll[t])) for t in torch.nonzero(mask).flatten().tolist());print(json.dumps(dict(image_id=int(row.image_id),nll=value)),flush=True)
            result=dict(images=len(values),complete=len(values)==93,nll=sum(values)/len(values),seconds=time.monotonic()-start,proofs=proofs,input_size=a.input_size,centerbias=a.centerbias,centerbias_source_raw_cache_sha256=digest(ROOT/'runs/dual_target_1011/human_control_raw_cache_for_ensemble.pt'),checkpoint_sha256=digest(a.checkpoint),checkpoint_url='https://github.com/matthias-k/DeepGaze/releases/download/v1.1.0/deepgaze3.pth',checkpoint_bytes=Path(a.checkpoint).stat().st_size,source_commit='c7db17e2d1d7ea6468ffdee2cfaddf141095dcff',torch_version=torch.__version__,torchvision_version=torchvision.__version__,backbone_loader='Installed torchvision DenseNet201 architecture, strict full original checkpoint loading; no extra backbone pretraining download',bos_sources=sources,protocol=f'{a.input_size[0]}x{a.input_size[1]} RGB0..255; {a.centerbias} centerbias; last4 strictly past observed coordinates, NaN padding; t0 frozen image-only arithmetic; pixel-center cell probability SUM grid24x48; fixed legacy16 val93 exploratory; no duration/future/boxes',license='Unresolved upstream code/weight license; local research only; no redistribution',natural_pretraining='SALICONmouseproxy spatialpretraining thenMIT1003 scanpaths; authorspaper PMC9055565; noCBIS documented')
            (out/'metrics.json').write_text(json.dumps(result,indent=2));pd.DataFrame(rows).to_csv(out/'per_step.csv',index=False)
    pd.DataFrame(rows).groupby(['image_id','patient_key'],as_index=False).nll.mean().to_csv(out/'per_image.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='/tmp/deepgaze_feasibility_1011');p.add_argument('--checkpoint',default='runs/dual_target_1011/deepgaze_public/assets/deepgaze3.pth');p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--device',default='cuda:0');p.add_argument('--input-size',type=int,nargs=2,default=[256,512]);p.add_argument('--centerbias',choices=['uniform','train'],default='uniform');run(p.parse_args())
