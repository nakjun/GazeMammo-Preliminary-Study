"""Feature extraction, MIL training, gaze training, generation, and evaluation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import random
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from src.data import ROOT,PatchDataset,TrajectoryDataset,ImageDataset
from src.models import backbone,PatchClassifier,TrajectoryModel,HeatmapModel,BACKBONES,trajectory_from_checkpoint
from src.metrics import classification_metrics,threshold_at_specificity,cluster_bootstrap,dtw_distance

def setup(args):
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; run outside sandbox or select --device cpu')
    return device

def manifest(path,split=None):
    frame=pd.read_csv(path)
    if split:frame=frame[frame.split==split]
    assert not frame.image_id.duplicated().any()
    return frame.reset_index(drop=True)

def save_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False))

def config(args,out):
    out.mkdir(parents=True,exist_ok=True)
    save_json(out/'config.json',dict(vars(args),torch_version=torch.__version__,
                                   manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()))

def validate_prediction_provenance(args,frame):
    if args.method not in ('trajectory','heatmap'):return
    path=Path(args.predictions)
    metadata=json.loads(path.with_suffix('.provenance.json').read_text())
    artifacts=metadata.get('artifacts',[metadata])
    for _,row in frame.iterrows():
        matching=[a for a in artifacts if int(row.image_id) in a['generated_image_ids']]
        if len(matching)!=1:raise ValueError(f'Missing/ambiguous provenance for {row.image_id}')
        if matching[0]['method']!=args.method:raise ValueError('Prediction method does not match feature selector')
        for key in ['generation_policy','generation_strategy']:
            if key in matching[0] and matching[0][key]!=getattr(args,key,None):raise ValueError(f'Prediction {key} does not match feature configuration')
        if matching[0].get('generation_policy') in ('hard','soft') and matching[0]['generation_radius']!=args.generation_radius:raise ValueError('Prediction visit radius does not match feature configuration')
        if row.patient_key in matching[0]['training_patient_keys']:
            raise ValueError(f'Predictor trained on patient {row.patient_key}; use cross-fitting')

def extract(args):
    device=setup(args);out=Path(args.output);config(args,out)
    frame=manifest(args.manifest)
    # Final test has to be explicitly requested after model selection.
    allowed=['train','val']+(['test'] if args.include_test else [])
    frame=frame[(frame.label>=0)&frame.split.isin(allowed)].reset_index(drop=True)
    if args.limit:frame=frame.groupby('split',group_keys=False).head(args.limit).reset_index(drop=True)
    validate_prediction_provenance(args,frame)
    dataset=PatchDataset(frame,args.method,args.k,args.crop_size,args.seed,args.predictions,args.min_distance,args.keep_predicted_centers)
    encoder=backbone().to(device).eval();features=[];masks=[];centers=[]
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers)
    started=time.time()
    with torch.inference_mode():
        for i,b in enumerate(loader):
            x=b['patches'];batch,k,c,h,w=x.shape
            flat=x.reshape(batch*k,c,h,w);encoded=[]
            for chunk in flat.split(args.encode_batch):
                encoded.append(encoder(chunk.to(device)).mean((-1,-2)).cpu())
            features.append(torch.cat(encoded).reshape(batch,k,-1));masks.append(b['mask']);centers.append(b['centers'])
            if i%10==0:print(f'extract {args.method}: {min((i+1)*args.batch_size,len(frame))}/{len(frame)}',flush=True)
    torch.save(dict(features=torch.cat(features),mask=torch.cat(masks),centers=torch.cat(centers),image_ids=frame.image_id.tolist(),
                    label=torch.tensor(frame.label.to_numpy(),dtype=torch.float32),config=vars(args)),out/'features.pt')
    frame.to_csv(out/'manifest.csv',index=False)
    save_json(out/'timing.json',dict(seconds=time.time()-started,n=len(frame),device=str(device)))

def cached(args):
    path=Path(args.features);payload=torch.load(path/'features.pt',map_location='cpu',weights_only=False)
    frame=manifest(path/'manifest.csv')
    assert frame.image_id.tolist()==payload['image_ids']
    return payload,frame

def evaluate_head(model,payload,indices,device,batch=32):
    probabilities=[];attention=[]
    model.eval()
    with torch.inference_mode():
        for ids in torch.tensor(indices).split(batch):
            logits,weights=model(payload['features'][ids].to(device),payload['mask'][ids].to(device))
            probabilities.extend(logits.sigmoid().cpu().tolist());attention.extend(weights.cpu().tolist())
    return probabilities,attention

def train_classifier(args):
    device=setup(args);out=Path(args.output);config(args,out);payload,frame=cached(args)
    train=frame.index[frame.split=='train'].to_numpy();val=frame.index[frame.split=='val'].to_numpy()
    if not len(train) or not len(val):raise ValueError('Training and validation features required')
    assert not set(frame.loc[train,'patient_key'])&set(frame.loc[val,'patient_key'])
    model=PatchClassifier(args.aggregation,k=payload['features'].shape[1]).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    loader=DataLoader(TensorDataset(payload['features'][train],payload['mask'][train],payload['label'][train]),batch_size=args.batch_size,shuffle=True)
    best=-float('inf');history=[];stale=0
    for epoch in range(args.epochs):
        model.train();losses=[]
        for features,mask,y in loader:
            opt.zero_grad();logit,_=model(features.to(device),mask.to(device));loss=nn.functional.binary_cross_entropy_with_logits(logit,y.to(device))
            loss.backward();opt.step();losses.append(float(loss.detach()))
        probs,_=evaluate_head(model,payload,val,device);metrics=classification_metrics(frame.loc[val,'label'],probs)
        score=metrics['auroc']
        if score is None:raise ValueError('Validation must contain both classes')
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation=metrics);history.append(record)
        print(json.dumps(record),flush=True)
        if score>best:
            best=score;stale=0;torch.save(dict(state=model.state_dict(),aggregation=args.aggregation,k=payload['features'].shape[1],args=vars(args)),out/'best.pt')
        else:stale+=1
        if stale>=args.patience:break
    checkpoint=torch.load(out/'best.pt',weights_only=False,map_location=device);model.load_state_dict(checkpoint['state'])
    probs,attention=evaluate_head(model,payload,val,device)
    threshold=threshold_at_specificity(frame.loc[val,'label'],probs,args.specificity)
    result=frame.loc[val].copy();result['probability']=probs;result['attention']=[json.dumps(a) for a in attention]
    result.to_csv(out/'validation_predictions.csv',index=False)
    save_json(out/'history.json',history);save_json(out/'threshold.json',dict(value=threshold,target_specificity=args.specificity,source='validation'))
    save_json(out/'validation_metrics.json',dict(metrics=classification_metrics(result.label,result.probability,threshold),auroc_ci=cluster_bootstrap(result,repetitions=args.bootstrap,seed=args.seed)))

def evaluate_classifier(args):
    if args.split=='test' and not args.final_test:raise ValueError('Use --final-test explicitly after selecting settings')
    device=setup(args);out=Path(args.output);config(args,out);payload,frame=cached(args)
    c=torch.load(Path(args.checkpoint)/'best.pt',map_location=device,weights_only=False)
    original_config=json.loads((Path(c['args']['features'])/'config.json').read_text())
    current_config=json.loads((Path(args.features)/'config.json').read_text())
    for key in ['method','k','crop_size','min_distance']:
        if original_config[key]!=current_config[key]:raise ValueError(f'Evaluation feature setting changed: {key}')
    if original_config.get('keep_predicted_centers',False)!=current_config.get('keep_predicted_centers',False):raise ValueError('Evaluation center projection setting changed')
    model=PatchClassifier(c['aggregation'],k=c['k']).to(device);model.load_state_dict(c['state'])
    ids=frame.index[frame.split==args.split].to_numpy()
    if not len(ids):raise ValueError('No evaluation features')
    # Check group separation against the checkpoint training feature manifest.
    training_manifest=manifest(Path(c['args']['features'])/'manifest.csv')
    if set(training_manifest[training_manifest.split=='train'].patient_key)&set(frame.loc[ids,'patient_key']):raise ValueError('Evaluation patient leakage')
    p,attention=evaluate_head(model,payload,ids,device);r=frame.loc[ids].copy();r['probability']=p;r['attention']=[json.dumps(a) for a in attention]
    threshold=json.loads((Path(args.checkpoint)/'threshold.json').read_text())['value']
    r.to_csv(out/'predictions.csv',index=False)
    save_json(out/'metrics.json',dict(metrics=classification_metrics(r.label,r.probability,threshold),auroc_ci=cluster_bootstrap(r,repetitions=args.bootstrap,seed=args.seed)))

def trajectory_frames(args):
    frame=manifest(args.manifest)
    train=frame[(frame.split=='train')&(frame.label>=0)]
    if args.holdout_fold is not None:train=train[train.crossfit_fold!=args.holdout_fold]
    val=frame[(frame.split=='val')&(frame.label>=0)]
    if args.limit:train=train.head(args.limit);val=val.head(args.limit)
    assert not set(train.patient_key)&set(val.patient_key)
    return frame,train,val

def gaze_loss(model,b,device,method):
    target=b['target'].to(device)
    if method=='trajectory':
        logits=model.forward_features(b['features'].to(device),target) if 'features' in b else model(b['image'].to(device),target)
        return nn.functional.cross_entropy(logits.flatten(0,1),target.flatten(),ignore_index=-100)
    logits=model(b['image'].to(device));valid=target!=-100
    hist=torch.zeros_like(logits).scatter_add_(1,target.clamp_min(0),valid.float())
    hist=hist/hist.sum(1,keepdim=True).clamp_min(1)
    return -(hist*logits.log_softmax(1)).sum(1).mean()

def train_gaze(args):
    if args.epochs<1 or args.accumulation<1:raise ValueError('epochs and accumulation must be positive')
    if args.finetune_backbone and (args.cache_visual or args.method!='trajectory'):raise ValueError('Backbone fine-tuning requires trajectory and uncached visual features')
    device=setup(args);out=Path(args.output);config(args,out);frame,train,val=trajectory_frames(args)
    grid=(args.grid_x,args.grid_y)
    model=(TrajectoryModel(grid,args.k,backbone_name=args.backbone,dim=args.decoder_dim,layers=args.decoder_layers,
                           heads=args.decoder_heads,ff_multiplier=args.ff_multiplier,dropout=args.dropout,finetune_backbone=args.finetune_backbone) if args.method=='trajectory' else HeatmapModel(grid)).to(device)
    if args.init_checkpoint:
        initial=torch.load(Path(args.init_checkpoint)/'best.pt',map_location='cpu',weights_only=False)
        if set(initial['training_patient_keys'])-set(train.patient_key):raise ValueError('Initializer has seen patients outside this training fold')
        if initial['grid']!=grid or initial['k']!=args.k or initial.get('backbone','resnet50')!=args.backbone:raise ValueError('Initializer architecture mismatch')
        model.load_state_dict(initial['state'])
    if args.finetune_backbone:
        visual_ids={id(p) for p in model.visual.parameters()}
        groups=[dict(params=list(model.visual.parameters()),lr=args.backbone_lr),dict(params=[p for p in model.parameters() if id(p) not in visual_ids],lr=args.lr)]
    else:groups=[dict(params=[p for p in model.parameters() if p.requires_grad],lr=args.lr)]
    opt=torch.optim.AdamW(groups,weight_decay=1e-4)
    loaders=[DataLoader(TrajectoryDataset(f,args.k,args.min_distance,grid),batch_size=args.batch_size,shuffle=i==0,num_workers=args.workers) for i,f in enumerate([train,val])]
    started=time.time()
    if args.cache_images:
        loaders=[DataLoader([dict(image=b['image'][j],target=b['target'][j],coords=b['coords'][j],image_id=b['image_id'][j]) for b in loader for j in range(len(b['image']))],batch_size=args.batch_size,shuffle=i==0) for i,loader in enumerate(loaders)]
        print('Cached input tensors only; backbone features recomputed at every update',flush=True)
    if args.cache_visual:
        if args.method!='trajectory':raise ValueError('--cache-visual currently supports trajectory only')
        cached_loaders=[]
        for i,loader in enumerate(loaders):
            examples=[]
            model.eval()
            with torch.no_grad():
                for b in loader:
                    visual=model.visual(b['image'].to(device)).cpu()
                    for j in range(len(visual)):
                        examples.append({'features':visual[j],'target':b['target'][j],'coords':b['coords'][j],'image_id':b['image_id'][j]})
            cached_loaders.append(DataLoader(examples,batch_size=args.batch_size,shuffle=i==0))
        loaders=cached_loaders
        print(f'Cached frozen {args.backbone} features: train={len(train)} val={len(val)}',flush=True)
    best=float('inf');history=[];stale=0;best_epoch=None;change=0.
    reference=next(model.visual.parameters()).detach().clone()
    def checkpoint(epoch):
        return dict(state=model.state_dict(),method=args.method,grid=grid,k=args.k,backbone=args.backbone,model_config=model.model_config if args.method=='trajectory' else {},args=vars(args),epoch=epoch,training_patient_keys=sorted(train.patient_key.unique()),training_image_ids=train.image_id.tolist(),selection_patient_keys=sorted(val.patient_key.unique()))
    def validation_loss():
        model.eval();v=[]
        with torch.inference_mode():
            for b in loaders[1]:
                with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.amp and device.type=='cuda'):
                    loss=gaze_loss(model,b,device,args.method)
                v.append((float(loss),len(b['target'])))
        return sum(a*n for a,n in v)/sum(n for _,n in v)
    if args.init_checkpoint:
        best=validation_loss();best_epoch=0;torch.save(checkpoint(0),out/'best.pt')
        history.append(dict(epoch=0,validation_loss=best));print(json.dumps(history[-1]),flush=True)
    for epoch in range(args.epochs):
        losses=[];model.train();opt.zero_grad()
        for step,b in enumerate(loaders[0]):
            with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.amp and device.type=='cuda'):loss=gaze_loss(model,b,device,args.method)
            group_size=min(args.accumulation,len(loaders[0])-(step//args.accumulation)*args.accumulation)
            (loss/group_size).backward();losses.append(float(loss.detach()))
            if (step+1)%args.accumulation==0 or step+1==len(loaders[0]):nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();opt.zero_grad()
        vl=validation_loss()
        change=float((next(model.visual.parameters()).detach()-reference).abs().mean())
        record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=vl,backbone_first_weight_mean_change=change);history.append(record);print(json.dumps(record),flush=True)
        if vl<best:
            best=vl;stale=0;best_epoch=epoch+1
            torch.save(checkpoint(epoch+1),out/'best.pt')
        else:stale+=1
        if stale>=args.patience:break
    save_json(out/'history.json',history)
    torch.save(checkpoint(max(h['epoch'] for h in history)),out/'last.pt')
    save_json(out/'training_summary.json',dict(backbone=args.backbone,decoder_config=model.model_config if args.method=='trajectory' else {},best_epoch=best_epoch,backbone_first_weight_mean_change=change,best_validation_nll=best,epochs=args.epochs if not history else max(h['epoch'] for h in history),seconds=time.time()-started,train_images=len(train),val_images=len(val),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters())))

def heatmap_centers(logits,grid,k):
    gx,gy=grid;score=logits.clone().reshape(gy,gx);points=[]
    for _ in range(k):
        cell=int(score.argmax());y,x=divmod(cell,gx);points.append(((x+.5)/gx,(y+.5)/gy))
        score[max(0,y-2):y+3,max(0,x-2):x+3]=-torch.inf
    return points

def generate(args):
    device=setup(args);out=Path(args.output);config(args,out)
    c=torch.load(Path(args.checkpoint)/'best.pt',map_location=device,weights_only=False)
    model=(trajectory_from_checkpoint(c) if c['method']=='trajectory' else HeatmapModel(c['grid'],pretrained=False)).to(device)
    model.load_state_dict(c['state']);model.eval()
    frame=manifest(args.manifest);frame=frame[frame.split.isin(args.splits.split(','))]
    if args.holdout_fold is not None:frame=frame[frame.crossfit_fold==args.holdout_fold]
    if args.limit:frame=frame.head(args.limit)
    if set(frame.patient_key)&set(c['training_patient_keys']):raise ValueError('Generating on training patients forbidden; use held-out cross-fitting fold')
    if 'test' in set(frame.split) and not args.final_test:raise ValueError('Test generation requires --final-test')
    dataset=TrajectoryDataset(frame,c['k'],c['args']['min_distance'],c['grid']) if args.with_gaze_evaluation else ImageDataset(frame)
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers)
    result={};metrics=[]
    torch.manual_seed(args.seed)
    with torch.inference_mode():
        for b in loader:
            if c['method']=='trajectory':p=model.generate(b['image'].to(device),args.strategy,args.temperature).cpu().numpy()
            else:p=np.array([heatmap_centers(x,c['grid'],c['k']) for x in model(b['image'].to(device)).cpu()])
            if args.with_gaze_evaluation and c['method']=='trajectory':
                teacher=model(b['image'].to(device),b['target'].to(device)).cpu()
                per_step_nll=nn.functional.cross_entropy(teacher.flatten(0,1),b['target'].flatten(),ignore_index=-100,reduction='none').reshape(b['target'].shape)
                cells=teacher.argmax(-1);gx,gy=c['grid']
                teacher_coords=torch.stack([((cells%gx)+.5)/gx,((cells//gx)+.5)/gy],-1)
            for i,image_id in enumerate(b['image_id'].tolist()):
                result[str(image_id)]=p[i].tolist()
                record=dict(image_id=image_id,unique_grid_locations=len(np.unique(p[i],axis=0)))
                if args.with_gaze_evaluation:
                    truth=b['coords'][i][b['target'][i]!=-100].numpy();record['normalized_dtw']=dtw_distance(truth,p[i])
                    if c['method']=='trajectory':
                        valid=b['target'][i]!=-100
                        record['teacher_forced_next_step_nll']=float(per_step_nll[i][valid].mean())
                        record['teacher_forced_normalized_location_error']=float(torch.linalg.vector_norm(teacher_coords[i][valid]-b['coords'][i][valid],dim=-1).mean())
                metrics.append(record)
    save_json(out/'predictions.json',result)
    save_json(out/'predictions.provenance.json',dict(checkpoint=str(Path(args.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256((Path(args.checkpoint)/'best.pt').read_bytes()).hexdigest(),
             method=c['method'],backbone=c.get('backbone','resnet50'),strategy=args.strategy,temperature=args.temperature,seed=args.seed,
             training_patient_keys=c['training_patient_keys'],selection_patient_keys=c['selection_patient_keys'],generated_image_ids=frame.image_id.tolist()))
    pd.DataFrame(metrics).to_csv(out/'trajectory_metrics.csv',index=False)

def merge_predictions(args):
    result={};artifacts=[]
    for filename in args.inputs:
        path=Path(filename);new=json.loads(path.read_text())
        if set(result)&set(new):raise ValueError('Duplicate generated image IDs')
        result.update(new);meta=json.loads(path.with_suffix('.provenance.json').read_text())
        artifacts.extend(meta.get('artifacts',[meta]))
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    save_json(out,result);save_json(out.with_suffix('.provenance.json'),{'artifacts':artifacts})

def compare(args):
    a=pd.read_csv(args.first);b=pd.read_csv(args.second)
    if set(a.image_id)!=set(b.image_id):raise ValueError('Paired comparison needs identical image IDs')
    joined=a.merge(b[['image_id','label','patient_key']],on='image_id',suffixes=('','_other'))
    assert (joined.label==joined.label_other).all() and (joined.patient_key==joined.patient_key_other).all()
    result=dict(first=classification_metrics(a.label,a.probability),second=classification_metrics(b.label,b.probability),paired_auroc_difference_ci=cluster_bootstrap(a,b,args.bootstrap,args.seed))
    save_json(args.output,result);print(json.dumps(result,indent=2))

def parser():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ['extract','train-classifier','evaluate','train-gaze','generate','compare','merge-predictions']:
        q=sub.add_parser(name);q.add_argument('--output',required=True);q.add_argument('--seed',type=int,default=42)
        q.add_argument('--manifest',default=str(ROOT/'manifests/classification.csv'));q.add_argument('--device',default='cuda:0')
        q.add_argument('--threads',type=int,default=4);q.add_argument('--batch-size',type=int,default=8);q.add_argument('--workers',type=int,default=0)
        q.add_argument('--limit',type=int,default=0);q.add_argument('--bootstrap',type=int,default=1000)
        if name=='extract':
            q.add_argument('--method',choices=['random','grid','observed','trajectory','heatmap','whole'],required=True)
            q.add_argument('--k',type=int,default=16);q.add_argument('--crop-size',type=int,default=224);q.add_argument('--encode-batch',type=int,default=32)
            q.add_argument('--predictions');q.add_argument('--min-distance',type=float,default=32);q.add_argument('--include-test',action='store_true')
            q.add_argument('--keep-predicted-centers',action='store_true')
            q.add_argument('--generation-policy',choices=['none','breast','hard','soft'],default='none')
            q.add_argument('--generation-radius',type=float,default=112);q.add_argument('--generation-strategy',choices=['argmax','sample'],default='argmax')
        if name in ['train-classifier','evaluate']:q.add_argument('--features',required=True)
        if name in ['evaluate','generate']:q.add_argument('--checkpoint',required=True);q.add_argument('--final-test',action='store_true')
        if name=='evaluate':q.add_argument('--split',choices=['val','test'],default='val')
        if name in ['train-classifier','train-gaze']:
            q.add_argument('--epochs',type=int,default=50);q.add_argument('--patience',type=int,default=10);q.add_argument('--lr',type=float,default=1e-3)
        if name=='train-classifier':
            q.add_argument('--aggregation',choices=['mil','sequence','shuffled'],default='mil');q.add_argument('--specificity',type=float,default=.9)
        if name in ['train-gaze','generate']:q.add_argument('--holdout-fold',type=int)
        if name=='train-gaze':
            q.add_argument('--backbone',choices=list(BACKBONES),default='resnet50');q.add_argument('--cache-visual',action='store_true')
            q.add_argument('--decoder-dim',type=int,default=128);q.add_argument('--decoder-layers',type=int,default=2)
            q.add_argument('--decoder-heads',type=int,default=4);q.add_argument('--ff-multiplier',type=int,default=2)
            q.add_argument('--dropout',type=float,default=.1)
            q.add_argument('--finetune-backbone',action='store_true');q.add_argument('--backbone-lr',type=float,default=1e-5)
            q.add_argument('--init-checkpoint');q.add_argument('--amp',action='store_true');q.add_argument('--cache-images',action='store_true')
            q.add_argument('--accumulation',type=int,default=1)
            q.add_argument('--method',choices=['trajectory','heatmap'],default='trajectory');q.add_argument('--k',type=int,default=16)
            q.add_argument('--min-distance',type=float,default=32);q.add_argument('--grid-x',type=int,default=24);q.add_argument('--grid-y',type=int,default=48)
        if name=='generate':
            q.add_argument('--splits',default='val');q.add_argument('--with-gaze-evaluation',action='store_true')
            q.add_argument('--strategy',choices=['argmax','sample'],default='argmax');q.add_argument('--temperature',type=float,default=1.)
        if name=='compare':q.add_argument('--first',required=True);q.add_argument('--second',required=True)
        if name=='merge-predictions':q.add_argument('--inputs',nargs='+',required=True)
    return p

if __name__=='__main__':
    args=parser().parse_args()
    {'extract':extract,'train-classifier':train_classifier,'evaluate':evaluate_classifier,'train-gaze':train_gaze,'generate':generate,'compare':compare,'merge-predictions':merge_predictions}[args.command](args)
