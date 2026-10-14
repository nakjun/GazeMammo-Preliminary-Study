"""Independent exploratory TRAIN/VAL study; fixed backbone and 16-patch budget."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.official_cbis_classifier_1012 import restore,metrics,digest,state_digest,paired_ci
from src.gaze_attention_1011 import GazeAttention1011,attention_loss,head_state


def diverse_indices(coords,count,anchors=None):
    """Farthest-first in physical canvas distance; tie uses cached order, no labels."""
    xy=coords*torch.tensor([1500.,3000.])
    chosen=[]
    reference=[] if anchors is None else list(anchors*torch.tensor([1500.,3000.]))
    if not reference:
        # Start nearest the coordinate centroid, then maximize spatial coverage.
        first=int(((xy-xy.mean(0))**2).sum(-1).argmin());chosen.append(first);reference.append(xy[first])
    while len(chosen)<count:
        distances=torch.cdist(xy,torch.stack(reference)).min(1).values
        if chosen: distances[chosen]=-1
        nxt=int(distances.argmax());chosen.append(nxt);reference.append(xy[nxt])
    return torch.tensor(chosen)


def build_bags(t):
    bags={name:(t[name+'_features'],t[name+'_coordinates']) for name in ('grid','gaze_baseline')}
    for source in ('random','gaze_baseline','gaze_improved'):
        fs=[];cs=[]
        for i in range(len(t['global_features'])):
            gi=diverse_indices(t['grid_coordinates'][i],8)
            gc=t['grid_coordinates'][i,gi]
            si=diverse_indices(t[source+'_coordinates'][i],8,gc)
            fs.append(torch.cat([t['grid_features'][i,gi],t[source+'_features'][i,si]]))
            cs.append(torch.cat([gc,t[source+'_coordinates'][i,si]]))
        bags['grid8_'+source+'8']=(torch.stack(fs),torch.stack(cs))
    return bags


def diagnostics(bags,frame):
    result={}
    for arm,(_,coords) in bags.items():
        unique=[];coverage=[]
        for c in coords:
            unique.append(len(torch.unique(c,dim=0)))
            d=torch.cdist(c*torch.tensor([1500.,3000.]),c*torch.tensor([1500.,3000.]))
            d.fill_diagonal_(float('inf'));coverage.append(float(d.min(1).values.mean()))
        result[arm]={split:dict(unique_patch_mean=float(np.mean(np.array(unique)[frame.split==split])),nearest_center_distance_px=float(np.mean(np.array(coverage)[frame.split==split]))) for split in ('train','val')}
    return result


def fit(t,frame,bag,arm,seed,global_ck,out,epochs,patience):
    torch.manual_seed(seed)
    model=GazeAttention1011(restore(global_ck),variant='mean_residual',input_dim=1024,dim=64,dropout=.2)
    frozen=state_digest(model.global_model.state_dict());features,coords=bag
    train=torch.tensor(np.flatnonzero(frame.split=='train'));val=torch.tensor(np.flatnonzero(frame.split=='val'))
    y=torch.tensor(frame.label.to_numpy(),dtype=torch.float32)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    out.mkdir();best=-1;stale=0;history=[]
    def forward(ids):return model(t['global_features'][ids],features[ids],coords[ids])
    for epoch in range(epochs+1):
        losses=[]
        if epoch:
            model.train();order=train[torch.randperm(len(train),generator=torch.Generator().manual_seed(seed+epoch))]
            for ids in order.split(32):
                opt.zero_grad();loss=attention_loss(forward(ids),y[ids])[0];loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():p=torch.cat([forward(ids)['logits'].sigmoid() for ids in val.split(32)]).numpy()
        score=metrics(y[val].numpy(),p);history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)) if losses else None,validation=score))
        if score['auroc']>best:
            best=score['auroc'];stale=0;bestp=p.copy();ck=dict(kind='gaze_coverage_exploratory_1012',arm=arm,seed=seed,epoch=epoch,state=copy.deepcopy(head_state(model)),global_state_sha256=frozen,model_config=model.config,patch_budget=16,threshold=.5)
            torch.save(ck,out/'best.pt')
        elif epoch:stale+=1
        if state_digest(model.global_model.state_dict())!=frozen:raise RuntimeError('Frozen global changed')
        print(json.dumps(dict(arm=arm,seed=seed,epoch=epoch,auroc=score['auroc'])),flush=True)
        if stale>=patience:break
    pred=frame.iloc[val.tolist()][['image_id','patient_key','label']].reset_index(drop=True);pred['probability']=bestp
    pred.to_csv(out/'validation_predictions.csv',index=False);(out/'history.json').write_text(json.dumps(history))
    return pred,dict(arm=arm,seed=seed,epoch=ck['epoch'],metrics=metrics(pred.label,pred.probability))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--heads',required=True);ap.add_argument('--output',required=True);ap.add_argument('--seeds',type=int,nargs='+',default=[42]);ap.add_argument('--epochs',type=int,default=60);ap.add_argument('--patience',type=int,default=12);ap.add_argument('--selected',nargs='+');a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    output=Path(a.output);output.mkdir(exist_ok=False)
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);frame=pd.DataFrame(cache['rows']);t=cache['tensors']
    assert cache['kind']=='official_cbis_features_1012' and cache['provenance']['split']=='trainval'
    assert len(frame)==2393 and (frame.split=='val').sum()==356 and set(frame.split)=={'train','val'}
    assert not frame.image_id.duplicated().any() and frame.groupby('patient_key').split.nunique().max()==1
    assert set(frame.label)=={0,1}
    bags=build_bags(t);(output/'coverage_diagnostics.json').write_text(json.dumps(diagnostics(bags,frame),indent=2))
    manifest=dict(state='running',cache=dict(path=str(Path(a.cache).resolve()),sha256=digest(a.cache)),source_sha256=digest(__file__),seeds=a.seeds,arms=a.selected or list(bags),patch_budget=16,threads=4,epochs=a.epochs,patience=a.patience,lr=.0003,weight_decay=.01,selection='validation AUROC, epoch zero included',test_used=False,teacher_oracle_used=False,limitations='Reused exploratory validation; in-sample gaze transfer; no new gaze training or teacher oracle.',selection_algorithm='Grid8: centroid then farthest-first physical coordinates. Other8: farthest-first from Grid8 and already selected source points. Cached order breaks ties; duplicates corrected by original head. No labels used in patch selection.',global_heads=[])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2));results=[];preds={}
    for seed in a.seeds:
        path=Path(a.heads)/f'global_seed{seed}'/'best.pt';global_ck=torch.load(path,map_location='cpu',weights_only=False);manifest['global_heads'].append(dict(path=str(path.resolve()),sha256=digest(path)))
        for arm in a.selected or bags:
            pred,score=fit(t,frame,bags[arm],arm,seed,global_ck,output/f'{arm}_seed{seed}',a.epochs,a.patience);results.append(score);preds[(arm,seed)]=pred
    report=dict(seed_results=results,paired={})
    for seed in a.seeds:
        for arm in a.selected or bags:
            ref=preds.get(('grid',seed))
            if ref is None:ref=pd.read_csv(Path(a.heads)/f'grid_seed{seed}'/'validation_predictions.csv')
            report['paired'][f'{arm}_seed{seed}_minus_grid']=paired_ci(preds[(arm,seed)],ref,1000)
            control=preds.get(('grid8_random8',seed))
            if control is not None and arm.startswith('grid8_gaze'):
                report['paired'][f'{arm}_seed{seed}_minus_grid8_random8']=paired_ci(preds[(arm,seed)],control,1000)
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False));manifest['state']='complete';(output/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
