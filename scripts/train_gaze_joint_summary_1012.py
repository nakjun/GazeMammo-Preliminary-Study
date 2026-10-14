"""Independent train-only normalized joint frozen-feature classifier, no TEST."""
import argparse,copy,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.official_cbis_classifier_1012 import metrics,digest,paired_ci
from scripts.train_gaze_contextual_improvement_1012 import build_bags

def summarize(global_features,patches):
    return torch.cat((global_features,patches.mean(1),patches.max(1).values),dim=1)

def normalize(x,train):
    mean=x[train].mean(0);std=x[train].std(0).clamp_min(.01)
    return ((x-mean)/std).clamp(-10,10),mean,std

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);p.add_argument('--seeds',nargs='+',type=int,default=[42]);p.add_argument('--configs',nargs='+',default=['linear01','linear1','mlp']);p.add_argument('--arms',nargs='+',default=['grid','gaze_baseline','random','grid8_gaze_baseline8']);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    out=Path(a.output);out.mkdir(exist_ok=False);cache=torch.load(a.cache,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors']
    assert cache['provenance']['split']=='trainval' and set(f.split)=={'train','val'} and len(f)==2393 and (f.split=='val').sum()==356
    assert f.groupby('patient_key').split.nunique().max()==1
    tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32)
    bags=build_bags(t);bags['random']=(t['random_features'],t['random_coordinates'])
    m=dict(state='running',stage='seed42_pilot' if a.seeds==[42] else 'three_seed_confirmation',cache_sha256=digest(a.cache),source_sha256=digest(__file__),patch_budget=16,normalization='train-only per-dimension mean/std, std floor .01, clamp 10',summary='global + patch mean + patch max',test_used=False,backbone='frozen ConvNeXtBase cached features',seeds=a.seeds,configs=a.configs,arms=a.arms,epochs=80,patience=15,batch_size=64,lr=.0003,results=[])
    def save(): (out/'manifest.json').write_text(json.dumps(m,indent=2))
    save();preds={}
    for arm in a.arms:
        x,mean,std=normalize(summarize(t['global_features'],bags[arm][0]),tr)
        for cfg in a.configs:
            for seed in a.seeds:
                torch.manual_seed(seed);d=out/f'{arm}_{cfg}_seed{seed}';d.mkdir()
                model=torch.nn.Linear(x.shape[1],1) if cfg.startswith('linear') else torch.nn.Sequential(torch.nn.Linear(x.shape[1],64),torch.nn.GELU(),torch.nn.Dropout(.5),torch.nn.Linear(64,1))
                opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1 if cfg=='linear1' else .01)
                history=[];best=-1;stale=0
                for epoch in range(81):
                    losses=[]
                    if epoch:
                        model.train();order=tr[torch.randperm(len(tr))]
                        for ids in order.split(64):
                            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[ids]).flatten(),y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
                    model.eval()
                    with torch.inference_mode(): prob=model(x[va]).flatten().sigmoid().numpy()
                    score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,validation=score))
                    if score['auroc']>best:
                        best=score['auroc'];stale=0;bp=prob.copy();be=epoch;bs=copy.deepcopy(model.state_dict())
                    else:stale+=1
                    if stale>=15:break
                torch.save(dict(state=bs,mean=mean,std=std,config=cfg,arm=arm,seed=seed,epoch=be),d/'best.pt')
                pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);(d/'history.json').write_text(json.dumps(history));preds[(arm,cfg,seed)]=pred
                result=dict(arm=arm,config=cfg,seed=seed,epoch=be,metrics=metrics(pred.label,pred.probability));m['results'].append(result);save();print(json.dumps(result),flush=True)
    report=dict(results=m['results'],means={},paired={})
    for arm in a.arms:
        for cfg in a.configs:
            report['means'][arm+'_'+cfg]=float(np.mean([r['metrics']['auroc'] for r in m['results'] if r['arm']==arm and r['config']==cfg]))
            for seed in a.seeds:
                if ('grid',cfg,seed) in preds:report['paired'][f'{arm}_{cfg}_{seed}_minus_grid']=paired_ci(preds[(arm,cfg,seed)],preds[('grid',cfg,seed)],1000)
    (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(a.cache)==m['cache_sha256'];assert digest(__file__)==m['source_sha256'];m['state']='complete';save()
if __name__=='__main__':main()
