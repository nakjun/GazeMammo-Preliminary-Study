"""Bag-supervised logsumexp MIL on pinned ConvNeXtBase cached features."""
import argparse,copy,json,sys,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.official_cbis_classifier_1012 import metrics,digest,paired_ci
from scripts.train_gaze_contextual_improvement_1012 import build_bags

class EvidenceMIL(torch.nn.Module):
    def __init__(self,dim=1024,temperature=.5):
        super().__init__();self.global_head=torch.nn.Linear(dim,1);self.patch_head=torch.nn.Linear(dim,1);self.dropout=torch.nn.Dropout(.2);self.temperature=temperature
    def forward(self,g,p):
        evidence=self.patch_head(self.dropout(p)).squeeze(-1)
        pooled=self.temperature*(torch.logsumexp(evidence/self.temperature,dim=1)-math.log(p.shape[1]))
        return self.global_head(self.dropout(g)).squeeze(-1)+pooled

def fit_normalization(g,p,train):
    gm=g[train].mean(0);gs=g[train].std(0).clamp_min(.01)
    pm=p[train].mean((0,1));ps=p[train].reshape(-1,p.shape[-1]).std(0).clamp_min(.01)
    return ((g-gm)/gs).clamp(-10,10),((p-pm)/ps).clamp(-10,10),dict(global_mean=gm,global_std=gs,patch_mean=pm,patch_std=ps)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',required=True);ap.add_argument('--output',required=True);ap.add_argument('--temperatures',nargs='+',type=float,default=[.5,2.]);ap.add_argument('--seeds',nargs='+',type=int,default=[42]);ap.add_argument('--arms',nargs='+',default=['grid','gaze_baseline','gaze_improved','grid8_gaze_baseline8','grid8_gaze_improved8','grid8_random8']);a=ap.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path(a.output);out.mkdir(exist_ok=False)
    cache=torch.load(a.cache,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);t=cache['tensors'];assert cache['provenance']['split']=='trainval' and len(f)==2393 and (f.split=='val').sum()==356 and set(f.split)=={'train','val'};assert f.groupby('patient_key').split.nunique().max()==1
    tr=torch.tensor(np.flatnonzero(f.split=='train'));va=torch.tensor(np.flatnonzero(f.split=='val'));y=torch.tensor(f.label.values,dtype=torch.float32);bags=build_bags(t)
    dependencies=['scripts/train_gaze_contextual_improvement_1012.py','src/official_cbis_classifier_1012.py']
    manifest=dict(state='running',stage='pilot42' if a.seeds==[42] else 'confirmation3seed',cache_path=str(Path(a.cache).resolve()),cache_sha256=digest(a.cache),source_sha256=digest(__file__),dependencies={p:digest(p) for p in dependencies},hypothesis='Sparse discriminative patch evidence diluted by feature means; bag-level MIL logsumexp may recover it.',loss='bag BCE only',backbone='frozen ConvNeXtBase',patch_budget=16,test_used=False,normalization='train-only global/patch moments; std floor .01; clamp10',temperatures=a.temperatures,seeds=a.seeds,arms=a.arms,lr=.0003,weight_decay=.01,dropout=.2,batch_size=64,epochs=80,patience=15,results=[])
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    save();preds={}
    for arm in a.arms:
        g,p,norm=fit_normalization(t['global_features'],bags[arm][0],tr)
        for temp in a.temperatures:
            for seed in a.seeds:
                torch.manual_seed(seed);d=out/f'{arm}_temp{temp}_seed{seed}';d.mkdir();model=EvidenceMIL(temperature=temp);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.01);best=-1;stale=0;history=[]
                for epoch in range(81):
                    losses=[]
                    if epoch:
                        model.train();order=tr[torch.randperm(len(tr))]
                        for ids in order.split(64):
                            opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(g[ids],p[ids]),y[ids]);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(float(loss.detach()))
                    model.eval()
                    with torch.inference_mode():prob=model(g[va],p[va]).sigmoid().numpy()
                    score=metrics(y[va].numpy(),prob);history.append(dict(epoch=epoch,loss=float(np.mean(losses)) if losses else None,validation=score))
                    if score['auroc']>best:best=score['auroc'];stale=0;bp=prob.copy();be=epoch;bs=copy.deepcopy(model.state_dict())
                    else:stale+=1
                    if stale>=15:break
                torch.save(dict(state=bs,normalization=norm,temperature=temp,arm=arm,seed=seed,epoch=be),d/'best.pt');pred=f.iloc[va.tolist()][['image_id','patient_key','label']].copy();pred['probability']=bp;pred.to_csv(d/'validation_predictions.csv',index=False);(d/'history.json').write_text(json.dumps(history));preds[(arm,temp,seed)]=pred
                r=dict(arm=arm,temperature=temp,seed=seed,epoch=be,metrics=metrics(pred.label,pred.probability));manifest['results'].append(r);save();print(json.dumps(r),flush=True)
    report=dict(results=manifest['results'],paired={})
    for (arm,temp,seed),pred in preds.items():
        ref=preds.get(('grid',temp,seed));control=preds.get(('grid8_random8',temp,seed))
        if ref is not None and arm!='grid':report['paired'][f'{arm}_{temp}_{seed}_minus_grid']=paired_ci(pred,ref,1000)
        if control is not None and arm.startswith('grid8_gaze'):report['paired'][f'{arm}_{temp}_{seed}_minus_matchedrandom']=paired_ci(pred,control,1000)
    (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(a.cache)==manifest['cache_sha256'];assert digest(__file__)==manifest['source_sha256'];assert all(digest(p)==h for p,h in manifest['dependencies'].items());manifest['state']='complete';save()
if __name__=='__main__':
    try:main()
    except Exception as e:
        if '--output' in sys.argv:
            p=Path(sys.argv[sys.argv.index('--output')+1]);
            if p.exists():(p/'failure.json').write_text(json.dumps(dict(state='failed',error=repr(e))))
        raise
