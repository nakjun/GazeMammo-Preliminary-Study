"""Train released HAT/ChestSearch decoders on a disclosed breast adaptation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,json,time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.data import ROOT
from src.reference_gaze import ReferenceGaze
from src.spatial import gaussian_targets

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['hat','chestsearch'],required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--epochs',type=int,default=30);p.add_argument('--patience',type=int,default=8)
    p.add_argument('--batch-size',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5);p.add_argument('--sigma',type=float,default=96)
    p.add_argument('--backbone-mode',choices=['pixel','last','all'],default='pixel');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(a.seed);np.random.seed(a.seed)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'best.pt').exists():raise FileExistsError('Use a fresh output directory')
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    payload=torch.load(ROOT/'runs/spatial_v2_cache/inputs.pt',map_location='cpu',weights_only=False)
    frame=pd.read_csv(ROOT/'manifests/all.csv');tr=frame[frame.split=='train'].sort_values('image_id');va=frame[(frame.split=='val')&(frame.label>=0)].sort_values('image_id')
    assert not set(tr.patient_key)&set(va.patient_key)
    train=DataLoader([payload['items'][int(i)] for i in tr.image_id],batch_size=a.batch_size,shuffle=True)
    valid=DataLoader([payload['items'][int(i)] for i in va.image_id],batch_size=a.batch_size)
    device=torch.device(a.device);model=ReferenceGaze(family=a.family).to(device);model.set_backbone(a.backbone_mode)
    optimizer=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=a.lr,weight_decay=1e-4)
    history=[];best=float('inf');stale=0;iterations=0
    for epoch in range(a.epochs):
        model.train();losses=[]
        for b in train:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                memory=model.maps(b['image'].to(device));coords=b['coords'].to(device);mask=b['mask'].to(device)
                batch,t,_=coords.shape
                features=tuple(x[:,None].expand(batch,t,*x.shape[1:]).reshape(batch*t,*x.shape[1:]) for x in memory['features'])
                previous=coords[:,None,:-1].expand(batch,t,t-1,2).reshape(batch*t,t-1,2)
                padding=(torch.arange(t-1,device=device)[None]>=torch.arange(t,device=device)[:,None])[None].expand(batch,-1,-1).reshape(batch*t,t-1)
                logits=model.prefix_logits({'features':features},previous,padding,return_dense=True).float()
                h,w=logits.shape[-2:];xy=coords.reshape(-1,2)
                cell=(xy*xy.new_tensor([w,h])).floor();centers=(cell+.5)/xy.new_tensor([w,h])
                target=gaussian_targets(centers,grid=(w,h),sigma_pixels=a.sigma).reshape(-1,h,w);target=target/target.amax((-1,-2),keepdim=True)
                probability=logits.sigmoid();positive=target==1
                value=-nn.functional.logsigmoid(logits)*(1-probability).square()*positive
                value-=nn.functional.logsigmoid(-logits)*probability.square()*(1-target).pow(4)*(~positive)
                value=value.flatten(1).sum(1)/positive.flatten(1).sum(1).clamp_min(1)
                loss=value[mask.flatten()].mean()
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(float(loss.detach()));iterations+=1
        model.eval();scores=[]
        with torch.inference_mode():
            for b in valid:
                coords=b['coords'].to(device);mask=b['mask'].to(device);logits=model(b['image'].to(device),coords)
                gx,gy=model.grid;ids=(coords[...,1]*gy).long().clamp(0,gy-1)*gx+(coords[...,0]*gx).long().clamp(0,gx-1)
                nll=-logits.gather(-1,ids[...,None]).squeeze(-1);scores.extend(((nll*mask).sum(1)/mask.sum(1)).cpu().tolist())
        score=float(np.mean(scores));r=dict(epoch=epoch+1,iterations=iterations,train_focal_loss=float(np.mean(losses)),validation_raw_nll=score)
        history.append(r);(out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(r),flush=True)
        if score<best:
            best=score;stale=0
            torch.save(dict(kind='reference_gaze',state=model.state_dict(),model_config=model.model_config,epoch=epoch+1,iterations=iterations,args=vars(a),
                training_patient_keys=sorted(tr.patient_key.unique()),selection_patient_keys=sorted(va.patient_key.unique()),adaptation='breast image-only, no duration, BOS initial fixation, common scoring grid'),out/'best.pt')
        else:stale+=1
        if stale>=a.patience:break
    (out/'summary.json').write_text(json.dumps(dict(best_validation_nll=best,iterations=iterations,test_used=False,
        note='Breast adaptation; pilot schedule is shorter than published 30000-iteration ChestSearch schedule'),indent=2))
