"""Cache fresh frozen visual features; train two prespecified causal heads."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.official_cbis_data_1012 import transform_points, validate_protocol
from src.official_cbis_trajectory_1012 import *

def cache_features(a):
    manifest=Path(a.manifest);frame=pd.read_csv(manifest)
    protocol=validate_protocol(Path(a.protocol).parent)
    if manifest.resolve()!=(Path(a.protocol).parent/'gaze.csv').resolve():raise ValueError('Requires protocol gaze manifest')
    if any(r.patient_key in set(protocol['test_patient_keys']) or protocol['patient_splits'].get(r.patient_key)!=r.split for r in frame.itertuples()):raise ValueError('Gaze protocol patient routing mismatch')
    if not set(frame.split)<= {'train','val'} or frame.file_name.duplicated().any():raise ValueError('Invalid train/val gaze allowlist')
    if a.limit:
        frame=pd.concat([g.head(max(1,a.limit//2)) for _,g in frame.groupby('split')]).reset_index(drop=True)
    train=set(frame[frame.split=='train'].patient_key);val=set(frame[frame.split=='val'].patient_key)
    if train&val:raise ValueError('Patient overlap')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    provenance=dict(kind='official_cbis_gaze_cache_1012',protocol_sha256=sha256(a.protocol),gaze_manifest_sha256=sha256(manifest),
      original_pretrained_sha256=sha256(PRETRAINED),training_patient_keys=sorted(train),selection_patient_keys=sorted(val),
      smoke=bool(a.limit),runtime_flags=dict(threads=4,cudnn_enabled=True,visual_tf32=True,matmul_tf32=False),source_hashes={str(p):sha256(p) for p in (Path(__file__),ROOT/'src/official_cbis_data_1012.py',ROOT/'src/gaze_nll_adaptation_1011.py',ROOT/'src/trajectory_1011.py',ROOT/'src/spatial.py',ROOT/'src/gaze_backbones.py',ROOT/'src/data.py')},gaze_source_sha256={p:sha256(p) for p in sorted(set(frame.gaze_csv_path))},source_sha256=sha256(ROOT/'src/official_cbis_trajectory_1012.py'),protocol_path=str(Path(a.protocol).resolve()),test_patient_keys=protocol['test_patient_keys'])
    state=out/'provenance.json'
    if state.exists() and json.loads(state.read_text())!=provenance:raise ValueError('Cache provenance changed')
    state.write_text(json.dumps(provenance,indent=2))
    payload={}
    for path,g in frame.groupby('gaze_csv_path'):payload.update(allowed_payloads(path,g.file_name))
    model=build_fresh_model(a.device).eval()
    for row in frame.itertuples():
        dest=out/f'{row.image_id}.pt'
        image_path=Path(row.gaze_canonical_path);affine_path=Path(str(image_path)+'.json')
        row_hash=sha256(image_path)+sha256(affine_path)+hashlib.sha256(json.dumps(row._asdict(),sort_keys=True).encode()).hexdigest()
        if dest.exists():
            saved=torch.load(dest,map_location='cpu',weights_only=False)
            if saved['row_hash']!=row_hash:raise ValueError('Cached row changed')
            continue
        image=Image.open(image_path).convert('L')
        if image.size!=(1500,3000):raise ValueError('Canonical image dimensions')
        parts=segments(transform_points(payload[row.file_name],json.loads(affine_path.read_text())),1500,3000,32)
        if not parts:raise ValueError('No valid chronological gaze segment')
        vectors=[]
        with torch.inference_mode():
            features=[x[0].cpu().float() for x in model.visual(whole_tensor(image)[None].to(a.device))]
            for part in parts:
                pieces=[]
                for start in range(0,len(part),a.crop_batch):
                    crops=torch.stack([patch_tensor(image,p,224) for p in part[start:start+a.crop_batch]]).to(a.device)
                    pieces.append(model.visual.patch_vector(crops).cpu().float())
                vectors.append(torch.cat(pieces))
        record=dict(image_id=int(row.image_id),patient_key=row.patient_key,split=row.split,row_hash=row_hash,
            parts=[torch.tensor(p/[1500,3000],dtype=torch.float32) for p in parts],vectors=vectors,features=features)
        tmp=dest.with_suffix('.tmp');torch.save(record,tmp);tmp.replace(dest)
        print(json.dumps(dict(cached=int(row.image_id),pool_points=sum(map(len,parts)))),flush=True)
    paths=[str(out/f'{i}.pt') for i in frame.image_id]
    index=dict(**provenance,records=paths,record_hashes={p:sha256(p) for p in paths},image_ids=[int(i) for i in frame.image_id],complete=True)
    torch.save(index,out/'index.pt')

def train(a):
    cache=torch.load(Path(a.cache)/'index.pt',map_location='cpu',weights_only=False)
    if not cache.get('complete'):raise ValueError('Incomplete cache')
    for p,h in cache['source_hashes'].items():
        if sha256(p)!=h:raise ValueError('Cache dependency source changed')
    for p,h in cache['record_hashes'].items():
        if sha256(p)!=h:raise ValueError('Cache record changed')
    if cache['source_sha256']!=sha256(ROOT/'src/official_cbis_trajectory_1012.py'):raise ValueError('Trajectory source changed after cache')
    if cache['original_pretrained_sha256']!=sha256(PRETRAINED):raise ValueError('Original visual hash changed')
    if sha256(cache['protocol_path'])!=cache['protocol_sha256']:raise ValueError('Protocol changed')
    protocol=validate_protocol(Path(cache['protocol_path']).parent)
    records=[torch.load(p,map_location='cpu',weights_only=False) for p in cache['records']]
    if any(r['patient_key'] in set(protocol['test_patient_keys']) or protocol['patient_splits'].get(r['patient_key'])!=r['split'] for r in records):raise ValueError('Cached patient routing mismatch')
    trainrows=[r for r in records if r['split']=='train'];valrows=[r for r in records if r['split']=='val']
    if [r['image_id'] for r in records]!=cache['image_ids'] or len(set(cache['image_ids']))!=len(records):raise ValueError('Cache image identity mismatch')
    if a.limit:trainrows=trainrows[:a.limit];valrows=valrows[:a.limit]
    cache=dict(cache,training_patient_keys=sorted({r['patient_key'] for r in trainrows}),selection_patient_keys=sorted({r['patient_key'] for r in valrows}))
    if a.max_epochs>40 or a.max_epochs<0:raise ValueError('max epochs must be 0..40')
    if not trainrows or not valrows:raise ValueError('Empty split')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    model=build_fresh_model(a.device)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    def batches(rows,epoch,training):
        order=torch.randperm(len(rows),generator=torch.Generator().manual_seed(42+epoch)).tolist() if training else range(len(rows))
        ordered=[rows[i] for i in order]
        for start in range(0,len(ordered),8):
            rs=ordered[start:start+8];inputs=[teacher_inputs(r,epoch,training and a.arm=='improved') for r in rs]
            coords,mask,vectors=[torch.stack([v[j] for v in inputs]).to(a.device) for j in range(3)]
            features=[torch.stack([r['features'][i] for r in rs]).to(a.device) for i in range(3)]
            logits=model.teacher_logits(model.maps_features(features),coords,vectors)
            loss=objective(logits,coords,mask,nll_weight=.8 if training and a.arm=='improved' else 1.,sigma=32)[0]
            yield loss,len(rs)
    best=float('inf');stale=0
    for epoch in range(a.max_epochs+1):
        model.train()
        if epoch:
            for loss,n in batches(trainrows,epoch,True):
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
        model.eval()
        with torch.inference_mode():values=[(float(loss),n) for loss,n in batches(valrows,0,False)]
        nll=sum(v*n for v,n in values)/sum(n for _,n in values)
        if not np.isfinite(nll):raise ValueError('Nonfinite validation NLL')
        improved=nll<best
        if improved:
            best=nll;stale=0;torch.save(checkpoint_payload(model,cache,epoch,nll,a.arm),out/'selected.pt')
        else:stale+=1
        line=dict(epoch=epoch,canonical_val_hard_nll=nll,best=best,selected=improved)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(line)+'\n')
        print(json.dumps(line),flush=True)
        if stale>=10:break

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('cache');c.add_argument('--manifest',required=True);c.add_argument('--protocol',required=True);c.add_argument('--output',required=True);c.add_argument('--limit',type=int,default=0);c.add_argument('--crop-batch',type=int,default=32)
    t=sub.add_parser('train');t.add_argument('--cache',required=True);t.add_argument('--output',required=True);t.add_argument('--arm',choices=['baseline','improved'],required=True);t.add_argument('--max-epochs',type=int,default=40);t.add_argument('--limit',type=int,default=0)
    for s in (c,t):s.add_argument('--device',default='cuda:0')
    a=p.parse_args();torch.set_num_threads(4);torch.backends.cudnn.enabled=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False;cache_features(a) if a.command=='cache' else train(a)
if __name__=='__main__':main()
