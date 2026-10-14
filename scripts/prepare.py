"""Create group-separated manifests and safe derived gaze/COCO artifacts."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import json
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split,StratifiedKFold
from src.data import ROOT,load_gazes,reduce_gaze

def prepare(args):
    out=ROOT/'manifests';out.mkdir(exist_ok=True)
    df=pd.read_csv(ROOT/'eda/image_inventory.csv')
    df['label']=df.image_label.map({'BENIGN':0,'MALIGNANT':1,'BENIGN+MALIGNANT':1}).fillna(-1).astype(int)
    df['patient_key']=df.inferred_group
    df['group_verified']=False
    df['group_evidence']='filename_only_unverified'
    meta=pd.read_csv(args.cbis_metadata)
    subjects=set(meta['Subject ID'])
    for idx,r in df[df.source=='CBIS-DDSM'].iterrows():
        prefix=re.sub(r'_(BENIGN|MALIGNANT)\.png$','',r.file_name)
        if prefix in subjects or any(s.startswith(prefix+'_') for s in subjects):
            df.loc[idx,'group_verified']=True
            df.loc[idx,'group_evidence']='TCIA_subject_ID_P_number'
        else:raise ValueError(f'CBIS subject not found in metadata: {r.file_name}')
    # INbreast CSV has removed Patient ID; filename hashes cannot prove patient independence.
    df['split']='quarantine'
    cbis=df[df.source=='CBIS-DDSM']
    train=cbis[cbis.filename_split=='Training'];test=cbis[cbis.filename_split=='Test']
    assert not set(train.patient_key)&set(test.patient_key),'Original split patient overlap'
    groups=train.groupby('patient_key').label.max()
    group_train,group_val=train_test_split(groups.index,test_size=args.val_fraction,random_state=args.seed,stratify=groups.values)
    df.loc[df.patient_key.isin(group_train),'split']='train'
    df.loc[df.patient_key.isin(group_val),'split']='val'
    df.loc[df.patient_key.isin(test.patient_key),'split']='test'
    df['crossfit_fold']=-1
    groups=df[df.split=='train'].groupby('patient_key').label.max()
    folds=StratifiedKFold(args.folds,shuffle=True,random_state=args.seed)
    for fold,(_,held) in enumerate(folds.split(groups.index,groups.values)):
        df.loc[(df.split=='train')&df.patient_key.isin(groups.index[held]),'crossfit_fold']=fold
    assert all(df.groupby('patient_key').split.nunique()==1)
    columns=['image_id','file_name','source','patient_key','group_verified','group_evidence','label','image_label','split','crossfit_fold','width','height','view','filename_split']
    df[columns].to_csv(out/'all.csv',index=False)
    df[df.label>=0][columns].to_csv(out/'classification.csv',index=False)
    df[(df.split.isin(['train','val','test']))&(df.label>=0)][columns].to_csv(out/'trajectory.csv',index=False)
    gazes=load_gazes();derived={}
    for _,r in df.iterrows():
        segments=reduce_gaze(gazes[r.file_name],r.width,r.height,args.k,args.min_distance)
        derived[str(r.image_id)]={'segments':[(p/[r.width,r.height]).tolist() for p in segments],
                                 'selection':'longest_contiguous_valid_segment'}
    (out/'gaze_sequences.json').write_text(json.dumps(derived))
    coco=json.loads((ROOT/'data/annotations/instances_all.json').read_text())
    dims={r['id']:(r['width'],r['height']) for r in coco['images']}
    for a in coco['annotations']:
        w,h=dims[a['image_id']];x,y,bw,bh=a['bbox'];x0,y0=max(0,x),max(0,y)
        x1,y1=min(w,x+bw),min(h,y+bh);a['bbox']=[x0,y0,max(0,x1-x0),max(0,y1-y0)];a['area']=a['bbox'][2]*a['bbox'][3]
    (out/'instances_clipped.json').write_text(json.dumps(coco))
    channel=[]
    for _,r in df.iterrows():
        with Image.open(ROOT/'data/image'/r.file_name) as im:
            a=np.asarray(im)
            equal=a.ndim==2 or (np.array_equal(a[:,:,0],a[:,:,1]) and np.array_equal(a[:,:,0],a[:,:,2]))
            channel.append({'image_id':r.image_id,'mode':im.mode,'channels_equal':equal})
    pd.DataFrame(channel).to_csv(out/'channel_audit.csv',index=False)
    audit=dict(seed=args.seed,k=args.k,min_distance=args.min_distance,split_counts=df.groupby(['source','split','label']).size().reset_index(name='n').to_dict('records'),
               cbis_metadata=str(Path(args.cbis_metadata).resolve()),cbis_metadata_sha256=hashlib.sha256(Path(args.cbis_metadata).read_bytes()).hexdigest(),
               cbis_group_verification='P number matched to TCIA Subject ID; diagnosis labels remain COCO-derived',
               inbreast='Patient ID removed in local CSV and empty in sampled DICOMs; quarantined until patient mapping is provided',
               sequence_semantics='stored_order; author README says collected during bounding-box annotation; session merge and chronological order not verified',
               rgb_channels_differ=int(sum(not r['channels_equal'] for r in channel)),
               source_reference='https://github.com/YanKong0408/GAA-DETR#dataset')
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=42);p.add_argument('--k',type=int,default=16)
    p.add_argument('--min-distance',type=float,default=32);p.add_argument('--val-fraction',type=float,default=.2)
    p.add_argument('--folds',type=int,default=3)
    p.add_argument('--cbis-metadata',default='/data/nas/Breast/Public_Mammo/CBIS-DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ/metadata.csv')
    prepare(p.parse_args())
