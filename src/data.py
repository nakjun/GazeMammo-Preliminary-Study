import ast
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
MEAN = torch.tensor([.485,.456,.406])[:,None,None]
STD = torch.tensor([.229,.224,.225])[:,None,None]

def seed_for(seed, image_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{image_id}'.encode()).digest()[:4], 'little')

@lru_cache(maxsize=1)
def load_gazes():
    result = {}
    for line in (ROOT/'data/gaze_data/merged_gaze_data.csv').read_text().splitlines():
        name, payload = line.split(';',1)
        samples = ast.literal_eval(payload)
        result[name] = np.array([np.asarray(s[0],dtype=float)/float(s[1]) for s in samples])
    return result

def reduce_gaze(points, width, height, k=16, min_distance=32.):
    """Preserve temporal segments; never create transitions across invalid samples."""
    segments, current, last = [], [], None
    for p in np.asarray(points):
        valid = np.isfinite(p).all() and 0<=p[0]<width and 0<=p[1]<height
        if not valid:
            if current: segments.append(np.array(current))
            current, last = [], None
        elif last is None or np.linalg.norm(p-last)>=min_distance:
            current.append(p);last=p
    if current:segments.append(np.array(current))
    reduced=[]
    for p in segments:
        if len(p)>k:p=p[np.linspace(0,len(p)-1,k).round().astype(int)]
        reduced.append(p)
    return reduced

def tissue_candidates(image, step=32):
    """Image-only foreground centers; shared by every selector."""
    a=np.asarray(image.convert('L'))
    ys,xs=np.mgrid[0:image.height:step,0:image.width:step]
    mask=a[ys,xs]>10
    candidates=np.stack([xs[mask],ys[mask]],axis=1).astype(float)
    if not len(candidates):raise ValueError('Empty foreground mask')
    return candidates

def select_centers(image, method, k, seed, points=None, prediction=None, min_distance=32):
    candidates=tissue_candidates(image)
    rng=np.random.default_rng(seed)
    if method=='random':
        p=candidates[rng.choice(len(candidates),min(k,len(candidates)),replace=False)]
    elif method=='grid':
        # Deterministic spatial coverage of the same foreground candidates.
        chosen=[int(np.argmin(((candidates-np.array([image.width/2,image.height/2]))**2).sum(1)))]
        distance=((candidates-candidates[chosen[0]])**2).sum(1)
        for _ in range(1,min(k,len(candidates))):
            idx=int(distance.argmax());chosen.append(idx)
            distance=np.minimum(distance,((candidates-candidates[idx])**2).sum(1))
        p=candidates[chosen]
    elif method=='observed':
        segments=reduce_gaze(points,image.width,image.height,k,min_distance)
        if not segments:raise ValueError('No valid gaze samples')
        # The longest contiguous valid segment avoids joining recording gaps.
        p=max(segments,key=len)
    elif method in ('trajectory','heatmap'):
        if prediction is None:raise ValueError('Prediction file required')
        p=np.asarray(prediction,dtype=float)*[image.width,image.height]
    else:raise ValueError(method)
    a=np.asarray(image.convert('L'))
    out=[]
    for center in p:
        center=np.clip(center,[0,0],[image.width-1,image.height-1])
        if a[int(center[1]),int(center[0])]<=10:
            center=candidates[np.argmin(((candidates-center)**2).sum(1))]
        out.append(center)
    p=np.array(out[:k]).reshape(-1,2)
    if not len(p):raise ValueError('No centers')
    return p

def patch_tensor(image, center, crop_size=224):
    x,y=np.round(center).astype(int);half=crop_size//2
    patch=image.crop((x-half,y-half,x-half+crop_size,y-half+crop_size)).resize((224,224),Image.Resampling.BILINEAR)
    a=np.asarray(patch.convert('L')).copy()
    x=torch.from_numpy(a).float()[None].repeat(3,1,1)/255
    return (x-MEAN)/STD

def whole_tensor(image,size=(256,512)):
    # Original images share 1:2 aspect ratio. No stretching.
    image=image.convert('L').resize(size,Image.Resampling.BILINEAR)
    x=torch.from_numpy(np.asarray(image).copy()).float()[None].repeat(3,1,1)/255
    return (x-MEAN)/STD

class PatchDataset(Dataset):
    def __init__(self, frame, method, k=16, crop_size=224, seed=42, predictions=None, min_distance=32,keep_predicted_centers=False):
        self.frame=frame.reset_index(drop=True);self.method=method;self.k=k
        self.crop_size=crop_size;self.seed=seed;self.min_distance=min_distance
        self.keep_predicted_centers=keep_predicted_centers
        self.gazes=load_gazes() if method=='observed' else {}
        self.predictions=json.loads(Path(predictions).read_text()) if predictions else {}
    def __len__(self):return len(self.frame)
    def __getitem__(self,idx):
        row=self.frame.iloc[idx]
        with Image.open(ROOT/'data/image'/row.file_name) as source:
            image=source.convert('L')
            if self.method=='whole':
                patches=whole_tensor(image)[None];mask=torch.ones(1,dtype=torch.bool);centers=torch.zeros(1,2)
            else:
                if self.keep_predicted_centers and self.method in ('trajectory','heatmap'):
                    p=np.asarray(self.predictions[str(row.image_id)],dtype=float)*[image.width,image.height]
                    if len(p)>self.k or not np.isfinite(p).all() or (p<0).any() or (p>=[image.width,image.height]).any():raise ValueError('Invalid predicted patch centers')
                else:p=select_centers(image,self.method,self.k,seed_for(self.seed,row.image_id),self.gazes.get(row.file_name),self.predictions.get(str(row.image_id)),self.min_distance)
                patches=torch.zeros(self.k,3,224,224);mask=torch.zeros(self.k,dtype=torch.bool)
                for i,c in enumerate(p):patches[i]=patch_tensor(image,c,self.crop_size);mask[i]=True
                centers=torch.zeros(self.k,2);centers[:len(p)]=torch.tensor(p/[image.width,image.height],dtype=torch.float32)
        return dict(patches=patches,mask=mask,centers=centers,label=torch.tensor(float(row.label)),image_id=int(row.image_id))

class TrajectoryDataset(Dataset):
    def __init__(self,frame,k=16,min_distance=32,grid=(24,48)):
        self.frame=frame.reset_index(drop=True);self.k=k;self.grid=grid;self.min_distance=min_distance
        self.gazes=load_gazes()
    def __len__(self):return len(self.frame)
    def __getitem__(self,idx):
        row=self.frame.iloc[idx]
        with Image.open(ROOT/'data/image'/row.file_name) as source:
            image=source.convert('L');whole=whole_tensor(image)
            segments=reduce_gaze(self.gazes[row.file_name],image.width,image.height,self.k,self.min_distance)
            p=max(segments,key=len)/[image.width,image.height]
        gx,gy=self.grid;cells=(np.minimum((p[:,1]*gy).astype(int),gy-1)*gx+np.minimum((p[:,0]*gx).astype(int),gx-1))
        target=torch.full((self.k,),-100,dtype=torch.long);target[:len(cells)]=torch.tensor(cells)
        coords=torch.zeros(self.k,2);coords[:len(p)]=torch.tensor(p,dtype=torch.float32)
        return dict(image=whole,target=target,coords=coords,image_id=int(row.image_id))

class ImageDataset(Dataset):
    """Inference uses only image pixels and image IDs, with no gaze dependency."""
    def __init__(self,frame):self.frame=frame.reset_index(drop=True)
    def __len__(self):return len(self.frame)
    def __getitem__(self,idx):
        row=self.frame.iloc[idx]
        with Image.open(ROOT/'data/image'/row.file_name) as image:tensor=whole_tensor(image)
        return dict(image=tensor,image_id=int(row.image_id))
