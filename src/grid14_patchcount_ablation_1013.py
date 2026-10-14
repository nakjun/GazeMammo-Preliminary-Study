"""Label-free nested source selection with immutable Grid14 and mask padding."""
import torch
import numpy as np
from scripts.train_gaze_coverage_quota_1012 import grid_coverage

KS=(0,1,2,4,8,16)

def nested_sequence(scores,coordinates,valid):
 grid=grid_coverage(coordinates[:16],14)
 xy=torch.round(coordinates.double()*torch.tensor([1500.,3000.]))
 xy_np=xy.numpy();squared=((xy_np[:,None]-xy_np[None,:])**2).sum(-1)
 seen={tuple(xy[i].tolist()) for i in grid};order=sorted(range(16,32),key=lambda j:(-float(abs(scores[j])),j));source=[];relaxed=[]
 for distance in (96,0):
  for j in order:
   key=tuple(xy[j].tolist())
   if not bool(valid[j]) or key in seen:continue
   if distance and any(squared[j,q]<distance*distance for q in grid+source):continue
   source.append(j);seen.add(key);relaxed.append(distance==0)
 return torch.tensor(grid),torch.tensor(source,dtype=torch.long),torch.tensor(relaxed,dtype=torch.bool)

def pack_nested(t,source,score_record):
 from scripts.train_gaze_diagnosis_selector_1012 import build_candidates
 features,coords,mask=build_candidates(t,source);n=len(features);indices=torch.zeros(n,30,dtype=torch.long);available=torch.zeros(n,30,dtype=torch.bool);relaxed=torch.zeros(n,16,dtype=torch.bool);counts=[]
 for i in range(n):
  grid,seq,relax=nested_sequence(score_record['scores'][i],coords[i],mask[i]);indices[i,:14]=grid;available[i,:14]=True;indices[i,14:14+len(seq)]=seq;available[i,14:14+len(seq)]=True;relaxed[i,:len(seq)]=relax;counts.append(len(seq))
 rows=torch.arange(n)[:,None];f=features[rows,indices];c=coords[rows,indices]
 f=torch.where(available[...,None],f,torch.zeros_like(f));c=torch.where(available[...,None],c,torch.zeros_like(c))
 return (f,c,available),dict(indices=indices,available=available,relaxed=relaxed,source_available=torch.tensor(counts))

def slice_bag(bag,k):return tuple(x[:,:14+k] for x in bag)
