"""Label-free nested coverage additions, retaining all original predicted gaze16."""
import torch

def additional_grid_sequence(gaze,grid,gaze_mask,grid_mask,max_k=4):
 scale=torch.tensor([1500.,3000.],dtype=torch.float64);gx=torch.round(gaze.double()*scale);xy=torch.round(grid.double()*scale);anchors=[p for p,m in zip(gx,gaze_mask) if bool(m)];seen={tuple(p.tolist()) for p in anchors};available=[i for i in range(len(grid)) if bool(grid_mask[i]) and tuple(xy[i].tolist()) not in seen];chosen=[]
 while available and len(chosen)<max_k:
  distances=torch.cdist(xy[available],torch.stack(anchors)).min(1).values if anchors else torch.ones(len(available));position=int(distances.argmax());j=available.pop(position);chosen.append(j);anchors.append(xy[j]);seen.add(tuple(xy[j].tolist()));available=[i for i in available if tuple(xy[i].tolist()) not in seen]
 return torch.tensor(chosen,dtype=torch.long)

def build_bags(t,ks=(0,1,2,4)):
 n=len(t['global_features']);indices=torch.zeros(n,4,dtype=torch.long);valid=torch.zeros(n,4,dtype=torch.bool)
 for i in range(n):
  chosen=additional_grid_sequence(t['gaze_baseline_coordinates'][i],t['grid_coordinates'][i],t['gaze_baseline_mask'][i],t['grid_mask'][i]);indices[i,:len(chosen)]=chosen;valid[i,:len(chosen)]=True
 rows=torch.arange(n)[:,None];extra=t['grid_features'][rows,indices];coords=t['grid_coordinates'][rows,indices];extra=torch.where(valid[...,None],extra,torch.zeros_like(extra));coords=torch.where(valid[...,None],coords,torch.zeros_like(coords));result={}
 for k in ks:result[k]=(torch.cat([t['gaze_baseline_features'],extra[:,:k]],1),torch.cat([t['gaze_baseline_coordinates'],coords[:,:k]],1),torch.cat([t['gaze_baseline_mask'],valid[:,:k]],1))
 return result,dict(indices=indices,valid=valid)
