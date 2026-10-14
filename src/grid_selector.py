"""Deterministic farthest-point grid selector used by the best fusion setting."""
import numpy as np
from .constraints import breast_mask,grid_mask

def grid_points(image,k=16,grid=(24,48),min_distance=112):
    ids=np.arange(grid[0]*grid[1])
    normalized=np.stack([((ids%grid[0])+.5)/grid[0],((ids//grid[0])+.5)/grid[1]],-1)
    mask=breast_mask(image)
    size=np.asarray(image.size)
    candidates=normalized[grid_mask(mask,grid)]*size
    if len(candidates)<k:raise ValueError('Insufficient estimated breast grid candidates')
    index=int(((candidates-size/2)**2).sum(1).argmin());chosen=[index]
    distance=((candidates-candidates[index])**2).sum(1)
    for _ in range(k-1):
        index=int(distance.argmax());chosen.append(index)
        distance=np.minimum(distance,((candidates-candidates[index])**2).sum(1))
    selected=candidates[chosen]
    d=np.linalg.norm(selected[:,None]-selected[None,:],axis=-1)
    if k>1 and d[np.triu_indices(k,1)].min()<min_distance:raise ValueError('Insufficient grid spacing')
    return selected/size
