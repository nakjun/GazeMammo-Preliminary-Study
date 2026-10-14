"""Observed training-path sampling; canonical validation never resamples."""
import numpy as np
import torch
from .data import seed_for


def selected_indices(length,image_id,epoch,seed=42,augmented=True):
    if length<1:raise ValueError('Empty chosen segment')
    if not augmented:return np.linspace(0,length-1,16).round().astype(int) if length>16 else np.arange(length)
    if length<=16:return np.arange(length)
    # Preserve exact path endpoints; draw one point in each disjoint interior stratum.
    rng=np.random.default_rng(seed_for(seed,int(image_id)+int(epoch)*100000))
    interior=np.array_split(np.arange(1,length-1),14)
    indices=np.array([0,*[int(rng.choice(s)) for s in interior],length-1])
    if len(indices)!=16 or np.any(np.diff(indices)<=0):raise ValueError('Invalid ordered stratum sample')
    return indices


def augmented_sample(pool,image_id,epoch,seed=42,augmented=True,split='train'):
    if augmented and split!='train':raise ValueError('Validation target resampling is forbidden')
    coords=pool['coords'];vectors=pool['patch_vectors'];indices=selected_indices(len(coords),image_id,epoch,seed,augmented)
    sample_coords=torch.zeros(16,2);sample_vectors=torch.zeros(16,vectors.shape[-1]);mask=torch.zeros(16,dtype=torch.bool);n=len(indices)
    sample_coords[:n]=coords[indices];mask[:n]=True
    if n>1:sample_vectors[1:n]=vectors[indices[:-1]]
    return dict(coords=sample_coords,vectors=sample_vectors,mask=mask,pool_indices=indices.tolist(),raw_indices=pool['raw_indices'][indices].tolist())
