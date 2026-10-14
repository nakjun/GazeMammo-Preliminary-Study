import torch,numpy as np
from src.gaze_path_augmentation_1011 import augmented_sample

def test_selected_dino_vectors_are_strictly_preceding_crops():
    m=41;pool=dict(coords=torch.stack([torch.linspace(0,1,m)]*2,-1),patch_vectors=torch.arange(m).float()[:,None].expand(m,384).clone(),raw_indices=torch.arange(m)*3)
    sample=augmented_sample(pool,62,4,42);indices=sample['pool_indices'];assert indices[0]==0 and indices[-1]==m-1 and np.all(np.diff(indices)>0);assert torch.equal(sample['vectors'][0],torch.zeros(384))
    for t in range(1,16):assert torch.equal(sample['vectors'][t],pool['patch_vectors'][indices[t-1]]) and sample['vectors'][t,0]!=indices[t]
    assert sample['raw_indices']==(torch.tensor(indices)*3).tolist()

def test_validation_resampling_rejected():
    pool=dict(coords=torch.zeros(41,2),patch_vectors=torch.zeros(41,384),raw_indices=torch.arange(41))
    try:augmented_sample(pool,64,1,42,split='val')
    except ValueError:pass
    else:raise AssertionError('Validation augmentation allowed')
