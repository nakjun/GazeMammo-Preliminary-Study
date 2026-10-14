import numpy as np
import torch
from PIL import Image
from src.gaze_mirror_augmentation_1011 import mirror_coordinates,mirror_patch_tensor,flip_for_epoch
from src.data import patch_tensor


def test_exact_coordinate_mask_and_pixel_crop_reflection():
    image=Image.fromarray(np.random.default_rng(42).integers(0,256,(301,501),dtype=np.uint8))
    coords=torch.tensor([[0.,0.],[.5,.5],[.000998004,.25],[.999,.9]])
    original=coords.clone();mirrored=mirror_coordinates(coords)
    assert torch.equal(coords,original) and torch.equal(mirrored[:,1],coords[:,1])
    assert torch.equal(mirrored[:,0],1-coords[:,0])
    for coord in coords:
        actual=mirror_patch_tensor(image,coord);expected=patch_tensor(image,coord.numpy()*image.size).flip(-1)
        assert torch.equal(actual,expected)
    mask=torch.tensor([1,1,0,0],dtype=torch.bool);masked=mirror_coordinates(coords,mask);assert torch.equal(masked[~mask],coords[~mask])


def test_independent_reproducible_train_flip_guard():
    assert flip_for_epoch(62,3)==flip_for_epoch(62,3)
    outcomes={flip_for_epoch(62,e) for e in range(1,20)};assert outcomes=={False,True}
    try:flip_for_epoch(64,3,split='val')
    except ValueError:pass
    else:raise AssertionError('Validation flip accepted')
