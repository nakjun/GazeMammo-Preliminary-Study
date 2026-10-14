import numpy as np,torch,cv2
from PIL import Image
from src.mammo_clip_visual_1011 import preprocess

def test_author_minmax_scalar_normalization_no_extra_255():
    image=Image.fromarray(np.array([[20,40],[60,100]],dtype=np.uint8));tensor,meta=preprocess(image,size=(2,2));expected=(np.asarray(image).astype(np.float32)-20)/80;expected=(expected-.3089279)/.25053555408335154;assert tensor.shape==(3,2,2) and torch.allclose(tensor[0],torch.from_numpy(expected),atol=1e-7);assert torch.equal(tensor[0],tensor[1]) and not meta['additional_cropping'] and not meta['thresholding']

def test_constant_source_image_explicitly_rejected():
    try:preprocess(Image.fromarray(np.zeros((3,4),dtype=np.uint8)),size=(4,3))
    except ValueError as error:assert 'Constant' in str(error)
    else:raise AssertionError('Undefinedminmax accepted')
