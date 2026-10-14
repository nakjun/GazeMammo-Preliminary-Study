import numpy as np
from PIL import Image
from scripts.visualize_grid14_gaze_case_1013 import crop_verified

def test_exact_inbounds_and_ties_even_rounding():
 image=Image.fromarray((np.arange(512*256).reshape(512,256)%256).astype(np.uint8));crop,metadata=crop_verified(image,np.array([111.5,255.5]));assert metadata['rectangle']==[0,144,224,368] and np.array_equal(crop,np.asarray(image)[144:368,0:224]) and not metadata['black_padding']

def test_outside_image_preserves_training_black_padding():
 image=Image.fromarray(np.full((300,300),17,dtype=np.uint8));crop,metadata=crop_verified(image,np.array([2.,2.]));assert metadata['rectangle']==[-110,-110,114,114] and metadata['black_padding'];assert (crop[:110]==0).all() and (crop[110:,110:]==17).all()
if __name__=='__main__':
 test_exact_inbounds_and_ties_even_rounding();test_outside_image_preserves_training_black_padding();print('Exactcrop/ties-even/boundarypadding/normalizationroundtrip tests passed')
