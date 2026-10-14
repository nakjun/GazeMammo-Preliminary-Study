import tempfile
from pathlib import Path
import numpy as np,torch
from PIL import Image
from src.data import patch_tensor,MEAN,STD
from src.official_cbis_classifier_1012 import digest
from scripts.extract_gaze_multiscale_1012 import validate_record

def test_crop_geometry_rounding_and_resize():
 image=Image.fromarray(np.arange(600*800,dtype=np.uint32).reshape(800,600).astype(np.uint8))
 point=np.array([231.6,398.4]);x,y=np.round(point).astype(int);expected=image.crop((x-224,y-224,x+224,y+224)).resize((224,224),Image.Resampling.BILINEAR)
 expected=torch.from_numpy(np.asarray(expected).copy()).float()[None].repeat(3,1,1)/255
 assert torch.equal(patch_tensor(image,point,448),(expected-MEAN)/STD)

def test_resume_rejects_changed_or_missing_record():
 with tempfile.TemporaryDirectory() as d:
  path=Path(d)/'one.pt';path.write_bytes(b'valid');sha=digest(path);validate_record(path,sha);path.write_bytes(b'changed')
  try:validate_record(path,sha)
  except ValueError:pass
  else:raise AssertionError('Changed record accepted')
  path.unlink()
  try:validate_record(path,sha)
  except ValueError:pass
  else:raise AssertionError('Missing record accepted')
if __name__=='__main__':
 test_crop_geometry_rounding_and_resize();test_resume_rejects_changed_or_missing_record();print('Crop geometry/rounding/resize and resume hash checks passed')
