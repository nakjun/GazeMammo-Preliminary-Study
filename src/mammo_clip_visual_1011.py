"""Strict full official UPMC Mammo-CLIP B5 image encoder, no clinical head."""
import importlib,sys,types,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
EXPECTED_SHA='0baccb0cc19cd6093ff8252f7c2b7a6efbc32973498640abcb311210c46fd02c'
SOURCE=Path(__file__).resolve().parents[1]/'runs/dual_target_1011/mammography_backbone_audit/source/src/codebase/breastclip/model/modules'

def source_class():
 name='mammo_official_modules_1011'
 if name not in sys.modules:
  package=types.ModuleType(name);package.__path__=[str(SOURCE)];sys.modules[name]=package
 return importlib.import_module(name+'.efficientnet_custom').EfficientNet

def preprocess(image,size=(912,1520),mean=.3089279,std=.25053555408335154):
 # Existing PNG is already border-trimmed; no added threshold/crop or coordinate change.
 rgb=np.asarray(image.convert('RGB'));resized=cv2.resize(rgb,tuple(size),interpolation=cv2.INTER_LINEAR).astype('float32');minimum=float(resized.min());resized-=minimum;maximum=float(resized.max())
 if maximum<=0:raise ValueError('Constant image: source minmax normalization undefined')
 normalized=(resized/maximum-mean)/std
 return torch.from_numpy(normalized.copy()).permute(2,0,1),dict(resize='cv2INTER_LINEAR as authorvalidResize',size=list(size),original=list(image.size),additional_cropping=False,thresholding=False,source_min=minimum,source_range=maximum,normalization='subtractmin/dividerange thenscalar(mean,std); noadditionaldiv255/ImageNetnorm',mean=mean,std=std)

class MammoClipVisual(nn.Module):
 def __init__(self,path):
  super().__init__();path=Path(path)
  if hashlib.sha256(path.read_bytes()).hexdigest()!=EXPECTED_SHA:raise ValueError('Official B5 checkpoint SHA mismatch')
  payload=torch.load(path,map_location='cpu',weights_only=False);config=payload['config'];encoder=config['model']['image_encoder']
  if encoder['source']!='cnn' or encoder['name']!='tf_efficientnet_b5_ns-detect' or set(config['data_train'])!={'upmc_wo_period'}:raise ValueError('Unexpected clinical pretraining variant')
  self.config=config;self.source_epoch=payload['epoch'];self.encoder=source_class().from_name('efficientnet-b5',num_classes=1)
  weights={k.removeprefix('image_encoder.'):v for k,v in payload['model'].items() if k.startswith('image_encoder.')}
  if set(weights)!=set(self.encoder.state_dict()):raise ValueError('Full imageencoder keyset mismatch')
  self.encoder.load_state_dict(weights,strict=True);self.loaded_keys=len(weights);self.checkpoint_sha256=EXPECTED_SHA;self.requires_grad_(False);self.eval()
 def train(self,mode=True):return super().train(False)
 def forward(self,tensor):
  with torch.no_grad():return self.encoder.extract_features(tensor)
 def patch_vector(self,tensor):return self(tensor).mean((-2,-1))
