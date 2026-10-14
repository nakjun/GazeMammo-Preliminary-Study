"""Frozen medical whole encoder plus independent frozen control past encoder."""
import torch
from torch import nn
from PIL import Image
from src.mammo_clip_visual_1011 import MammoClipVisual,preprocess
from src.gaze_nll_adaptation_1011 import GazeNLLAdaptation1011
from src.trajectory_1011 import restore
from src.proposal_utility_1011 import state_sha256,file_sha256

class MammoSplitVisual(nn.Module):
 channels=[2048,2048,2048]
 def __init__(self,whole,past):
  super().__init__();self.whole=whole;self.past=past;self.requires_grad_(False);self.eval()
 def forward(self,tensor):
  with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):feature=self.whole(tensor)
  return(feature,feature,feature)
 def patch_vector(self,tensor):
  with torch.no_grad(),torch.backends.cudnn.flags(enabled=True,allow_tf32=True):return self.past.patch_vector(tensor)
 def is_last_parameter(self,name):return False

class MammoGaze(GazeNLLAdaptation1011):
 def __init__(self,medical_checkpoint,past_checkpoint):
  # Restore genuine encoders before introducing any clinical gaze parameters.
  whole=MammoClipVisual(medical_checkpoint);control=restore(torch.load(past_checkpoint,map_location='cpu',weights_only=False))
  super().__init__(grid=(24,48),k=16,dim=256,foveal=True,fpn=False,pretrained=False,backbone_name='convnext_base',motion_bias=True,input_size=(912,1520),spatial_history=True,history_length=4)
  self.visual=MammoSplitVisual(whole,control.visual);self.lateral=nn.ModuleList([nn.Conv2d(2048,256,1) for _ in range(3)]);self.lateral[0].requires_grad_(False);self.lateral[1].requires_grad_(False);self.backbone_name='mammo_clip_whole_control_past';self.model_config['backbone_name']=self.backbone_name;self.feature_memory_format='contiguous_nchw';self.set_backbone('frozen');self.input_protocol='PILwholecv2resize912x1520/minmax/scalarstats; controlpastImageNetnormalized224'
 def maps(self,image):
  if isinstance(image,Image.Image):
   tensor,metadata=preprocess(image);memory=self.maps_features(self.visual(tensor[None].to(next(self.parameters()).device)));memory['input_metadata']=metadata;return memory
  return self.maps_features(self.visual(image))
 def maps_features(self,features):
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):return super().maps_features(features)
 def teacher_logits(self,memory,coords,patch_vectors=None,**kwargs):
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):return super().teacher_logits(memory,coords,patch_vectors,**kwargs)
 def step_history(self,memory,prefix,hidden=None,patch_vector=None):
  with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):return super().step_history(memory,prefix,hidden,patch_vector)
 def head_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('visual.')}
 def inventory(self):return dict(trainable_head=sum(p.numel() for p in self.parameters() if p.requires_grad),frozen_whole=sum(p.numel() for p in self.visual.whole.parameters()),frozen_past=sum(p.numel() for p in self.visual.past.parameters()),frozen_unused_lateral0_1=sum(p.numel() for layer in self.lateral[:2] for p in layer.parameters()),whole_native=[2048,48,29],past_vector=1024,dim=256,fpn=False)

def restore_mammo_gaze(checkpoint,device='cpu'):
 if checkpoint['kind']!='mammo_clip_gaze_1011':raise ValueError('Wrong medicalgaze checkpoint')
 config=checkpoint['config']
 for path,key in [(config['medical_checkpoint'],'medical_checkpoint_sha256'),(config['past_checkpoint'],'past_checkpoint_sha256')]:
  if file_sha256(path)!=config[key]:raise ValueError('Medical/past dependency changed')
 for path,sha in config.get('source_code_sha256',{}).items():
  if file_sha256(path)!=sha:raise ValueError('Medicalgaze source code changed')
 model=MammoGaze(config['medical_checkpoint'],config['past_checkpoint']).to(device)
 if set(checkpoint['state'])!=set(model.head_state()):raise ValueError('Exact complete gazehead keyset mismatch')
 model.load_state_dict(checkpoint['state'],strict=False)
 if state_sha256(model.visual.whole.state_dict())!=checkpoint['whole_state_sha256'] or state_sha256(model.visual.past.state_dict())!=checkpoint['past_state_sha256']:raise ValueError('Frozen medical/past state changed')
 return model.eval()
