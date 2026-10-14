"""Frozen official original DINO ViT-S/8 and a separate causal gaze head."""
import importlib.util,sys
from pathlib import Path
import torch
from torch import nn
from src.gaze_nll_adaptation_1011 import GazeNLLAdaptation1011

class DinoVisual(nn.Module):
    channels=[384,384,384]
    def __init__(self,source,checkpoint):
        super().__init__();source=Path(source)
        spec=importlib.util.spec_from_file_location('dino_utils_1011',source/'utils.py');utils=importlib.util.module_from_spec(spec);spec.loader.exec_module(utils);old=sys.modules.get('utils');sys.modules['utils']=utils
        try:
            spec=importlib.util.spec_from_file_location('dino_vit_1011',source/'vision_transformer.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        finally:
            if old is None:sys.modules.pop('utils',None)
            else:sys.modules['utils']=old
        self.encoder=module.vit_small(patch_size=8);self.encoder.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False),strict=True)
        self.requires_grad_(False);self.eval()
    def forward(self,image):
        with torch.no_grad():tokens=self.encoder.get_intermediate_layers(image,n=1)[0][:,1:];feature=tokens.transpose(1,2).reshape(len(image),384,image.shape[-2]//8,image.shape[-1]//8)
        return (feature,feature,feature) # Single final patch feature; FPN is disabled.
    def patch_vector(self,image):return self(image)[-1].mean((-2,-1))
    def is_last_parameter(self,name):return False

class DinoGaze(GazeNLLAdaptation1011):
    def __init__(self,source,checkpoint):
        super().__init__(grid=(24,48),k=16,dim=256,foveal=True,fpn=False,pretrained=False,backbone_name='convnext_base',motion_bias=True,input_size=(256,512),spatial_history=True,history_length=4)
        self.visual=DinoVisual(source,checkpoint);self.lateral=nn.ModuleList([nn.Conv2d(384,256,1) for _ in range(3)]);self.patch=nn.Linear(384,256);self.backbone_name='dino_vits8';self.model_config['backbone_name']='dino_vits8';self.feature_memory_format='contiguous_nchw';self.set_backbone('frozen')

def restore_dino(checkpoint,device='cpu'):
    import hashlib
    if checkpoint['kind']!='dino_gaze_1011':raise ValueError('Wrong DINO gaze checkpoint')
    source=checkpoint['config']['source']
    for path,expected_sha in checkpoint['config'].get('source_code_sha256',{}).items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected_sha:raise ValueError('Pinned DINO code changed')
    if hashlib.sha256(Path(source['checkpoint']).read_bytes()).hexdigest()!=source['checkpoint_sha256']:raise ValueError('DINO dependency changed')
    model=DinoGaze(source['source'],source['checkpoint']).to(device)
    expected={k for k in model.state_dict() if not k.startswith('visual.')}
    if set(checkpoint['state'])!=expected:raise ValueError('DINO head state mismatch')
    model.load_state_dict(checkpoint['state'],strict=False)
    from src.proposal_utility_1011 import state_sha256
    if state_sha256(model.visual.state_dict())!=checkpoint['visual_state_sha256']:raise ValueError('DINO frozen state changed')
    return model.eval()
