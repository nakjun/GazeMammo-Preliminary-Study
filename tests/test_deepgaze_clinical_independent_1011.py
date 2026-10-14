"""Independent causal and shared-graph-gradient checks for clinical DG readouts."""
import copy,sys,types
import torch
from torch import nn
from src.deepgaze_clinical_1011 import history_batch,prepare_saliency,forward_saliency,freeze_features

class Selection(nn.Module):
    def __init__(self):super().__init__();self.conv=nn.Conv2d(2,1,1)
    def forward(self,pair):return self.conv(torch.cat(pair,1))
class Finalizer(nn.Module):
    def forward(self,x,bias):
        q=x[:,0]+bias;return q-q.flatten(1).logsumexp(1)[:,None,None]
class Dummy(nn.Module):
    def __init__(self):
        super().__init__();self.downsample=self.readout_factor=1;self.features=nn.Sequential(nn.Conv2d(1,1,1),nn.BatchNorm2d(1));self.saliency_networks=nn.ModuleList([nn.Conv2d(1,1,1)]);self.scanpath_networks=nn.ModuleList([nn.Conv2d(1,1,1)]);self.fixation_selection_networks=nn.ModuleList([Selection()]);self.finalizers=nn.ModuleList([Finalizer()])

def test_history_current_future_and_short_prefix():
    coords=torch.rand(2,16,2);changed=coords.clone();changed[:,7:]=1-changed[:,7:]
    x,y=history_batch(coords,7,(512,1024));xx,yy=history_batch(changed,7,(512,1024));assert torch.equal(x,xx) and torch.equal(y,yy)
    x,y=history_batch(coords,1,(512,1024));assert torch.isnan(x[:,1:]).all() and torch.isnan(y[:,1:]).all();assert torch.equal(x[:,0],coords[:,0,0]*512)

def test_detached_saliency_accumulation_matches_joint_backward():
    original=sys.modules.get('deepgaze_pytorch.modules');stub=types.ModuleType('deepgaze_pytorch.modules')
    stub.encode_scanpath_features=lambda hx,hy,size,device: hx[:,:1,None,None].expand(-1,1,*size)
    sys.modules['deepgaze_pytorch.modules']=stub
    try:
        torch.manual_seed(42);direct=Dummy();shared=copy.deepcopy(direct);features=[torch.randn(2,1,2,2)];bias=torch.zeros(2,2,2);saliency=prepare_saliency(direct,features,(2,2));loss=0
        for t in range(1,4):loss=loss-forward_saliency(direct,saliency,(2,2),bias,torch.full((2,1),float(t)),torch.zeros(2,1))[:,0,0,0].mean()
        loss.backward();saliency=prepare_saliency(shared,features,(2,2));leaves=[x.detach().requires_grad_(True) for x in saliency]
        for t in range(1,4):(-forward_saliency(shared,leaves,(2,2),bias,torch.full((2,1),float(t)),torch.zeros(2,1))[:,0,0,0].mean()).backward()
        torch.autograd.backward(saliency,[x.grad for x in leaves])
        for (name,a),(other,b) in zip(direct.named_parameters(),shared.named_parameters()):
            assert name==other
            if a.grad is not None:assert torch.allclose(a.grad,b.grad,atol=1e-6),name
    finally:
        if original is None:sys.modules.pop('deepgaze_pytorch.modules',None)
        else:sys.modules['deepgaze_pytorch.modules']=original

def test_frozen_encoder_bn_eval():
    model=freeze_features(Dummy());model.train();model.features.eval();before=model.features[1].running_mean.clone();model.features(torch.rand(2,1,2,2));assert torch.equal(before,model.features[1].running_mean);assert not any(p.requires_grad for p in model.features.parameters())

def test_native_cache_rejects_target_variant_bos_and_incomplete_cohort():
    import json,tempfile
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_deepgaze_clinical_1011 import load_native_cache
    from train_attention_1011 import digest
    for failure in ['coords','mask','variant','bos','incomplete']:
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir();(source/'provenance.json').write_text('{}');checkpoint=root/'parent.pt';checkpoint.write_bytes(b'parent')
            record=dict(image_id=64,patient_key='p',split='val',image_sha256='image',coords=torch.full((16,2),.5),mask=torch.ones(16,dtype=torch.bool));item={**record,'flipped':False,'features':[torch.zeros(c,2,1) for c in [1888,128,32]],'bos_logp':torch.full((1152,),-torch.log(torch.tensor(1152.)))}
            if failure=='coords':item['coords']=item['coords'].clone();item['coords'][0,0]=.6
            if failure=='mask':item['mask']=item['mask'].clone();item['mask'][0]=False
            if failure=='variant':item['flipped']=True
            if failure=='bos':item['bos_logp']=torch.zeros(1152)
            torch.save(item,root/'64.pt');meta=dict(name='64',image_id=64,flipped=False,split='val',file_sha256=digest(root/'64.pt'))
            header=dict(kind='deepgaze_clinical_native_cache_1011',checkpoint_sha256=digest(checkpoint),feature_state_sha256='feature',source_cache_provenance_sha256=digest(source/'provenance.json'),input_size=[64,128],smoke=False if failure=='incomplete' else True,records=[] if failure=='incomplete' else [meta]);(root/'provenance.json').write_text(json.dumps(header));(root/'status.json').write_text('{"state":"complete"}')
            try:load_native_cache(SimpleNamespace(native_cache=str(root),checkpoint=str(checkpoint),cache=str(source),smoke_limit=1),[record],'feature')
            except ValueError:pass
            else:raise AssertionError(f'Accepted malformed {failure}')

def test_centerbias_rejects_hash_valid_nonpositive_tensor():
    import json,tempfile
    from pathlib import Path
    from types import SimpleNamespace
    import pandas as pd
    from train_deepgaze_clinical_1011 import verified_centerbias
    from train_attention_1011 import digest
    root=Path(__file__).resolve().parents[1];provenance=json.loads((root/'runs/dual_target_1011/deepgaze_public/train_centerbias_provenance.json').read_text());frame=pd.read_csv(root/'manifests/all.csv',usecols=['image_id','split']);records=[dict(image_id=int(r.image_id),split='train') for _,r in frame[frame.split=='train'].iterrows()]
    with tempfile.TemporaryDirectory() as folder:
        tensor=Path(folder)/'bias.pt';torch.save(torch.zeros(48,24),tensor);provenance['probability_tensor_sha256']=digest(tensor);metadata=Path(folder)/'bias.json';metadata.write_text(json.dumps(provenance))
        try:verified_centerbias(SimpleNamespace(centerbias=str(tensor),centerbias_provenance=str(metadata)),records)
        except ValueError as error:assert 'positive normalized' in str(error)
        else:raise AssertionError('Accepted nonpositive hash-valid KDE')

def test_restore_rejects_encoder_state_in_readout_checkpoint():
    import tempfile
    from pathlib import Path
    import train_deepgaze_clinical_1011 as trainer
    from src.proposal_utility_1011 import state_sha256
    from train_attention_1011 import digest
    from src.deepgaze_clinical_1011 import head_state
    original=trainer.load_deepgaze;model=Dummy();trainer.load_deepgaze=lambda *args:copy.deepcopy(model)
    try:
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder)/'source.pt';parent.write_bytes(b'source');state=head_state(model);state['features.0.weight']=model.features[0].weight.detach().clone();checkpoint=dict(kind='deepgaze_clinical_readout_1011',source_checkpoint=str(parent),source_checkpoint_sha256=digest(parent),source='unused',feature_state_sha256=state_sha256(model.features.state_dict()),state=state);path=Path(folder)/'head.pt';torch.save(checkpoint,path)
            try:trainer.load_clinical(path)
            except ValueError as error:assert 'readout-only' in str(error)
            else:raise AssertionError('Accepted encoder override in clinical head')
    finally:trainer.load_deepgaze=original
