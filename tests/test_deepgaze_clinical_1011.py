import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_deepgaze3_1011 import load_deepgaze,cell_mass
from src.deepgaze_clinical_1011 import freeze_features,history_batch,prepare_saliency,forward_saliency
from src.deepgaze_context_1011 import forward_native
from src.proposal_utility_1011 import state_sha256


def test_native_prepared_forward_causal_and_probability_mass():
    torch.set_num_threads(4);model=freeze_features(load_deepgaze('/tmp/deepgaze_feasibility_1011',ROOT/'runs/dual_target_1011/deepgaze_public/assets/deepgaze3.pth','cpu'));features=[torch.randn(1,c,2,1) for c in (1888,128,32)];size=(64,128);bias=torch.zeros(1,128,64);coords=torch.rand(1,4,2)
    hx,hy=history_batch(coords,2,size);assert torch.isnan(hx[:,2:]).all();changed=coords.clone();changed[:,2:]=1-changed[:,2:];cx,cy=history_batch(changed,2,size)
    with torch.no_grad():
        saliency=prepare_saliency(model,features,size);q=cell_mass(forward_saliency(model,saliency,size,bias,hx,hy));reference=cell_mass(forward_native(model,features,size,bias,hx,hy));other=cell_mass(forward_saliency(model,saliency,size,bias,cx,cy))
    assert torch.allclose(q,reference,atol=1e-6,rtol=1e-6) and torch.equal(q,other)
    assert torch.allclose(q.exp().sum(-1),torch.ones(1),atol=1e-6)
    try:history_batch(coords,0,size)
    except ValueError:pass
    else:raise AssertionError('Unsupported fake first fixation allowed')


def test_streaming_saliency_gradient_and_feature_freeze():
    torch.set_num_threads(4);model=freeze_features(load_deepgaze('/tmp/deepgaze_feasibility_1011',ROOT/'runs/dual_target_1011/deepgaze_public/assets/deepgaze3.pth','cpu'));before=state_sha256(model.features.state_dict());features=[torch.randn(1,c,2,1) for c in (1888,128,32)];size=(64,128);bias=torch.zeros(1,128,64);coords=torch.rand(1,3,2)
    saliency=prepare_saliency(model,features,size);leaves=[s.detach().requires_grad_(True) for s in saliency]
    for t in (1,2):
        hx,hy=history_batch(coords,t,size);q=cell_mass(forward_saliency(model,leaves,size,bias,hx,hy));(-q[:,100].mean()/3).backward()
    torch.autograd.backward(saliency,[leaf.grad for leaf in leaves])
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.saliency_networks.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.scanpath_networks.parameters())
    assert all(p.grad is None for p in model.features.parameters()) and state_sha256(model.features.state_dict())==before
