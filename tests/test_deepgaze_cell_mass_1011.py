import torch
from scripts.evaluate_deepgaze3_1011 import cell_mass,histories

def test_cell_mass_conserves_uniform_probability():
    q=cell_mass(torch.full((1,1,96,48),-torch.log(torch.tensor(96*48.))))
    assert torch.allclose(q.exp(),torch.full((1,1152),1/1152),atol=1e-8)

def test_history_never_contains_current_or_future():
    coords=torch.arange(32).reshape(1,16,2).float()/32
    x,y=histories(coords,2,(256,512));assert torch.isnan(x[:,2:]).all()
    assert x[0,0]==coords[0,1,0]*256 and y[0,1]==coords[0,0,1]*512
