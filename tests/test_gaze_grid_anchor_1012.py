import torch
from scripts.train_gaze_grid_anchor_1012 import BoundedResidual

def test_anchor_preserved_at_initialization():
 m=BoundedResidual(4).eval();x=torch.randn(8,4);anchor=torch.randn(8)
 assert torch.equal(anchor+m(x),anchor)

def test_correction_bounded_even_with_large_weights():
 m=BoundedResidual(4).eval()
 with torch.no_grad():m.net[-1].weight.fill_(100);m.net[-1].bias.fill_(100)
 assert bool((m(torch.randn(8,4)).abs()<=.5).all())

if __name__=='__main__':
 test_anchor_preserved_at_initialization();test_correction_bounded_even_with_large_weights();print('2 anchor checks passed')
