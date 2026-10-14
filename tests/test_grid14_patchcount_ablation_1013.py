import torch
from src.grid14_patchcount_ablation_1013 import nested_sequence,pack_nested,slice_bag,KS
from scripts.train_gaze_coverage_quota_1012 import quota_indices

def test_nested_fixed_grid_and_old_k2_identity():
 grid=torch.cartesian_prod(torch.linspace(.1,.9,4),torch.linspace(.1,.9,4));source=grid+.04;coords=torch.cat([grid,source]);scores=torch.arange(32.);valid=torch.ones(32,dtype=torch.bool);g,s,r=nested_sequence(scores,coords,valid);old,_=quota_indices(scores,coords,valid)
 assert torch.equal(torch.cat([g,s[:2]]),old)
 for k in KS:assert torch.equal(torch.cat([g,s[:k]])[:14],g) and torch.equal(s[:min(k,2)],s[:2][:min(k,2)])

def test_padding_gather_alignment_and_mask_budget():
 grid=torch.cartesian_prod(torch.linspace(.1,.9,4),torch.linspace(.1,.9,4));t={'grid_features':torch.arange(16*3.).reshape(1,16,3),'grid_coordinates':grid[None],'grid_mask':torch.ones(1,16,dtype=torch.bool)};t.update(gaze_baseline_features=torch.ones(1,16,3),gaze_baseline_coordinates=grid[0].repeat(1,16,1),gaze_baseline_mask=torch.ones(1,16,dtype=torch.bool));record={'scores':torch.arange(32.)[None]};bag,a=pack_nested(t,'gaze_baseline',record)
 assert a['source_available'][0]<=1
 for k in KS:
  b=slice_bag(bag,k);assert b[0].shape[1]==14+k and torch.equal(b[0][:,:14],bag[0][:,:14]);assert torch.equal(b[0][~b[2]],torch.zeros_like(b[0][~b[2]]));assert b[2].sum()==14+min(k,int(a['source_available'][0]))
 assert torch.equal(bag[0][0,:14],t['grid_features'][0,a['indices'][0,:14]])
if __name__=='__main__':
 test_nested_fixed_grid_and_old_k2_identity();test_padding_gather_alignment_and_mask_budget();print('FixedGrid14/nestedK/oldK2/gather/maskpadding checks passed')
