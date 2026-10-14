import torch
from scripts.train_gaze_coverage_quota_1012 import grid_coverage,quota_indices

def test_fixedquota_alignment_and_label_free():
 grid=torch.cartesian_prod(torch.linspace(.1,.9,4),torch.linspace(.1,.9,4));source=grid+.01;coords=torch.cat([grid,source]);scores=torch.arange(32.);mask=torch.ones(32,dtype=torch.bool);ids,a=quota_indices(scores,coords,mask)
 assert len(ids)==16 and int((ids<16).sum())==14 and int((ids>=16).sum())==2
 assert len(torch.unique(torch.round(coords[ids]*torch.tensor([1500.,3000.])),dim=0))==16
 # No label argument exists; gathering preserves original candidate coordinates/vectors.
 values=torch.arange(32*3).reshape(32,3);assert torch.equal(values[ids][:14],values[grid_coverage(grid)])

def test_zero_source_fallback_preserves16():
 grid=torch.cartesian_prod(torch.linspace(.1,.9,4),torch.linspace(.1,.9,4));covered=grid_coverage(grid);source=grid[covered[0]].repeat(16,1);ids,a=quota_indices(torch.arange(32.),torch.cat([grid,source]),torch.ones(32,dtype=torch.bool));assert a['source_count']==0 and a['grid_fallback_count']==2 and len(torch.unique(ids))==16 and (ids<16).all()
if __name__=='__main__':
 test_fixedquota_alignment_and_label_free();test_zero_source_fallback_preserves16();print('Fixed14/2budget/alignment and zero-sourcefallback checks passed')
