import torch
from src.gaze16_grid_addition_1013 import additional_grid_sequence,build_bags

def test_gaze16_identical_everyk_and_gridgather_prefix():
 t={'global_features':torch.zeros(2,3)}
 for i,s in enumerate(['grid','gaze_baseline']):
  t[s+'_features']=torch.arange(2*16*3.).reshape(2,16,3)+i*100;t[s+'_coordinates']=torch.rand(2,16,2);t[s+'_mask']=torch.ones(2,16,dtype=torch.bool)
 bags,a=build_bags(t)
 for k,b in bags.items():
  assert torch.equal(b[0][:,:16],t['gaze_baseline_features']) and torch.equal(b[1][:,:16],t['gaze_baseline_coordinates']) and torch.equal(b[2][:,:16],t['gaze_baseline_mask']);assert torch.equal(b[0][:,16:],bags[4][0][:,16:16+k])
 assert torch.equal(bags[4][0][0,16:],t['grid_features'][0,a['indices'][0]])

def test_coverage_farthest_and_pixelduplicate_removed():
 gaze=torch.tensor([[.5,.5]]).repeat(16,1);grid=torch.tensor([[.5,.5],[0.,0.],[1.,1.],[0.,1.],[1.,0.]]);ids=additional_grid_sequence(gaze,grid,torch.ones(16,dtype=torch.bool),torch.ones(5,dtype=torch.bool));assert ids[0]==1 and 0 not in ids and len(torch.unique(ids))==4
if __name__=='__main__':
 test_gaze16_identical_everyk_and_gridgather_prefix();test_coverage_farthest_and_pixelduplicate_removed();print('Gaze16identity/nestedgridgather/coverage/pixelduplicate checks passed')
