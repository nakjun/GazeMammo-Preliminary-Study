import torch
from scripts.evaluate_gaze_shared_grid_1012 import bags

def test_extra_patch_control_budget_and_original_grid_preserved():
 t={}
 for i,a in enumerate(['grid','gaze_baseline','gaze_improved','random']):
  t[a+'_features']=torch.full((2,16,3),float(i));t[a+'_coordinates']=torch.rand(2,16,2);t[a+'_mask']=torch.ones(2,16,dtype=torch.bool)
 b=bags(t)
 for a in ['gaze_baseline','gaze_improved','random']:
  z=b['grid16_'+a+'16'];assert z[0].shape[1]==32 and z[1].shape[1]==32 and z[2].shape[1]==32
  assert torch.equal(z[0][:,:16],t['grid_features']) and torch.equal(z[0][:,16:],t[a+'_features'])
 assert torch.equal(b['grid'][0],t['grid_features'])
if __name__=='__main__':
 test_extra_patch_control_budget_and_original_grid_preserved();print('Matched32patch and original-grid invariance check passed')
