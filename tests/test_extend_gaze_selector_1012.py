import tempfile
from pathlib import Path
import torch
from scripts.extend_gaze_selector_classifier_1012 import fixed_bags

def test_fixedbags_reuse_exact_vectors_and_reject_wrong_ids():
 t={};n=2
 for i,a in enumerate(['grid','gaze_baseline','random']):
  t[a+'_features']=torch.full((n,16,3),float(i));t[a+'_coordinates']=torch.rand(n,16,2);t[a+'_mask']=torch.ones(n,16,dtype=torch.bool)
 cache={'rows':[{'image_id':1},{'image_id':2}],'tensors':t}
 with tempfile.TemporaryDirectory() as temp:
  for source in ['gaze_baseline','random']:
   p=Path(temp)/f'selector_{source}';p.mkdir();torch.save(dict(image_ids=[1,2],indices=torch.arange(16,32).repeat(n,1),kind='patientOOFTRAIN_fullTRAINVAL_hardselection',candidate_source=source),p/'selected_indices.pt')
  bags,_=fixed_bags(cache,temp);assert torch.equal(bags['selected_gaze_baseline'][0],t['gaze_baseline_features']);assert torch.equal(bags['selected_random'][1],t['random_coordinates']);assert torch.equal(bags['grid'][0],t['grid_features'])
  cache['rows'].reverse()
  try:fixed_bags(cache,temp)
  except ValueError:pass
  else:raise AssertionError('Wrongimageorderaccepted')
if __name__=='__main__':
 test_fixedbags_reuse_exact_vectors_and_reject_wrong_ids();print('Exactfrozenbags/identitymisalignmentguard passed')
