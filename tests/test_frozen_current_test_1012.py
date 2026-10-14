import importlib.util,io,unittest
from unittest.mock import patch
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('frozen_test','scripts/evaluate_frozen_current_test_1012.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class TestFrozenEvaluation(unittest.TestCase):
 def test_cohort_metadata(self):
  rows,boxed=m.metadata();self.assertEqual(len(rows),150);self.assertEqual(len(boxed),131);self.assertEqual(len({int(r['image_id']) for r in rows}-boxed),19);self.assertTrue(all(r['source']=='CBIS-DDSM' and r['split']=='test' for r in rows))
 def test_allowed_payload_only_and_canonical_equivalence(self):
  row=m.metadata()[0][0];samples=[([float(i*40+10),float(i*60+10)],1) for i in range(20)];text='INbreast_do_not_parse;invalid forbidden payload\n'+row['file_name']+';'+repr(samples)+'\n';original=m.Path.open
  def opened(path,*args,**kwargs):return io.StringIO(text) if str(path).endswith('merged_gaze_data.csv') else original(path,*args,**kwargs)
  with patch.object(m.Path,'open',opened):sample,error=m.canonical(row['file_name'],(1500,3000))
  points=np.array([s[0] for s in samples]);parts=[p/(1500,3000) for p in m.segments(points,1500,3000,32.)];expected=m.window_sample(parts,int(row['image_id']),42,0,'legacy16',16,8,.5);self.assertIsNone(error);self.assertTrue(torch.equal(sample['coords'],expected['coords'][:16]));self.assertTrue(torch.equal(sample['mask'],expected['loss_mask'][:16]))
 def test_missing_is_excluded_not_synthetic(self):
  original=m.Path.open
  def opened(path,*args,**kwargs):return io.StringIO('unrelated;never parse\n') if str(path).endswith('merged_gaze_data.csv') else original(path,*args,**kwargs)
  with patch.object(m.Path,'open',opened):sample,error=m.canonical(m.metadata()[0][0]['file_name'],(1500,3000))
  self.assertIsNone(sample);self.assertEqual(error,'missing_gaze')
if __name__=='__main__':unittest.main()
