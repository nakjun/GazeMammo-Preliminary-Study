import tempfile,json
from pathlib import Path
import torch
from scripts.export_deepgaze_clinical_raw_1011 import terminal_trial,alignment,canonical_targets

def test_export_refuses_nonterminal_trial():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'config.json').write_text('{"epochs":15}')
        try:terminal_trial(root)
        except ValueError as error:assert 'not terminal' in str(error)
        else:raise AssertionError('Live trial accepted')

def test_parent_alignment_rejects_changed_target_and_mask():
    coords=torch.full((16,2),.5);record=dict(patient_key='trainpatient',split='train',coords=coords,loss_mask=torch.ones(16,dtype=torch.bool),target=canonical_targets(coords))
    alignment(record,record)
    for key in ['target','loss_mask']:
        altered={**record,key:record[key].clone()};altered[key][0]=0
        try:alignment(record,altered)
        except ValueError:pass
        else:raise AssertionError(f'Mismatch {key} accepted')
