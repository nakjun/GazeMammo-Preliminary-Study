import torch
from src.deepgaze_context_1011 import dense_centerbias

def test_dense_bias_normalizes_probability_not_log_values():
    p=torch.tensor([[.1,.2],[.3,.4]])
    dense=dense_centerbias(p,(4,8),'cpu')
    assert torch.allclose(dense.exp().sum(),torch.tensor(1.),atol=1e-6)
    assert torch.isfinite(dense).all()
    assert dense[0,-1,-1]>dense[0,0,0]

def test_centerbias_rejects_incomplete_training_cohort():
    import tempfile
    from pathlib import Path
    import pandas as pd
    from src.deepgaze_context_1011 import train_centerbias
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'manifest.csv';pd.DataFrame([dict(image_id=i,patient_key=str(i),split='train') for i in range(412)]).to_csv(path,index=False)
        try:train_centerbias([],path)
        except ValueError as error:assert '412' in str(error)
        else:raise AssertionError('Incomplete cohort accepted')
