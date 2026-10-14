import torch
from scripts.compare_deepgaze_clinical_mixtures_1011 import mix,verify_parity

def test_three_prespecified_mixtures_normalize():
    a=torch.tensor([[.2,.8]]).log();b=torch.tensor([[.8,.2]]).log()
    for weight in [.25,.5,.75]:assert torch.allclose(mix(a,b,weight).exp(),a.exp()*(1-weight)+b.exp()*weight,atol=1e-7)

def test_incomplete_actual_parity_rejected():
    try:verify_parity({'actual_val_parity':{'requested':True,'images':1,'records':[{'image_id':64,'max_logp_error':0.}]}})
    except ValueError:pass
    else:raise AssertionError('Incomplete93 parityaccepted')
