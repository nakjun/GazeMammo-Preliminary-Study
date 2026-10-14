import inspect
import torch
from src.proposal_policy_1011 import policy_distribution,proposal_budget_indices,rollout


def geometry(n):return dict(geometry_valid=True,background=torch.linspace(0,1,n),edge=torch.linspace(1,0,n))


def test_normalization_and_alpha_zero():
    raw=torch.randn(1,12).log_softmax(-1);r=torch.rand(1,12);g=geometry(12)
    original=policy_distribution(raw,r,g,'P0')['q']
    for policy in ('P0','P1','P2','P3','P4','P5'):
        d=policy_distribution(raw,r,g,policy)
        assert torch.isfinite(d['q']).all()
        assert torch.allclose(d['q'].sum(-1),torch.ones(1))
    assert torch.equal(policy_distribution(raw,r,g,'P3',alpha=0)['q'],original)
    assert torch.equal(policy_distribution(raw,r,g,'P4',alpha=0)['q'],original)
    expected=(raw+2*r-3*g['background']-.5*g['edge']*(1-r)).softmax(-1)
    assert torch.allclose(original,expected,atol=1e-7,rtol=1e-6)


def test_budget_nonoverlap():
    rect=torch.tensor([[0.,0,10,10],[1,1,11,11],[20,20,30,30],[40,40,50,50]])
    assert proposal_budget_indices(torch.tensor([.9,.8,.7,.6]),rect)==[0,2,3]


def test_rollout_has_no_annotation_argument():
    assert set(inspect.signature(rollout).parameters)=={'model','memory','image','geometry','policy','steps','seed','alpha','tau','strategy','roi_relevance'}


def test_invalid_proposal_falls_back():
    raw=torch.randn(1,12).log_softmax(-1);g=geometry(12)
    baseline=policy_distribution(raw,torch.zeros(1,12),g,'P0',proposal_valid=False)['q']
    for policy in ('P3','P4'):
        q=policy_distribution(raw,torch.full((1,12),float('nan')),g,policy)['q']
        assert torch.equal(q,baseline)
        assert torch.isfinite(q).all()


def test_manifest_routes_exact_comparable_validation():
    import csv,sys
    from argparse import Namespace
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/'scripts'))
    from proposal_policy_1011 import image_records
    args=Namespace(manifest=str(root/'manifests/all.csv'),split='val',image_ids=None)
    records=image_records(args)
    with (root/'manifests/trajectory.csv').open() as source:
        expected={int(r['image_id']) for r in csv.DictReader(source) if r['split']=='val'}
    assert len(records)==93
    assert {r['image_id'] for r in records}==expected
    assert args.requested_routing_count==102 and args.comparable_routing_count==93
