import importlib.util
from pathlib import Path
import pandas as pd
spec=importlib.util.spec_from_file_location('proposal_report',Path(__file__).parents[1]/'scripts/report_proposal_visit_1011.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def complete_frame():
    return pd.DataFrame([dict(image_id=i,patient_key=f'p{i//2}',policy=f'P{p}',seed=seed,**{key:1. for key in m.PRIMARY}) for i in range(93) for p in range(6) for seed in (42,43,44)])


def test_complete_and_duplicate_denominators():
    frame=complete_frame();assert len(m.validate_sampling(frame,'original'))==93
    duplicate=pd.concat([frame,frame.iloc[:1]],ignore_index=True)
    try:m.validate_sampling(duplicate,'original')
    except ValueError as error:assert 'duplicate' in str(error)
    else:raise AssertionError('duplicate accepted')


def test_missing_one_seed_rejected():
    frame=complete_frame().iloc[1:]
    try:m.validate_sampling(frame,'original')
    except ValueError as error:assert 'incomplete' in str(error)
    else:raise AssertionError('missing seed accepted')


def test_missing_artifacts_never_write_partial_report(tmp_path):
    output=tmp_path/'final'
    try:m.build(tmp_path,output)
    except FileNotFoundError as error:assert 'Final report blocked' in str(error)
    else:raise AssertionError('missing artifacts accepted')
    assert not output.exists()
