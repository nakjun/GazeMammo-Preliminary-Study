#!/usr/bin/env python
"""Explicit missing-source TEST recovery; original metadata, models and freeze stay pinned."""
import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'runs/official_cbis_v2_experiment_1012'
META = ROOT / 'runs/official_cbis_preprocess_v2_1012'
REC = OUT / 'missing_source_recovery'


def original_runner():
    spec = importlib.util.spec_from_file_location('pinned_v2_runner', OUT / 'runner.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(body, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def prepare(approved):
    import pandas as pd
    from src import official_cbis_classifier_1012 as c
    runner = original_runner()
    state = json.loads((OUT / 'run_manifest.json').read_text())
    runner.verify(state)
    if any(state['stages'].get(s, {}).get('status') != 'complete' for s in ('baseline','improved','classifiers','freeze')):
        raise ValueError('All original training and freeze stages must be complete')
    freeze = c.verify_freeze(META / 'freeze_v2.json')
    train, test = c.population(META)
    # A mount outage must never be interpreted as per-image missingness.
    roots = {str(Path(p).parents[3]) for p in test.raw_path}
    if not all(Path(p).is_dir() for p in roots):
        raise ValueError('DICOM source mount/cohort root unavailable; run on the host')
    absent = test.loc[~test.raw_path.map(lambda p: Path(p).is_file())]
    if set(absent.image_id) != set(approved):
        raise ValueError(f'Actual missing IDs {absent.image_id.tolist()} differ from approved IDs {approved}; if none are missing, resume the original 645-image runner')
    selected = test.loc[~test.image_id.isin(approved)].copy()
    body = dict(kind='authorized_missing_source_test_subset', reason='User authorized exclusion of unavailable original TEST DICOM only', original_count=len(test), available_count=len(selected), original_test_sha256=c.digest(META/'test.csv'), original_freeze_sha256=freeze['sha256'], recovery_source_sha256=c.digest(__file__), exclusions=absent.to_dict('records'), selected_image_ids=selected.image_id.tolist(), patient_count=int(selected.patient_key.nunique()), class_counts={str(k):int(v) for k,v in selected.label.value_counts().sort_index().items()}, original_patient_count=int(test.patient_key.nunique()), source_roots_checked=sorted(roots))
    path = REC / 'authorization.json'
    if path.exists() and json.loads(path.read_text()) != body:
        raise ValueError('Recovery authorization changed')
    write(path, body)
    snapshot = REC/'test_available.csv'
    if snapshot.exists() and not pd.read_csv(snapshot).equals(selected.reset_index(drop=True)):
        raise ValueError('Recovery identity snapshot changed')
    selected.to_csv(snapshot,index=False)
    return runner, c, train, selected, body


@contextlib.contextmanager
def authorized_population(c, selected):
    original = c.population
    def subset(metadata):
        if Path(metadata).resolve() != META.resolve():
            raise ValueError('Only original pinned metadata is allowed')
        train, full = original(metadata)
        expected = full.loc[full.image_id.isin(selected.image_id)].reset_index(drop=True)
        if not expected.equals(selected.reset_index(drop=True)):
            raise ValueError('Authorized subset identity/labels changed')
        return train, expected
    c.population = subset
    try:
        yield
    finally:
        c.population = original


def audit(runner,c,selected,body):
    import numpy as np
    import pandas as pd
    import torch
    cache=torch.load(REC/'test.pt',map_location='cpu',weights_only=False)
    c.validate_cache(cache,META,'test')
    report=json.loads((REC/'test_evaluation/metrics.json').read_text())
    freeze=c.verify_freeze(META/'freeze_v2.json')
    if report['freeze_sha256']!=freeze['sha256'] or report['threshold']!=.5 or report['tta'] is not False:
        raise ValueError('Frozen evaluation settings changed')
    expected=selected[['image_id','patient_key','label']].reset_index(drop=True)
    comparison={}
    for arm in c.ARMS:
        old_scores=[]; new_scores=[]
        for seed in c.SEEDS:
            frame=pd.read_csv(REC/f'test_evaluation/{arm}_seed{seed}.csv')
            if not frame[['image_id','patient_key','label']].equals(expected):
                raise ValueError('Recovery prediction identity mismatch')
            actual=c.metrics(frame.label,frame.probability.to_numpy(dtype=np.float32))
            for key,value in actual.items():
                if not runner.metric_equal(key,value,report['arms'][arm]['single_heads'][str(seed)][key]):
                    raise ValueError('Recovery metric recomputation failed')
            old=pd.read_csv(ROOT/f'runs/official_cbis_trajectory_1012/test_evaluation/{arm}_seed{seed}.csv')
            old=old.set_index('image_id').loc[expected.image_id].reset_index()
            if not old[['image_id','patient_key','label']].equals(expected):
                raise ValueError('v1 common subset identity mismatch')
            old.to_csv(REC/f'v1_common/{arm}_seed{seed}.csv',index=False)
            old_scores.append(c.metrics(old.label,old.probability.to_numpy(dtype=np.float32)))
            new_scores.append(actual)
        comparison[arm]=dict(v1_common_single_heads={str(s):m for s,m in zip(c.SEEDS,old_scores)},v2_common_single_heads={str(s):m for s,m in zip(c.SEEDS,new_scores)},mean_head_auroc_delta_v2_minus_v1=float(np.mean([m['auroc'] for m in new_scores])-np.mean([m['auroc'] for m in old_scores])))
    bootstrap=report['paired_patient_bootstrap']
    if bootstrap['repetitions']!=5000 or bootstrap['seed']!=42:
        raise ValueError('Bootstrap registration changed')
    for reference in ('gaze_baseline','global','grid'):
        result=bootstrap['comparisons'][reference]['mean_head_auroc']
        delta=np.mean([report['arms']['gaze_improved']['single_heads'][str(s)]['auroc']-report['arms'][reference]['single_heads'][str(s)]['auroc'] for s in c.SEEDS])
        if not np.isclose(delta,result['delta']) or result['valid_replicates']<1 or result['low']>result['high']:
            raise ValueError('Paired bootstrap audit failed')
    runner.verify(json.loads((OUT/'run_manifest.json').read_text()))
    write(REC/'v1_v2_comparison.json',dict(common_test_count=len(expected),same_identity_labels=True,estimand='Mean single-head AUROC point difference on identical available images; no v1-v2 CI',arms=comparison))
    write(REC/'closing_audit.json',dict(complete=True,authorization=body,test_count=len(expected),heads=15,trajectories=2,original_pins_verified=True,original_freeze_verified=True,metrics_recomputed=True,common_v1_identity_verified=True,paired_bootstrap_verified=True))


def mark(stage, status='running', error=None):
    import time
    write(REC/'run_manifest.json',dict(kind='missing_source_recovery',stage=stage,status=status,updated=time.time(),error=error))


def run():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--approved-missing-image-id',type=int,action='append',required=True)
    parser.add_argument('--through',choices=['prepare','convert_test','test_cache','test_evaluation','audit'],default='audit')
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    import fcntl
    lock=(OUT/'runner.py').open('r')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    mark('prepare','initialized')
    mark('prepare')
    runner,c,train,selected,body=prepare(args.approved_missing_image_id)
    write(REC/'cohort_note.json',dict(original_test_count=body['original_count'],actual_test_count=len(selected),missing_source_exclusion_count=len(body['exclusions']),excluded_image_ids=args.approved_missing_image_id,limitation='Unavailable original DICOM excluded by explicit user authorization; all metrics use the exact available-image subset; no TEST tuning'))
    if args.through=='prepare':
        mark('prepare','complete');return
    from concurrent.futures import ProcessPoolExecutor
    from scripts.prepare_official_cbis_preprocess_v2_1012 import _convert_row
    mark('convert_test')
    with ProcessPoolExecutor(max_workers=8) as pool:
        results=list(pool.map(_convert_row,selected.to_dict('records')))
    write(REC/'conversion_test.json',dict(complete=True,authorization_sha256=c.digest(REC/'authorization.json'),row_hashes={str(identity):meta['canonical_sha256'] for identity,meta in results}))
    if args.through=='convert_test':
        mark('convert_test','complete');return
    mark('test_cache')
    with authorized_population(c,selected):
        if not (REC/'test.pt').exists():
            c.build_cache(META,OUT/'baseline/selected.pt',OUT/'improved/selected.pt',REC/'test.pt','test',args.device,META/'freeze_v2.json')
        cache=__import__('torch').load(REC/'test.pt',map_location='cpu',weights_only=False)
        c.validate_cache(cache,META,'test')
        binding=dict(cache_sha256=c.digest(REC/'test.pt'),authorization_sha256=c.digest(REC/'authorization.json'),snapshot_sha256=c.digest(REC/'test_available.csv'))
        binding_path=REC/'cache_binding.json'
        if binding_path.exists() and json.loads(binding_path.read_text())!=binding:
            raise ValueError('Recovery cache binding changed')
        write(binding_path,binding)
        if args.through=='test_cache':
            mark('test_cache','complete');return
        mark('test_evaluation')
        if not (REC/'test_evaluation').exists():
            c.evaluate_test(REC/'test.pt',META,META/'freeze_v2.json',REC/'test_evaluation')
        if args.through=='test_evaluation':
            mark('test_evaluation','complete');return
        mark('audit')
        (REC/'v1_common').mkdir(exist_ok=True)
        audit(runner,c,selected,body)
        mark('audit','complete')


def main():
    try:
        run()
    except BlockingIOError:
        raise
    except BaseException as error:
        path=REC/'run_manifest.json'
        stage=json.loads(path.read_text()).get('stage','prepare') if path.exists() else 'prepare'
        mark(stage,'failed',repr(error))
        raise

if __name__=='__main__':main()
