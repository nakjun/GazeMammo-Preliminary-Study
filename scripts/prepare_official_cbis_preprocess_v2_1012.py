#!/usr/bin/env python
"""Separate v2 cohort, RAW conversion and image-only GAA registration runner."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.official_cbis_data_1012 import (RUN_ROOT, load_manifest, sha256_file,
    validate_protocol, _write_csv, _write_json)
from src.official_cbis_preprocess_v2_1012 import canonical_image_v2, PREPROCESS_V2
from src.official_cbis_registration_v2_1012 import estimate_registration_v2
DEFAULT_OUTPUT = RUN_ROOT.parent / 'official_cbis_preprocess_v2_1012'
VERSION = PREPROCESS_V2['version']


def dependencies():
    return [Path(__file__).resolve(), Path(sys.modules[canonical_image_v2.__module__].__file__).resolve(),
            Path(sys.modules[estimate_registration_v2.__module__].__file__).resolve()]


def cohort(source=RUN_ROOT, output=DEFAULT_OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError('v2 output must differ from frozen source')
    old = validate_protocol(source)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'cohort_v2.json'
    hashes = {str((source / name).resolve()):sha256_file(source / name)
              for name in ('protocol.json', 'trainval.csv', 'test.csv', 'gaze.csv', 'audit.json')}
    hashes.update({str(p):sha256_file(p) for p in dependencies()})
    if state_path.exists():
        state = check_state(output)
        if state['source_hashes'] != hashes:
            raise ValueError('v2 cohort sources changed')
        return state
    counts = {}
    for name in ('trainval', 'test', 'gaze'):
        rows = load_manifest(source / f'{name}.csv')
        counts[name] = len(rows)
        for row in rows:
            row['canonical_path'] = str(output / 'canonical/raw' / f"{row['image_id']}.png")
            if name == 'gaze':
                row['gaze_canonical_path'] = str(output / 'canonical/gaze' / f"{row['source_image_id']}.png")
        _write_csv(output / ('gaze_candidates.csv' if name == 'gaze' else f'{name}.csv'), rows)
    state = dict(preprocessing_version=VERSION, source=str(source), source_hashes=hashes,
                 counts=counts, source_protocol=old,
                 manifest_hashes={n:sha256_file(output / n) for n in ('trainval.csv','test.csv','gaze_candidates.csv')})
    _write_json(state_path, state)
    return state


def check_state(output):
    output = Path(output)
    state = json.loads((output / 'cohort_v2.json').read_text())
    for path, digest in state['source_hashes'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'v2 source changed: {path}')
    for name, digest in state['manifest_hashes'].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f'v2 manifest changed: {name}')
    return state


def convert_one(source, target):
    source, target = Path(source), Path(target)
    digest = sha256_file(source)
    sidecar = Path(str(target)+'.json')
    if target.exists() or sidecar.exists():
        if not target.exists() or not sidecar.exists():
            raise ValueError('Incomplete canonical artifact')
        meta = json.loads(sidecar.read_text())
        if (meta['source_sha256'] != digest or meta['canonical_sha256'] != sha256_file(target)
                or meta['preprocessing'] != PREPROCESS_V2):
            raise ValueError('v2 canonical provenance mismatch')
        mask_path = Path(str(target)+'.mask.png')
        if not mask_path.exists() or meta.get('mask_sha256') != sha256_file(mask_path):
            raise ValueError('v2 canonical mask provenance mismatch')
        return meta
    image, meta, mask = canonical_image_v2(source, return_mask=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix('.partial.png')
    image.save(tmp)
    tmp.replace(target)
    mask_path = Path(str(target)+'.mask.png')
    mask.save(mask_path)
    meta.update(source_path=str(source), source_sha256=digest, canonical_sha256=sha256_file(target),
                mask_path=str(mask_path), mask_sha256=sha256_file(mask_path))
    _write_json(sidecar, meta)
    return meta


def _convert_row(row):
    return row['image_id'], convert_one(row['raw_path'], row['canonical_path'])

def convert(output=DEFAULT_OUTPUT, split='trainval', limit=None, frozen_test=False, workers=4):
    output = Path(output)
    check_state(output)
    if split not in ('trainval','test'):
        raise ValueError('Invalid split')
    if split == 'test':
        validate_protocol(output)
        freeze = json.loads((output / 'freeze_v2.json').read_text()) if (output / 'freeze_v2.json').exists() else {}
        if not frozen_test or freeze.get('frozen') is not True or freeze.get('protocol_sha256') != sha256_file(output / 'protocol.json'):
            raise ValueError('TEST requires explicit flag and matching v2 freeze')
    rows = load_manifest(output / f'{split}.csv')
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    selected = rows if limit is None else rows[:limit]
    if workers < 1:
        raise ValueError('Workers must be positive')
    state = dict(preprocessing_version=VERSION, split=split, complete=False, row_hashes={}, failed_rows={})
    state_path = output / f'conversion_{split}.json'
    progress_path = output / f'conversion_{split}_progress.jsonl'
    def record(identity, meta):
        state['row_hashes'][str(identity)] = meta['canonical_sha256']
        if len(state['row_hashes']) % 25 == 0:
            _write_json(state_path, state)
            with progress_path.open('a') as log:
                log.write(json.dumps(dict(converted=len(state['row_hashes']), total=len(selected), image_id=identity))+'\n')
    try:
        if workers == 1:
            for row in selected:
                try:
                    identity, meta = _convert_row(row)
                    record(identity,meta)
                except Exception as exc:
                    state['failed_rows'][str(row['image_id'])]=str(exc)
                    raise
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                # Ordered results preserve reproducible cohort progress and fail without dropping rows.
                futures = [(row,pool.submit(_convert_row,row)) for row in selected]
                for row, future in futures:
                    try:
                        identity,meta = future.result()
                        record(identity,meta)
                    except Exception as exc:
                        state['failed_rows'][str(row['image_id'])] = str(exc)
                        for _,pending in futures: pending.cancel()
                        raise
    except Exception as exc:
        state['status']='failed'
        state['failure_reason']=str(exc)
        _write_json(state_path,state)
        raise
    state['complete'] = limit is None and len(selected) == len(rows)
    state['status']='complete' if state['complete'] else 'partial'
    _write_json(state_path, state)
    return state


def register_pair(row, output):
    raw = Path(row['canonical_path'])
    if not raw.exists():
        raise ValueError('RAW canonical conversion must precede registration')
    gaa, gaa_meta, gaa_mask = canonical_image_v2(row['source_path'], return_mask=True)
    with Image.open(raw) as im, Image.open(str(raw)+'.mask.png') as raw_mask:
        matrix, quality = estimate_registration_v2(gaa, im.convert('L'), gaze_mask=np.asarray(gaa_mask)>0, raw_mask=np.asarray(raw_mask)>0)
    raw_meta_path = Path(str(raw)+'.json')
    raw_source_sha = json.loads(raw_meta_path.read_text()).get('source_sha256') if raw_meta_path.exists() else None
    audit = dict(image_id=row['image_id'], source_image_id=row['source_image_id'], quality=quality,
                 gaa_source_sha256=sha256_file(row['source_path']), raw_canonical_sha256=sha256_file(raw),
                 raw_source_sha256=raw_source_sha,
                 registration_source_sha256=sha256_file(dependencies()[2]), pair_matrix=np.asarray(matrix).tolist())
    if quality.get('passed'):
        affine = np.vstack([np.asarray(gaa_meta['affine']), [0,0,1]])
        composed = np.asarray(matrix) @ affine
        target = Path(row['gaze_canonical_path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if sha256_file(target) != sha256_file(raw):
                raise ValueError('Existing gaze alias differs from RAW canonical')
        else:
            try:
                os.link(raw, target)
            except OSError:
                shutil.copyfile(raw, target)
        meta = dict(gaa_meta, affine=composed[:2].tolist(), gaa_preprocessing_affine=gaa_meta['affine'], source_path=row['source_path'],
                    source_sha256=audit['gaa_source_sha256'], canonical_sha256=audit['raw_canonical_sha256'],
                    raw_canonical_path=str(raw), registration=audit, preprocessing_version=VERSION)
        _write_json(str(target)+'.json', meta)
    return audit


def register(output=DEFAULT_OUTPUT, limit=None):
    output = Path(output)
    state = check_state(output)
    conversion = json.loads((output / 'conversion_trainval.json').read_text())
    if not conversion['complete']:
        raise ValueError('Complete trainval RAW conversion required')
    rows = load_manifest(output / 'gaze_candidates.csv')
    accepted, audits = [], []
    if limit is not None and limit < 1:
        raise ValueError('Limit must be positive')
    for row in rows if limit is None else rows[:limit]:
        audit_path = output / 'registration' / f"{row['source_image_id']}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            if audit_path.exists():
                audit = json.loads(audit_path.read_text())
                if (audit.get('gaa_source_sha256') != sha256_file(row['source_path'])
                    or audit.get('raw_canonical_sha256') != sha256_file(row['canonical_path'])
                    or audit.get('registration_source_sha256') != sha256_file(dependencies()[2])):
                    raise ValueError('Registration resume provenance mismatch')
                if audit['quality'].get('passed') and (not Path(row['gaze_canonical_path']).exists() or sha256_file(row['gaze_canonical_path']) != audit['raw_canonical_sha256']):
                    raise ValueError('Registration resume alias mismatch')
            else:
                audit = register_pair(row, output)
        except (ValueError, OSError) as exc:
            if 'resume' in str(exc):
                raise
            audit = dict(image_id=row['image_id'], source_image_id=row['source_image_id'],
                         quality=dict(passed=False, failure_reason=str(exc)),
                         gaa_source_sha256=sha256_file(row['source_path']),
                         raw_canonical_sha256=sha256_file(row['canonical_path']),
                         registration_source_sha256=sha256_file(dependencies()[2]))
        _write_json(audit_path, audit)
        audits.append(audit)
        if len(audits) % 25 == 0:
            with (output/'registration_progress.jsonl').open('a') as log:
                log.write(json.dumps(dict(processed=len(audits), total=len(rows), source_image_id=row['source_image_id']))+'\n')
        if audit['quality'].get('passed'):
            accepted.append(row)
    report = dict(original=len(rows), processed=len(audits), accepted=len(accepted), rejected=len(audits)-len(accepted),
                  complete=limit is None and len(audits)==len(rows), failures=dict(Counter(a['quality'].get('failure_reason') for a in audits if not a['quality'].get('passed'))), pairs=audits)
    _write_json(output / 'registration_report.json', report)
    if report['complete']:
        if not accepted:
            raise ValueError('No reliable registration pairs; protocol remains unpublished')
        _write_csv(output / 'gaze.csv', accepted)
        _write_json(output / 'audit.json', dict(counts=state['counts'], registration={k:v for k,v in report.items() if k!='pairs'}))
        old = state['source_protocol']
        protocol = dict(old, preprocessing=PREPROCESS_V2, preprocessing_version=VERSION,
                        source_hashes=state['source_hashes'], test_pixels_read=False,
                        artifact_hashes={n:sha256_file(output/n) for n in ('trainval.csv','test.csv','gaze.csv','audit.json','registration_report.json','conversion_trainval.json')})
        _write_json(output / 'protocol.json', protocol)
        validate_protocol(output)
    return report



def qa(output=DEFAULT_OUTPUT):
    """Development-only before/after figures; image agreement is not clinical validation."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    check_state(output)
    report = json.loads((output / 'registration_report.json').read_text())
    rows = load_manifest(output / 'gaze_candidates.csv')
    by_id = {r['source_image_id']:r for r in rows}
    pairs = report['pairs']
    worst = sorted(pairs,key=lambda a:a['quality'].get('registration_confidence',0))[:3]
    chosen = list(dict.fromkeys([r['source_image_id'] for r in rows[:3]]+[p['source_image_id'] for p in worst]))
    selected = []
    for sid in chosen:
        row = by_id[sid]
        audit = next((a for a in pairs if a['source_image_id']==sid),None)
        if audit is None: continue
        gaa,_ = canonical_image_v2(row['source_path'])
        with Image.open(row['canonical_path']) as im: raw=np.asarray(im.convert('L'))
        ga=np.asarray(gaa)
        matrix=np.asarray(audit.get('pair_matrix',[]))
        aligned=cv2.warpAffine(ga,matrix[:2],(raw.shape[1],raw.shape[0])) if matrix.shape==(3,3) and np.isfinite(matrix).all() else np.zeros_like(raw)
        fig,axes=plt.subplots(1,4,figsize=(12,6))
        for ax,arr,title in zip(axes,[ga,raw,aligned,np.abs(raw.astype(float)-aligned)],['GAA canonical','RAW canonical','Registered GAA','Absolute difference']):
            ax.imshow(arr,cmap='gray'); ax.set_title(title); ax.axis('off')
        fig.suptitle(f"Source {sid}: {audit['quality']} (image agreement only)",fontsize=8)
        (output/'qa').mkdir(exist_ok=True)
        path=output/'qa'/f'{sid}.png'; fig.savefig(path,dpi=100,bbox_inches='tight'); plt.close(fig)
        selected.append(dict(source_image_id=sid,figure=str(path),quality=audit['quality']))
    summary=dict(scope='development_trainval_only',interpretation='Image agreement metrics do not establish clinical success',selected=selected)
    _write_json(output/'qa_report.json',summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['cohort','converttrainval','register','qa','status','converttest'])
    parser.add_argument('--source', type=Path, default=RUN_ROOT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--frozen-test', action='store_true')
    args = parser.parse_args()
    if args.stage == 'cohort': result = cohort(args.source,args.output)
    elif args.stage in ('converttrainval','converttest'): result = convert(args.output, 'test' if args.stage=='converttest' else 'trainval', args.limit,args.frozen_test,args.workers)
    elif args.stage == 'register': result = register(args.output,args.limit)
    elif args.stage == 'qa': result = qa(args.output)
    else:
        check_state(args.output)
        result = {p.name:json.loads(p.read_text()) for p in args.output.glob('*report.json')}
    if args.stage == 'cohort':
        result = {k:result[k] for k in ('preprocessing_version','source','counts')}
    if args.stage == 'register':
        result = {k:v for k,v in result.items() if k != 'pairs'}
    print(json.dumps(result, indent=2))

if __name__ == '__main__': main()
