"""Independent official-CBIS cohort and image-only canonical preprocessing.

Metadata may be audited for TEST, but pixel conversion requires a frozen run.
Neither pathology masks nor cropped DICOM series enter image preprocessing.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = Path('/data/nas/Breast/Public_Mammo/CBIS-DDSM/CBIS-DDSM-All-doiJNLP-zzWs5zfZ')
RUN_ROOT = ROOT / 'runs/official_cbis_trajectory_1012'
CANVAS = (1500, 3000)
PREPROCESS = dict(canvas=list(CANVAS), foreground_threshold=10,
                  percentile=[0.5, 99.5], monochrome1_inversion=True,
                  interpolation='bilinear', coordinate_convention='pixel_centers_half_pixel',
                  foreground='image_intensity_only', roi_used=False)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path):
    with Path(path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ('image_id', 'label', 'source_image_id'):
            if key in r:
                r[key] = int(r[key])
    return rows


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f'Empty manifest: {path}')
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def pathology_label(value):
    mapping = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}
    value = str(value).strip().upper()
    if value not in mapping:
        raise ValueError(f'Unknown official pathology: {value!r}')
    return mapping[value]


def source_patient(sn):
    m = re.fullmatch(r'(Mass|Calc)-(Training|Test)_P_(\d{5})_(LEFT|RIGHT)_(CC|MLO)', sn)
    if not m:
        raise ValueError(f'Not a full mammogram subject: {sn!r}')
    return 'CBIS_' + m[3]


def patient_split(rows, seed=42, validation_fraction=.15):
    """One stratified split of patients; all images of a patient stay together."""
    from sklearn.model_selection import train_test_split
    labels = defaultdict(int)
    for r in rows:
        labels[r['patient_key']] = max(labels[r['patient_key']], int(r['label']))
    patients = sorted(labels)
    tr, va = train_test_split(patients, test_size=validation_fraction,
                             random_state=seed, stratify=[labels[p] for p in patients])
    return {p: 'train' for p in tr} | {p: 'val' for p in va}


def purge_test_patients(train, test):
    forbidden = {r['patient_key'] for r in test}
    return ([r for r in train if r['patient_key'] not in forbidden],
            [r for r in train if r['patient_key'] in forbidden])


def build_cohort(metadata_dir, raw_root, output, gaze_manifest=None, strict=True):
    metadata_dir, raw_root, output = map(Path, (metadata_dir, raw_root, output))
    gaze_manifest = Path(gaze_manifest or ROOT / 'manifests/all.csv')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'protocol.json').exists():
        validate_protocol(output)
        return json.loads((output / 'protocol.json').read_text())
    dependencies = [raw_root / 'metadata.csv', gaze_manifest,
                    metadata_dir / 'download_provenance.json', Path(__file__)]
    downloaded = json.loads((metadata_dir / 'download_provenance.json').read_text())
    for item in downloaded:
        p = metadata_dir / item['path']
        if sha256_file(p) != item['sha256']:
            raise ValueError(f'Official CSV download hash mismatch: {p}')
        dependencies.append(p)
    labels, lesion_count = defaultdict(list), Counter()
    for category in ('mass', 'calc'):
        for split in ('train', 'test'):
            path = metadata_dir / f'{category}_case_description_{split}_set.csv'
            for r in load_manifest(path):
                sn = r['image file path'].strip().replace('\\', '/').split('/')[0]
                expected = 'Training' if split == 'train' else 'Test'
                if f'-{expected}_' not in sn or sn.split('-')[0].lower() != category:
                    raise ValueError(f'Case CSV routing mismatch: {sn}')
                source_patient(sn)
                labels[sn].append(pathology_label(r['pathology']))
                lesion_count[sn] += 1
    metadata = load_manifest(raw_root / 'metadata.csv')
    full = [r for r in metadata if r['Series Description'] == 'full mammogram images']
    by_subject = defaultdict(list)
    for r in full:
        by_subject[r['Subject ID']].append(r)
    if any(len(v) != 1 for v in by_subject.values()):
        raise ValueError('Multiple full mammogram series for a subject')
    missing_labels = sorted(set(by_subject) - set(labels))
    missing_images = sorted(set(labels) - set(by_subject))
    if missing_labels:
        raise ValueError(f'Full images without official pathology: {missing_labels}')
    if strict and missing_images != ['Calc-Training_P_00673_LEFT_MLO']:
        raise ValueError(f'Unexpected missing full mammograms: {missing_images}')
    rows = []
    for identity, sn in enumerate(sorted(by_subject), start=1):
        mr = by_subject[sn][0]
        folder = (raw_root / mr['File Location']).resolve()
        if not folder.is_relative_to(raw_root.resolve() / 'CBIS-DDSM'):
            raise ValueError('DICOM path escaped raw full-image root')
        files = sorted(folder.glob('*.dcm'))
        if len(files) != 1 or not files[0].is_file():
            raise ValueError(f'Expected exactly one full-image DICOM: {folder}')
        official = 'train' if '-Training_' in sn else 'test'
        rows.append(dict(image_id=identity, source_sn=sn, patient_key=source_patient(sn),
                         patient_id=source_patient(sn), label=max(labels[sn]),
                         split=official, official_split=official,
                         raw_path=str(files[0]), canonical_path=str(output.resolve() / 'canonical/raw' / f'{identity}.png')))
    train, test = [r for r in rows if r['split'] == 'train'], [r for r in rows if r['split'] == 'test']
    safe, purged = purge_test_patients(train, test)
    assignments = patient_split(safe)
    for r in safe:
        r['split'] = assignments[r['patient_key']]
    safe_by_sn = {r['source_sn']: r for r in safe}
    gaze, exclusion = [], Counter()
    for old in load_manifest(gaze_manifest):
        if old['source'] != 'CBIS-DDSM':
            exclusion['not_cbis'] += 1
            continue
        if old['filename_split'] != 'Training':
            exclusion['not_official_training_filename'] += 1
            continue
        sn = re.sub(r'_(BENIGN_WITHOUT_CALLBACK|MALIGNANT|BENIGN|UNKNOWN)\.png$', '', old['file_name'])
        if sn not in safe_by_sn:
            exclusion['not_safe_official_training_image'] += 1
            continue
        r = dict(safe_by_sn[sn])
        p = (ROOT / 'data/image' / old['file_name']).resolve()
        if not p.is_file():
            raise ValueError(f'Missing GAA source PNG: {p}')
        r.update(file_name=old['file_name'], source_image_id=old['image_id'], source_path=str(p),
                 gaze_png_path=str(p), gaze_filename=old['file_name'],
                 gaze_csv_path=str(ROOT / 'data/gaze_data/merged_gaze_data.csv'),
                 gaze_canonical_path=str(output.resolve() / 'canonical/gaze' / f'{old["image_id"]}.png'))
        gaze.append(r)
    if len({r['image_id'] for r in gaze}) != len(gaze):
        raise ValueError('Multiple gaze PNGs map to one official image')
    _write_csv(output / 'trainval.csv', safe)
    _write_csv(output / 'test.csv', test)
    _write_csv(output / 'gaze.csv', gaze)
    audit = dict(official_train_images=len(train), official_test_images=len(test),
                 official_train_patients=len({r['patient_key'] for r in train}),
                 official_test_patients=len({r['patient_key'] for r in test}),
                 overlap_patients=sorted({r['patient_key'] for r in purged}),
                 purged_training_images=len(purged), purged_subjects=[r['source_sn'] for r in purged],
                 safe_training_images=len(safe), safe_training_patients=len(assignments),
                 classification_splits=dict(Counter(r['split'] for r in safe)),
                 gaze_splits=dict(Counter(r['split'] for r in gaze)),
                 gaze_exclusions=dict(exclusion), missing_full_subjects=missing_images,
                 multi_lesion_subjects={sn:n for sn,n in lesion_count.items() if n > 1},
                 conflicting_lesion_labels=sorted(sn for sn,v in labels.items() if len(set(v)) > 1),
                 metadata_series_counts=dict(Counter(r['Series Description'] for r in metadata)))
    if strict and (len(train), len(test), len(purged), len(safe), len(assignments)) != (2457,645,64,2393,1217):
        raise ValueError(f'Unexpected official cohort counts: {audit}')
    _write_json(output / 'audit.json', audit)
    manifests = [output / x for x in ('trainval.csv', 'test.csv', 'gaze.csv', 'audit.json')]
    protocol = dict(kind='official_cbis_1012_v1', seed=42, validation_fraction=.15,
                    split_unit='patient', test_patient_exclusion=True, initialization='ImageNet_only',
                    preprocessing=PREPROCESS, raw_root=str(raw_root.resolve()),
                    source_hashes={str(p.resolve()):sha256_file(p) for p in dependencies},
                    artifact_hashes={p.name:sha256_file(p) for p in manifests},
                    patient_splits=assignments, test_patient_keys=sorted({r['patient_key'] for r in test}),
                    development_patient_keys=sorted(assignments), test_pixels_read=False)
    _write_json(output / 'protocol.json', protocol)
    return protocol


def validate_protocol(output):
    output = Path(output)
    protocol = json.loads((output / 'protocol.json').read_text())
    if protocol['kind'] != 'official_cbis_1012_v1':
        raise ValueError('Unknown protocol kind')
    for path, digest in protocol['source_hashes'].items():
        if sha256_file(path) != digest:
            raise ValueError(f'Protocol source changed: {path}')
    for name, digest in protocol['artifact_hashes'].items():
        if sha256_file(output / name) != digest:
            raise ValueError(f'Protocol artifact changed: {name}')
    rows = load_manifest(output / 'trainval.csv')
    test_patients = set(protocol['test_patient_keys'])
    if any(r['patient_key'] in test_patients for r in rows):
        raise ValueError('TEST patient leaked into development cohort')
    if any(protocol['patient_splits'][r['patient_key']] != r['split'] for r in rows):
        raise ValueError('Patient split changed')
    gaze = load_manifest(output / 'gaze.csv')
    if any(r['patient_key'] in test_patients or protocol['patient_splits'][r['patient_key']] != r['split'] for r in gaze):
        raise ValueError('Gaze/classification patient split disagreement')
    return protocol


def canonical_image(image_or_dicom):
    """Image-only crop/resize/pad; return affine from original pixel centers.

    PNG intensities remain in their original 8-bit scale. DICOMs use modality
    rescale, MONOCHROME1 inversion, then finite .5/99.5 percentile clipping.
    """
    is_dicom = not isinstance(image_or_dicom, Image.Image) and Path(image_or_dicom).suffix.lower() == '.dcm'
    dicom_meta = dict(input_kind='png', monochrome1=False)
    if is_dicom:
        import pydicom
        ds = pydicom.dcmread(image_or_dicom)
        a = np.asarray(ds.pixel_array, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError('Full mammogram must be a single grayscale frame')
        a = a * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
        finite = np.isfinite(a)
        if not finite.any():
            raise ValueError('DICOM has no finite pixels')
        mono1 = str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1'
        if mono1:
            a = np.nanmax(a[finite]) + np.nanmin(a[finite]) - a
        lo, hi = np.percentile(a[finite], PREPROCESS['percentile'])
        if hi <= lo:
            raise ValueError('DICOM percentile intensity range is empty')
        a = np.clip((np.where(finite, a, lo) - lo) * (255. / (hi - lo)), 0, 255).astype(np.uint8)
        source = Image.fromarray(a)
        dicom_meta = dict(input_kind='dicom', monochrome1=mono1,
                          intensity_low=float(lo), intensity_high=float(hi))
    else:
        if isinstance(image_or_dicom, Image.Image):
            source = image_or_dicom.convert('L')
        else:
            with Image.open(image_or_dicom) as im:
                source = im.convert('L')
        a = np.asarray(source)
    yy, xx = np.nonzero(a > PREPROCESS['foreground_threshold'])
    if not len(xx):
        raise ValueError('Image-only foreground is empty')
    bbox = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
    cw, ch = bbox[2] - bbox[0], bbox[3] - bbox[1]
    scale = min(CANVAS[0] / cw, CANVAS[1] / ch)
    # Python round intentionally uses ties-to-even, recorded in affine metadata.
    resized = (max(1, min(CANVAS[0], round(cw * scale))), max(1, min(CANVAS[1], round(ch * scale))))
    sx, sy = resized[0] / cw, resized[1] / ch
    left, top = (CANVAS[0] - resized[0]) // 2, (CANVAS[1] - resized[1]) // 2
    output = Image.new('L', CANVAS, 0)
    output.paste(source.crop(bbox).resize(resized, Image.Resampling.BILINEAR), (left, top))
    affine = [[sx, 0., left + (0.5 - bbox[0]) * sx - 0.5],
              [0., sy, top + (0.5 - bbox[1]) * sy - 0.5]]
    metadata = dict(preprocessing=PREPROCESS, original_size=list(source.size), crop_bbox=list(bbox),
                    resized_size=list(resized), padding=[left, top], affine=affine,
                    rounding='Python ties-to-even', **dicom_meta)
    return output, metadata


def transform_points(points, metadata):
    p = np.asarray(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError('Points must have shape N,2')
    affine = np.asarray(metadata['affine'], dtype=np.float64)
    if affine.shape != (2,3) or not np.isfinite(affine).all():
        raise ValueError('Invalid coordinate affine')
    # Preserve recording gaps and off-canvas samples as invalid segment breaks.
    valid = np.isfinite(p).all(axis=1)
    w, h = metadata['original_size']
    valid &= (p[:,0] >= 0) & (p[:,0] < w) & (p[:,1] >= 0) & (p[:,1] < h)
    transformed = p @ affine[:,:2].T + affine[:,2]
    transformed[~valid] = np.nan
    return transformed


def _convert_one(source, target):
    source, target = Path(source), Path(target)
    meta_path = Path(str(target) + '.json')
    source_hash = sha256_file(source)
    if target.exists() or meta_path.exists():
        if not target.exists() or not meta_path.exists():
            raise ValueError(f'Incomplete canonical artifact: {target}')
        meta = json.loads(meta_path.read_text())
        if meta['source_sha256'] != source_hash or meta['canonical_sha256'] != sha256_file(target) or meta['preprocessing'] != PREPROCESS:
            raise ValueError(f'Canonical artifact provenance mismatch: {target}')
        return meta
    image, meta = canonical_image(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.partial.png')
    image.save(temporary)
    temporary.replace(target)
    meta.update(source_path=str(source), source_sha256=source_hash,
                canonical_sha256=sha256_file(target))
    _write_json(meta_path, meta)
    return meta


def convert_manifest(output, split='trainval', limit=None, test_authorized=False):
    output = Path(output)
    validate_protocol(output)
    if split not in ('trainval', 'test'):
        raise ValueError('Conversion split must be trainval or test')
    if split == 'test':
        freeze_path = output / 'freeze.json'
        if not freeze_path.exists():
            raise ValueError('TEST pixel conversion blocked before classifier freeze')
        freeze = json.loads(freeze_path.read_text())
        if freeze.get('protocol_sha256') != sha256_file(output / 'protocol.json'):
            raise ValueError('TEST pixel conversion blocked: freeze protocol mismatch')
        if not (freeze.get('frozen') is True or test_authorized):
            raise ValueError('TEST pixel conversion blocked: invalid freeze marker')
    rows = load_manifest(output / f'{split}.csv')
    if limit is not None:
        if limit < 1:
            raise ValueError('Conversion limit must be positive')
        rows = rows[:limit]
    state = dict(kind='official_cbis_conversion_v1', split=split,
                 protocol_sha256=sha256_file(output / 'protocol.json'),
                 complete=False, row_hashes={})
    state_path = output / f'conversion_{split}.json'
    for r in rows:
        meta = _convert_one(r['raw_path'], r['canonical_path'])
        state['row_hashes'][str(r['image_id'])] = meta['canonical_sha256']
        _write_json(state_path, state)
    if split == 'trainval':
        allowed = {r['image_id'] for r in rows}
        for r in load_manifest(output / 'gaze.csv'):
            if r['image_id'] in allowed:
                _convert_one(r['source_path'], r['gaze_canonical_path'])
    state['complete'] = limit is None
    _write_json(state_path, state)
    return state
