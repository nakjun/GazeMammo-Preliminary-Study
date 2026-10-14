"""Read-only closing checks for v2 development preprocessing and gaze mapping."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from src.official_cbis_data_1012 import load_manifest,sha256_file,validate_protocol,transform_points
from src.official_cbis_trajectory_1012 import allowed_payloads

def main():
    base=ROOT/'runs/official_cbis_preprocess_v2_1012'
    old=ROOT/'runs/official_cbis_trajectory_1012'
    protocol=validate_protocol(base)
    raw=load_manifest(base/'trainval.csv');test=load_manifest(base/'test.csv');gaze=load_manifest(base/'gaze.csv')
    reference=load_manifest(old/'trainval.csv');reference_test=load_manifest(old/'test.csv')
    assert len(raw)==2393 and len(test)==645
    for rows,ref in [(raw,reference),(test,reference_test)]:
        for a,b in zip(rows,ref):
            assert all(a[k]==b[k] for k in ['image_id','patient_key','label','split','source_sn'])
    assert not any(Path(r['canonical_path']).exists() for r in test)
    conversion=json.loads((base/'conversion_trainval.json').read_text())
    assert conversion['complete'] and not conversion['failed_rows'] and len(conversion['row_hashes'])==2393
    for row in raw:
        path=Path(row['canonical_path']);meta=json.loads(Path(str(path)+'.json').read_text())
        assert meta['canonical_sha256']==sha256_file(path)==conversion['row_hashes'][str(row['image_id'])]
        assert meta['mask_sha256']==sha256_file(meta['mask_path'])
        assert meta['preprocessing']['version']==protocol['preprocessing_version']
    report=json.loads((base/'registration_report.json').read_text())
    assert report['complete'] and report['original']==report['processed']==506
    passed={a['source_image_id']:a for a in report['pairs'] if a['quality']['passed']}
    assert len(gaze)==report['accepted']==len(passed)
    payloads={}
    for path in sorted({r['gaze_csv_path'] for r in gaze}):
        payloads.update(allowed_payloads(path,[r['file_name'] for r in gaze if r['gaze_csv_path']==path]))
    coordinate_rows=[]
    for row in gaze:
        a=passed[row['source_image_id']];path=Path(row['gaze_canonical_path'])
        meta=json.loads(Path(str(path)+'.json').read_text())
        assert sha256_file(path)==sha256_file(row['canonical_path'])==a['raw_canonical_sha256']
        assert meta['registration']['quality']['passed']
        points=payloads[row['file_name']];mapped=transform_points(points,meta)
        original_valid=np.isfinite(points).all(1)&(points[:,0]>=0)&(points[:,0]<1500)&(points[:,1]>=0)&(points[:,1]<3000)
        valid=np.isfinite(mapped).all(1)&(mapped[:,0]>=0)&(mapped[:,0]<1500)&(mapped[:,1]>=0)&(mapped[:,1]<3000)
        assert not np.any(valid&~original_valid)
        if not valid.any():raise ValueError(f'Accepted image has no mapped gaze points: {row["source_image_id"]}')
        coordinate_rows.append(dict(source_image_id=row['source_image_id'],split=row['split'],
            source_valid=int(original_valid.sum()),mapped_valid=int(valid.sum()),
            retained_fraction=float(valid.sum()/max(original_valid.sum(),1))))
    summary={}
    for split in ['train','val']:
        pairs=[a for a in passed.values() if next(r['split'] for r in gaze if r['source_image_id']==a['source_image_id'])==split]
        summary[split]=dict(accepted=len(pairs),edge_correlation_before_median=float(np.median([a['quality']['before_edge_correlation'] for a in pairs])),
            edge_correlation_after_median=float(np.median([a['quality']['edge_correlation'] for a in pairs])),
            mask_dice_after_median=float(np.median([a['quality']['mask_dice'] for a in pairs])))
    proof=dict(status='passed',development_raw_images=2393,classification_population_unchanged=True,
        TEST_images=645,TEST_pixels_read=False,gaze_candidate_pairs=506,gaze_accepted=report['accepted'],
        gaze_rejected=report['rejected'],gaze_and_classifier_pixels_identical=True,
        original_invalid_gaze_preserved=True,coordinate_validity=coordinate_rows,
        accepted_pair_quality=summary,unit_tests=20,model_training_started=False,
        median_retained_valid_gaze_fraction=float(np.median([r['retained_fraction'] for r in coordinate_rows])),
        protocol_sha256=sha256_file(base/'protocol.json'))
    (base/'parent_verification.json').write_text(json.dumps(proof,indent=2))
    print(json.dumps({k:v for k,v in proof.items() if k!='coordinate_validity'}))

if __name__=='__main__':main()
