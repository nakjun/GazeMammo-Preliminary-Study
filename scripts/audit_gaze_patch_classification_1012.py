"""Independent closing checks for the development smoke; no training or inference."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from src.gaze_patch_features_1012 import ARMS, SOURCE_SHA256, validate_feature_cache
from src.gaze_patch_classifier_1012 import restore, forward, digest


def main():
    torch.set_num_threads(4)
    feature_dir = ROOT/'runs/gaze_patch_classification_1012_smoke8'
    heads = ROOT/'runs/gaze_patch_classifier_1012_smoke'
    cache = torch.load(feature_dir/'features.pt', map_location='cpu', weights_only=False)
    validate_feature_cache(cache)
    frame = pd.DataFrame(cache['rows'])
    assert frame.groupby(['split', 'label']).size().to_dict() == {
        ('train', 0): 2, ('train', 1): 2, ('val', 0): 2, ('val', 1): 2}
    assert cache['provenance']['source_config_sha256'] == SOURCE_SHA256
    assert cache['provenance']['population'] == 'smoke8'
    assert cache['provenance']['pixel_only']
    manifest = pd.read_csv(ROOT/'manifests/all.csv')
    tr = set(manifest.loc[manifest.split == 'train', 'patient_key'])
    va = set(manifest.loc[manifest.split == 'val', 'patient_key'])
    for proof in cache['provenance']['source_patient_audit']:
        assert set(proof['training_patient_keys']) <= tr
        assert set(proof['selection_patient_keys']) <= va
    for row in cache['rows']:
        meta = row['selection_metadata']
        assert meta['proposal']['actual_roi_budget'] == 16
        assert not meta['proposal']['gaze_component_used']
        assert meta['gaze']['actual_roi_budget'] == 0
        assert meta['gaze']['conditional_gaze_steps'] == 16
        assert meta['hybrid']['actual_roi_budget'] == 14
        assert meta['hybrid']['conditional_gaze_steps'] == 2
    report = json.loads((heads/'report.json').read_text())
    assert report['smoke'] and not report['performance_evidence']
    assert report['seeds'] == [42] and report['k'] == 16
    assert len(report['seed_results']) == 6
    assert json.loads((heads/'status.json').read_text())['state'] == 'complete'
    global_ck = torch.load(heads/'global_seed42/best.pt', map_location='cpu', weights_only=False)
    errors = {}
    for arm in ('global',) + ARMS:
        ck = torch.load(heads/f'{arm}_seed42/best.pt', map_location='cpu', weights_only=False)
        assert ck['arm'] == arm and ck['source_provenance']['cache_sha256'] == digest(feature_dir/'features.pt')
        assert set(ck['training_patient_keys']) == set(frame.loc[frame.split == 'train', 'patient_key'])
        assert set(ck['selection_patient_keys']) == set(frame.loc[frame.split == 'val', 'patient_key'])
        model = restore(ck, None if arm == 'global' else global_ck)
        prediction = pd.read_csv(heads/f'{arm}_seed42/validation_predictions.csv')
        indices = torch.tensor([int(frame.index[frame.image_id == i][0]) for i in prediction.image_id])
        with torch.inference_mode():
            output = forward(model, cache['tensors'], indices, arm, 16)
        errors[arm] = float(np.max(np.abs(output['logits'].sigmoid().numpy() - prediction.probability)))
        assert errors[arm] < 1e-6
        if arm != 'global':
            assert not any(p.requires_grad for p in model.global_model.parameters())
            assert not any(k.startswith('global_model.') for k in ck['state'])
    pixel = json.loads((heads/'image_only_case64.json').read_text())
    index = int(frame.index[frame.image_id == 64][0])
    coordinate_error = float((torch.tensor(pixel['coordinates']) - cache['tensors']['gaze_coordinates'][index]).abs().max())
    cached_probability = float(pd.read_csv(heads/'gaze_seed42/validation_predictions.csv').set_index('image_id').loc[64, 'probability'])
    probability_error = abs(pixel['probability'] - cached_probability)
    assert coordinate_error < 1e-6 and probability_error < 1e-5
    assert pixel['image_sha256'] == frame.loc[index, 'image_sha256']
    proof = dict(status='passed',scope='development smoke only, not performance evidence',
        images=8,train_images=4,validation_images=4,patch_arms=list(ARMS),
        patches=8*5*16,classifiers=6,epochs=2,head_seed=42,path_seed=42,
        source_config_sha256=SOURCE_SHA256,original_control_encoder_common=True,
        source_patient_audits=len(cache['provenance']['source_patient_audit']),
        TEST_or_INbreast_used=False,compact_patch_heads=True,
        checkpoint_csv_probability_max_errors=errors,
        image_only_case64_coordinate_max_error=coordinate_error,
        image_only_case64_probability_error=probability_error,
        cache_sha256=digest(feature_dir/'features.pt'),report_sha256=digest(heads/'report.json'),
        source_hashes_checked=True)
    (heads/'parent_verification.json').write_text(json.dumps(proof, indent=2))
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
