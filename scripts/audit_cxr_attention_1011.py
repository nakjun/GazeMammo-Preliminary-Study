"""Post-hoc parent audit. Annotations/gaze enter only after frozen predictions."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gaze_mammo_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.gaze_cxr_1011 import restore_head, inbounds_fraction
from src.data import load_gazes, reduce_gaze


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def arguments(tensors, indices):
    return [tensors[k][indices] for k in ('global_features', 'gaze_features',
            'gaze_coordinates', 'grid_features', 'grid_coordinates')]


def difference(a, b):
    return float((a-b).abs().max())


@torch.inference_mode()
def main(a):
    directory = Path(a.run)
    destination = Path(a.output) if a.output else directory
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(directory / f'{a.variant}_seed42/best.pt',
                            map_location='cpu', weights_only=False)
    provenance = checkpoint['source_provenance']
    global_checkpoint = torch.load(provenance['global_checkpoint'], map_location='cpu', weights_only=False)
    cache = torch.load(provenance['cache_path'], map_location='cpu', weights_only=False)
    frame = pd.DataFrame(cache['rows'])
    selected = np.flatnonzero(frame.split.to_numpy() == 'val')
    data = arguments(cache['tensors'], selected)
    model = restore_head(checkpoint, global_checkpoint).eval()
    output = model(*data)
    assert torch.isfinite(output['logits']).all()
    original = pd.read_csv(directory / f'{a.variant}_seed42/validation_predictions.csv')
    expected = original.set_index('image_id').loc[frame.iloc[selected].image_id].probability.to_numpy()
    errors = {'saved_probability_max_error': float(np.max(np.abs(expected-output['logits'].sigmoid().numpy())))}
    for branch, feature_index, coordinate_index in [('gaze', 1, 2), ('grid', 3, 4)]:
        permutation = torch.arange(data[feature_index].shape[1]-1, -1, -1)
        changed = list(data)
        changed[feature_index] = data[feature_index][:, permutation]
        changed[coordinate_index] = data[coordinate_index][:, permutation]
        errors[branch+'_permutation_max_logit_error'] = difference(model(*changed)['logits'], output['logits'])
        copied = list(data)
        copied[feature_index] = torch.cat([data[feature_index], data[feature_index][:, :1]], 1)
        copied[coordinate_index] = torch.cat([data[coordinate_index], data[coordinate_index][:, :1]], 1)
        errors[branch+'_duplicate_max_logit_error'] = difference(model(*copied)['logits'], output['logits'])
    absent = model(*data, force_gaze_absent=True, force_grid_absent=True)
    errors['all_absent_global_max_error'] = difference(absent['logits'], output['global_logits'])
    errors['signed_decomposition_max_error'] = difference(output['gaze_signed'].sum(1)+output['grid_signed'].sum(1)+output['residual_bias'], output['residual'])
    assert max(errors.values()) < 2e-6, errors
    # Ground truth is deliberately loaded only after predictions and invariance checks.
    annotations = json.loads((ROOT/'data/annotations/instances_all.json').read_text())
    boxes = {}
    for annotation in annotations['annotations']:
        boxes.setdefault(annotation['image_id'], []).append(annotation['bbox'])
    rows = []
    for j, index in enumerate(selected):
        row = frame.iloc[index]
        for branch, coordinate_index in [('gaze', 2), ('grid', 4)]:
            coords = data[coordinate_index][j]
            centers = np.round(coords.numpy().astype(np.float64)*[1500, 3000])
            overlap = np.zeros(len(centers), dtype=bool)
            for x, y, w, h in boxes.get(int(row.image_id), []):
                intersection = np.maximum(0, np.minimum(centers+112, [x+w, y+h])-np.maximum(centers-112, [x, y])).prod(1)
                overlap |= intersection > 0
            attention = output[branch+'_attention'][j].numpy()
            quality = inbounds_fraction(coords).numpy()
            eq = np.max(np.abs(coords.numpy()[:, None]-coords.numpy()[None, :]), axis=-1) <= 1e-6
            uniform = 1/eq.sum(1)
            uniform /= uniform.sum()
            rows.append(dict(image_id=int(row.image_id), label=int(row.label), source=branch,
                             source_gate=float(output['beta'][j, 0 if branch=='gaze' else 1]),
                             any_crop_touches_box=bool(overlap.any()),
                             annotated_box_count=len(boxes.get(int(row.image_id), [])),
                             attention_mass_on_box_touching_crops=float(attention[overlap].sum()),
                             uniform_mass_on_box_touching_crops=float(uniform[overlap].sum()),
                             attention_weighted_padding_fraction=float((attention*(1-quality)).sum()),
                             uniform_weighted_padding_fraction=float((uniform*(1-quality)).sum())))
    diagnostics = pd.DataFrame(rows)
    diagnostics.to_csv(destination/'parent_roi_padding_audit.csv', index=False)
    summary = diagnostics.groupby('source').mean(numeric_only=True).drop(columns=['image_id', 'label', 'annotated_box_count']).to_dict('index')
    save(destination/'parent_verification.json', dict(errors=errors,posthoc_source_summary=summary,
         roi_caveat='Any crop/box intersection is a coarse post-hoc diagnostic; not lesion localization accuracy. Gaze and boxes never enter prediction.',
         checkpoint=str(directory/f'{a.variant}_seed42/best.pt')))
    observed = load_gazes()
    for image_id in (64, 73):
        index = int(np.flatnonzero(frame.image_id.to_numpy()==image_id)[0])
        j = int(np.flatnonzero(selected==index)[0])
        row = frame.iloc[index]
        with Image.open(ROOT/'data/image'/row.file_name) as source:
            image = source.convert('L')
        segments = reduce_gaze(observed[row.file_name], *image.size)
        truth = max(segments, key=len)
        fig, axes = plt.subplots(1, 3, figsize=(13, 9))
        for ax in axes:
            ax.imshow(image, cmap='gray', vmin=0, vmax=255)
            ax.set(xlim=(0, image.width), ylim=(image.height, 0))
            ax.set_xticks([]); ax.set_yticks([])
            for x,y,w,h in boxes.get(image_id, []):
                ax.add_patch(Rectangle((x,y), w,h, fill=False, edgecolor='#00ffab', linewidth=2))
        axes[0].plot(truth[:,0], truth[:,1], '-o', color='#00c9ff', linewidth=1, markersize=3)
        axes[0].set_title('Observed gaze: reduced longest segment\nGreen = annotated lesion box (GT)')
        for ax, branch, ci, bi in [(axes[1], 'gaze', 2, 0), (axes[2], 'grid', 4, 1)]:
            centers = data[ci][j].numpy()*image.size
            attention = output[branch+'_attention'][j].numpy()
            effective = attention*float(output['beta'][j,bi])
            for k, ((x,y), weight) in enumerate(zip(centers, effective)):
                color = plt.cm.plasma(weight/max(effective.max(), 1e-12))
                ax.add_patch(Rectangle((x-112,y-112),224,224,fill=False,edgecolor=color,linewidth=.7+2*weight/max(effective.max(),1e-12)))
                ax.annotate(str(k+1), (x,y), color='white', fontsize=7,
                            bbox=dict(facecolor='black', alpha=.5, pad=.3, edgecolor='none'))
            if branch=='gaze':
                ax.plot(centers[:,0],centers[:,1],'-',color='#ffa752',alpha=.4,linewidth=.7)
            ax.set_title(f'{branch}: effective weight = source gate x attention\nsource gate={float(output["beta"][j,bi]):.3f}')
        probability=float(output['logits'][j].sigmoid())
        fig.suptitle(f'Fixed validation case {image_id} | GT={"malignant" if row.label else "benign"} | P(malignant)={probability:.3f}\n'
                     f'{a.variant} seed42 | Gaze and boxes shown only after prediction', fontsize=12)
        fig.tight_layout(rect=(0,.035,1,.94))
        fig.text(.5,.012,'Colors normalized within each branch. Attention is not a lesion probability.',ha='center',fontsize=10)
        fig.savefig(destination/f'case{image_id}_gt_comparison.png',dpi=160);plt.close(fig)
    print(json.dumps({'verification':errors,'posthoc_source_summary':summary},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--variant', default='dual_cross_drop_quality');parser.add_argument('--output')
    args=parser.parse_args();torch.set_num_threads(4);main(args)
