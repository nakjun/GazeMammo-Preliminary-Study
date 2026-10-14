"""Compare saved fixed-case rollouts with their exact legacy16 evaluation GT."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/gaze_mammo_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patheffects import withStroke
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'runs/full_1011_seed42/visual_checks'
GT_COLOR = '#56f5ba'
PRED_COLOR = '#ff64c3'


def trace(axis, points, size, color, label, offset):
    pixels = np.asarray(points) * np.asarray(size)
    axis.plot(*pixels.T, '-o', color=color, linewidth=1.5, markersize=4,
              alpha=.88, label=label)
    axis.scatter(*pixels[0], s=110, facecolors='none', edgecolors=color,
                 linewidths=2, zorder=5)
    axis.scatter(*pixels[-1], s=65, marker='X', color=color,
                 edgecolors='black', linewidths=.5, zorder=5)
    for i, (x, y) in enumerate(pixels):
        axis.annotate(str(i + 1), (x, y), xytext=offset,
                      textcoords='offset points', fontsize=7, color=color,
                      path_effects=[withStroke(linewidth=1.8, foreground='black')])


def main():
    source = DEST / 'final_15_case_64.metadata.json'
    meta = json.loads(source.read_text())
    rows = pd.read_csv(ROOT / 'manifests/all.csv')
    row = rows[(rows.image_id == meta['image_id']) & (rows.split == 'val')]
    assert len(row) == 1
    with Image.open(ROOT / 'data/image' / row.iloc[0].file_name) as image:
        background = image.convert('L')
    assert list(background.size) == meta['image_size']
    truth = np.asarray(meta['truth']['legacy16'])
    outputs = []
    for variant, strategy, tag in [
        ('improved_last', 'sample', 'last15_sample'),
        ('improved_best', 'sample', 'best12_sample'),
        ('improved_last', 'argmax', 'last15_argmax'),
    ]:
        model = meta['models'][variant]
        diag = model['diagnostics'][strategy]
        points = np.asarray(diag['predicted_normalized_coordinates'])
        assert truth.shape == points.shape == (16, 2)
        assert np.isfinite(points).all() and ((points >= 0) & (points < 1)).all()
        path = DEST / f'gt_vs_{tag}_case_64.png'
        if path.exists():
            raise FileExistsError(path)
        fig, axes = plt.subplots(1, 3, figsize=(12, 8.7))
        for axis in axes:
            axis.imshow(background, cmap='gray', vmin=0, vmax=255)
            axis.set(xlim=(0, background.width), ylim=(background.height, 0))
            axis.set_aspect('equal')
            axis.set_xticks([])
            axis.set_yticks([])
        trace(axes[0], truth, background.size, GT_COLOR, 'GT', (4, 3))
        trace(axes[1], points, background.size, PRED_COLOR, 'Prediction', (4, 3))
        trace(axes[2], truth, background.size, GT_COLOR, 'GT', (-11, 3))
        trace(axes[2], points, background.size, PRED_COLOR, 'Prediction', (4, -8))
        axes[0].set_title('GT: legacy16\n16 samples across the observed segment')
        axes[1].set_title(f'Prediction: epoch {model["checkpoint_epoch"]}, {strategy}\n'
                          f'Unique positions: {16 - diag["exact_repeat_count"]}/16')
        axes[2].set_title('GT + prediction overlay\nSame image, coordinates and scale')
        axes[2].legend(loc='upper left', fontsize=9)
        fig.suptitle(f'Validation case 64 | {variant} | free rollout, policy none, CPU seed 42\n'
                     f'Legacy16 normalized DTW: {diag["normalized_dtw_to_legacy16"]:.4f} '
                     '(lower is better)', fontsize=12)
        fig.text(.5, .018, 'Green = GT; pink = prediction. Numbers = step order. '
                 'Ring = start; X = end. Overlapping predictions may hide step labels.',
                 ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .035, 1, .93))
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    provenance = DEST / 'gt_comparison_case_64.json'
    provenance.write_text(json.dumps({
        'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'truth': 'legacy16 subsampled observed segment; not contiguous first16',
        'predictions': 'Saved CPU free rollouts; no new generation or GT conditioning',
        'outputs': outputs,
    }, indent=2))
    print(json.dumps(outputs, indent=2))


if __name__ == '__main__':
    main()
