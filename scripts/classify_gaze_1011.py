"""Train/evaluate binary heads on the fixed 1011 gaze backbone and predicted crops."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.gaze_classifier_1011 import GazeDiagnosisHead, image_features, validate_population
from src.trajectory_1011 import restore
from src.metrics import classification_metrics, cluster_bootstrap


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def checkpoint_path(path):
    path = Path(path)
    return path / 'best.pt' if path.is_dir() else path


def load_gaze(path, device):
    path = checkpoint_path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = restore(checkpoint).to(device).eval().requires_grad_(False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return model, checkpoint, digest


def cache_features(a, out):
    model, checkpoint, digest = load_gaze(a.gaze_checkpoint, a.device)
    manifest = pd.read_csv(ROOT / 'manifests/all.csv')
    frame = validate_population(manifest, checkpoint)
    if a.smoke:
        frame = frame.groupby(['split', 'label'], group_keys=False).head(2).reset_index(drop=True)
    frame.to_csv(out / 'population.csv', index=False)
    data = []
    started = time.monotonic()
    for index, row in frame.iterrows():
        with Image.open(ROOT / 'data/image' / row.file_name) as source:
            features = image_features(model, source.convert('L'), int(row.image_id), a.seed)
        data.append(features)
        if (index + 1) % 10 == 0 or index + 1 == len(frame):
            print(json.dumps(dict(event='extract', images=index + 1, total=len(frame),
                                  seconds=round(time.monotonic() - started, 1))), flush=True)
    tensors = {key: torch.stack([d[key] for d in data]) for key in data[0]}
    if not all(torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError('Nonfinite extracted features')
    provenance = dict(gaze_checkpoint=str(checkpoint_path(a.gaze_checkpoint).resolve()),
                      gaze_checkpoint_sha256=digest, gaze_epoch=checkpoint['epoch'],
                      gaze_training_patient_keys=checkpoint['training_patient_keys'],
                      gaze_selection_patient_keys=checkpoint['selection_patient_keys'],
                      seed=a.seed, steps=16, policy='none', strategy='sample', patch_size=224,
                      label_mapping={'benign': 0, 'malignant_or_mixed': 1},
                      input='image and past predicted gaze only; no GT gaze or lesion boxes',
                      protocol='Frozen supervised gaze transfer on shared training patients; not out-of-fold features',
                      independent_holdout=False, final_test_used=False,
                      extraction_seconds=time.monotonic() - started)
    torch.save(dict(tensors=tensors, rows=frame.to_dict('records'), provenance=provenance), out / 'features.pt')
    save(out / 'feature_provenance.json', provenance)
    del model
    if str(a.device).startswith('cuda'):
        torch.cuda.empty_cache()
    return tensors, frame, provenance


def forward(model, tensors, indices):
    name = model.mode
    return model(tensors['global_features'][indices],
                 None if name == 'global' else tensors[name + '_features'][indices],
                 None if name == 'global' else tensors[name + '_coordinates'][indices])


@torch.inference_mode()
def evaluate(model, tensors, indices, batch_size=64):
    model.eval()
    probabilities, attention = [], []
    for chunk in indices.split(batch_size):
        logits, weights = forward(model, tensors, chunk)
        probabilities.append(logits.sigmoid().cpu())
        if weights is not None:
            attention.append(weights.cpu())
    return torch.cat(probabilities).numpy(), torch.cat(attention).numpy() if attention else None


def train_head(a, out, tensors, frame, provenance, mode):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    directory = out / mode
    directory.mkdir()
    model = GazeDiagnosisHead(input_dim=tensors['global_features'].shape[-1], mode=mode,
                              dim=a.dim, dropout=a.dropout).to(a.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    tr = torch.tensor(np.flatnonzero(frame.split.to_numpy() == 'train'), device=a.device)
    va = torch.tensor(np.flatnonzero(frame.split.to_numpy() == 'val'), device=a.device)
    labels = torch.tensor(frame.label.to_numpy(), dtype=torch.float32, device=a.device)
    selection = frame.iloc[va.cpu().numpy()].reset_index(drop=True)
    best, stale, history = -float('inf'), 0, []
    started = time.monotonic()
    for epoch in range(1, a.epochs + 1):
        model.train()
        generator = torch.Generator().manual_seed(a.seed + epoch)
        order = tr[torch.randperm(len(tr), generator=generator).to(a.device)]
        losses = []
        for chunk in order.split(a.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = forward(model, tensors, chunk)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[chunk])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite classifier loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append((float(loss.detach()), len(chunk)))
        probs, _ = evaluate(model, tensors, va)
        metrics = classification_metrics(selection.label, probs, .5)
        record = dict(epoch=epoch, train_bce=sum(v * n for v, n in losses) / len(tr), validation=metrics)
        history.append(record)
        save(directory / 'history.json', history)
        score = metrics['auroc']
        if score > best:
            best, stale = score, 0
            checkpoint = dict(kind='gaze_diagnosis_1011', state=model.state_dict(),
                              model_config=model.config, epoch=epoch, args=vars(a),
                              training_patient_keys=sorted(set(frame[frame.split == 'train'].patient_key)
                                                           | set(provenance['gaze_training_patient_keys'])),
                              classifier_training_patient_keys=sorted(frame[frame.split == 'train'].patient_key.unique()),
                              selection_patient_keys=sorted(set(selection.patient_key)
                                                            | set(provenance['gaze_selection_patient_keys'])),
                              gaze_provenance=provenance, threshold=.5)
            torch.save(checkpoint, directory / 'best.pt')
        else:
            stale += 1
        print(json.dumps(dict(event='train', model=mode, epoch=epoch,
                              auroc=round(score, 6), best=round(best, 6), stale=stale)), flush=True)
        if stale >= a.patience:
            break
    selected = torch.load(directory / 'best.pt', map_location=a.device, weights_only=False)
    model.load_state_dict(selected['state'])
    probs, attention = evaluate(model, tensors, va)
    predictions = selection[['image_id', 'patient_key', 'file_name', 'label']].copy()
    predictions['probability'] = probs
    predictions['prediction_at_0_5'] = (probs >= .5).astype(int)
    predictions.to_csv(directory / 'validation_predictions.csv', index=False)
    if attention is not None:
        save(directory / 'validation_attention.json', {
            str(image_id): dict(weights=weights.tolist(), coordinates=coords.tolist())
            for image_id, weights, coords in zip(selection.image_id, attention,
                tensors[mode + '_coordinates'][va].cpu().numpy())})
    result = dict(mode=mode, best_epoch=selected['epoch'], epochs_completed=len(history),
                  metrics=classification_metrics(predictions.label, probs, .5),
                  auroc_ci=cluster_bootstrap(predictions, repetitions=a.bootstrap, seed=a.seed),
                  seconds=time.monotonic() - started, trainable_parameters=sum(p.numel() for p in model.parameters()),
                  independent_holdout=False, threshold_selection='fixed 0.5, not tuned')
    save(directory / 'validation_metrics.json', result)
    return result, predictions


def make_report(out, results, predictions, a):
    comparisons = {name: cluster_bootstrap(predictions['gaze'], predictions[name],
                                          repetitions=a.bootstrap, seed=a.seed)
                   for name in ('global', 'grid')}
    for name, values in comparisons.items():
        values['delta'] = results['gaze']['metrics']['auroc'] - results[name]['metrics']['auroc']
    save(out / 'paired_auroc_difference.json', comparisons)
    lines = ['# Gaze 기반 benign/malignant 분류', '',
             '최적 gaze epoch12의 frozen ConvNeXt 특징과 image-only sampling16개 패치로 분류기를 학습했다. '
             '분류 학습은 GT gaze/병변 박스를 읽지 않는다. global은 전체 영상, gaze는 전체+예측 gaze 패치 순서 GRU/attention, '
             'grid는 같은 분류 구조에 foreground grid16개 패치를 사용한다.', '',
             '## Validation 결과', '',
             '| 모델 | best epoch | AUROC | 환자 bootstrap95%CI | AUPRC | Sensitivity@0.5 | Specificity@0.5 |',
             '|---|---:|---:|---|---:|---:|---:|']
    for name, r in results.items():
        m, ci = r['metrics'], r['auroc_ci']
        lines.append(f'| {name} | {r["best_epoch"]} | {m["auroc"]:.4f} | '
                     f'[{ci["low"]:.4f}, {ci["high"]:.4f}] | {m["auprc"]:.4f} | '
                     f'{m["sensitivity"]:.4f} | {m["specificity"]:.4f} |')
    lines += ['', '## 동일 영상·환자 기준 AUROC 차이', '']
    for name, ci in comparisons.items():
        lines.append(f'- gaze − {name}: {ci["delta"]:+.4f}, 95%CI [{ci["low"]:+.4f}, {ci["high"]:+.4f}]')
    lines += ['', '## 해석 범위', '',
              '학습 seed42의 frozen-feature transfer 실험이다. train 환자는 gaze 사전학습에도 사용됐으므로 OOF/cross-fitting이 아니다. '
              'validation 환자는 gaze gradient 학습에는 없지만 gaze checkpoint 선택과 분류 checkpoint 선택에 재사용됐다. '
              '위 수치는 탐색적 validation 성능이며 독립 test 성능이 아니다. bootstrap은 고정된 모델에 조건부이며 선택 편향과 '
              '학습 seed 불확실성을 제거하지 않는다. 최종 test와 INbreast quarantine은 평가하지 않았다.', '',
              '분류 대상은 COCO에서 유도된 영상 단위 benign(0)/malignant 또는 혼합(1) 라벨이다. unknown(-1)은 제외했다. '
              '확률은 별도 calibration을 거치지 않은 sigmoid 출력이다. gaze 예측 자체를 분류 loss로 미세조정하지 않았다.', '',
              '![Classification learning curves and ROC](classification_curves.png)', '']
    (out / 'report.md').write_text('\n'.join(lines))
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gaze_mammo_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in results:
        history = json.loads((out / name / 'history.json').read_text())
        axes[0].plot([r['epoch'] for r in history], [r['validation']['auroc'] for r in history], label=name)
        data = predictions[name]
        x, y, _ = roc_curve(data.label, data.probability)
        axes[1].plot(x, y, label=f'{name}: {results[name]["metrics"]["auroc"]:.3f}')
    axes[0].set(xlabel='Epoch', ylabel='Validation AUROC', title='Frozen gaze backbone; trainable diagnosis heads')
    axes[1].plot([0, 1], [0, 1], '--', color='gray')
    axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='Selected checkpoints on reused validation')
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(out / 'classification_curves.png', dpi=160); plt.close(fig)


def train(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Use a fresh output directory')
    out.mkdir(parents=True)
    save(out / 'config.json', vars(a))
    save(out / 'status.json', dict(stage='feature_extraction', complete=False))
    started = time.monotonic()
    tensors, frame, provenance = cache_features(a, out)
    tensors = {k: v.to(a.device) for k, v in tensors.items()}
    results, predictions = {}, {}
    for mode in ('global', 'gaze', 'grid'):
        save(out / 'status.json', dict(stage='train_' + mode, complete=False))
        results[mode], predictions[mode] = train_head(a, out, tensors, frame, provenance, mode)
    save(out / 'comparison.json', results)
    make_report(out, results, predictions, a)
    save(out / 'status.json', dict(stage='complete', complete=True, seconds=time.monotonic() - started,
                                  training_images=int((frame.split == 'train').sum()),
                                  validation_images=int((frame.split == 'val').sum()), smoke=a.smoke))


def predict(a):
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Use a fresh output directory')
    state = torch.load(checkpoint_path(a.classifier), map_location='cpu', weights_only=False)
    if state['kind'] != 'gaze_diagnosis_1011':
        raise ValueError('Wrong classifier checkpoint')
    gaze, _, digest = load_gaze(a.gaze_checkpoint, a.device)
    if digest != state['gaze_provenance']['gaze_checkpoint_sha256']:
        raise ValueError('Classifier and gaze checkpoint mismatch')
    head = GazeDiagnosisHead(**state['model_config']).to(a.device).eval()
    head.load_state_dict(state['state'])
    with Image.open(a.image) as source:
        image = source.convert('L')
    mode = head.mode
    features = image_features(gaze, image, a.image_id, state['gaze_provenance']['seed'], include_grid=mode == 'grid')
    features = {k: v[None].to(a.device) for k, v in features.items()}
    with torch.inference_mode():
        logits, weights = forward(head, features, torch.tensor([0], device=a.device))
    probability = float(logits.sigmoid().item())
    coordinates = features['gaze_coordinates' if mode == 'global' else mode + '_coordinates'][0].cpu().numpy()
    record = dict(malignant_probability=probability, benign_probability=1 - probability,
                  prediction='malignant' if probability >= state['threshold'] else 'benign',
                  threshold=state['threshold'], probability_calibrated=False, classifier_mode=mode,
                  classifier_epoch=state['epoch'], gaze_checkpoint_sha256=digest,
                  image=str(Path(a.image).resolve()), image_id=a.image_id,
                  seed=state['gaze_provenance']['seed'], coordinates=coordinates.tolist(),
                  attention=weights[0].cpu().tolist() if weights is not None else None,
                  inputs='Image pixels and supplied RNG image ID only; no gaze, label, manifest or boxes read')
    out.mkdir(parents=True)
    save(out / 'prediction.json', record)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/gaze_mammo_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    points = coordinates * image.size
    fig, ax = plt.subplots(figsize=(6, 11))
    ax.imshow(image, cmap='gray'); ax.plot(*points.T, color='cyan', linewidth=1, alpha=.7)
    ax.scatter(*points.T, c=np.arange(len(points)), cmap='turbo', s=25)
    if weights is not None:
        attention = weights[0].cpu().numpy()
        for index in np.argsort(attention)[-3:]:
            x, y = points[index]
            ax.add_patch(Rectangle((x - 112, y - 112), 224, 224, fill=False, edgecolor='yellow', linewidth=1.5))
            ax.annotate(f'{index + 1}: {attention[index]:.2f}', (x, y), color='yellow', fontsize=9)
    ax.set(xlim=(0, image.width), ylim=(image.height, 0))
    ax.set_title(f'{mode}: P(malignant)={probability:.3f}\nImage-only predicted gaze; yellow = top 3 attention crops')
    ax.axis('off'); fig.tight_layout(); fig.savefig(out / 'classification.png', dpi=160); plt.close(fig)
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['train', 'predict'])
    p.add_argument('--gaze-checkpoint', default=str(ROOT / 'runs/full_1011_seed42/improved'))
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--dropout', type=float, default=.3)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--classifier')
    p.add_argument('--image')
    p.add_argument('--image-id', type=int, default=0)
    a = p.parse_args()
    if a.epochs < 1 or a.patience < 1 or a.bootstrap < 1:
        p.error('epochs, patience, bootstrap must be positive')
    if a.command == 'predict' and (not a.classifier or not a.image):
        p.error('predict requires --classifier and --image')
    torch.set_num_threads(a.threads)
    train(a) if a.command == 'train' else predict(a)


if __name__ == '__main__':
    main()
