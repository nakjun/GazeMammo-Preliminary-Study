"""Image-only GAA-to-RAW registration; never accepts labels, gaze, or ROI data.

Matrices map pixel centers from source canonical pixels to RAW canonical pixels.
Quality gates are fixed a priori and failed registrations remain explicitly failed.
"""
from __future__ import annotations
import cv2
import numpy as np

REGISTRATION_VERSION = 'image_only_affine_v2_1012'
# Fixed heuristic gates; not a calibrated probability or validation-tuned cutoff.
MIN_MASK_DICE = 0.70
MIN_EDGE_CORRELATION = 0.12


def _gray(image):
    a = np.asarray(image).astype(np.float32)
    if a.ndim == 3:
        a = a[..., :3].mean(axis=2)
    if a.ndim != 2 or min(a.shape) < 16 or not np.isfinite(a).all():
        raise ValueError('Registration requires finite 2-D images of at least 16 pixels')
    lo, hi = np.percentile(a, (1, 99))
    return np.clip((a-lo)/max(float(hi-lo), 1e-6), 0, 1).astype(np.float32)


def _mask(a, provided):
    if provided is not None:
        m = np.asarray(provided, dtype=np.uint8)
        if m.shape != a.shape:
            raise ValueError('Image-only mask shape differs from image')
        return m > 0
    u = (a*255).astype(np.uint8)
    _, m = cv2.threshold(u, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1:
        return np.zeros(a.shape, bool)
    return labels == (1+np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def _edges(a):
    a = cv2.GaussianBlur(a, (0, 0), 1.2)
    x = cv2.Sobel(a, cv2.CV_32F, 1, 0)
    y = cv2.Sobel(a, cv2.CV_32F, 0, 1)
    e = np.sqrt(x*x+y*y)
    return np.clip(e/max(float(np.percentile(e, 98)), 1e-6), 0, 1)


def _bbox(mask):
    y, x = np.where(mask)
    if len(x) < 64:
        raise ValueError('Insufficient image foreground for registration')
    return float(x.min()), float(y.min()), float(x.max()), float(y.max())


def _score(src_edges, src_mask, dst_edges, dst_mask, matrix):
    size = (dst_mask.shape[1], dst_mask.shape[0])
    m = cv2.warpAffine(src_mask.astype(np.uint8), matrix[:2].astype(np.float32), size,
                       flags=cv2.INTER_NEAREST) > 0
    e = cv2.warpAffine(src_edges, matrix[:2].astype(np.float32), size)
    dice = 2*float((m & dst_mask).sum())/max(float(m.sum()+dst_mask.sum()), 1)
    # Exclude the shared silhouette: shape agreement alone cannot prove pairing.
    region = cv2.erode((m & dst_mask).astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    x, y = e[region], dst_edges[region]
    corr = float(np.corrcoef(x, y)[0, 1]) if len(x)>1 and x.std()>1e-6 and y.std()>1e-6 else 0.0
    return dice, corr if np.isfinite(corr) else 0.0


def estimate_registration_v2(gaze_canonical, raw_canonical, gaze_mask=None, raw_mask=None,
                             *, allow_horizontal_flip=True, max_side=512):
    """Return ``(source_to_raw_3x3, quality_dict)`` from paired images alone.

    Horizontal reflection is an explicit candidate, selected using image agreement.
    ECC refines an anisotropic foreground-bounds initialization. No failed pair is
    silently assigned identity. The returned matrix is retained for QA even on fail.
    """
    source, target = _gray(gaze_canonical), _gray(raw_canonical)
    sm, tm = _mask(source, gaze_mask), _mask(target, raw_mask)
    def down(a, m):
        scale = min(1., max_side/max(a.shape))
        size = (max(16, round(a.shape[1]*scale)), max(16, round(a.shape[0]*scale)))
        d = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
        dm = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)>0
        # OpenCV resizing maps centers: x_small=(x_original+.5)*scale-.5.
        sx, sy = size[0]/a.shape[1], size[1]/a.shape[0]
        s = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1.]])
        return d, dm, s
    a, am, sa = down(source, sm)
    b, bm, sb = down(target, tm)
    candidates = []
    before_dice, before_corr = _score(_edges(a), am, _edges(b), bm, sb @ np.linalg.inv(sa))
    try:
        bx0, by0, bx1, by1 = _bbox(bm)
        for flip in ([False, True] if allow_horizontal_flip else [False]):
            aa, mm = (a[:, ::-1].copy(), am[:, ::-1].copy()) if flip else (a, am)
            reflection = np.array([[-1., 0, a.shape[1]-1], [0, 1, 0], [0, 0, 1]]) if flip else np.eye(3)
            ax0, ay0, ax1, ay1 = _bbox(mm)
            sx, sy = (bx1-bx0)/max(ax1-ax0, 1), (by1-by0)/max(ay1-ay0, 1)
            initial = np.array([[sx, 0, bx0-sx*ax0], [0, sy, by0-sy*ay0], [0, 0, 1.]])
            ae, be = _edges(aa), _edges(b)
            # ECC accepts target->input; invert source->target initialization.
            transforms = [(initial, False)]
            try:
                inv = np.linalg.inv(initial)[:2].astype(np.float32)
                _, inv = cv2.findTransformECC(be, ae, inv, cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 150, 1e-5), None, 5)
                refined = np.linalg.inv(np.vstack([inv, [0, 0, 1]]))
                relative = refined[:2, :2] @ np.linalg.inv(initial[:2, :2])
                singular = np.linalg.svd(relative, compute_uv=False)
                if np.linalg.det(relative)>0 and singular.min()>=0.6 and singular.max()<=1.6:
                    transforms.append((refined, True))
            except (cv2.error, np.linalg.LinAlgError):
                pass
            for transform, converged in transforms:
                dice, corr = _score(ae, mm, be, bm, transform)
                candidates.append((0.55*dice+0.45*max(corr, 0), transform@reflection, flip, dice, corr, converged))
        _, matrix, flip, dice, corr, converged = max(candidates, key=lambda c: c[0])
        matrix = np.linalg.inv(sb) @ matrix @ sa
        passed = bool(dice >= MIN_MASK_DICE and corr >= MIN_EDGE_CORRELATION)
        return matrix, dict(version=REGISTRATION_VERSION, passed=passed,
            registration_confidence=float(min(dice, max(0., corr))), mask_dice=dice,
            edge_correlation=corr, before_mask_dice=before_dice, before_edge_correlation=before_corr,
            horizontal_flip=flip, ecc_converged=converged,
            failure_reason=None if passed else 'image_agreement_below_fixed_quality_gate')
    except (ValueError, np.linalg.LinAlgError) as exc:
        return np.full((3, 3), np.nan), dict(version=REGISTRATION_VERSION, passed=False,
            registration_confidence=0., horizontal_flip=False, before_mask_dice=before_dice,
            before_edge_correlation=before_corr, failure_reason=str(exc))


def transform_gaze_to_raw(points_original_gaa, gaa_meta, pair_matrix, *, raw_size=(1500, 3000), registration_quality=None):
    """Compose original GAA->canonical->RAW; preserve missing rows as NaNs."""
    extent = gaa_meta.get('original_size', gaa_meta.get('source_size'))
    if extent is None or tuple(extent) != (1500, 3000):
        raise ValueError('Expected original GAA coordinate extent (1500, 3000)')
    affine = gaa_meta.get('original_to_canonical', gaa_meta.get('affine_original_to_canonical', gaa_meta.get('affine')))
    if affine is None:
        raise ValueError('Missing original_to_canonical pixel-center matrix')
    affine = np.asarray(affine, float)
    if affine.shape == (2, 3):
        affine = np.vstack([affine, [0, 0, 1]])
    if affine.shape != (3, 3) or not np.isfinite(affine).all() or not np.allclose(affine[2], [0, 0, 1]):
        raise ValueError('Canonical affine must be finite with final row [0,0,1]')
    pair = np.asarray(pair_matrix, float)
    if pair.shape != (3, 3):
        raise ValueError('Pair affine must have shape 3 x 3')
    if np.isfinite(pair).all() and not np.allclose(pair[2], [0, 0, 1]):
        raise ValueError('Pair affine must have final row [0,0,1]')
    if len(raw_size) != 2 or min(raw_size) <= 0:
        raise ValueError('RAW canvas size must be positive width,height')
    points = np.asarray(points_original_gaa, float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Expected N x 2 gaze coordinates')
    result = np.full(points.shape, np.nan)
    if not np.isfinite(pair).all() or (registration_quality is not None and not registration_quality.get('passed', False)):
        return result
    matrix = pair @ affine
    valid = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0) &
             (points[:, 0] < 1500) & (points[:, 1] >= 0) & (points[:, 1] < 3000))
    mapped = np.column_stack([points[valid], np.ones(valid.sum())]) @ matrix.T
    good = np.isfinite(mapped).all(axis=1) & (np.abs(mapped[:, 2])>1e-12)
    xy = np.full((len(mapped), 2), np.nan)
    xy[good] = mapped[good, :2]/mapped[good, 2:3]
    good &= (xy[:, 0] >= 0) & (xy[:, 0] < raw_size[0]) & (xy[:, 1] >= 0) & (xy[:, 1] < raw_size[1])
    indices = np.flatnonzero(valid)[good]
    result[indices] = xy[good]
    return result
