"""Independent image-only v2 preprocessing; no labels, ROI or gaze are accepted.

The preset is fixed before evaluation. Geometry preserves source orientation.
A failed quality gate raises rather than silently producing a questionable crop.
"""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

CANVAS = (1500, 3000)
PREPROCESS_V2 = dict(version='official_cbis_image_only_v2_1012', canvas=list(CANVAS),
    segmentation_max_side=1024, opening_radius=18, threshold_range_fraction=.04,
    horizontal_band_max_fraction=.06, horizontal_band_min_width=.85,
    horizontal_band_transition_width=.65, vertical_rail_min_height=.90,
    vertical_rail_max_width_fraction=.03, normalization_percentiles=[0.5,99.5],
    foreground='image_only_largest_opened_component', roi_used=False,
    orientation='preserve_source_no_flip', coordinate_convention='pixel_centers_half_pixel',
    preset='fixed_development_only', interpolation='bilinear')


def _pixels(source):
    if isinstance(source, Image.Image):
        return np.asarray(source.convert('L'), dtype=np.float32), dict(input_kind='png', monochrome1=False)
    if Path(source).suffix.lower() != '.dcm':
        with Image.open(source) as im:
            return _pixels(im)
    import pydicom
    ds = pydicom.dcmread(source)
    a = np.asarray(ds.pixel_array, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError('Full mammogram must be a single grayscale frame')
    a = a * float(getattr(ds, 'RescaleSlope', 1)) + float(getattr(ds, 'RescaleIntercept', 0))
    finite = np.isfinite(a)
    if not finite.all():
        raise ValueError('Nonfinite mammogram pixels')
    inverted = str(getattr(ds, 'PhotometricInterpretation', '')) == 'MONOCHROME1'
    if inverted:
        a = a.max() + a.min() - a
    return a, dict(input_kind='dicom', monochrome1=inverted)


def _segment(a):
    h, w = a.shape
    ratio = min(1., 1024 / max(h, w))
    size = (max(1, round(w * ratio)), max(1, round(h * ratio)))
    small = np.asarray(Image.fromarray(a).resize(size, Image.Resampling.BILINEAR))
    lo, hi = np.percentile(small, [1, 99.5])
    if hi <= lo:
        raise ValueError('Empty image intensity range')
    # A low threshold keeps the thin peripheral skin rather than using Otsu,
    # which can bisect low density breasts. Opening severs thin film frames.
    threshold = float(lo + .04 * (hi - lo))
    foreground = small > threshold
    # Wide horizontal film bands can survive opening. Suppress only a near-edge
    # band with a sharp transition from nearly full width to mostly background.
    suppressed_bands = []
    for reverse in (False, True):
        work = foreground[::-1] if reverse else foreground
        widths = work.mean(axis=1)
        limit = max(2, int(work.shape[0] * .06))
        for stop in range(1, limit):
            if widths[:stop].max() > .85 and widths[:stop].mean() > .75 and widths[stop] < .65:
                work[:stop] = False
                suppressed_bands.append(['bottom' if reverse else 'top', stop])
                break
    # A narrow almost full-height strip is a film rail, whereas a genuine
    # broad chestwall remains intact. Suppression is bounded to 3% width.
    rail_columns = foreground.mean(axis=0) > .90
    rail_labels, rail_count = ndi.label(rail_columns)
    suppressed_rails = []
    for index in range(1, rail_count + 1):
        columns = np.flatnonzero(rail_labels == index)
        if len(columns) <= max(2, int(size[0] * .03)):
            foreground[:, columns] = False
            suppressed_rails.append([int(columns[0]), int(columns[-1]) + 1])
    r = PREPROCESS_V2['opening_radius']
    padded = np.pad(foreground, r + 1, constant_values=False)
    eroded = ndi.distance_transform_edt(padded)[r+1:-(r+1), r+1:-(r+1)] > r
    if not eroded.any():
        raise ValueError('No substantial image-only breast core')
    opened = ndi.distance_transform_edt(~eroded) <= r
    labels, n = ndi.label(opened)
    counts = np.bincount(labels.ravel()); counts[0] = 0
    if not n or counts.max() < small.size * .025:
        raise ValueError('No substantial image-only breast component')
    core = labels == counts.argmax()
    # Restore the opening's peripheral loss, without flood filling back into
    # connected film borders. Remaining holes are filled image-geometrically.
    mask = ndi.binary_fill_holes((ndi.distance_transform_edt(~core) <= r) & foreground)
    yy, xx = np.nonzero(mask)
    bbox = [max(0, int(np.floor(xx.min() * w / size[0]))),
            max(0, int(np.floor(yy.min() * h / size[1]))),
            min(w, int(np.ceil((xx.max()+1) * w / size[0]))),
            min(h, int(np.ceil((yy.max()+1) * h / size[1])))]
    fraction = float(mask.mean())
    occupancy = float(mask[yy.min():yy.max()+1,xx.min():xx.max()+1].mean())
    issues = []
    if occupancy < .25: issues.append('sparse_component')
    if fraction > .98: issues.append('no_visible_background')
    if issues:
        raise ValueError('Segmentation quality failed: ' + ', '.join(issues))
    full_mask = np.asarray(Image.fromarray(mask).resize((w,h), Image.Resampling.NEAREST), dtype=bool)
    return full_mask, bbox, dict(status='pass', confidence='heuristic_not_calibrated',
        foreground_fraction=fraction, bbox_occupancy=occupancy, threshold=threshold,
        segmentation_size=list(size), suppressed_horizontal_bands=suppressed_bands, suppressed_vertical_rails=suppressed_rails, issues=issues)


def canonical_image_v2(image_or_dicom, *, return_mask=False):
    """Return grayscale PIL canvas and JSON-safe source-to-canvas metadata."""
    a, source_meta = _pixels(image_or_dicom)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite two dimensional mammogram')
    mask, bbox, quality = _segment(a)
    low, high = np.percentile(a[mask], PREPROCESS_V2['normalization_percentiles'])
    if high <= low:
        raise ValueError('Empty within-breast intensity range')
    normalized = np.clip((a-low)*255/(high-low), 0,255).astype(np.uint8)
    normalized[~mask] = 0
    source = Image.fromarray(normalized)
    cw, ch = bbox[2]-bbox[0], bbox[3]-bbox[1]
    scale = min(CANVAS[0]/cw, CANVAS[1]/ch)
    resized = (max(1,min(CANVAS[0],round(cw*scale))), max(1,min(CANVAS[1],round(ch*scale))))
    sx, sy = resized[0]/cw, resized[1]/ch
    left, top = (CANVAS[0]-resized[0])//2, (CANVAS[1]-resized[1])//2
    output = Image.new('L',CANVAS,0)
    output.paste(source.crop(bbox).resize(resized,Image.Resampling.BILINEAR),(left,top))
    affine = [[sx,0.,left+(0.5-bbox[0])*sx-0.5], [0.,sy,top+(0.5-bbox[1])*sy-0.5]]
    metadata = dict(preprocessing=dict(PREPROCESS_V2), original_size=list(source.size),
        crop_bbox=bbox,resized_size=list(resized),padding=[left,top],affine=affine,
        inverse_affine=np.linalg.inv(np.vstack([affine,[0,0,1]]))[:2].tolist(),
        flip=False,intensity_low=float(low),intensity_high=float(high),
        segmentation_quality=quality,rounding='Python ties-to-even',**source_meta)
    if return_mask:
        mask_canvas = Image.new('L', CANVAS, 0)
        mask_canvas.paste(Image.fromarray(mask.astype(np.uint8)*255).crop(bbox).resize(
            resized, Image.Resampling.NEAREST), (left, top))
        return output, metadata, mask_canvas
    return output, metadata


def canonical_mask_v2(image_or_dicom, metadata=None):
    """Return the exact image-only segmentation mask in canonical geometry.

    Input is the original image, not an already normalized canvas. Optional
    metadata guards against a changed preset or a different source geometry.
    """
    _, actual, mask = canonical_image_v2(image_or_dicom, return_mask=True)
    if metadata is not None:
        for key in ('original_size', 'crop_bbox', 'affine', 'preprocessing'):
            if actual[key] != metadata[key]:
                raise ValueError('Canonical mask metadata mismatch: ' + key)
    return np.asarray(mask, dtype=bool)

