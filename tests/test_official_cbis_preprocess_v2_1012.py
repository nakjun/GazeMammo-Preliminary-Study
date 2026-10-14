import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from official_cbis_preprocess_v2_1012 import canonical_image_v2


def breast(frame=False, large=False):
    y,x=np.mgrid[:600,:400]
    mask=((x/ (370 if large else 260))**2 + ((y-300)/280)**2)<1
    a=np.zeros((600,400),dtype=np.uint8)
    a[mask]=(60+x[mask]*.3+y[mask]*.07).astype(np.uint8)
    if frame:
        a[:2,:]=255; a[-2:,:]=255; a[:,-2:]=255
        a[100:110,350:375]=220
        a[300:302,:]=255  # thin frame connected to breast
    return Image.fromarray(a)


def test_connected_frame_and_disconnected_text():
    _, m=canonical_image_v2(breast(True))
    assert m['crop_bbox'][2] < 280
    assert m['crop_bbox'][1] > 10
    assert m['crop_bbox'][3] < 590


def test_large_breast_and_chestwall_preserved():
    _,m=canonical_image_v2(breast(large=True))
    assert m['crop_bbox'][0]==0
    assert m['crop_bbox'][2]>365
    assert m['flip'] is False


def test_half_pixel_affine_and_inverse():
    out,m=canonical_image_v2(breast())
    assert out.size==(1500,3000)
    a=np.vstack([m['affine'],[0,0,1]])
    inv=np.vstack([m['inverse_affine'],[0,0,1]])
    np.testing.assert_allclose(a@inv,np.eye(3),atol=1e-10)
    x,y=m['crop_bbox'][:2]; sx=a[0,0]; sy=a[1,1]
    np.testing.assert_allclose(a@np.array([x-.5,y-.5,1]),
        [m['padding'][0]-.5,m['padding'][1]-.5,1])


def test_monochrome1_matches_png():
    import pydicom
    original=np.asarray(breast(),dtype=np.uint16)
    ds=SimpleNamespace(pixel_array=4095-original,PhotometricInterpretation='MONOCHROME1')
    from unittest.mock import patch
    patcher=patch.object(pydicom,'dcmread',lambda p:ds)
    patcher.start()
    d,m=canonical_image_v2(Path('/tmp/image.dcm'))
    p,_=canonical_image_v2(breast())
    patcher.stop()
    assert m['monochrome1']
    np.testing.assert_array_equal(np.asarray(d),np.asarray(p))


def test_empty_rejected(value):
    with unittest.TestCase().assertRaises(ValueError):
        canonical_image_v2(Image.new('L',(400,600),value))


def test_only_image_argument():
    with unittest.TestCase().assertRaises(TypeError):
        canonical_image_v2(breast(),roi=np.ones((600,400)))


class PreprocessV2Tests(unittest.TestCase):
    def test_frames(self): test_connected_frame_and_disconnected_text()
    def test_wide_connected_vertical_rail(self):
        y,x=np.mgrid[:600,:400]
        mask=((x-200)/160)**2+((y-300)/210)**2<1
        a=np.zeros((600,400),dtype=np.uint8)
        a[mask]=(60+x[mask]*.3+y[mask]*.07).astype(np.uint8)
        a[:,190:210]=230
        _,metadata=canonical_image_v2(Image.fromarray(a))
        self.assertGreater(metadata['crop_bbox'][1],40)
        self.assertLess(metadata['crop_bbox'][3],560)
    def test_exact_canonical_mask(self):
        from official_cbis_preprocess_v2_1012 import canonical_mask_v2
        source=breast()
        _,metadata,mask=canonical_image_v2(source,return_mask=True)
        actual=canonical_mask_v2(source,metadata)
        np.testing.assert_array_equal(actual,np.asarray(mask,dtype=bool))
        self.assertEqual(actual.shape,(3000,1500))
    def test_large_breast(self): test_large_breast_and_chestwall_preserved()
    def test_affine(self): test_half_pixel_affine_and_inverse()
    def test_monochrome1(self): test_monochrome1_matches_png()
    def test_empty(self):
        for value in [0,255]: test_empty_rejected(value)
    def test_no_roi_input(self): test_only_image_argument()

if __name__ == '__main__': unittest.main()
