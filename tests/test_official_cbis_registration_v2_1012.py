import unittest
import cv2
import numpy as np
from src.official_cbis_registration_v2_1012 import estimate_registration_v2, transform_gaze_to_raw


def image():
    rng = np.random.default_rng(17)
    yy, xx = np.mgrid[:240, :180]
    m = ((xx-55)/65)**2+((yy-120)/105)**2 < 1
    a = cv2.GaussianBlur(rng.random((240,180)).astype(np.float32), (0,0), 2)
    return (a*.6+.3)*m, m


class RegistrationV2Tests(unittest.TestCase):
    def test_anisotropic_registration(self):
        a, m = image()
        truth = np.array([[.8, 0, 22], [0, .9, 10], [0, 0, 1.]])
        b = cv2.warpAffine(a, truth[:2], (180,240))
        bm = cv2.warpAffine(m.astype(np.uint8), truth[:2], (180,240), flags=cv2.INTER_NEAREST)>0
        got, quality = estimate_registration_v2(a, b, m, bm)
        assert quality['passed'], quality
        points = np.array([[55,120,1], [75,70,1], [30,175,1]])
        assert np.max(np.abs(points @ got.T-points @ truth.T)) < 3


    def test_flip_detected_from_image_only(self):
        a, m = image()
        got, quality = estimate_registration_v2(a, a[:,::-1].copy(), m, m[:,::-1].copy())
        assert quality['passed'] and quality['horizontal_flip'], quality
        assert np.max(np.abs(got-np.array([[-1,0,179],[0,1,0],[0,0,1]]))) < 1


    def test_known_coordinate_composition_and_nan_gaps(self):
        meta = {'original_size': [1500,3000], 'affine': [[.5,0,10],[0,.25,20]]}
        pair = np.array([[2,0,5],[0,3,7],[0,0,1]])
        result = transform_gaze_to_raw([[100,200],[np.nan,30]], meta, pair)
        np.testing.assert_allclose(result[0], [125,217])
        assert np.isnan(result[1]).all()


    def test_blank_pair_fails_without_identity_fallback(self):
        matrix, quality = estimate_registration_v2(np.zeros((100,100)),np.zeros((100,100)))
        assert not quality['passed']
        assert np.isnan(matrix).all()


    def test_unrelated_texture_fails(self):
        a,m=image()
        rng=np.random.default_rng(91)
        b=cv2.GaussianBlur(rng.random(a.shape).astype(np.float32),(0,0),2)*m
        _,quality=estimate_registration_v2(a,b,m,m)
        assert not quality['passed'],quality

    def test_out_of_bounds_and_failed_pair_preserve_gaps(self):
        meta={'original_size':[1500,3000],'affine': [[1,0,0],[0,1,0]]}
        points=[[-1,20],[1500,20],[10,3000],[100,200],[1490,2990]]
        pair=np.array([[1,0,20],[0,1,20],[0,0,1]])
        result=transform_gaze_to_raw(points,meta,pair)
        assert np.isnan(result[[0,1,2,4]]).all()
        np.testing.assert_allclose(result[3],[120,220])
        assert np.isnan(transform_gaze_to_raw(points,meta,np.full((3,3),np.nan))).all()
        assert np.isnan(transform_gaze_to_raw(points,meta,pair,registration_quality={'passed':False})).all()

    def test_invalid_affine_rejected(self):
        meta={'original_size':[1500,3000],'affine': [[1,0,0],[0,1,0],[0,.1,1]]}
        with self.assertRaises(ValueError):
            transform_gaze_to_raw([[10,20]],meta,np.eye(3))

if __name__ == '__main__':
    unittest.main()
