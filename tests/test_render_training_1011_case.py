import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
spec=importlib.util.spec_from_file_location('render_case',Path(__file__).resolve().parents[1]/'scripts/render_training_1011_case.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)

class VisualTests(unittest.TestCase):
    def test_pixel_roundtrip(self):
        points=np.array([[.1,.2],[.9,.8]]);size=(1500,3000)
        pixels=renderer.normalized_to_pixels(points,size)
        np.testing.assert_allclose(pixels,[[150,600],[1350,2400]])
        np.testing.assert_allclose(pixels/size,points)
    def test_predictions_are_image_only(self):
        class Model:
            foveal=False;input_size=(8,16);cell_xy=torch.tensor([[.25,.25],[.75,.75]])
            def eval(self):return self
            def maps(self,image):return {}
            def step_history(self,memory,prefix,hidden,vector):return torch.tensor([[100.,0.]]),None,dict(gates=torch.tensor([[1.,0.,0.]]))
        with patch.object(renderer,'load_gazes',side_effect=AssertionError('Future gaze access')):
            points,gates=renderer.predict_image_only(Model(),Image.new('L',(20,40)),64,strategy='argmax',steps=3)
        np.testing.assert_allclose(points,np.full((3,2),.25));self.assertEqual(len(gates),3)

if __name__=='__main__':unittest.main()
