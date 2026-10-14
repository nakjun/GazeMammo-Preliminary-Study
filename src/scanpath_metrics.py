"""Geometric metrics using the reference authors' released implementations.

No clinical duration is inferred. MultiMatch's duration dimension is excluded.
"""
import importlib.util
from pathlib import Path
import numpy as np
from .data import ROOT

SOURCE=ROOT/'literature/code/GazeSearch/src/common'

def load(name):
    spec=importlib.util.spec_from_file_location('gaze_reference_'+name,SOURCE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

ScanMatch=load('scanmatch').ScanMatch
multimatch=load('multimatch').docomparison

class ScanpathScorer:
    def __init__(self,size=(1500,3000),bins=(12,8)):
        self.size=np.asarray(size)
        self.matcher=ScanMatch(Xres=size[0],Yres=size[1],Xbin=bins[0],Ybin=bins[1],TempBin=0,Threshold=3.5)
    def __call__(self,prediction,truth):
        prediction=np.asarray(prediction,dtype=float)*self.size;truth=np.asarray(truth,dtype=float)*self.size
        a=self.matcher.fixationToSequence(prediction).astype(np.int32);b=self.matcher.fixationToSequence(truth).astype(np.int32)
        score=self.matcher.match(a,b)[0]
        # Unit placeholders satisfy the reference API. Grouping is disabled;
        # shape/direction/length/position do not use duration. Never report time.
        first=np.column_stack([prediction,np.ones(len(prediction))]);second=np.column_stack([truth,np.ones(len(truth))])
        with np.errstate(divide='ignore',invalid='ignore'):
            mm=np.asarray(multimatch(first,second,sz=self.size.tolist(),grouping=False))[0,:4]
        names=['multimatch_vector','multimatch_direction','multimatch_length','multimatch_position']
        return dict(scanmatch_no_duration=float(score),**{k:float(v) if np.isfinite(v) else None for k,v in zip(names,mm)},
            multimatch_geometry_mean=float(mm.mean()) if np.isfinite(mm).all() else None)
