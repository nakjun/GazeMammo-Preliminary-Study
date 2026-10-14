import importlib.util
from pathlib import Path
import torch
spec=importlib.util.spec_from_file_location('prefix_cache',Path(__file__).parents[1]/'scripts/cache_gaze_prefix_tail_1011.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_tail_vectors_preserve_stage_order_and_normalization():
    class Visual:
        def __init__(self):self.features=torch.nn.Sequential(*[torch.nn.Identity() for _ in range(6)],torch.nn.Conv2d(2,4,1),torch.nn.ReLU());self.norm=torch.nn.Identity()
    visual=Visual();features=torch.randn(3,2,14,14)
    expected=visual.features[7](visual.features[6](features.contiguous(memory_format=torch.channels_last))).mean((-1,-2))
    assert torch.equal(m.tail_vectors(visual,features),expected)


def test_mmap_indexing_selected_points_matches_original(tmp_path):
    tensor=torch.randn(30,2,14,14);path=tmp_path/'prefix.pt';torch.save({'crop_feature5':tensor},path)
    loaded=torch.load(path,map_location='cpu',mmap=True,weights_only=False);indices=torch.tensor([0,7,18,29])
    assert torch.equal(loaded['crop_feature5'][indices],tensor[indices])
