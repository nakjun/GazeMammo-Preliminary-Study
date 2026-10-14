"""Compare user-selected encoders on the best grid + global/local fusion setup."""
import json
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'runs/grid_fusion_backbones_seed42'
BACKBONES=['densenet161','efficientnet_b5','convnext_base','swin_b','vit_b_16']

def worker(index):
    # Separate large models across GPUs; each run retains batch2 / accumulation4.
    for name in BACKBONES[index::2]:
        out=BASE/name;out.mkdir(parents=True,exist_ok=True)
        with (out/'run.log').open('w') as log:
            subprocess.run([sys.executable,str(ROOT/'scripts/spatial_pipeline.py'),'fusion',
                '--features',str(ROOT/'runs/spatial_v2_seed42/grid_control/features'),
                '--output',str(out),'--backbone',name,'--epochs','12','--patience','4',
                '--device',f'cuda:{index}','--seed','42'],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(name,'completed',flush=True)

if __name__=='__main__':
    BASE.mkdir(parents=True,exist_ok=True)
    (BASE/'protocol.json').write_text(json.dumps(dict(backbones=BACKBONES,baseline='resnet50',
        selector='image_only_grid_control',k=16,crop_size=224,whole_input_hw=[512,256],
        seed=42,batch_size=2,gradient_accumulation=4,backbone_lr=1e-5,head_lr=1e-4,
        max_epochs=12,patience=4,selection='validation fusion AUROC',test_used=False,
        finetuning='all encoder parameters; BN running statistics fixed',
        note='Common input resolution; EfficientNet-B5 native pretrained resolution 456 is not used; torchvision ImageNet-1K weights'),indent=2))
    with ThreadPoolExecutor(2) as pool:list(pool.map(worker,range(2)))
    subprocess.run([sys.executable,str(ROOT/'scripts/report_fusion_backbones.py')],check=True)
    (BASE/'completed.json').write_text(json.dumps(dict(status='complete',test_used=False,backbones=BACKBONES),indent=2))
