"""Single-image inference without actual gaze, annotations, or diagnosis labels."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
from PIL import Image
import torch
from src.data import whole_tensor,select_centers,patch_tensor
from src.models import TrajectoryModel,HeatmapModel,PatchClassifier,JointPatchClassifier,backbone,trajectory_from_checkpoint
from pipeline import heatmap_centers
from src.constraints import breast_mask,grid_mask,rollout

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--image',required=True);p.add_argument('--gaze-checkpoint',required=True)
    p.add_argument('--classifier-checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    device=torch.device(a.device);g=torch.load(Path(a.gaze_checkpoint)/'best.pt',map_location=device,weights_only=False)
    c=torch.load(Path(a.classifier_checkpoint)/'best.pt',map_location=device,weights_only=False)
    feature_config=json.loads((Path(c['args']['features'])/'config.json').read_text())
    if feature_config['method']!=g['method'] or c['k']!=g['k']:raise ValueError('Incompatible gaze selector and classifier')
    model=(trajectory_from_checkpoint(g) if g['method']=='trajectory' else HeatmapModel(g['grid'],pretrained=False)).to(device).eval()
    model.load_state_dict(g['state']);joint=c.get('end_to_end',False)
    classifier=(JointPatchClassifier(k=c['k'],pretrained=False) if joint else PatchClassifier(c['aggregation'],k=c['k'])).to(device).eval();classifier.load_state_dict(c['state'])
    encoder=None if joint else backbone().to(device).eval()
    with Image.open(a.image) as source:image=source.convert('L')
    if image.size!=(1500,3000):raise ValueError('Expected the dataset preprocessing size 1500x3000; transform new acquisitions consistently first')
    with torch.inference_mode():
        whole=whole_tensor(image)[None].to(device)
        if g['method']=='trajectory':
            policy=feature_config.get('generation_policy','none');strategy=feature_config.get('generation_strategy','argmax')
            if policy=='none':predicted=model.generate(whole,strategy=strategy)[0].cpu().numpy()
            else:
                memory=model.memory(whole);allowed=torch.tensor(grid_mask(breast_mask(image),g['grid'])[None],device=device)
                result=rollout(model,memory,allowed,torch.tensor([[image.width,image.height]],device=device),kind=policy,radius=feature_config.get('generation_radius',112),strategy=strategy)
                predicted=result['coords'][0][result['valid'][0]].cpu().numpy()
        else:predicted=np.array(heatmap_centers(model(whole)[0].cpu(),g['grid'],g['k']))
        centers=predicted*[image.width,image.height] if feature_config.get('keep_predicted_centers',False) else select_centers(image,g['method'],g['k'],42,prediction=predicted)
        patches=torch.stack([patch_tensor(image,center,feature_config['crop_size']) for center in centers])
        if joint:
            patches=patches[None].to(device)
            if len(centers)<c['k']:patches=torch.nn.functional.pad(patches,(0,0,0,0,0,0,0,c['k']-len(centers)))
            valid=(torch.arange(patches.shape[1],device=device)<len(centers))[None]
            logit,attention=classifier(patches,valid)
        else:
            features=encoder(patches.to(device)).mean((-1,-2))[None]
            if len(centers)<c['k']:features=torch.nn.functional.pad(features,(0,0,0,c['k']-len(centers)))
            valid=(torch.arange(features.shape[1],device=device)<len(centers))[None]
            logit,attention=classifier(features,valid)
        probability=float(logit.sigmoid().item())
    threshold=json.loads((Path(a.classifier_checkpoint)/'threshold.json').read_text())['value']
    output=dict(image=str(Path(a.image).resolve()),malignant_probability=probability,validation_threshold=threshold,
                predicted_label=int(probability>=threshold),raw_normalized_trajectory=predicted.tolist(),
                patch_centers_pixels=centers.tolist(),patch_attention=attention[0,:len(centers)].cpu().tolist(),
                note='Research output; annotation-present image classification, not clinical screening validation')
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
