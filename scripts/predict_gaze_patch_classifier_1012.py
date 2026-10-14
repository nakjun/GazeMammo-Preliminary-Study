"""Predict from one preprocessed image, frozen source and compact head only."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from src.gaze_patch_classifier_1012 import restore,digest,forward
from src.gaze_patch_features_1012 import load_frozen_source,extract_image_features

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',required=True);p.add_argument('--image-id',required=True,type=int);p.add_argument('--checkpoint',required=True)
    p.add_argument('--global-checkpoint');p.add_argument('--source-config',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cpu')
    a=p.parse_args();output=Path(a.output)
    if output.exists(): raise ValueError('Output must be new')
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    global_checkpoint=torch.load(a.global_checkpoint,map_location='cpu',weights_only=False) if a.global_checkpoint else None
    model=restore(checkpoint,global_checkpoint)
    provenance=checkpoint['source_provenance']
    if digest(a.source_config)!=provenance['source_config_file_sha256']: raise ValueError('Source config hash mismatch')
    for dependency in provenance['dependencies']:
        if digest(dependency['path'])!=dependency['sha256']: raise ValueError('Source dependency hash mismatch')
    for path, sha in provenance['source_code_sha256'].items():
        if digest(Path(__file__).resolve().parents[1]/path)!=sha: raise ValueError('Source code hash mismatch')
    bundle=load_frozen_source(a.source_config,a.device)
    if bundle[3]['classification_encoder_sha256']!=provenance['classification_encoder_sha256']: raise ValueError('Classification encoder hash mismatch')
    arm=checkpoint['arm'];features=extract_image_features(bundle,Image.open(a.image).convert('L'),a.image_id,seed=provenance['path_seed'],arms=() if arm=='global' else (arm,))
    tensors={'global_features':features['global_features'][None]}
    if arm!='global':
        bag=features.get(arm,features.get('arms',{}).get(arm))
        if bag is None: raise ValueError('Missing selected arm features')
        for name in ('features','coordinates','rectangles','mask'): tensors[arm+'_'+name]=bag[name][None]
    with torch.inference_mode(): out=forward(model,tensors,torch.tensor([0]),arm,checkpoint['k'])
    result=dict(image_id=a.image_id,image_sha256=digest(a.image),arm=arm,probability=float(out['logits'][0].sigmoid()),global_logit=float(out['global_logits'][0]),residual_logit=float(out['residual'][0]),threshold=.5,source_provenance=provenance,head_sha256=digest(a.checkpoint),note='Attention and signed contribution are pooling explanations, not patch malignancy probabilities or lesion truth')
    if arm!='global': result.update(coordinates=tensors[arm+'_coordinates'][0,:checkpoint['k']].tolist(),rectangles=tensors[arm+'_rectangles'][0,:checkpoint['k']].tolist(),attention=out['attention'][0].tolist(),signed_contribution=out['signed_contribution'][0].tolist())
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False))
if __name__=='__main__': main()
