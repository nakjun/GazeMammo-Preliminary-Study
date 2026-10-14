"""Three compatible classifier heads share one image-only extraction; average probabilities."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.gaze_cxr_1011 import restore_head
from src.gaze_classifier_1011 import image_features
from src.trajectory_1011 import restore
from train_attention_1011 import digest,save
from predict_cxr_attention_1011 import prediction_record


def validate_ensemble(checkpoints):
    if len(checkpoints)!=3:raise ValueError('Exactly three heads required')
    first=checkpoints[0];sources=first['source_provenance']
    if len({head['training_seed'] for head in checkpoints})!=3:raise ValueError('Head seeds must be distinct')
    for head in checkpoints:
        if head['kind']!='gaze_cxr_1011' or head['model_config']!=first['model_config']:raise ValueError('Incompatible ensemble architecture')
        provenance=head['source_provenance']
        for key in ('cache_sha256','global_checkpoint_sha256','gaze_checkpoint_sha256'):
            if provenance[key]!=sources[key]:raise ValueError(f'Incompatible ensemble source: {key}')
        for key in ('training_patient_keys','selection_patient_keys'):
            if set(head[key])!=set(first[key]):raise ValueError(f'Incompatible head patients: {key}')
        for key in ('gaze_training_patient_keys','gaze_selection_patient_keys'):
            if set(provenance['gaze_provenance'][key])!=set(sources['gaze_provenance'][key]):raise ValueError(f'Incompatible gaze patients: {key}')
        for key in ('seed','steps','strategy','policy','patch_size'):
            if provenance['gaze_provenance'][key]!=sources['gaze_provenance'][key]:raise ValueError(f'Incompatible generation protocol: {key}')
        if provenance['final_test_used'] or provenance['gaze_provenance']['final_test_used']:raise ValueError('Unexpected test provenance')
    return sources


def probability_ensemble(logits):
    """Arithmetic sigmoid-probability mean, never sigmoid of the mean logit."""
    return torch.stack([logit.sigmoid() for logit in logits]).mean(0)


def predict(a):
    output=Path(a.output)
    if output.exists():raise FileExistsError('Use a fresh output directory')
    heads=[torch.load(path,map_location='cpu',weights_only=False) for path in a.heads];source=validate_ensemble(heads)
    global_path=Path(a.global_checkpoint or source['global_checkpoint']);gaze_path=Path(a.gaze_checkpoint or source['gaze_checkpoint'])
    if digest(global_path)!=source['global_checkpoint_sha256'] or digest(gaze_path)!=source['gaze_checkpoint_sha256']:raise ValueError('Actual source checkpoint SHA mismatch')
    global_checkpoint=torch.load(global_path,map_location='cpu',weights_only=False)
    for key in ('training_patient_keys','selection_patient_keys'):
        if set(global_checkpoint[key])!=set(heads[0][key]):raise ValueError('Global/head patient mismatch')
    if a.image:image_path=Path(a.image)
    else:
        manifest=pd.read_csv(ROOT/'manifests/all.csv');selected=manifest[(manifest.image_id==a.image_id)&(manifest.split=='val')&(manifest.label>=0)]
        if len(selected)!=1:raise ValueError('Fixed case must be labelled validation')
        image_path=ROOT/'data/image'/selected.iloc[0].file_name
    with Image.open(image_path) as src:image=src.convert('L')
    gaze_checkpoint=torch.load(gaze_path,map_location='cpu',weights_only=False)
    for key,sourcekey in [('training_patient_keys','gaze_training_patient_keys'),('selection_patient_keys','gaze_selection_patient_keys')]:
        if set(gaze_checkpoint[key])!=set(source['gaze_provenance'][sourcekey]):raise ValueError('Gaze/source patient mismatch')
    gaze=restore(gaze_checkpoint).eval().requires_grad_(False)
    features=image_features(gaze,image,a.image_id,seed=source['gaze_provenance']['seed'],include_grid=True);del gaze,gaze_checkpoint
    cached=None;comparison=None
    if a.compare_cache:
        cache_path=Path(a.features or source['cache_path'])
        if digest(cache_path)!=source['cache_sha256']:raise ValueError('Actual source cache SHA mismatch')
        payload=torch.load(cache_path,map_location='cpu',weights_only=False);frame=pd.DataFrame(payload['rows']);matches=np.flatnonzero(frame.image_id.to_numpy()==a.image_id)
        if len(matches)!=1:raise ValueError('Cache image ID not unique')
        cached={key:value[int(matches[0])] for key,value in payload['tensors'].items()}
    outputs=[];cached_outputs=[]
    with torch.inference_mode():
        for head in heads:
            model=restore_head(head,global_checkpoint).eval()
            outputs.append(model(*[features[key][None] for key in ('global_features','gaze_features','gaze_coordinates','grid_features','grid_coordinates')]))
            if cached is not None:cached_outputs.append(model(*[cached[key][None] for key in ('global_features','gaze_features','gaze_coordinates','grid_features','grid_coordinates')]))
            del model
        probability=float(probability_ensemble([out['logits'] for out in outputs]))
        if cached is not None:
            cached_probability=float(probability_ensemble([out['logits'] for out in cached_outputs]));coordinate_error=max(float((features[name+'_coordinates']-cached[name+'_coordinates']).abs().max()) for name in ('gaze','grid'))
            comparison=dict(cached_ensemble_probability=cached_probability,probability_absolute_difference=abs(probability-cached_probability),coordinate_max_absolute_difference=coordinate_error,tolerance=a.tolerance)
            if coordinate_error>1e-6 or comparison['probability_absolute_difference']>a.tolerance:raise ValueError(f'Image-only/cache ensemble mismatch: {comparison}')
    # Labels are display-only, obtained after every head prediction.
    label=int(frame.iloc[int(matches[0])].label) if cached is not None else (int(selected.iloc[0].label) if not a.image else None)
    records=[dict(checkpoint=str(Path(path).resolve()),checkpoint_sha256=digest(path),**prediction_record(out,head['epoch'],head['training_seed'],label)) for path,head,out in zip(a.heads,heads,outputs)]
    result=dict(ensemble_malignant_probability=probability,benign_probability=1-probability,prediction=int(probability>=.5),threshold=.5,ground_truth_display_only=label,image_id=a.image_id,image=str(image_path),variant=heads[0]['model_config']['variant'],head_seeds=[head['training_seed'] for head in heads],aggregation='arithmetic mean of sigmoid probabilities, not mean logits',head_diagnostics=records,gaze_coordinates=features['gaze_coordinates'].tolist(),grid_coordinates=features['grid_coordinates'].tolist(),cached_comparison=comparison,inputs='One image-only extraction shared by three heads; RNG image ID only; no actual gaze/label/ROI/duration inputs',interpretation='Signed contributions describe each head residual logit separately. Probability-ensemble averaging has no claimed signed-logit decomposition or mean-attention explanation.',source_provenance=source)
    output.mkdir(parents=True,exist_ok=False);save(output/'prediction.json',result);print(json.dumps(dict(output=str(output),ensemble_probability=probability,cached_comparison=comparison)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--heads',nargs=3,required=True);p.add_argument('--output',required=True);p.add_argument('--image');p.add_argument('--image-id',type=int,default=64);p.add_argument('--compare-cache',action='store_true');p.add_argument('--features');p.add_argument('--global-checkpoint');p.add_argument('--gaze-checkpoint');p.add_argument('--tolerance',type=float,default=5e-4);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);predict(a)
if __name__=='__main__':main()
