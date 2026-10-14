"""Image-only attention inference, cached equivalence, and fixed-case crop galleries."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from PIL import Image
import torch
from src.gaze_attention_1011 import restore_head
from src.gaze_classifier_1011 import image_features
from src.trajectory_1011 import restore
from train_attention_1011 import digest,save


def plot_case(image,coordinates,result,output,title):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.patheffects import withStroke
    points=np.asarray(coordinates)*image.size;weights=result['attention'][0].numpy();signed=result['signed_contribution'][0].numpy()
    fig,axis=plt.subplots(figsize=(6,11));axis.imshow(image,cmap='gray',vmin=0,vmax=255)
    maximum=max(float(weights.max()),1e-12)
    for i,((x,y),weight,contribution) in enumerate(zip(points,weights,signed)):
        color=plt.cm.viridis(weight/maximum);axis.add_patch(Rectangle((x-112,y-112),224,224,fill=False,edgecolor=color,linewidth=.6+2*weight/maximum));axis.scatter(x,y,s=12,color=color)
        axis.annotate(f'{i+1} a={weight:.3f}',(x,y),color='white',fontsize=7,path_effects=[withStroke(linewidth=1.4,foreground='black')])
    axis.set(xlim=(0,image.width),ylim=(image.height,0),title=title);axis.axis('off');fig.tight_layout();fig.savefig(output/'overlay.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(12,12))
    for i,(axis,(x,y)) in enumerate(zip(axes.flat,points)):
        x,y=np.round([x,y]).astype(int);crop=image.crop((x-112,y-112,x+112,y+112));axis.imshow(crop,cmap='gray',vmin=0,vmax=255)
        axis.set_title(f'Step{i+1}: attention={weights[i]:.4f}\nsigned residual={signed[i]:+.4f}',fontsize=9);axis.axis('off')
    fig.suptitle(title+'\nAttention is normalized importance, not malignancy probability',fontsize=11);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/'crop_gallery.png',dpi=150);plt.close(fig)


def predict(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    checkpoint=torch.load(a.head,map_location='cpu',weights_only=False);provenance=checkpoint['source_provenance']
    global_path=Path(a.global_checkpoint or provenance['global_checkpoint']);gaze_path=Path(a.gaze_checkpoint or provenance['gaze_checkpoint'])
    if digest(global_path)!=provenance['global_checkpoint_sha256'] or digest(gaze_path)!=provenance['gaze_checkpoint_sha256']:raise ValueError('Inference source checkpoint SHA mismatch')
    global_checkpoint=torch.load(global_path,map_location='cpu',weights_only=False)
    if set(global_checkpoint['training_patient_keys'])!=set(checkpoint['training_patient_keys']) or set(global_checkpoint['selection_patient_keys'])!=set(checkpoint['selection_patient_keys']):raise ValueError('Inference global patient mismatch')
    model=restore_head(checkpoint,global_checkpoint).eval();image_id=a.image_id;label=None;cache=None
    if a.image:
        image_path=Path(a.image)
    else:
        manifest=pd.read_csv(ROOT/'manifests/all.csv');row=manifest[(manifest.image_id==image_id)&(manifest.split=='val')&(manifest.label>=0)]
        if len(row)!=1:raise ValueError('Fixed case must be labelled validation')
        image_path=ROOT/'data/image'/row.iloc[0].file_name
    with Image.open(image_path) as source:image=source.convert('L')
    if a.cached_only or a.compare_cache:
        cache_path=Path(a.features or provenance['cache_path'])
        if digest(cache_path)!=provenance['cache_sha256']:raise ValueError('Inference cache SHA mismatch')
        cache=torch.load(cache_path,map_location='cpu',weights_only=False);rows=pd.DataFrame(cache['rows']);matches=np.flatnonzero(rows.image_id.to_numpy()==image_id)
        if len(matches)!=1:raise ValueError('Image ID not unique in source cache')
        index=int(matches[0]);cached={key:value[index] for key,value in cache['tensors'].items()}
    if a.cached_only:features=cached;route='cached features, image-only provenance (no new encoder rollout)'
    else:
        gaze_checkpoint=torch.load(gaze_path,map_location='cpu',weights_only=False)
        if gaze_checkpoint['training_patient_keys']!=provenance['gaze_provenance']['gaze_training_patient_keys'] or gaze_checkpoint['selection_patient_keys']!=provenance['gaze_provenance']['gaze_selection_patient_keys']:raise ValueError('Inference gaze patient mismatch')
        gaze=restore(gaze_checkpoint).eval().requires_grad_(False)
        features=image_features(gaze,image,image_id,seed=42,include_grid=model.source=='grid');route='image pixels and supplied RNG image ID only; no gaze/label/box input';del gaze,gaze_checkpoint
    with torch.inference_mode():
        result=model(features['global_features'][None],features[model.source+'_features'][None],features[model.source+'_coordinates'][None]);comparison=None
        if a.compare_cache:
            cached_result=model(cached['global_features'][None],cached[model.source+'_features'][None],cached[model.source+'_coordinates'][None])
            coord_delta=float((features[model.source+'_coordinates']-cached[model.source+'_coordinates']).abs().max())
            probability_delta=float(abs(result['logits'].sigmoid()-cached_result['logits'].sigmoid()))
            comparison=dict(coordinate_max_absolute_difference=coord_delta,probability_absolute_difference=probability_delta,tolerance=a.tolerance)
            if coord_delta>1e-6 or probability_delta>a.tolerance:raise ValueError(f'Cached/image-only mismatch: {comparison}')
    probability=float(result['logits'].sigmoid());coordinates=features[model.source+'_coordinates'].numpy()
    # Ground-truth display is after prediction and never enters the model.
    if cache is not None:label=int(rows.iloc[index].label)
    elif not a.image:label=int(row.iloc[0].label)
    record=dict(image_id=image_id,image=str(image_path),ground_truth_display_only=label,malignant_probability=probability,threshold=.5,prediction=int(probability>=.5),head_seed=checkpoint['training_seed'],selected_epoch=checkpoint['epoch'],attention_untrained=checkpoint['epoch']==0,attention=result['attention'][0].tolist(),signed_contribution=result['signed_contribution'][0].tolist(),residual_bias=float(result['residual_bias']),residual_logit=float(result['residual']),alpha=float(result['alpha']),coordinates=coordinates.tolist(),inputs=route,cached_comparison=comparison,source_provenance=provenance)
    if not np.isclose(sum(record['signed_contribution'])+record['residual_bias'],record['residual_logit'],atol=2e-6):raise ValueError('Signed contribution mismatch')
    save(output/'prediction.json',record)
    caveat='epoch0: local attention untrained / residual zero' if checkpoint['epoch']==0 else f'selected epoch{checkpoint["epoch"]}'
    title=f'{model.variant}, head seed{checkpoint["training_seed"]}; {caveat}\nP(malignant)={probability:.3f}; GT(display only)={label}'
    plot_case(image,coordinates,result,output,title);print(json.dumps(dict(output=str(output),probability=probability,cached_comparison=comparison)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--head',required=True);p.add_argument('--output',required=True);p.add_argument('--image');p.add_argument('--image-id',type=int,default=64);p.add_argument('--global-checkpoint');p.add_argument('--gaze-checkpoint');p.add_argument('--features');p.add_argument('--cached-only',action='store_true');p.add_argument('--compare-cache',action='store_true');p.add_argument('--tolerance',type=float,default=5e-4);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);predict(a)
if __name__=='__main__':main()
