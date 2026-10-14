"""Image-only dual-source inference and cheap cached interim crop visual checks."""
import argparse,os,json,sys
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


def plot_case(image,features,result,output,title):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl');import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.patheffects import withStroke
    fig,axes=plt.subplots(1,2,figsize=(10,10));gallery,gallery_axes=plt.subplots(8,4,figsize=(12,21))
    for source_index,source in enumerate(('gaze','grid')):
        axis=axes[source_index];axis.imshow(image,cmap='gray',vmin=0,vmax=255);points=features[source+'_coordinates'].cpu().numpy()*image.size
        attention=result[source+'_attention'][0].cpu().numpy();beta=float(result['beta'][0,source_index]);effective=attention*beta;signed=result[source+'_signed'][0].cpu().numpy();quality=result[source+'_quality'][0].cpu().numpy();valid=result[source+'_mask'][0].cpu().numpy()
        for i,((x,y),weight,contribution,q) in enumerate(zip(points,effective,signed,quality)):
            color=plt.cm.viridis(min(weight*16,1));axis.add_patch(Rectangle((x-112,y-112),224,224,fill=False,edgecolor=color,linewidth=.5+min(weight*32,3)));axis.annotate(str(i+1),(x,y),color='white',fontsize=7,path_effects=[withStroke(linewidth=1.4,foreground='black')])
            ga=gallery_axes.flat[source_index*16+i];cx,cy=np.round([x,y]).astype(int);ga.imshow(image.crop((cx-112,cy-112,cx+112,cy+112)),cmap='gray',vmin=0,vmax=255)
            ga.set_title(f'{source} step{i+1}: effective a={weight:.4f}\nsigned={contribution:+.4f}; q={q:.3f}; valid={bool(valid[i])}',fontsize=8);ga.axis('off')
        axis.set(xlim=(0,image.width),ylim=(image.height,0),title=f'{source}: source gate={beta:.3f}');axis.axis('off')
    fig.suptitle(title+'\nEffective attention = source gate × within-source attention',fontsize=10);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/'overlay.png',dpi=140);plt.close(fig)
    gallery.suptitle(title+'\nRaw patch values; importance is not gaze or malignancy probability',fontsize=11);gallery.tight_layout(rect=(0,0,1,.96));gallery.savefig(output/'crop_gallery.png',dpi=130);plt.close(gallery)


def prediction_record(result,epoch,seed,label=None):
    probability=float(result['logits'].sigmoid());record=dict(malignant_probability=probability,prediction=int(probability>=.5),threshold=.5,epoch=epoch,head_seed=seed,ground_truth_display_only=label,attention_untrained=epoch==0,alpha=float(result['alpha']),residual=float(result['residual']),residual_bias=float(result['residual_bias']),source_gate=result['beta'][0].tolist())
    for source in ('gaze','grid'):record[source]=dict(attention=result[source+'_attention'][0].tolist(),effective_attention=(result[source+'_attention'][0]*result['beta'][0,0 if source=='gaze' else 1]).tolist(),signed_contribution=result[source+'_signed'][0].tolist(),quality=result[source+'_quality'][0].tolist())
    total=sum(record['gaze']['signed_contribution'])+sum(record['grid']['signed_contribution'])+record['residual_bias']
    if not np.isclose(total,record['residual'],atol=3e-6):raise ValueError('Inference signed residual mismatch')
    return record

@torch.inference_mode()
def render_cached_case(model,tensors,frame,image_id,output,current_epoch,best_epoch):
    rows=frame[(frame.image_id==image_id)&(frame.split=='val')]
    if len(rows)!=1:raise ValueError('Interim case must be validation')
    index=int(rows.index[0]);features={key:value[index] for key,value in tensors.items()};output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=model(features['global_features'][None],features['gaze_features'][None],features['gaze_coordinates'][None],features['grid_features'][None],features['grid_coordinates'][None])
    label=int(rows.iloc[0].label);record=prediction_record(result,current_epoch,42,label);record.update(image_id=image_id,variant=model.variant,best_epoch_so_far=best_epoch,route='cached image-only features; current weights, not selection/tuning evidence')
    save(output/'prediction.json',record)
    with Image.open(ROOT/'data/image'/rows.iloc[0].file_name) as src:image=src.convert('L')
    caveat='epoch0 local attention untrained / residual zero' if current_epoch==0 else f'current epoch{current_epoch}; best so far{best_epoch}'
    plot_case(image,features,result,output,f'{model.variant}: {caveat}; case{image_id}\nP(malignant)={record["malignant_probability"]:.3f}; GT(display only)={label}')


def predict(a):
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);head_checkpoint=torch.load(a.head,map_location='cpu',weights_only=False);provenance=head_checkpoint['source_provenance']
    global_path=Path(a.global_checkpoint or provenance['global_checkpoint']);gaze_path=Path(a.gaze_checkpoint or provenance['gaze_checkpoint'])
    if digest(global_path)!=provenance['global_checkpoint_sha256'] or digest(gaze_path)!=provenance['gaze_checkpoint_sha256']:raise ValueError('Inference source SHA mismatch')
    global_checkpoint=torch.load(global_path,map_location='cpu',weights_only=False)
    if set(head_checkpoint['training_patient_keys'])!=set(global_checkpoint['training_patient_keys']) or set(head_checkpoint['selection_patient_keys'])!=set(global_checkpoint['selection_patient_keys']):raise ValueError('Inference global patients mismatch')
    model=restore_head(head_checkpoint,global_checkpoint).eval();label=None;cached=None
    if a.image:image_path=Path(a.image)
    else:
        frame=pd.read_csv(ROOT/'manifests/all.csv');row=frame[(frame.image_id==a.image_id)&(frame.split=='val')&(frame.label>=0)]
        if len(row)!=1:raise ValueError('Fixed case must be labelled validation')
        image_path=ROOT/'data/image'/row.iloc[0].file_name
    with Image.open(image_path) as source:image=source.convert('L')
    if a.cached_only or a.compare_cache:
        cache_path=Path(a.features or provenance['cache_path'])
        if digest(cache_path)!=provenance['cache_sha256']:raise ValueError('Inference cache SHA mismatch')
        payload=torch.load(cache_path,map_location='cpu',weights_only=False);cache_frame=pd.DataFrame(payload['rows']);match=np.flatnonzero(cache_frame.image_id.to_numpy()==a.image_id)
        if len(match)!=1:raise ValueError('Cache image ID not unique')
        cached={key:value[int(match[0])] for key,value in payload['tensors'].items()}
    if a.cached_only:features=cached;route='cached source image-only features'
    else:
        gaze_checkpoint=torch.load(gaze_path,map_location='cpu',weights_only=False)
        for key,sourcekey in [('training_patient_keys','gaze_training_patient_keys'),('selection_patient_keys','gaze_selection_patient_keys')]:
            if set(gaze_checkpoint[key])!=set(provenance['gaze_provenance'][sourcekey]):raise ValueError('Inference gaze patients mismatch')
        gaze=restore(gaze_checkpoint).eval().requires_grad_(False);features=image_features(gaze,image,a.image_id,seed=42,include_grid=True);route='image pixels and supplied RNG image ID only; no gaze/label/ROI/duration inputs';del gaze,gaze_checkpoint
    with torch.inference_mode():
        args=[features[key][None] for key in ('global_features','gaze_features','gaze_coordinates','grid_features','grid_coordinates')];options=dict(force_gaze_absent=a.no_gaze,force_grid_absent=a.no_grid);result=model(*args,**options);comparison=None
        if a.compare_cache:
            cached_result=model(*[cached[key][None] for key in ('global_features','gaze_features','gaze_coordinates','grid_features','grid_coordinates')],**options)
            coords=max(float((features[source+'_coordinates']-cached[source+'_coordinates']).abs().max()) for source in ('gaze','grid'));prob_delta=float(abs(result['logits'].sigmoid()-cached_result['logits'].sigmoid()));comparison=dict(coordinate_max_absolute_difference=coords,probability_absolute_difference=prob_delta,tolerance=a.tolerance)
            if coords>1e-6 or prob_delta>a.tolerance:raise ValueError(f'Cache comparison failed: {comparison}')
    if cached is not None:label=int(cache_frame.iloc[int(match[0])].label)
    elif not a.image:label=int(row.iloc[0].label)
    record=prediction_record(result,head_checkpoint['epoch'],head_checkpoint['training_seed'],label);record.update(image_id=a.image_id,image=str(image_path),variant=model.variant,inputs=route,cached_comparison=comparison,source_provenance=provenance)
    for source in ('gaze','grid'):record[source]['coordinates']=features[source+'_coordinates'].tolist()
    save(output/'prediction.json',record);caveat='epoch0: local attention untrained / residual zero' if head_checkpoint['epoch']==0 else f'selected epoch{head_checkpoint["epoch"]}'
    plot_case(image,features,result,output,f'{model.variant}: {caveat}\nP(malignant)={record["malignant_probability"]:.3f}; GT(display only)={label}')
    print(json.dumps(dict(output=str(output),probability=record['malignant_probability'],cached_comparison=comparison)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--head',required=True);p.add_argument('--output',required=True);p.add_argument('--image');p.add_argument('--image-id',type=int,default=64);p.add_argument('--global-checkpoint');p.add_argument('--gaze-checkpoint');p.add_argument('--features');p.add_argument('--cached-only',action='store_true');p.add_argument('--compare-cache',action='store_true');p.add_argument('--no-gaze',action='store_true');p.add_argument('--no-grid',action='store_true');p.add_argument('--tolerance',type=float,default=5e-4);p.add_argument('--threads',type=int,default=4);a=p.parse_args();torch.set_num_threads(a.threads);predict(a)
if __name__=='__main__':main()
