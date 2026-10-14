"""Reproducible random correctly diagnosed VAL case, full model inputs + real crops."""
import argparse,json,sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from PIL import Image
from src.data import patch_tensor,MEAN,STD
from src.official_cbis_classifier_1012 import digest
from src.grid14_patchcount_ablation_1013 import nested_sequence
from scripts.train_gaze_multiview_1012 import pair_views

BLUE='#2672b5';ORANGE='#d77b20'

def crop_verified(image,center):
 x,y=np.round(center).astype(int);rect=[int(x-112),int(y-112),int(x+112),int(y+112)];raw=np.asarray(image.convert('L'));left,top,right,bottom=rect;expected=np.zeros((224,224),dtype=np.uint8);il,it,ir,ib=max(0,left),max(0,top),min(image.width,right),min(image.height,bottom)
 if ir>il and ib>it:expected[it-top:ib-top,il-left:ir-left]=raw[it:ib,il:ir]
 crop=np.asarray(image.crop(tuple(rect)).convert('L'));assert crop.shape==(224,224) and np.array_equal(crop,expected)
 recovered=((patch_tensor(image,center,224)*STD+MEAN)*255).round().clamp(0,255).to(torch.uint8);assert all(np.array_equal(ch.numpy(),crop) for ch in recovered)
 return crop,dict(rectangle=rect,visible_intersection=[il,it,ir,ib],black_padding=bool(left<0 or top<0 or right>image.width or bottom>image.height),pixel_array_exact=True,original_helper_normalization_roundtrip_exact=True)

def read_ensemble(ablation,k,mode,frame):
 name='multiview_validation_predictions.csv' if mode=='multiview' else 'validation_predictions.csv';rows=[];paths=[]
 for seed in [42,43,44]:
  path=ablation/f'gaze_baseline_k{k}_seed{seed}'/name;p=pd.read_csv(path).set_index('image_id').loc[frame.image_id].reset_index();assert p[['image_id','patient_key','label']].equals(frame.reset_index(drop=True));rows.append(p.probability.to_numpy());paths.append(path)
 out=frame.reset_index(drop=True).copy();out['probability']=np.mean(rows,axis=0);return out,paths

def draw_original(ax,image,patches):
 import matplotlib.patches as mp
 ax.imshow(image,cmap='gray',vmin=0,vmax=255);ax.set_xlim(-220,1720);ax.set_ylim(3000,0);ax.axis('off')
 for p in patches:
  x0,y0,x1,y1=p['rectangle'];ax.add_patch(mp.Rectangle((x0,y0),224,224,fill=False,edgecolor=p['color'],linewidth=1.1,alpha=.9))
 for side in [0,1]:
  subset=sorted([p for p in patches if (p['center_pixels'][0]>=750)==bool(side)],key=lambda p:p['center_pixels'][1]);positions=np.linspace(120,2880,len(subset)) if len(subset)>1 else [1500]
  for p,yy in zip(subset,positions):
   xx=1660 if side else -160;ax.annotate(p['name'],xy=p['center_pixels'],xytext=(xx,yy),ha='center',va='center',fontsize=7,color=p['color'],bbox=dict(boxstyle='round,pad=.12',fc='white',ec=p['color'],lw=.5,alpha=.9),arrowprops=dict(arrowstyle='-',color=p['color'],lw=.5,alpha=.8))

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--k',type=int,default=4,choices=[2,4],help='Supported complete valid bags: K=2 or K=4 (default). Mask-padded K=16 is not visualized.');ap.add_argument('--random-seed',type=int,default=20261013);ap.add_argument('--output',default='runs/grid14_gaze_case_visualization_1013');ap.add_argument('--pdf',action='store_true',help='Also export the same figure as PDF');a=ap.parse_args();torch.set_num_threads(4);out=Path(a.output);out.mkdir(exist_ok=True);ablation=Path('runs/grid14_patchcount_ablation_1013');manifest=json.loads((ablation/'manifest.json').read_text());assert manifest['state']=='complete';cp=Path('runs/official_cbis_v2_experiment_1012/development.pt');assert digest(cp)==manifest['cache_sha256'];cache=torch.load(cp,map_location='cpu',weights_only=False,mmap=True);f=pd.DataFrame(cache['rows']);assert set(f.split)=={'train','val'};val=f[f.split=='val'][['image_id','patient_key','label']].reset_index(drop=True);assert len(val)==356 and val.patient_key.nunique()==183;multi,multi_paths=read_ensemble(ablation,a.k,'multiview',val);single,single_paths=read_ensemble(ablation,a.k,'singleview',val);correct=multi[(multi.probability>=.5).astype(int)==multi.label].sort_values('image_id').reset_index(drop=True);chosen=int(np.random.default_rng(a.random_seed).choice(correct.image_id.to_numpy()));i=int(np.flatnonzero(f.image_id==chosen)[0]);partners,side,view=pair_views(f);views=[i]+partners[i];assert len(views)<=2
 selection_path=ablation/'gaze_baseline_nested_selection.pt';assert digest(selection_path)==manifest['dependencies'][str(selection_path)];selection=torch.load(selection_path,map_location='cpu',weights_only=False);scorepath=Path('runs/gaze_diagnosis_selector_1012_round10_pilot42/selector_gaze_baseline/selected_indices.pt');assert digest(scorepath)==manifest['dependencies'][str(scorepath)];scores=torch.load(scorepath,map_location='cpu',weights_only=False);assert selection['image_ids']==scores['image_ids']==f.image_id.tolist();t=cache['tensors'];all_coords=torch.cat([t['grid_coordinates'],t['gaze_baseline_coordinates']],1);all_valid=torch.cat([t['grid_mask'],t['gaze_baseline_mask']],1)
 pm=multi.set_index('image_id');ps=single.set_index('image_id');metadata=dict(selection_rule='UniformrandomimageIDamongcorrectVALpairedensemblebinarydiagnoses,afterstableIDsort',random_seed=a.random_seed,k=a.k,combination=f'FixedGrid14+predictedgaze{a.k}',candidate_correct_count=len(correct),total_val_images=356,classifier_seeds=[42,43,44],threshold=.5,chosen_image_id=chosen,patient_key=f.iloc[i].patient_key,true_label=int(f.iloc[i].label),true_class='Malignant' if f.iloc[i].label else 'Benign',predicted_class='Malignant' if pm.loc[chosen].probability>=.5 else 'Benign',malignant_probability_paired_ensemble=float(pm.loc[chosen].probability),malignant_probability_current_alone_ensemble=float(ps.loc[chosen].probability),paired_information_used=bool(partners[i]),views=[],test_used=False,oracle_used=False,model_retrained=False,lesion_ground_truth_used=False,source_sha256=digest(__file__),dependencies={str(p):digest(p) for p in [cp,selection_path,scorepath,ablation/'manifest.json',Path('src/data.py'),Path('src/grid14_patchcount_ablation_1013.py'),Path('scripts/train_gaze_multiview_1012.py'),*multi_paths,*single_paths]})
 os.environ['MPLCONFIGDIR']='/tmp/grid14_gaze_case_mpl';import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig=plt.figure(figsize=(17,5.1*len(views)+1.3));outer=fig.add_gridspec(len(views),2,width_ratios=[1.3,4.9],left=.025,right=.985,top=.88,bottom=.065,wspace=.1,hspace=.32)
 for row_index,j in enumerate(views):
  row=f.iloc[j];assert row.split=='val' and row.patient_key==f.iloc[i].patient_key and side.iloc[j]==side.iloc[i]
  if j!=i:assert view.iloc[j]!=view.iloc[i] and view.iloc[j] in ['CC','MLO']
  grid,source,relaxed=nested_sequence(scores['scores'][j],all_coords[j],all_valid[j]);expected=torch.cat([grid,source[:a.k]]);valid=selection['available'][j,:14+a.k];indices=selection['indices'][j,:14+a.k];assert valid.all() and len(source)>=a.k and torch.equal(indices,expected);assert (indices[:14]<16).all() and (indices[14:]>=16).all();image_path=Path(row.canonical_path);assert digest(image_path)==row.image_sha256;image=Image.open(image_path).convert('L');assert image.size==(1500,3000);patches=[];patch_arrays=[]
  for pindex,candidate in enumerate(indices.tolist()):
   coords=all_coords[j,candidate];center=coords.double().numpy()*np.array(image.size);crop,audit=crop_verified(image,center);cache_arm='grid' if candidate<16 else 'gaze_baseline';cache_slot=candidate if candidate<16 else candidate-16;assert audit['rectangle']==t[cache_arm+'_rectangles'][j,cache_slot].tolist();audit['cached_crop_rectangle_exact']=True;is_grid=pindex<14;name=f'G{pindex+1:02d}' if is_grid else f'P{pindex-13:02d}';entry=dict(name=name,source='Grid' if is_grid else 'Predictedgaze',candidate_index=candidate,normalized_center=coords.tolist(),center_pixels=center.tolist(),color=BLUE if is_grid else ORANGE,**audit);patches.append(entry);patch_arrays.append(crop)
  original_ax=fig.add_subplot(outer[row_index,0]);draw_original(original_ax,image,patches);original_ax.set_title(f'Model input / preprocessed full image\nID {row.image_id} | {side.iloc[j]} {view.iloc[j]}\nCurrent-view P(M)={ps.loc[row.image_id].probability:.4f}',fontsize=10)
  cols=6;nrows=int(np.ceil(len(patches)/cols));tiles=outer[row_index,1].subgridspec(nrows,cols,hspace=.2,wspace=.1)
  for pindex,(entry,crop) in enumerate(zip(patches,patch_arrays)):
   ax=fig.add_subplot(tiles[pindex//cols,pindex%cols]);ax.imshow(crop,cmap='gray',vmin=0,vmax=255);ax.set_xticks([]);ax.set_yticks([]);ax.set_title(f"{entry['name']} | {entry['source']} | 224 x 224",fontsize=8,color=entry['color'])
   for spine in ax.spines.values():spine.set_edgecolor(entry['color']);spine.set_linewidth(1.8)
  metadata['views'].append(dict(image_id=int(row.image_id),side=side.iloc[j],view=view.iloc[j],canonical_path=str(image_path),canonical_sha256=row.image_sha256,true_label=int(row.label),malignant_probability_current_alone_ensemble=float(ps.loc[row.image_id].probability),patch_count=len(patches),patches=patches));metadata['dependencies'][str(image_path)]=digest(image_path)
 fig.suptitle(f"{metadata['combination']} | Random correct VAL case ID {chosen} | {side.iloc[i]} {view.iloc[i]}\nTrue: {metadata['true_class']} | Predicted: {metadata['predicted_class']} | Paired ensemble P(M)={metadata['malignant_probability_paired_ensemble']:.4f}",fontsize=15,y=.97)
 figure_information=f"Paired CC/MLO ensemble uses both displayed views ({len(views)*(14+a.k)} selected crops)." if partners[i] else f"No opposite-view partner; current-view fallback ({14+a.k} selected crops)."
 fig.text(.02,.018,f'{figure_information} Blue=Grid; orange=predicted gaze. Sampling boxes are not lesion annotations.\nCanonical model-input contrast preserved; 224-pixel crops verified against original helper. Uniform correct-case draw, RNG seed {a.random_seed}.',fontsize=9);fig.savefig(out/'case.png',dpi=180)
 if a.pdf:fig.savefig(out/'case.pdf')
 plt.close(fig);metadata['figure_png_sha256']=digest(out/'case.png');assert all(digest(p)==h for p,h in metadata['dependencies'].items());(out/'case.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(dict(image_id=chosen,true_class=metadata['true_class'],predicted_class=metadata['predicted_class'],paired_p_m=metadata['malignant_probability_paired_ensemble'],current_alone_p_m=metadata['malignant_probability_current_alone_ensemble'],correct_candidates=len(correct),views=[dict(image_id=x['image_id'],view=x['view'],patches=x['patch_count']) for x in metadata['views']],figure=str(out/'case.png'))))
if __name__=='__main__':main()
