"""Expert-gaze ORACLE diagnostic on matched subset only, not deployable/goal."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from src.trajectory_1011 import window_sample
from src.official_cbis_classifier_1012 import digest,restore,metrics,paired_ci,state_digest
from src.official_cbis_trajectory_1012 import build_fresh_model

def expert_bag(record):
 # Cache vectors are current-point crops. Never use shifted teacher_inputs.
 w=window_sample([p.numpy() for p in record['parts']],record['image_id'],mode='legacy16',epoch=0,seed=42)
 coords=w['coords'][:16];mask=w['valid'][:16];vectors=torch.zeros(16,1024);positions=w['positions'][:16]
 vectors[mask]=record['vectors'][w['segment']][positions[mask]]
 assert torch.isfinite(coords[mask]).all() and ((coords[mask]>=0)&(coords[mask]<1)).all() and torch.isfinite(vectors).all()
 return vectors,coords,mask,dict(segment=w['segment'],positions=positions.tolist(),valid=int(mask.sum()),first_vector_norm=float(vectors[0].norm()),duplicates=int(mask.sum())-len(torch.unique(coords[mask],dim=0)))

@torch.inference_mode()
def infer(model,g,bag,ids):
 return torch.cat([model(g[x],bag[0][x],bag[1][x],bag[2][x])['logits'].sigmoid() for x in ids.split(32)]).numpy()

def main():
 torch.set_num_threads(4);torch.set_num_interop_threads(1);out=Path('runs/expert_gaze_oracle_1012_round8');out.mkdir(exist_ok=False);cp='runs/official_cbis_v2_experiment_1012/development.pt';cache=torch.load(cp,map_location='cpu',weights_only=False);f=pd.DataFrame(cache['rows']);indexp=Path('runs/official_cbis_v2_experiment_1012/gaze_cache/index.pt');index=torch.load(indexp,map_location='cpu',weights_only=False);fresh=build_fresh_model('cpu');teacher_visual_sha=state_digest(fresh.visual.state_dict());assert teacher_visual_sha==cache['provenance']['visual_state_sha256'];del fresh;gaze=pd.read_csv('runs/official_cbis_preprocess_v2_1012/gaze.csv').set_index('image_id');assert len(index['records'])==497 and index['complete'];assert index['original_pretrained_sha256']==digest('/home/njsung/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth');assert all(digest(p)==h for p,h in index['source_hashes'].items());
 m=dict(state='running',stage='teacher_cache_alignment_audit',oracle_only=True,deployable=False,goal_metric=False,test_used=False,subset='expert-available TRAIN/VAL only; fullVAL356 not evaluated with oracle',seeds=[42,43,44],patch_budget=16,selection='Fixed legacy16 longest segment/downsample positions, no diagnosis labels',source_sha256=digest(__file__),cache_sha256=digest(cp),index_sha256=digest(indexp),teacher_cached_encoder='Frozen pretrained ConvNeXtBase 1024dim with identical weightSHA; currentpointvectors, not shiftedteacherinputs',teacher_visual_state_sha256=teacher_visual_sha,known_runtime=index['runtime_flags'],dependencies={},results=[])
 def save():(out/'manifest.json').write_text(json.dumps(m,indent=2))
 save();ids=[];vs=[];cs=[];masks=[];audit=[];hashdisagreements=[]
 for path in index['records']:
  assert digest(path)==index['record_hashes'][path];r=torch.load(path,map_location='cpu',weights_only=False);found=np.flatnonzero(f.image_id==r['image_id']);assert len(found)==1;i=int(found[0]);row=f.iloc[i];g=gaze.loc[r['image_id']];assert r['patient_key']==row.patient_key and r['split']==row.split and r['split'] in ['train','val'];assert digest(g.gaze_canonical_path)==row.image_sha256 and digest(row.canonical_path)==row.image_sha256;meta=json.loads(Path(str(g.gaze_canonical_path)+'.json').read_text());assert meta['registration']['quality']['passed'];assert all(p.shape[0]==v.shape[0] and v.shape[1]==1024 for p,v in zip(r['parts'],r['vectors']))
  v,c,mask,a=expert_bag(r);ids.append(i);vs.append(v);cs.append(c);masks.append(mask);audit.append(dict(image_id=r['image_id'],split=r['split'],**a))
 chosen=torch.tensor(ids);frame=f.iloc[ids].reset_index(drop=True);t=cache['tensors'];globalfeatures=t['global_features'][chosen];mask=torch.stack(masks);bags={'expert_oracle':(torch.stack(vs),torch.stack(cs),mask)}
 for arm in ['grid','gaze_baseline','gaze_improved','random']:bags[arm]=(t[arm+'_features'][chosen],t[arm+'_coordinates'][chosen],t[arm+'_mask'][chosen]&mask)
 summaries={split:dict(n=int((frame.split==split).sum()),patients=int(frame[frame.split==split].patient_key.nunique()),positive=int(frame[frame.split==split].label.sum()),valid_patches_mean=float(mask[frame.split.values==split].sum(1).float().mean()),less_than16=sum(x['valid']<16 for x in audit if x['split']==split)) for split in ['train','val']};(out/'alignment_audit.json').write_text(json.dumps(dict(population=summaries,per_image=audit,all497_raw_gaze_pixels_same_sha=True,offset_corrected_by_direct_current_vectors=True,invalid_vectors_zero_masked=True),indent=2));m['population']=summaries;m['stage']='shared_scorer_inference';save();print(json.dumps(summaries),flush=True);preds={};heads=Path('runs/official_cbis_v2_experiment_1012/heads')
 for seed in m['seeds']:
  gp=heads/f'global_seed{seed}/best.pt';hp=heads/f'grid_seed{seed}/best.pt';m['dependencies'].update({str(p):digest(p) for p in [gp,hp]});gc=torch.load(gp,map_location='cpu',weights_only=False);hc=torch.load(hp,map_location='cpu',weights_only=False);assert hc['source_provenance']['visual_state_sha256']==cache['provenance']['visual_state_sha256'];model=restore(hc,gc)
  for arm,bag in bags.items():
   for split in ['train','val']:
    where=torch.tensor(np.flatnonzero(frame.split==split));p=infer(model,globalfeatures,bag,where);pred=frame.iloc[where.tolist()][['image_id','patient_key','label']].copy();pred['probability']=p;pred.to_csv(out/f'{arm}_{split}_seed{seed}_predictions.csv',index=False);preds[(arm,split,seed)]=pred;r=dict(arm=arm,split=split,seed=seed,metrics=metrics(pred.label,pred.probability));m['results'].append(r);save()
 report=dict(results=m['results'],means={},paired={},limitations='Expert gaze is oracle input; matched VAL77 only, small patient-bootstrap confidence intervals. Classifier trained on fullTRAIN including expert subset, TRAIN scores in-sample. No fullVAL356 goal claim.')
 for arm in bags:
  for split in ['train','val']:report['means'][arm+'_'+split]=float(np.mean([r['metrics']['auroc'] for r in m['results'] if r['arm']==arm and r['split']==split]))
 for seed in m['seeds']:
  for arm in ['grid','gaze_baseline','gaze_improved','random']:report['paired'][f'oracle_minus_{arm}_val_seed{seed}']=paired_ci(preds[('expert_oracle','val',seed)],preds[(arm,'val',seed)],1000)
 (out/'report.json').write_text(json.dumps(report,indent=2));assert digest(cp)==m['cache_sha256'] and digest(indexp)==m['index_sha256'] and digest(__file__)==m['source_sha256'] and all(digest(p)==h for p,h in m['dependencies'].items());m['state']='complete';m['stage']='report_complete';save();print(json.dumps({'means':report['means'],'paired':report['paired']}),flush=True)
if __name__=='__main__':main()
