"""Independent cross-view terminal export/fixed mixtures/paired patient CI."""
import json,sys
from pathlib import Path
import pandas as pd,numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.proposal_utility_1011 import file_sha256,state_sha256
from src.dual_output_1011 import config_sha256
from src.gaze_crossview_1011 import restore_crossview
sys.path.insert(0,str(ROOT/'scripts'))
from gaze_nll_adaptation_1011 import load_base

def main():
    torch.set_num_threads(4);trial=ROOT/'runs/dual_target_1011/gaze_crossview_seed42';folder=trial/'fixed_export';summary=json.loads((folder/'summary.json').read_text());assert summary['state']=='complete';training=json.loads((trial/'summary.json').read_text());assert training['state']=='complete' and training['epochs']==15;ck=torch.load(trial/'best.pt',map_location='cpu',weights_only=False);config=ck['config'];assert file_sha256(trial/'best.pt')==summary['checkpoint_sha256'];assert file_sha256(config['source_checkpoint'])==config['source_checkpoint_sha256'];assert file_sha256(Path(config['prefix_cache'])/'provenance.json')==config['prefix_cache_provenance_sha256'];assert file_sha256(ROOT/'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/provenance.json')==config['canonical_cache_provenance_sha256'];assert file_sha256(ROOT/'runs/dual_target_1011/train_gaze_pool_cache/provenance.json')==config['pool_cache_provenance_sha256'];assert file_sha256(ROOT/'runs/dual_target_1011/augmentation_pool_audit_v2/train_pool_labels.pt')==config['pool_labels_sha256']
    for path,sha in config['source_code_sha256'].items():assert file_sha256(ROOT/path)==sha
    base,_=load_base(config['source_checkpoint']);model=restore_crossview(ck,base);assert state_sha256(model.compact_state())==state_sha256(ck['state']) and state_sha256(model.frozen_state())==ck['frozen_state_sha256'];del model,base
    current=json.loads((ROOT/'runs/dual_target_1011/dino_mdn_unified_runtime/config.json').read_text());assert config_sha256(current['config'])==current['sha256']==summary['current_parent_config_sha256'];recipe=current['config'];recipes=[]
    while True:
        recipes.append(recipe);child=recipe.get('parent_config',recipe.get('baseline_config'))
        if child is None:break
        assert child['sha256']==config_sha256(child['config']);recipe=child['config']
    for recipe in recipes:
        for path,sha in recipe['source_code_sha256'].items():assert file_sha256(ROOT/path)==sha
        dependencies=recipe['probability']['parents']+[recipe['roi']['guided'],recipe['roi']['offset']]
        for d in dependencies:assert file_sha256(d['path'])==d['sha256']
    for d in current['config']['reference_caches']:assert file_sha256(d['raw_cache'])==d['raw_cache_sha256']
    frame=pd.read_csv(folder/'per_image.csv');allframe=pd.read_csv(ROOT/'manifests/all.csv',usecols=['image_id','patient_key','split']);trajectory=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','patient_key','split']);route=pd.concat([allframe[allframe.split=='train'],trajectory[trajectory.split=='val']]).set_index('image_id');assert len(frame)==505 and frame.image_id.is_unique and set(frame.image_id)==set(route.index);assert all(r.patient_key==route.loc[r.image_id].patient_key and r.split==route.loc[r.image_id].split for _,r in frame.iterrows());raw=torch.load(folder/'best_raw_cache.pt',map_location='cpu',weights_only=False);assert raw['base_checkpoint_sha256']==summary['checkpoint_sha256'] and raw['grid']==[24,48];records={int(r['image_id']):r for r in raw['records']};assert len(records)==len(raw['records'])==505 and set(records)==set(route.index)
    for image_id,r in records.items():
        assert r['patient_key']==route.loc[image_id].patient_key and r['split']==route.loc[image_id].split;assert r['raw_logp'].shape==(16,1152) and torch.isfinite(r['raw_logp']).all() and r['raw_logp'].logsumexp(-1).abs().max()<=1e-5;reference=torch.load(ROOT/f'runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache/{image_id}.pt',map_location='cpu',weights_only=False);assert torch.equal(r['coords'],reference['coords']) and torch.equal(r['loss_mask'],reference['mask']);target=(r['coords'][:,1]*48).long().clamp(0,47)*24+(r['coords'][:,0]*24).long().clamp(0,23);assert torch.equal(target,r['target'])
    recomputed={}
    for split,count in [('train',412),('val',93)]:
        subset=frame[frame.split==split];assert len(subset)==count;recomputed[split]={}
        for key in ['candidate_nll','current_nll','weight_0.25','weight_0.5','weight_0.75']:
            value=float(subset[key].mean());assert abs(value-summary['splits'][split][key])<1e-10;recomputed[split][key]=value
    proofs=json.loads((folder/'actual93_parity.json').read_text())['records'];val_ids=set(route[route.split=='val'].index);assert len(proofs)==93 and len({r['image_id'] for r in proofs})==93 and {r['image_id'] for r in proofs}==val_ids;maximum=max(r['max_logp_error'] for r in proofs);assert maximum<=1e-4 and maximum==summary['actual93_max_logp_error'];actual=frame[frame.split=='val'].actual_candidate_nll;assert actual.notna().all() and abs(float(actual.mean())-summary['actual93_nll'])<1e-10
    val=frame[frame.split=='val'].copy();val['delta']=val['weight_0.25']-val.current_nll;groups=val.groupby('patient_key').delta.agg(['sum','count']);assert len(groups)==77;draw=np.random.default_rng(42).multinomial(77,np.full(77,1/77),size=10000);sample=draw@groups['sum'].to_numpy()/(draw@groups['count'].to_numpy());ci=np.quantile(sample,[.025,.975]);bootstrap=dict(delta_candidate_weight025_minus_current=float(val.delta.mean()),ci95=ci.tolist(),patients=77,images=93,resamples=10000,seed=42,aggregation='Pairedpatientclusterswithreplacement; allimageswithmultiplicity; image-macro weighting',claim='Exploratorymanypriorselections; intervalunadjusted; noupdatedunifiedROIclaim');result=dict(status='passed',exporter_terminal_confirmed_by_root=True,source_and_current_all_nested_config_SHA_verified=True,compact_selected_and_frozen_SHA_verified=True,raw_records=505,train=412,val=93,actual93_candidate_nll=float(actual.mean()),actual93_max_logp_error=maximum,tolerance=1e-4,fixed3_image_macro=recomputed,paired_patient_bootstrap=bootstrap,ROI_integration_verified=False,goal_met=False);output=folder/'independent_terminal_review.json';assert not output.exists();output.write_text(json.dumps(result,indent=2));(folder/'paired_patient_bootstrap.json').write_text(json.dumps(bootstrap,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
