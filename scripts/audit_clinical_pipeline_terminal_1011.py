"""Independent read-only terminal audit of saved actual clinical dual pipeline."""
import json,hashlib,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.dual_output_1011 import config_sha256

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    folder=ROOT/'runs/dual_target_1011/clinical_unified_runtime';payload=json.loads((folder/'config.json').read_text());config=payload['config'];tag=config_sha256(config);assert tag==payload['sha256'];nested=config['baseline_config'];assert nested['sha256']==config_sha256(nested['config']);state=json.loads((folder/'actual_state.json').read_text());assert state['status']=='complete' and state['combined_config_sha256']==tag and all(s['status']=='complete' and s['returncode']==0 for s in state['stages']);checks={}
    for recipe in [config,nested['config']]:
        for path,expected in recipe['source_code_sha256'].items():assert sha(ROOT/path)==expected;checks[path]=expected
    dependencies=config['probability']['parents']+[config['roi']['guided'],config['roi']['offset']]
    for d in dependencies:
        assert sha(d['path'])==d['sha256'];checks[d['path']]=d['sha256']
        if 'raw_cache' in d:assert sha(d['raw_cache'])==d['raw_cache_sha256'];checks[d['raw_cache']]=d['raw_cache_sha256']
    frame=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','patient_key','split']);val=frame[frame.split=='val'];ids=set(val.image_id.astype(int));assert len(ids)==93;patients=dict(zip(val.image_id.astype(int),val.patient_key));nll=pd.read_csv(folder/'actual_nll/per_image_nll.csv');assert len(nll)==93 and nll.image_id.is_unique and set(nll.image_id.astype(int))==ids and (nll.cached_reference_max_logp_error<=1e-4).all();assert all(r.patient_key==patients[int(r.image_id)] for _,r in nll.iterrows());accept=json.loads((folder/'all93_parity_acceptance.json').read_text());assert accept['status']=='passed' and accept['images']==93 and accept['tolerance']==1e-4 and abs(nll.cached_reference_max_logp_error.max()-accept['maximum_logp_error'])<1e-12
    keys=set();prefix=0
    with (folder/'actual_generation/traces.jsonl').open() as stream:
        for line in stream:
            trace=json.loads(line);key=(trace['image_id'],trace['seed']);assert key not in keys and key[0] in ids and key[1] in [42,43,44];keys.add(key);assert trace['combined_config_sha256']==tag and trace['crop_size']==224 and len(trace['coordinates'])==16 and trace['grid']==[24,48]
            for flag in ['annotation_used','annotations_used','gaze_used','observed_gaze_used','labels_used']:assert trace[flag] is False
            previous=json.loads((ROOT/f"runs/dual_target_1011/center_offset_K14_confirmation/O_spatial_p1_s0p5_K14_nms0p15_image{key[0]}_seed{key[1]}.json").read_text());assert trace['coordinates'][:14]==previous['coordinates'][:14] and trace['source_anchor_indices'][:14]==previous['source_anchor_indices'][:14];prefix+=1
    assert keys=={(i,s) for i in ids for s in [42,43,44]};metrics=json.loads((folder/'actual_posthoc/metrics.json').read_text());summary=metrics['summaries']['DUAL_ROI14_RAW_CLINICAL025'];capture=summary['actual_recall_05_step16'];actual_nll=float(nll.nll.mean());old_nll=5.787841889;assert abs(actual_nll-state['outcome']['canonical_actual_image_macro_nll'])<1e-12 and capture==state['outcome']['physical_capture50'];result=dict(kind='independent_dual_target_status_1011',pipeline='clinical025plusoriginalROI14',complete=True,config_sha256=tag,canonical_fixed_val93_nll=actual_nll,physical_box50_image_macro_capture=capture,normalized_possible_coverage_recall50_separate=summary['recall_05_step16'],targets=dict(nll_max=5.3,physical_capture_min=.85,nll_met=actual_nll<=5.3,physical_capture_met=capture>=.85,both_met=actual_nll<=5.3 and capture>=.85),previous_pipeline_nll=old_nll,nll_change=actual_nll-old_nll,denominators=dict(images=93,patient_clusters=77,generation_seeds=[42,43,44],paths=len(keys),exact_original_ROI14_prefixes=prefix),actual_parity=dict(images=93,max_logp_error=float(nll.cached_reference_max_logp_error.max()),tolerance=1e-4,passed=True),compute=dict(current_whole_encoders=4,current_previous_crop_encoders=2,previous_whole_encoders=3,previous_previous_crop_encoders=2,DG_whole_input=[512,1024],baseline_ROI_input=[256,512],physical_patch=224,steps=16),source_dependency_sha256=checks,claims='Exploratoryvalidation; physicalcaptureisactualboxfraction>=.5, notnormalizedcoverage; NLLoncanonicalobservedprefix, notforcedROIpath; noindependenttestclaim; originalsourcehashesverifiedunchanged')
    output=folder/'independent_dual_target_status.json';assert not output.exists();output.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k in ['targets','canonical_fixed_val93_nll','physical_box50_image_macro_capture','nll_change','denominators','actual_parity']}))
if __name__=='__main__':main()
