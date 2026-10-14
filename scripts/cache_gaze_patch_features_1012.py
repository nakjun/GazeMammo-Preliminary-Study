"""Create a new pixel-only development feature cache; --smoke selects metadata-first8."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.gaze_patch_features_1012 import (ARMS,DEFAULT_CONFIG,SOURCE_SHA256,_dependencies,load_frozen_source,validate_development_population,extract_image_features,validate_feature_cache)
from src.proposal_utility_1011 import file_sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default=str(DEFAULT_CONFIG));p.add_argument('--manifest',default=str(ROOT/'manifests/all.csv'));p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=42);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(a.threads)
    payload=json.loads(Path(a.config).read_text());frame=validate_development_population(pd.read_csv(a.manifest),payload)
    if a.smoke:
        frame=pd.concat([frame[(frame.split==split)&(frame.label==label)].head(2) for split in ('train','val') for label in (0,1)],ignore_index=True)
        if len(frame)!=8: raise ValueError('Smoke8 population unavailable')
    patient_audit=payload['source_patient_audit'];bundle=load_frozen_source(a.config,a.device);payload=bundle[-1];rows=[];values={'global_features':[]}
    for arm in ARMS:
        for suffix in ('features','coordinates','rectangles','mask'): values[f'{arm}_{suffix}']=[]
    for row in frame.to_dict('records'):
        path=(ROOT/'data/image'/row['file_name']).resolve()
        with Image.open(path) as source:
            image=source.convert('L');features=extract_image_features(bundle,image,int(row['image_id']),a.seed);size=list(image.size)
        rows.append(dict(image_id=int(row['image_id']),patient_key=row['patient_key'],split=row['split'],label=int(row['label']),file_name=row['file_name'],image_path=str(path),image_sha256=file_sha256(path),image_size=size,timing=features['timing'],selection_metadata=features['selection_metadata']))
        values['global_features'].append(features['global_features'])
        for arm in ARMS:
            for suffix in ('features','coordinates','rectangles','mask'): values[f'{arm}_{suffix}'].append(features[arm][suffix])
        print(json.dumps({'completed':len(rows),'total':len(frame),'image_id':int(row['image_id'])}),flush=True)
    code={str(path.relative_to(ROOT)):file_sha256(path) for path in (ROOT/'src/gaze_patch_features_1012.py',Path(__file__).resolve())}
    def source_codes(v):
        if isinstance(v,dict):
            code.update(v.get('source_code_sha256',{}))
            for x in v.values():source_codes(x)
        elif isinstance(v,list):
            for x in v:source_codes(x)
    source_codes(payload)
    provenance=dict(source_config_sha256=SOURCE_SHA256,source_config_path=payload['source_config_path'],source_config_file_sha256=payload['source_config_file_sha256'],classification_encoder_sha256=payload['classification_encoder_sha256'],dependencies=_dependencies(payload),source_code_sha256=code,manifest_path=str(Path(a.manifest).resolve()),manifest_sha256=file_sha256(a.manifest),population='smoke8' if a.smoke else 'development469',path_seed=a.seed,crop_size=224,feature_dim=1024,arms=list(ARMS),classification_encoder='classification_encoder(probability): unwrap visual.parent -> original PatchBundle.encoders[0]',global_input_size=[256,512],pixel_only=True,frozen_transfer=True,source_compute=payload['config']['compute'],source_patient_audit=patient_audit,numerics=dict(matmul_tf32=False,cudnn_enabled=True,classification_visual_tf32=True,guided_visual_tf32=True))
    cache=dict(kind='gaze_patch_classification_features_1012',rows=rows,tensors={k:torch.stack(v) for k,v in values.items()},provenance=provenance)
    validate_feature_cache(cache);torch.save(cache,output/'features.pt');(output/'provenance.json').write_text(json.dumps(provenance,indent=2));(output/'status.json').write_text(json.dumps({'state':'complete','images':len(rows)}))

if __name__=='__main__': main()
