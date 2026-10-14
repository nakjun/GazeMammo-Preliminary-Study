"""Audit final HPO refits and reproduce one image-only prediction per backbone."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import subprocess
import re
import torch
import pandas as pd
from src.data import ROOT
from src.fusion_backbones import SPECS
from src.spatial import JointGlobalLocal
from src.data import whole_tensor,patch_tensor
from src.grid_selector import grid_points
from PIL import Image
import numpy as np

if __name__=='__main__':
    torch.set_num_threads(4);base=ROOT/'runs/fusion_hpo_seed42'
    frame=pd.read_csv(ROOT/'manifests/classification.csv');train=set(frame[frame.split=='train'].patient_key);hold=set(frame[frame.split!='train'].patient_key)
    example=frame[frame.split=='val'].sort_values('image_id').iloc[0];report={}
    samples=frame[frame.split=='val'].sort_values('image_id').iloc[:2];images=[];patches=[]
    for row in samples.itertuples():
        with Image.open(ROOT/'data/image'/row.file_name) as raw:image=raw.convert('L')
        images.append(whole_tensor(image));patches.append(torch.stack([patch_tensor(image,p) for p in grid_points(image)*image.size]))
    images=torch.stack(images).to('cuda:1');patches=torch.stack(patches).to('cuda:1');mask=torch.ones(2,16,dtype=torch.bool,device='cuda:1')
    for name,(_,weight) in SPECS.items():
        selected=json.loads((base/name/'selected_final.json').read_text());path=base/name/selected['refit_dir']
        assert selected['epoch']>selected['config']['warmup']
        c=torch.load(path/'best.pt',map_location='cpu',weights_only=False);assert set(c['training_patient_keys'])==train and not train&hold
        original=torch.load(Path(torch.hub.get_dir())/'checkpoints'/weight.url.rsplit('/',1)[-1],map_location='cpu',weights_only=True)
        if name=='densenet161':
            pattern=re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            original={pattern.sub(r'\1\2',key):value for key,value in original.items()}
        key='features.conv0.weight' if name=='densenet161' else 'conv_proj.weight' if name=='vit_b_16' else 'features.0.0.weight'
        delta=float((c['state']['visual.'+key]-original[key]).norm());assert delta>0
        stats=[float((value-original[k[len('visual.'):]]).abs().max()) for k,value in c['state'].items() if k.startswith('visual.') and k.endswith(('running_mean','running_var'))]
        assert all(x==0 for x in stats)
        output=ROOT/'results'/f'hpo_{name}_single_image.json'
        subprocess.run([sys.executable,str(ROOT/'scripts/predict_grid_fusion.py'),'--checkpoint',str(path),'--image',str(ROOT/'data/image'/example.file_name),'--output',str(output),'--device','cuda:1'],check=True)
        prediction=json.loads(output.read_text());preds=pd.read_csv(path/'fusion_logits_predictions.csv');expected=float(preds.loc[preds.image_id==example.image_id,'probability'].iloc[0])
        diff=abs(prediction['probabilities']['fusion_logits']-expected);assert diff<1e-3
        model=JointGlobalLocal(pretrained=False,**c['model_config']).to('cuda:1').eval();model.load_state_dict(c['state'])
        with torch.inference_mode():batch=model(images,patches,mask)['fusion_logits'].sigmoid().cpu().numpy()
        reference=preds.set_index('image_id').loc[samples.image_id,'probability'].to_numpy()
        np.testing.assert_allclose(batch,reference,atol=1e-6,rtol=1e-5)
        precise_difference=None
        if diff>1e-4:
            original_tf32=torch.backends.cudnn.allow_tf32;torch.backends.cudnn.allow_tf32=False
            with torch.inference_mode():
                full=model(images,patches,mask)['fusion_logits'].sigmoid()[0].item()
                single=model(images[:1],patches[:1],mask[:1])['fusion_logits'].sigmoid().item()
            torch.backends.cudnn.allow_tf32=original_tf32
            precise_difference=abs(full-single);assert precise_difference<1e-5
        del model;torch.cuda.empty_cache()
        report[name]=dict(selected_epoch=c['epoch'],warmup=selected['config']['warmup'],first_conv_weight_change_l2=delta,
            bn_buffers_compared=len(stats),bn_max_change=max(stats,default=0),training_patients=len(train),heldout_overlap=False,singleton_prediction_absolute_difference=diff,
            same_batch_predictions_match_csv=True,fp32_without_tf32_singleton_difference=precise_difference)
        print(name,report[name],flush=True)
    (base/'verification.json').write_text(json.dumps(dict(unit_tests_passed=33,models=report,test_used=False),indent=2)+'\n')
