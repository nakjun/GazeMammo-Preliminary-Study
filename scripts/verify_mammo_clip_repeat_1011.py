"""Same actual inputs/strides and five identical GPU repeats; no tolerance change."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.mammo_clip_dual_output_1011 import load_mammo_clip_dual
from src.dual_output_1011 import config_sha256
from mammo_clip_dual_output_1011 import teacher_vectors
from gaze_nll_adaptation_1011 import routing

def tensor_sha(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;out=ROOT/'runs/dual_target_1011/mammo_clip_unified_runtime';payload=json.loads((out/'config.json').read_text());model,_,_=load_mammo_clip_dual(payload['config'],'cuda:0');raw=torch.load(payload['config']['reference_caches'][0]['raw_cache'],weights_only=False)['records'];refs={int(r['image_id']):r for r in raw};routes={int(r['image_id']):r for r in routing()};proof=[]
with torch.no_grad():
 for identity in [64,73]:
  coords=refs[identity]['coords'][None].cuda();mask=refs[identity]['loss_mask']
  with Image.open(ROOT/'data/image'/routes[identity]['file_name']) as source:image=source.convert('L');memory=model.maps(image);vectors=teacher_vectors(model,image,coords,mask,'cuda:0')
  repeated=[model.teacher_logits(memory,coords,vectors) for _ in range(5)];repeat=max(float((a-b).abs().max()) for a in repeated for b in repeated);changed=coords.clone();changed[:,8:]=1-changed[:,8:];cv=vectors.clone();cv[:,9:]=torch.randn_like(cv[:,9:]);future=model.teacher_logits(memory,changed,cv);causal=max(float((r[:,:9]-future[:,:9]).abs().max()) for r in repeated)
  same_coords=torch.equal(coords[:,:8],changed[:,:8]);same_vectors=torch.equal(vectors[:,:9],cv[:,:9]);same_strides=coords[:,:8].stride()==changed[:,:8].stride() and vectors[:,:9].stride()==cv[:,:9].stride()
  if not(same_coords and same_vectors and same_strides) or max(repeat,causal)>1e-5:raise ValueError('Sameinput/repeat source causal proof failed')
  proof.append(dict(image_id=identity,identical_input_repeats=5,max_pairwise_repeated_logp_error=repeat,max_future_vs_sameinput_logp_error=causal,exact_observed_prefix_coordinates_equal=same_coords,exact_past_vectors_through_step8_equal=same_vectors,strides_equal=same_strides,coordinate_prefix_sha256=tensor_sha(coords[:,:8]),changed_coordinate_prefix_sha256=tensor_sha(changed[:,:8]),past_vector_prefix_sha256=tensor_sha(vectors[:,:9]),changed_past_vector_prefix_sha256=tensor_sha(cv[:,:9]),coordinate_stride=list(coords[:,:8].stride()),vector_stride=list(vectors[:,:9].stride())))
report=dict(combined_config_sha256=payload['sha256'],records=proof,tolerance=1e-5,tolerance_unchanged=True,source_inputs='actualsameimage/trueobservedprefix/actualpastvectors; no modifiedcurrenttargetinput');(out/'same_input_repeat_proof.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
