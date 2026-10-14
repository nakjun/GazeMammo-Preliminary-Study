import argparse,json,sys
from pathlib import Path
import torch,numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from export_deepgaze_clinical_raw_1011 import predict
from train_deepgaze_clinical_1011 import load_clinical,load_native_cache,verified_centerbias,group_data
from gaze_nll_adaptation_1011 import cache_records
from src.deepgaze_context_1011 import native_features
from src.proposal_utility_1011 import state_sha256

a=argparse.Namespace(**json.load(open('runs/dual_target_1011/deepgaze_clinical_seed42/config.json')));a.device='cuda:0';a.smoke_limit=0;torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
model,ck=load_clinical('runs/dual_target_1011/deepgaze_clinical_seed42/best.pt',a.device);records,_=cache_records(a);native,header=load_native_cache(a,records,state_sha256(model.features.state_dict()));bias=verified_centerbias(a,records);size=ck['input_size'];val=[r for r in records if r['split']=='val'];group=val[:4]
with torch.no_grad():
 f,c,m,b,cb=group_data(group,native,bias,size,a.device);q4=predict(model,f,c,b,cb,size)[0];one=[x[:1] for x in f];q1=predict(model,one,c[:1],b[:1],cb[:1],size)[0];repeat=predict(model,one,c[:1],b[:1],cb[:1],size)[0]
 with Image.open(ROOT/'data/image'/group[0]['file_name']) as image:rgb=torch.from_numpy(np.asarray(image.convert('L').resize(tuple(size),Image.Resampling.BILINEAR)).copy()).float()[None,None].repeat(1,3,1,1).to(a.device)
 fresh=native_features(model,rgb);qfresh=predict(model,fresh,c[:1],b[:1],cb[:1],size)[0];qcontig=predict(model,[x.contiguous() for x in fresh],c[:1],b[:1],cb[:1],size)[0]
 with torch.backends.cudnn.flags(enabled=True,allow_tf32=False):
  qieee4=predict(model,f,c,b,cb,size)[0];qieee1=predict(model,one,c[:1],b[:1],cb[:1],size)[0];qieeefresh=predict(model,[x.contiguous() for x in fresh],c[:1],b[:1],cb[:1],size)[0]
result=dict(image_id=int(group[0]['image_id']),feature_max=[float((x-y).abs().max()) for x,y in zip(one,fresh)],cached_stride=[list(x.stride()) for x in one],fresh_stride=[list(x.stride()) for x in fresh],batch4_vs1=float((q4-q1).abs().max()),same_input_repeat=float((q1-repeat).abs().max()),cached1_vsfresh=float((q1-qfresh).abs().max()),cached1_vscontiguousfresh=float((q1-qcontig).abs().max()),ieee_batch4_vs1=float((qieee4-qieee1).abs().max()),ieee_cached1_vscontiguousfresh=float((qieee1-qieeefresh).abs().max()),feature_sha=state_sha256(model.features.state_dict()),expected_feature_sha=header['feature_state_sha256']);print(json.dumps(result,indent=2));Path('runs/dual_target_1011/deepgaze_clinical_parity_diagnostic.json').write_text(json.dumps(result,indent=2))
