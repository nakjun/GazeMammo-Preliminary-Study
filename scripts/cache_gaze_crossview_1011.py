"""New whole-only frozen prefix before ConvNeXt stage5 block24, exact505."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from PIL import Image
from src.data import whole_tensor
from src.gaze_crossview_1011 import CrossViewGaze
from src.proposal_utility_1011 import file_sha256,state_sha256
from gaze_nll_adaptation_1011 import load_base,cache_records

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--cache',default='runs/gaze_nll_adaptation_trials_1011_v2/control_visual_cache');p.add_argument('--base',default='runs/full_1011_seed42/control/best.pt');a=p.parse_args();torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;guard=json.loads((ROOT/'runs/dual_target_1011/gaze_crossview_guard_v3/proof.json').read_text());causal_path=ROOT/'runs/dual_target_1011/gaze_crossview_guard_v3/causal_past_proof.json';causal=json.loads(causal_path.read_text());assert causal['state']=='passed';records,provenance=cache_records(a);base,_=load_base(a.base);model=CrossViewGaze(base).cuda(1).eval();frozen=state_sha256(model.frozen_state())
 if guard['state']!='passed' or frozen!=guard['frozen_state_sha256'] or guard['source_checkpoint_sha256']!=file_sha256(a.base) or len(records)!=505:raise ValueError('Actualguard/exact505 prerequisite')
 out=Path(a.output);out.mkdir(exist_ok=False,parents=True);entries=[];start=time.monotonic();header=dict(kind='gaze_crossview_whole_prefix_1011',causal_past_proof_sha256=file_sha256(causal_path),base_checkpoint=str(Path(a.base).resolve()),base_checkpoint_sha256=file_sha256(a.base),frozen_state_sha256=frozen,source_cache_provenance_sha256=file_sha256(Path(a.cache)/'provenance.json'),source_code_sha256=file_sha256(ROOT/'src/gaze_crossview_1011.py'),cut='beforevisual.whole.features.5.24',input_size=[256,512],native_shape=[512,32,16],dtype='float32',visual_TF32=True,records=entries)
 with torch.no_grad():
  for r in records:
   image_id=int(r['image_id']);path=ROOT/'data/image'/r['file_name']
   if file_sha256(path)!=r['image_sha256']:raise ValueError('SourceimageSHA')
   with Image.open(path) as im:image=im.convert('L')
   f3,prefix=model.visual.prefix_features(whole_tensor(image,(256,512))[None].cuda(1));error=float((f3[0].cpu()-r['features'][0]).abs().max())
   if error>1e-4 or prefix.shape!=(1,512,32,16) or not torch.isfinite(prefix).all():raise ValueError('Nativeprefix/frozenfeature3 source mismatch')
   file=out/f'{image_id}.pt';torch.save(dict(image_id=image_id,patient_key=r['patient_key'],split=r['split'],image_sha256=r['image_sha256'],prefix5_23=prefix[0].cpu()),file);entries.append(dict(image_id=image_id,patient_key=r['patient_key'],split=r['split'],sha256=file_sha256(file),feature3_source_error=error));(out/'provenance.json').write_text(json.dumps(header,indent=2));(out/'state.json').write_text(json.dumps(dict(state='running',completed=len(entries))));print(json.dumps(dict(completed=len(entries),total=505)),flush=True)
 if state_sha256(model.frozen_state())!=frozen:raise ValueError('Frozenprefix/past drift')
 (out/'state.json').write_text(json.dumps(dict(state='complete',completed=505,seconds=time.monotonic()-start,bytes=sum(p.stat().st_size for p in out.glob('*.pt')),frozen_state_unchanged=True)))
if __name__=='__main__':main()
