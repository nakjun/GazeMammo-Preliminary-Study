"""Read-only terminal audit of frozen final TEST evaluator artifacts/dependencies."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from src.proposal_utility_1011 import file_sha256
base=ROOT/'runs/frozen_current_test_1012';protocol=json.loads((base/'protocol.json').read_text());payload=json.loads(Path(protocol['model_config']).read_text());sources={}
def collect(obj):
 if isinstance(obj,dict):
  if 'source_code_sha256' in obj:
   for name,sha in obj['source_code_sha256'].items():
    path=Path(name);path=path if path.is_absolute() else ROOT/path
    if str(path) in sources and sources[str(path)]!=sha:raise ValueError('Conflicting pinned source')
    sources[str(path)]=sha
  for value in obj.values():collect(value)
 elif isinstance(obj,list):
  for value in obj:collect(value)
collect(payload)
for dep in protocol['source_patient_audit']:
 if file_sha256(dep['path'])!=dep['sha256']:raise ValueError('Checkpoint changed')
 ck=torch.load(dep['path'],map_location='cpu',weights_only=False);collect(ck.get('config',{}))
for path,sha in sources.items():
 if file_sha256(path)!=sha:raise ValueError(('Pinned source changed',path))
if file_sha256(ROOT/'scripts/evaluate_frozen_current_test_1012.py')!=protocol['evaluator_sha256'] or file_sha256(protocol['model_config'])!=protocol['model_config_file_sha256']:raise ValueError('Frozen evaluator/config changed')
for path,key in [('manifests/all.csv','all_manifest_sha256'),('manifests/trajectory.csv','trajectory_manifest_sha256'),('data/gaze_data/merged_gaze_data.csv','gaze_source_sha256')]:
 if file_sha256(ROOT/path)!=protocol[key]:raise ValueError('Routing/gaze source changed')
for phase in ['nll','generation']:
 if json.loads((base/phase/'state.json').read_text())['state']!='complete':raise ValueError('Evaluation incomplete')
cache=torch.load(base/'nll/raw_probabilities.pt',map_location='cpu',weights_only=False);records=cache['records'];ids=[r['image_id'] for r in records];excluded=cache['excluded'];expected=set(protocol['test_ids'])
if len(ids)!=len(set(ids)) or set(ids)|{r['image_id'] for r in excluded}!=expected or set(ids)&{r['image_id'] for r in excluded}:raise ValueError('TEST gaze cohort mismatch')
with (ROOT/'manifests/all.csv').open() as f:route={int(r['image_id']):r for r in csv.DictReader(f) if r['split']=='test' and r['source']=='CBIS-DDSM'}
maxnorm=0
for r in records:
 q=r['raw_logp'];mask=r['loss_mask'];coords=r['coords'];target=(coords[:,1]*48).long().clamp(0,47)*24+(coords[:,0]*24).long().clamp(0,23)
 if r['patient_key']!=route[r['image_id']]['patient_key'] or r['file_name']!=route[r['image_id']]['file_name'] or file_sha256(ROOT/'data/image'/r['file_name'])!=r['image_sha256']:raise ValueError('TEST identity/image changed')
 if r['split']!='test' or q.shape!=(16,1152) or mask.shape!=(16,) or not mask.any() or not torch.equal(target,r['target']) or not torch.isfinite(q).all() or r['patient_key'] not in protocol['test_patient_keys']:raise ValueError('TEST teacher tensor/identity mismatch')
 maxnorm=max(maxnorm,float(q.logsumexp(-1).abs().max()))
 if abs(float(-q[mask,target[mask]].mean())-r['nll'])>1e-6:raise ValueError('NLL record mismatch')
if maxnorm>1e-5:raise ValueError('Probability normalization')
seen=set();paths=0;smoke=json.loads((base/'engineering_smoke/traces.jsonl').read_text().splitlines()[0]);repeat=False
with (base/'generation/traces.jsonl').open() as f:
 for line in f:
  r=json.loads(line);key=(r['image_id'],r['seed'])
  if key in seen or r['image_id'] not in expected or r['seed'] not in [42,43,44] or r['combined_config_sha256']!=protocol['model_config_sha256'] or r['evaluation_split']!='test' or any(r[k] for k in ['annotations_used','labels_used','observed_gaze_used','gaze_used']) or len(r['coordinates'])!=16 or r['crop_size']!=224:raise ValueError('TEST image-only path provenance/cohort mismatch')
  if r['patient_key']!=route[r['image_id']]['patient_key'] or Path(r['image'])!=(ROOT/'data/image'/route[r['image_id']]['file_name']).resolve():raise ValueError('TEST generation image/patient mismatch')
  if key==(smoke['image_id'],smoke['seed']):
   repeat=r['coordinates']==smoke['coordinates'] and r['source_anchor_indices']==smoke['source_anchor_indices']
   if not repeat:raise ValueError('Engineering/full repeat mismatch')
  seen.add(key);paths+=1
if seen!={(iid,s) for iid in expected for s in [42,43,44]}:raise ValueError('TEST generation incomplete')
report=dict(state='passed',config_sha256=protocol['model_config_sha256'],checkpoints_unchanged=len(protocol['source_patient_audit']),pinned_source_files_unchanged=len(sources),evaluator_config_manifests_gaze_source_unchanged=True,test_patients=len(protocol['test_patient_keys']),test_images=150,gaze_images=len(records),gaze_excluded=excluded,normalized_q_max_error=maxnorm,image_macro_nll=sum(r['nll'] for r in records)/len(records),image_only_paths=paths,engineering_full_repeat_exact=repeat,seeds=[42,43,44],annotated_images=131,unannotated_images=19,INbreast_payloads_used=False,training_or_tuning=False,goal_remains_paused=True,physical_posthoc='Independent strict boxes/coordinate validation delegated to test_metrics_1012.py',artifact_sha256={'raw_probabilities.pt':file_sha256(base/'nll/raw_probabilities.pt'),'traces.jsonl':file_sha256(base/'generation/traces.jsonl')})
(base/'frozen_source_cohort_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
