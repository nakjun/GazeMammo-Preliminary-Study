"""Three prespecified normalized arithmetic clinical mixtures, strict canonical alignment."""
import argparse,json,sys,time,math
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from export_deepgaze_clinical_raw_1011 import alignment
from train_attention_1011 import digest

def verify_parity(payload):
    proof=payload.get('actual_val_parity',{});records=proof.get('records',[]);manifest=pd.read_csv(ROOT/'manifests/trajectory.csv',usecols=['image_id','split']);allowed=set(manifest[manifest.split=='val'].image_id.astype(int))
    if proof.get('requested') is not True or proof.get('images')!=93 or len(records)!=93 or {r['image_id'] for r in records}!=allowed or max(r['max_logp_error'] for r in records)>1e-4:raise ValueError('Clinical all93 actual-image parity not proven')

def mix(log_a,log_b,weight):
    q=torch.logaddexp(log_a+math.log(1-weight),log_b+math.log(weight));return q-q.logsumexp(-1,keepdim=True)

def run(a):
    torch.set_num_threads(4);paths=[ROOT/'runs/dual_target_1011'/x for x in ['human_control_raw_cache_for_ensemble.pt','human_improved_raw_cache.pt']]+[Path(a.clinical)];payloads=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];verify_parity(payloads[2]);sources=[{int(r['image_id']):r for r in p['records']} for p in payloads]
    for p in payloads:
        if digest(p['base_checkpoint'])!=p['base_checkpoint_sha256'] or p['grid']!=[24,48]:raise ValueError('Checkpoint dependency changed')
    if any(set(s)!=set(sources[0]) or len(s)!=505 for s in sources):raise ValueError('Exact505 matchedsources required')
    result=[];rows=[]
    for image_id,r in sources[0].items():
        alignment(r,sources[1][image_id]);alignment(r,sources[2][image_id]);logps=[s[image_id]['raw_logp'].float() for s in sources]
        if any(q.shape!=(16,1152) or not torch.isfinite(q).all() or not torch.allclose(q.logsumexp(-1),torch.zeros(16),atol=1e-5) for q in logps):raise ValueError('Malformed normalized source')
        baseline=torch.logaddexp(logps[0]+math.log(.75),logps[1]+math.log(.25));baseline-=baseline.logsumexp(-1,keepdim=True)
        for label,q,weight in [('baseline',baseline,0.),('clinical',logps[2],1.)]+[(f'clinical_{weight:.2f}',mix(baseline,logps[2],weight),weight) for weight in [.25,.5,.75]]:
            nll=float((-q.gather(-1,r['target'][:,None]).squeeze(-1))[r['loss_mask'].bool()].mean());rows.append(dict(policy=label,clinical_weight=weight,image_id=image_id,patient_key=r['patient_key'],split=r['split'],nll=nll))
    table=pd.DataFrame(rows)
    for label,group in table.groupby('policy',sort=False):result.append(dict(policy=label,clinical_weight=float(group.clinical_weight.iloc[0]),train_images=int((group.split=='train').sum()),val_images=int((group.split=='val').sum()),train_nll=float(group[group.split=='train'].nll.mean()),val_nll=float(group[group.split=='val'].nll.mean())))
    if any(r['train_images']!=412 or r['val_images']!=93 for r in result):raise ValueError('Wrong splitdenominators')
    baseline=next(r for r in result if r['policy']=='baseline');best=min(result,key=lambda r:r['val_nll']);output=Path(a.output);output.mkdir(parents=True,exist_ok=False);configs=[]
    for row in result:
        if row['clinical_weight'] in [.25,.5,.75]:configs.append(dict(policy=row['policy'],weights=[.75*(1-row['clinical_weight']),.25*(1-row['clinical_weight']),row['clinical_weight']],sources=[dict(raw_cache=str(p.resolve()),raw_cache_sha256=digest(p),checkpoint=v['base_checkpoint'],checkpoint_sha256=v['base_checkpoint_sha256']) for p,v in zip(paths,payloads)],temperature=1.,geometry_coefficient=0.,grid=[24,48],normalization='Arithmeticprobabilitysum thenrenormalize',clinical_runtime='t0actualfrozenarithBOS;t>=1DGIIItruthfulpast',selection='Threefixedweights exploratoryval93',val_nll=row['val_nll']))
    metrics=dict(results=result,best=best,baseline=baseline,goal_met=best['val_nll']<=5.3,parity=payloads[2]['actual_val_parity'],source_hashes={str(p):digest(p) for p in paths},runtime_configs=configs);(output/'metrics.json').write_text(json.dumps(metrics,indent=2));table.to_csv(output/'per_image.csv',index=False);(output/'runtime_configs.json').write_text(json.dumps(configs,indent=2));lines=['# Fixed clinical arithmetic mixtures','','All505 alignedcanonicaltrain412/val93, normalized rawprobabilities andimage-macroNLL. Only clinicalweight.25/.5/.75 tested; no temperature/geometry/sweep. Clinical all93 actual/native updated-headparity verified≤1e−4. Exploratoryvalidation, noindependenttest claim.','','| Clinical weight | Train NLL | Val NLL |','|---:|---:|---:|'];lines += [f"| {r['clinical_weight']:.2f} | {r['train_nll']:.6f} | {r['val_nll']:.6f} |" for r in result];lines += ['',f"Best {best['policy']}: {best['val_nll']:.6f}; baseline {baseline['val_nll']:.6f}. Target5.3 {'met' if metrics['goal_met'] else 'unmet'}. Sourcehashes and explicit3parent normalizedarithmetic runtimeconfigs saved; anypromotion stillrequires actualcombinedpipeline proof."];(output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--clinical',default='runs/dual_target_1011/deepgaze_clinical_best_raw_cache.pt');p.add_argument('--output',default='runs/dual_target_1011/deepgaze_clinical_fixed_mixtures');run(p.parse_args())
