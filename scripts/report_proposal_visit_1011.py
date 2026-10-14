"""Merge completed exploratory proposal experiments; never create a partial final report."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

SOURCES=('original','occupancy','utility')
PRIMARY=('recall_05_step16','recall_08_step16','actual_recall_05_step16','actual_recall_08_step16','actual_box_coverage','any_overlap','normalized_dtw','mean_foreground','mean_inbounds','unique_locations','mean_max_previous_crop_iou')


def validate_sampling(frame,source,expected_ids=None):
    required={'image_id','policy','seed','patient_key',*PRIMARY}
    if not required.issubset(frame):raise ValueError(f'{source}: missing columns {required-set(frame)}')
    if frame.duplicated(['image_id','policy','seed']).any():raise ValueError(f'{source}: duplicate image/policy/seed')
    policies=frame.policy.unique();expected_count=6 if source=='original' else 4
    if len(policies)!=expected_count:raise ValueError(f'{source}: expected {expected_count} policies')
    reference=None
    for policy,group in frame.groupby('policy'):
        if group.image_id.nunique()!=93 or set(group.seed)!={42,43,44} or len(group)!=279:raise ValueError(f'{source}/{policy}: incomplete val93 × seeds42/43/44')
        if not group.groupby('image_id').seed.apply(lambda v:set(v)=={42,43,44}).all():raise ValueError('Incomplete per-image seeds')
        ids=set(group.image_id)
        if reference is None:reference=ids
        if ids!=reference or (expected_ids is not None and ids!=expected_ids):raise ValueError('Mismatched image sets')
    return reference


def load_required(root):
    artifacts={};missing=[]
    for source in SOURCES:
        for name in ('per_image.csv','metrics.json'):
            path=root/f'{source}_sampling_posthoc'/name
            if not path.is_file():missing.append(str(path))
        for name in ('per_image.csv','summary.json'):
            path=root/f'{source}_teacher_nll'/name
            if not path.is_file():missing.append(str(path))
    if missing:raise FileNotFoundError('Final report blocked; required artifacts missing: '+', '.join(missing))
    manifest=pd.read_csv(Path(__file__).resolve().parents[1]/'manifests/trajectory.csv',usecols=['image_id','split'])
    ids=set(int(i) for i in manifest[manifest.split=='val'].image_id)
    if len(ids)!=93:raise ValueError('Fixed trajectory validation cohort is not93')
    for source in SOURCES:
        frame=pd.read_csv(root/f'{source}_sampling_posthoc/per_image.csv');ids=validate_sampling(frame,source,ids)
        metrics=json.loads((root/f'{source}_sampling_posthoc/metrics.json').read_text())
        nll=pd.read_csv(root/f'{source}_teacher_nll/per_image.csv')
        all_manifest=pd.read_csv(Path(__file__).resolve().parents[1]/'manifests/all.csv',usecols=['image_id','split'])
        allowed=set(all_manifest[all_manifest.split=='val'].image_id)
        excluded=set(nll.image_id)-ids
        if not excluded.issubset(allowed):raise ValueError(f'{source}: nonvalidation teacher NLL rows')
        nll=nll[nll.image_id.isin(ids)]
        if nll.duplicated(['image_id','policy']).any() or any(set(g.image_id)!=ids for _,g in nll.groupby('policy')):raise ValueError(f'{source}: incomplete/duplicate teacher likelihood')
        summary=json.loads((root/f'{source}_teacher_nll/summary.json').read_text())
        if not {'P0','raw'}.issubset(set(nll.policy)):raise ValueError(f'{source}: missing P0/raw teacher likelihood')
        summary['fixed93_routing']=dict(excluded_image_ids=sorted(int(i) for i in excluded),original_summary_images=summary.get('images'),analyzed_images=93)
        summary['fixed93_image_macro_nll']=nll.groupby('policy').nll.mean().to_dict()
        artifacts[source]=dict(frame=frame,metrics=metrics,nll=nll,teacher_summary=summary)
    return artifacts


def build(root,output):
    root=Path(root);output=Path(output);artifacts=load_required(root)
    rows=[];case=[];baseline_checks={};reference=artifacts['original']['frame'];p0=reference[reference.policy=='P0'].set_index(['image_id','seed']).sort_index()
    for source,data in artifacts.items():
        frame=data['frame'];base=frame[frame.policy=='P0'].set_index(['image_id','seed']).sort_index()
        differences={metric:float((base[metric]-p0[metric]).abs().max()) for metric in PRIMARY}
        verified=0;coordinate_mismatches=[]
        for image_id,seed in base.index:
            name=f'P0_image{image_id}_seed{seed}.json';first_path=root/'original_sampling'/name;other_path=root/f'{source}_sampling'/name
            if first_path.exists() and other_path.exists():
                first=json.loads(first_path.read_text());other=json.loads(other_path.read_text());verified+=1
                if first.get('coordinates')!=other.get('coordinates') or first.get('indices')!=other.get('indices'):coordinate_mismatches.append([int(image_id),int(seed)])
        baseline_checks[source]=dict(max_absolute_difference=differences,exact_metric_match=bool(all(v==0 for v in differences.values())),coordinate_index_pairs_checked=verified,coordinate_index_mismatches=coordinate_mismatches,exact_coordinate_index_match=bool(verified==279 and not coordinate_mismatches),interpretation='Exact saved P0 coordinate/index comparisons where tiny per-path files exist; paired study P0 remains the comparator.')
        for policy,group in frame.groupby('policy'):
            means=group.groupby('image_id')[list(PRIMARY)].mean().mean()
            row=dict(source=source,policy=policy,comparison=f'{source}/{policy}',images=93,generation_seeds=3,known_annotated_images=int(group[group.known_annotation==1].image_id.nunique()),**means.to_dict())
            likelihood=data['nll'];matched=likelihood[likelihood.policy==policy]
            row['teacher_nll']=float(matched.nll.mean()) if len(matched) else np.nan
            screen=data['metrics'].get('engineering_screen',{}).get(policy,{})
            row.update({k:screen.get(k) for k in ('recall50_delta','foreground_proxy_drop','dtw_delta','passes')})
            comparison=data['metrics'].get('paired_patient_bootstrap',{}).get(policy,{})
            for metric in ('recall_05_step16','mean_foreground','normalized_dtw'):
                result=comparison.get(metric)
                row[metric+'_delta_ci_low']=result['ci95'][0] if result else np.nan
                row[metric+'_delta_ci_high']=result['ci95'][1] if result else np.nan
            rows.append(row)
        selected=frame[frame.image_id==73][['image_id','policy','seed','recall_05_step16','actual_box_coverage']].copy();selected.insert(0,'source',source);case.append(selected)
    comparison=pd.DataFrame(rows);case=pd.concat(case,ignore_index=True)
    argmax={}
    for source in SOURCES:
        path=root/f'{source}_argmax_posthoc/per_image.csv'
        if path.exists():
            frame=pd.read_csv(path)
            if frame.duplicated(['image_id','policy','seed']).any() or any(g.image_id.nunique()!=93 or len(g)!=93 for _,g in frame.groupby('policy')):raise ValueError(f'{source}: incomplete argmax')
            argmax[source]=frame.groupby('policy')[list(PRIMARY)].mean().to_dict('index')
    original_teacher=artifacts['original']['teacher_summary']['fixed93_image_macro_nll']
    teacher_checks={name:dict(actual=float(original_teacher.get(name,np.nan)),reference=target,absolute_difference=float(abs(original_teacher.get(name,np.nan)-target))) for name,target in [('P0',5.907845),('raw',6.187455)]}
    output.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(output/'comparison.csv',index=False);case.to_csv(output/'case73_comparison.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');colors={'original':'#4267a5','occupancy':'#d98e36','utility':'#39836a'}
    for source,group in comparison.groupby('source'):
        axes[0].scatter(group.normalized_dtw,group.recall_05_step16,label=source,color=colors[source],s=35)
        for _,r in group.iterrows():axes[0].annotate(r.policy.replace('_occupancy_mse','').replace('_utility_mse',''),(r.normalized_dtw,r.recall_05_step16),fontsize=7,xytext=(3,3),textcoords='offset points')
    axes[0].set(xlabel='Normalized DTW (lower better)',ylabel='Image-macro normalized lesion recall50',ylim=(0,1));axes[0].legend(fontsize=8)
    x=np.arange(len(comparison));labels=[r.source[0].upper()+':'+r.policy.replace('_occupancy_mse','').replace('_utility_mse','') for _,r in comparison.iterrows()]
    axes[1].bar(x,comparison.mean_foreground,color=[colors[v] for v in comparison.source]);axes[1].set(ylabel='Estimated foreground crop fraction',ylim=(0,1));axes[1].set_xticks(x,labels,rotation=90,fontsize=7)
    axes[2].bar(x,comparison.teacher_nll,color=[colors[v] for v in comparison.source]);axes[2].set(ylabel='Observed-prefix teacher NLL (lower better)');axes[2].set_xticks(x,labels,rotation=90,fontsize=7)
    fig.savefig(output/'comparison.png');plt.close(fig)
    metadata=dict(status='complete',scope='All required sampling posthoc + teacher NLL artifacts validated; argmax optional',sources={s:dict(sampling_policies=data['frame'].policy.unique().tolist(),cohort_routing=data['metrics'].get('cohort_routing'),teacher_summary=data['teacher_summary']) for s,data in artifacts.items()},baseline_checks=baseline_checks,teacher_reference_checks=teacher_checks,argmax=argmax)
    (output/'comparison_metadata.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    lines=['# Proposal 후보와 실제 방문 비교','', '반복 검토한 validation93장의 탐색 실험이다. 독립 test와 INbreast는 평가하지 않았다. 영상별 생성 seed42/43/44를 먼저 평균하며, 환자 단위 paired bootstrap은 각 실험의 기존 평가 결과를 사용한다.','', 'original은 기존 occupancy/BCE proposal이다. occupancy는 새 head의 occupancy/MSE 대조이며 utility는 같은 초기화·MSE에서224px crop coverage target을 학습했다. 가능 coverage 기준 Recall50은 실제 crop coverage를 해당 병변에서 grid 후보가 달성할 수 있는 최대 coverage로 나눈 값이0.5이상인 비율이다. 실제 box50% recall은 원 박스 면적의 절반 이상이 한224px crop에 들어오는 비율이다. 큰 박스에서는 두 지표가 다르며 normalized recall을 실제 박스 절반 포착률로 해석하지 않는다. Original 대비 차이는 head·loss 변경을 함께 포함한다. 새 occupancy와 utility의 비교가 target 차이를 더 분리하지만 독립 임상 효능 검증은 아니다.','','| 실험/정책 | 가능 coverage 기준 Recall50 | 가능 coverage 기준 Recall80 | 실제 box50% recall | 실제 box80% recall | 실제 box coverage | Foreground proxy | DTW | Teacher NLL |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for _,r in comparison.iterrows():lines.append('| '+r.comparison+' | '+' | '.join('—' if pd.isna(r[k]) else f'{r[k]:.4f}' for k in ['recall_05_step16','recall_08_step16','actual_recall_05_step16','actual_recall_08_step16','actual_box_coverage','mean_foreground','normalized_dtw','teacher_nll'])+' |')
    original_rows=comparison[comparison.source=='original'].set_index('policy')
    if {'P0','P5'}.issubset(original_rows.index):
        first=original_rows.loc['P0'];budget=original_rows.loc['P5']
        lines+=['',f"기존 proposal의 서로 다른 top3 crop을 먼저 방문하고 나머지13개를 sampling하는 P5는 실제 box50% recall을 {first.actual_recall_05_step16:.1%}에서 {budget.actual_recall_05_step16:.1%}로 바꿨다. 가능 coverage 기준 Recall50은 {first.recall_05_step16:.1%}→{budget.recall_05_step16:.1%}이다. 이 결과는 proposal 후보를 방문하도록 경로를 강제한 효과이며 사람 gaze 예측 개선과 별도로 평가한다."]
    lines+=['','각 sampling 분석 군은 사전에 고정한93장×3seed를 완료했다. 일부 생성 단계에서 all.csv val102장이 처리되었으면 추가9장은 고정 trajectory val93과의 교집합으로 제외했으며 평가 성능에 따른 선택이 아니다. 제외 IDs와 원본 trace 수는 source별 metrics.json에 남겼다. 박스 없는 영상은 unknown이며 lesion recall 분모에서 제외한다. Foreground는 영상에서 추정한 proxy로 임상 조직 판정이 아니다. P5는 경로 강제 선택 heuristic으로 teacher NLL을 보고하지 않는다.','', '공학적 진입 기준은 해당 실험 P0 대비 recall50 +5pp 이상, foreground proxy 감소2pp 이하, DTW 증가0.005 이하이다. 기준은 탐색용이며 CI는 comparison.csv에 기록했다.']
    for _,r in comparison.iterrows():lines.append(f"- {r.comparison}: Δrecall50={r.recall50_delta:+.4f}, foreground drop={r.foreground_proxy_drop:+.4f}, ΔDTW={r.dtw_delta:+.4f}, 통과={r.passes}")
    lines+=['','Case73은 원인 발견 사례이며 독립 선택 기준이 아니다. 각 seed의 recall50와 실제 coverage는 case73_comparison.csv에 저장했다.','', '| 실험/정책 | Seed42 recall/coverage | Seed43 recall/coverage | Seed44 recall/coverage |','|---|---|---|---|']
    for (source,policy),group in case.groupby(['source','policy']):
        values=[f"{g.recall_05_step16:.0f}/{g.actual_box_coverage:.3f}" for _,g in group.sort_values('seed').iterrows()];lines.append('| '+source+'/'+policy+' | '+' | '.join(values)+' |')
    lines+=['','P0의 source별 최대 metric 차이와279개 저장 경로의 좌표/index 재현 검증은 comparison_metadata.json에 기록했다. GPU 간 변화가 있으면 source별 paired P0 비교를 유지한다. Teacher NLL의 기존 기준은 P0≈5.907845, raw≈6.187455이며 실제 재계산 차이도 metadata에 기록했다.','',f"Argmax 완료 source: {', '.join(argmax) or '없음'} (sampling과 분리).",'', '![비교](comparison.png)']
    if (output/'case73_comparison.png').exists():lines+=['','![Case73 proposal와 실제 방문](case73_comparison.png)']
    (output/'report.md').write_text('\n'.join(lines)+'\n');return comparison

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-root',default='runs/proposal_visit_1011');parser.add_argument('--output');args=parser.parse_args();build(args.run_root,args.output or args.run_root)
