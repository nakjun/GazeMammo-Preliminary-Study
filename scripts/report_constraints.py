"""Report geometric policy effects and optional cross-fitted classification results."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT

def paired_ci(a,b,column,seed=42):
    pair=a[['image_id','patient_key',column]].merge(b[['image_id',column]],on='image_id',suffixes=('_a','_b'))
    pair['difference']=pair[column+'_a'].astype(float)-pair[column+'_b'].astype(float)
    groups=pair.patient_key.unique();by_group={g:pair[pair.patient_key==g].difference.to_numpy() for g in groups}
    rng=np.random.default_rng(seed);values=[]
    for _ in range(1000):values.append(float(np.concatenate([by_group[g] for g in rng.choice(groups,len(groups),replace=True)]).mean()))
    return dict(mean=float(pair.difference.mean()),low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)))

if __name__=='__main__':
    base=ROOT/'runs/trajectory_constraints_seed42';d=pd.read_csv(base/'comparison.csv')
    report='# 유방 영역 및 방문 이력 제약: 검증 결과\n\n'
    report+='큰 trajectory checkpoint 4종(ResNet50, ResNet101, ConvNeXt-Base, Swin-B)을 재학습하지 않고 동일 validation 93장에서 생성 정책만 비교했다. K=16, grid=24×48, seed=42다. 같은 batch/seed로 기존·유방 마스크·거리 억제 조건을 재생성했다. 과거 sampling 표와는 batch 및 random stream이 달라 이번 기준 조건을 재계산했다.\n\n'
    report+='## 구현과 보장 범위\n\n- grayscale >10에서 가장 큰 연결 성분을 선택하고 구멍 채우기와 closing으로 추정 유방 마스크를 만든다. 마스크는 이미지에서만 계산하며 박스와 시선을 사용하지 않는다.\n- breast: grid 중심이 추정 유방 밖이면 logit을 -inf로 차단한다. **의학적 정답 유방 segmentation에 대한 보장은 아니다.**\n- hard: 방문한 grid 셀을 금지하고, 이전 모든 위치와 원본 PNG 기준 최소 거리 64/112/224픽셀을 적용한다.\n- 후보가 없어지면 거리만 단계적으로 완화한다. 유방 밖 셀 및 이미 방문한 셀 금지는 유지한다. 모든 후보를 방문하면 해당 경로는 종료한다. padding은 결과에 포함하지 않는다.\n- soft112: 최근접 방문 거리의 Gaussian penalty를 logit에서 차감한다(lambda=3). 재방문이 가능하므로 hard와 보장 범위가 다르다.\n- 좌표 중심을 제한하며 224×224 패치 전체가 유방 안에 있음을 보장하지 않는다. 패치의 유방 포함 면적도 측정한다.\n\n'
    report+='## Swin-B의 주요 비교\n\n| 생성 | 정책 | 유방 밖 중심 | 서로 다른 위치 /16 | 쌍별 평균 패치 겹침 | 병변 hit 영상 | DTW ↓ |\n|---|---|---:|---:|---:|---:|---:|\n'
    main=d[(d.backbone=='swin_b')&d.policy.isin(['none','breast','hard112'])]
    for strategy in ['argmax','sample']:
        for policy in ['none','breast','hard112']:
            r=main[(main.strategy==strategy)&(main.policy==policy)].iloc[0]
            report+=f'| {strategy} | {policy} | {r.out_of_breast_fraction:.1%} | {r.unique_locations:.2f} | {r.mean_pairwise_patch_overlap:.1%} | {r.any_lesion_hit:.1%} | {r.normalized_dtw:.4f} |\n'
    report+='\n쌍별 겹침은 모든 패치 쌍의 intersection / 224²를 평균한 값이다. 병변 hit는 한 패치라도 하나의 COCO 박스와 면적으로 겹치는 영상 비율이다. 이는 병변 검출 정확도나 악성 분류 정확도가 아니다. 박스는 사후 평가에만 사용했다.\n\n'
    per_image=pd.read_csv(base/'swin_b/per_image_metrics.csv');cis={}
    for strategy in ['argmax','sample']:
        a=per_image[(per_image.strategy==strategy)&(per_image.policy=='hard112')]
        b=per_image[(per_image.strategy==strategy)&(per_image.policy=='none')]
        cis[strategy]={column:paired_ci(a,b,column) for column in ['out_of_breast_fraction','mean_pairwise_patch_overlap','any_lesion_hit','normalized_dtw']}
    (base/'paired_geometry_ci.json').write_text(json.dumps(cis,indent=2))
    report+='hard112 − none, 환자 단위 paired bootstrap 95% CI:\n\n'
    for strategy,columns in cis.items():
        for col,r in columns.items():report+=f'- {strategy}, {col}: {r["mean"]:+.4f} [{r["low"]:+.4f}, {r["high"]:+.4f}].\n'
    report+='\n## 거리와 soft penalty 비교 (Swin-B)\n\n| 생성 | 정책 | 최소 중심 거리 평균(px) | 병변 hit 영상 | 유방 면적 커버리지 | 패치 유방 포함률 | 거리 완화 단계 평균 |\n|---|---|---:|---:|---:|---:|---:|\n'
    for _,r in d[d.backbone=='swin_b'].iterrows():report+=f'| {r.strategy} | {r.policy} | {r.minimum_center_distance:.1f} | {r.any_lesion_hit:.1%} | {r.breast_area_covered:.1%} | {r.mean_patch_breast_fraction:.1%} | {r.relaxed_steps:.2f} |\n'
    qc=pd.read_csv(base/'swin_b/mask_audit.csv')
    report+=f'\n관측된 축약 gaze의 추정 마스크 밖 비율은 평균 {qc.observed_gaze_outside.mean():.2%}다. 실제 배경 관찰인지 마스크 오류인지는 정답 segmentation 없이 확정할 수 없다. `examples.png`에서 고정된 첫 3장의 경계와 경로를 검토할 수 있다.\n\n'
    report+='## 다른 백본의 hard112 효과\n\n| 백본 | 생성 | 위치 수: 기존 → hard | 패치 겹침: 기존 → hard | 병변 hit: 기존 → hard |\n|---|---|---|---|---|\n'
    for backbone in d.backbone.unique():
        for strategy in ['argmax','sample']:
            subset=d[(d.backbone==backbone)&(d.strategy==strategy)];a=subset[subset.policy=='none'].iloc[0];b=subset[subset.policy=='hard112'].iloc[0]
            report+=f'| {backbone} | {strategy} | {a.unique_locations:.2f} → {b.unique_locations:.2f} | {a.mean_pairwise_patch_overlap:.1%} → {b.mean_pairwise_patch_overlap:.1%} | {a.any_lesion_hit:.1%} → {b.any_lesion_hit:.1%} |\n'
    fig,axs=plt.subplots(1,3,figsize=(14,4));s=main[main.strategy=='sample'].set_index('policy').loc[['none','breast','hard112']]
    for ax,col,title in zip(axs,['out_of_breast_fraction','mean_pairwise_patch_overlap','any_lesion_hit'],['Outside estimated breast','Mean pairwise patch overlap','Any annotated lesion hit']):
        ax.bar(s.index,s[col]*100);ax.set_title(title);ax.set_ylabel('%')
    fig.tight_layout();fig.savefig(base/'policy_effects.png',dpi=160);plt.close(fig)
    classification=ROOT/'runs/constraint_classification_seed42'
    if (classification/'comparison.csv').exists():
        result=pd.read_csv(classification/'comparison.csv');paired=json.loads((classification/'paired_auroc_difference.json').read_text())
        report+='\n## 악성 분류: 내부 cross-fitting 비교\n\nSwin-B + 512차원·6층 decoder를 동일 학습 설정으로 train 내부 3-fold cross-fitting했다. held-out train 환자 경로와 full-train predictor의 validation 경로를 사용한다. 모두 sample seed42, K=16의 **원본 예측 중심**에서 224×224 패치를 추출했다. 별도의 background snap은 적용하지 않았다. 각 정책의 동일 고정 ResNet50 patch encoder와 attention MIL을 train 376장으로 학습하고 val 93장에서 비교했다. 박스와 실제 시선은 classifier 입력에 사용하지 않는다.\n\n'
        report+='| 정책 | AUROC | 환자 bootstrap 95% CI | AUPRC |\n|---|---:|---|---:|\n'
        for _,r in result.iterrows():report+=f'| {r.policy} | {r.auroc:.3f} | {r.auroc_ci_low:.3f}–{r.auroc_ci_high:.3f} | {r.auprc:.3f} |\n'
        for policy,r in paired.items():report+=f'\n- {policy} − none AUROC 차이 95% CI: {r["low"]:+.3f}–{r["high"]:+.3f}.\n'
        report+='\nvalidation은 predictor와 classifier checkpoint 선택에 사용됐다. 이 CI는 선택 편향을 보정하지 않으며 독립 test 성능이 아니다. 단일 seed이며, geometry 제약이 learned gaze 정확도를 개선했다는 주장과 분류 결과는 구분한다.\n'
    else:report+='\n악성 분류의 cross-fitting 실험은 진행 중이며 완료 수치가 나오기 전까지 분류 개선을 주장하지 않는다.\n'
    argmax_dir=ROOT/'runs/constraint_classification_argmax_seed42'
    if (argmax_dir/'comparison.csv').exists():
        result=pd.read_csv(argmax_dir/'comparison.csv');paired=json.loads((argmax_dir/'paired_auroc_difference.json').read_text())
        report+='\n## Argmax 경로의 악성 분류 (같은 CF predictor 재사용)\n\n위 실험의 동일 fold checkpoint를 재사용하고 생성만 argmax로 바꿨다. classifier는 동일 설정으로 별도로 학습한다.\n\n| 정책 | AUROC | 환자 bootstrap 95% CI | AUPRC |\n|---|---:|---|---:|\n'
        for _,r in result.iterrows():report+=f'| {r.policy} | {r.auroc:.3f} | {r.auroc_ci_low:.3f}–{r.auroc_ci_high:.3f} | {r.auprc:.3f} |\n'
        for policy,r in paired.items():report+=f'\n- {policy} − none AUROC 차이 95% CI: {r["low"]:+.3f}–{r["high"]:+.3f}.\n'
        report+='\nargmax에서 마스크 조건만 적용한 경로가 기존 경로와 같으면 분류 결과도 동일할 수 있다. 방문 억제의 경로 다양성 효과와 분류 효과를 구분한다.\n'
    if (classification/'comparison.csv').exists() and (argmax_dir/'comparison.csv').exists():
        fig,axs=plt.subplots(1,2,figsize=(10,4))
        for ax,folder,title in zip(axs,[classification,argmax_dir],['Sampling','Argmax']):
            frame=pd.read_csv(folder/'comparison.csv').set_index('policy').loc[['none','breast','hard112']]
            ax.bar(frame.index,frame.auroc);ax.errorbar(np.arange(3),frame.auroc,yerr=np.stack([frame.auroc-frame.auroc_ci_low,frame.auroc_ci_high-frame.auroc]),fmt='none',ecolor='black',capsize=4)
            ax.set(ylim=(.4,.9),ylabel='Validation AUROC',title=title)
        fig.tight_layout();fig.savefig(base/'classification_auroc.png',dpi=160);plt.close(fig)
        report+='\n샘플링에서 hard112는 AUROC 0.572→0.666, argmax에서는 0.613→0.630이었다. 두 조건 모두 hard112−none의 paired 차이 CI가 0을 포함한다. 따라서 현재는 위치 제약과 패치 다양성의 효과는 확인했지만 악성 분류 개선을 확정하지 않는다.\n'
    report+='\n## 해석\n\n유방 마스크와 hard 방문 억제는 추정 마스크 밖 이동 및 동일 grid 반복을 직접 막는 효과가 있다. 이는 생성 정책의 변화이며 predictor가 새로운 시선 행동을 학습한 결과가 아니다. 공간 커버리지가 늘어나도 실제 시선에는 재방문이 있으므로 DTW가 악화될 수 있다. 거리값은 이번 validation의 개발 비교이며 최종 값을 선택한 후 별도 test에서 검증해야 한다. 모든 실험에서 test 131장과 INbreast quarantine은 사용하지 않았다.\n'
    (base/'report.md').write_text(report)
