"""Create a standalone development-baseline report from saved results."""
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

if __name__=='__main__':
    base=ROOT/'runs/baseline_seed42';out=ROOT/'results';out.mkdir(exist_ok=True)
    d=pd.read_csv(base/'comparison.csv')
    fig,ax=plt.subplots(figsize=(8,5));y=d.auroc.to_numpy()
    ax.bar(d.method,y,color=['#7c8ca4','#40a090','#c77b36','#5674b9'])
    ax.errorbar(np.arange(len(d)),y,yerr=np.stack([y-d.ci_low,d.ci_high-y]),fmt='none',ecolor='black',capsize=5)
    ax.axhline(.5,color='gray',linestyle='--');ax.set(ylim=(.4,1),ylabel='Validation AUROC',title='Frozen ResNet50 baseline | patient-separated validation (n=93)')
    fig.tight_layout();fig.savefig(out/'baseline_auroc.png',dpi=180);plt.close(fig)
    difference=json.loads((out/'gaze_vs_random.json').read_text())['paired_auroc_difference_ci']
    report='# 첫 baseline 결과 및 구현 상태\n\n'
    report+='CBIS-DDSM 환자 단위 분할: 학습 376장, 검증 93장. 원본 test 131장은 사용하지 않았다. seed 42, 고정 ImageNet ResNet50 특징, attention MIL, 최대 50 epochs와 validation early stopping을 적용했다.\n\n'
    report+='| 선택 방식 | 검증 AUROC | 환자 bootstrap 95% CI | AUPRC | sensitivity / specificity* |\n|---|---:|---|---:|---|\n'
    for _,r in d.iterrows():
        report+=f'| {r.method} | {r.auroc:.3f} | {r.ci_low:.3f}–{r.ci_high:.3f} | {r.auprc:.3f} | {r.sensitivity:.3f} / {r.specificity:.3f} |\n'
    report+='\n*validation specificity ≥90%를 만족하는 threshold를 같은 validation에서 선택한 결과다. 독립 test 성능이 아니다. checkpoint 선택에도 validation을 사용했으므로 신뢰구간은 선택 편향을 보정하지 않는다.\n\n'
    report+=f'실제 gaze − random의 AUROC 차이는 {float(d[d.method=="observed"].auroc.iloc[0]-d[d.method=="random"].auroc.iloc[0]):.3f}, paired 환자 bootstrap 95% CI는 {difference["low"]:.3f}–{difference["high"]:.3f}다. 첫 설정에서는 gaze 분류 이점이 확인되지 않았다. 단일 seed 결과를 방법의 일반적 우열로 단정하지 않는다.\n\n'
    for method in ['random','observed']:
        c=json.loads((out/f'{method}_coverage/summary.json').read_text())
        report+=f'- {method}: 검증 영상의 병변 patch hit 비율 {c["any_lesion_hit"]["val"]:.1%}, 병변별 hit rate {c["lesion_hit_rate"]["val"]:.1%}, 병변당 최선 단일 패치 면적 커버리지 평균 {c["mean_best_single_patch_lesion_coverage"]["val"]:.1%}.\n'
    report+='\n시선이 병변에 접근하는 효과는 관찰됐지만 분류 이점으로 이어지지 않았다. 원본 224 패치의 문맥 부족, ImageNet 고정 특징, 근접 패치의 중복 등을 다음 실험에서 분리해 검토해야 한다.\n\n'
    report+='## 구현과 검증\n\n- random/grid/observed/whole baseline: 전체 train/validation 학습 완료.\n- trajectory와 heatmap: 별도 8장 train/8장 validation, 1 epoch smoke 학습·생성 검증. 성능 결과로 해석하지 않는다.\n- cross-fitting → 예측 경로 → 패치 → MIL/sequence/shuffled → 평가: 소규모 integration smoke 결과는 `runs/integration_smoke/completed.json`에 기록한다.\n- 환자 분할, 좌표 경계, padding, masking, causal decoding, threshold ties, paired bootstrap 단위 검사.\n- 단일 영상 추론 CLI와 test 명시 실행, 예측 provenance 및 학습 환자 중복 거부 구현.\n\n'
    report+='## 다음 연구 실험\n\n1. 실제 gaze와 random/grid의 448→224 문맥 패치 비교.\n2. 동일 조건에서 patch encoder 일부 미세 조정 및 주요 비교 3 seeds.\n3. 위 결과를 근거로 전체 trajectory/heatmap cross-fitting 학습과 분류 실험 실행.\n4. 모델 설정 고정 후 원본 test 평가. INbreast는 환자 메타데이터 확인 전 quarantine 유지.\n\n실행 안내: `../IMPLEMENTATION.md`.\n'
    (out/'baseline_report.md').write_text(report)
