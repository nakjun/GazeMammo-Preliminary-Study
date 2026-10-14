"""Report separate trajectory-backbone and patch-backbone fine-tuning experiments."""
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
from src.metrics import cluster_bootstrap

if __name__=='__main__':
    out=ROOT/'runs/finetuning_report_seed42';out.mkdir(exist_ok=True)
    definitions=[('Frozen / Frozen','constraint_classification_seed42/hard112/classifier'),
                 ('FT trajectory / Frozen patches','finetune_constraint_classification_seed42/hard112/classifier'),
                 ('Frozen trajectory / FT patches','patch_finetune_frozen_trajectory_seed42'),
                 ('FT trajectory / FT patches','patch_finetune_finetuned_trajectory_seed42')]
    rows=[];paired={};reference=pd.read_csv(ROOT/'runs'/definitions[0][1]/'validation_predictions.csv')
    for name,folder in definitions:
        path=ROOT/'runs'/folder
        if not (path/'validation_metrics.json').exists():continue
        r=json.loads((path/'validation_metrics.json').read_text());row=dict(model=name,folder=str(path),**r['metrics'],ci_low=r['auroc_ci']['low'],ci_high=r['auroc_ci']['high'],best_epoch=r.get('best_epoch'))
        rows.append(row)
        if name!=definitions[0][0]:
            pred=pd.read_csv(path/'validation_predictions.csv');paired[name]=cluster_bootstrap(pred,reference,1000,42)
    frame=pd.DataFrame(rows);frame.to_csv(out/'comparison.csv',index=False);(out/'paired_auroc_difference.json').write_text(json.dumps(paired,indent=2))
    trajectory=json.loads((ROOT/'runs/trajectory_finetune_seed42/swin_b/training_summary.json').read_text())
    folds=[]
    for i in range(3):
        path=ROOT/f'runs/finetune_constraint_classification_seed42/fold{i}/model/training_summary.json'
        if path.exists():folds.append(dict(fold=i,**json.loads(path.read_text())))
    (out/'trajectory_summary.json').write_text(json.dumps(dict(full=trajectory,folds=folds),indent=2))
    report='# 백본 파인튜닝 비교\n\n'
    report+='CBIS-DDSM train 376장 / 환자 독립 validation 93장, seed42, sample trajectory + 추정 유방 mask + 방문 억제 112px + K16 + 224×224 패치를 사용했다. test와 INbreast quarantine은 사용하지 않았다.\n\n'
    report+='## 학습 구성\n\n- Swin-B: 전체 백본과 512차원·6층 decoder를 gaze CE loss로 업데이트. backbone lr=1e-5, decoder lr=1e-4, batch8, 최대20 epochs, patience6, BF16 AMP. 학습 환자를 동일하게 본 기존 frozen checkpoint에서 시작했다.\n- CF predictor: 각 fold는 해당 fold를 제외한 기존 frozen-fold checkpoint에서 초기화했다. full-train checkpoint를 fold initializer로 사용하지 않았다.\n- ResNet50 + attention MIL: 모든 백본/헤드 가중치를 분류 loss로 업데이트. backbone lr=1e-5, MIL lr=1e-4, microbatch2 × accumulation4, 최대15 epochs, patience5, BF16 AMP. 검증은 FP32다. BatchNorm의 running statistics만 고정하고 affine 가중치는 업데이트한다.\n- 입력 영상/패치를 RAM에 저장하지만 backbone 특징은 매 update 다시 계산한다. 전체 crop pipeline을 하나의 분류 loss로 미분하는 실험이 아니라 두 단계별 파인튜닝이다.\n- 초기 상태(epoch0)도 검증하며, 개선이 없으면 초기 checkpoint를 유지한다. 모델 설정의 finetune flag만 보고 선택된 가중치가 바뀌었다고 주장하지 않는다.\n\n'
    report+='## Trajectory 학습 결과\n\n'
    report+=f'전체 predictor의 학습 가능한 파라미터는 {trajectory["trainable_parameters"]/1e6:.2f}M이다. 첫 백본 가중치의 최종 평균 변화량은 {trajectory["backbone_first_weight_mean_change"]:.6g}로, 실제 업데이트를 확인했다. 그러나 validation 선택은 epoch **{trajectory["best_epoch"]}**였고 best NLL은 {trajectory["best_validation_nll"]:.4f}였다. BF16 selection loss이므로 과거 FP32 NLL과 작은 수치 차이를 개선으로 해석하지 않는다.\n\n'
    report+='| CF fold | 선택 epoch | 백본 첫 가중치 변화량(학습 마지막) |\n|---|---:|---:|\n'
    for f in folds:report+=f'| {f["fold"]} | {f["best_epoch"]} | {f["backbone_first_weight_mean_change"]:.6g} |\n'
    report+='\n선택 epoch0은 초기 frozen 가중치를 유지한 경우다. 이때 val 경로가 기존 경로와 같을 수 있다. CF에서 일부 선택 checkpoint만 업데이트됐다면 학습 경로의 변화가 predictor 전반의 검증 개선을 뜻하지 않는다.\n\n'
    report+='## 악성 분류\n\nFT는 파인튜닝을 시도한 학습 프로토콜을 뜻한다. 초기 상태를 선택한 경우 최종 가중치는 초기와 같다.\n\n| Trajectory / Patch encoder | AUROC | 환자 bootstrap 95% CI | AUPRC | Patch FT 선택 epoch |\n|---|---:|---|---:|---:|\n'
    for _,r in frame.iterrows():
        epoch='—' if pd.isna(r.best_epoch) else str(int(r.best_epoch))
        report+=f'| {r.model} | {r.auroc:.3f} | {r.ci_low:.3f}–{r.ci_high:.3f} | {r.auprc:.3f} | {epoch} |\n'
    report+='\n기존 frozen/frozen 대비 paired AUROC 차이의 95% CI:\n\n'
    for name,r in paired.items():report+=f'- {name}: {r["low"]:+.3f}–{r["high"]:+.3f}.\n'
    report+='\ncheckpoint와 threshold 선택에 동일 validation을 사용한 개발 결과이며 CI는 선택 편향을 보정하지 않는다. 단일 seed 결과로 일반화하지 않으며, 차이 CI가 0을 포함하면 확정적 개선으로 해석하지 않는다.\n\n'
    if len(rows)==4:report+='현재 설정에서 가장 높은 AUROC는 frozen trajectory + fine-tuned ResNet50의 0.693이다. 두 단계 fine-tuning 프로토콜의 ResNet50은 초기 상태(epoch0)를 선택하여 추가 개선이 없었다. Swin-B full 모델도 초기 상태를 선택했으며 CF fold2만 epoch1을 선택했다. 즉, backbone 학습이 실제로 수행됐다는 사실과 선택된 모델이 fine-tuned weights를 사용한다는 사실을 구분해야 한다.\n\n'
    report+='## 산출물\n\ntrajectory full/fold별 history·config·best checkpoint, CF 좌표 및 학습 환자 provenance, patch fine-tuning best/last checkpoint, 영상별 예측, 실제 백본 변화량 로그를 저장했다. 첫 full trajectory 실행은 last checkpoint 저장 기능 추가 전이라 last.pt가 없으며, CF와 patch fine-tuning 실행은 last.pt를 제공한다.\n'
    (out/'report.md').write_text(report)
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(np.arange(len(frame)),frame.auroc)
    ax.errorbar(np.arange(len(frame)),frame.auroc,yerr=np.stack([frame.auroc-frame.ci_low,frame.ci_high-frame.auroc]),fmt='none',ecolor='black',capsize=4)
    ax.set_xticks(np.arange(len(frame)),frame.model,rotation=20,ha='right');ax.set(ylim=(.4,.9),ylabel='Validation AUROC',title='Trajectory / patch backbone fine-tuning | n=93')
    fig.tight_layout();fig.savefig(out/'auroc_comparison.png',dpi=160);plt.close(fig)
