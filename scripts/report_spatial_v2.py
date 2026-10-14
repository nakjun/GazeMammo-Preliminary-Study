"""Refresh all V2 ablations and compare against strong matched classification controls."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from src.data import ROOT
from src.metrics import cluster_bootstrap

if __name__=='__main__':
    base=ROOT/'runs/spatial_v2_seed42'
    model_dirs={'spatial_hard':base/'spatial_hard/validation','spatial_soft':base/'spatial_soft/validation','fpn_soft_batch8':base/'fpn_soft/validation',
                'fpn_soft_batch2':base/'fpn_soft_batch2/validation','foveal_soft_batch2':base/'foveal_soft/validation','static_heatmap':base/'static_validation'}
    records=[]
    for name,path in model_dirs.items():
        if not (path/'metrics.csv').exists():continue
        data=pd.read_csv(path/'metrics.csv');records.append(dict(model=name,**data.drop(columns=['image_id','patient_key']).mean().to_dict()))
    trajectory=pd.DataFrame(records);trajectory.to_csv(base/'trajectory_comparison_refreshed.csv',index=False)
    definitions=[('V1 patch-only fine-tuned',ROOT/'runs/patch_finetune_frozen_trajectory_seed42','metrics'),
                 ('V2 static heatmap + fusion',base/'static_classification/fusion','fusion_logits'),
                 ('V2 foveal sequence + fusion',base/'sequence_classification/fusion','fusion_logits'),
                 ('V2 matched global-only',base/'global_only_matched','global_logits'),
                 ('V2 grid + fusion',base/'grid_control/fusion','fusion_logits')]
    rows=[];predictions={}
    for name,path,head in definitions:
        metrics_path=path/('validation_metrics.json' if head=='metrics' else 'metrics.json')
        if not metrics_path.exists():continue
        obj=json.loads(metrics_path.read_text());r=obj if head=='metrics' else obj['heads'][head]
        rows.append(dict(model=name,**r['metrics'],ci_low=r['auroc_ci']['low'],ci_high=r['auroc_ci']['high']))
        predictions[name]=pd.read_csv(path/('validation_predictions.csv' if head=='metrics' else f'{head}_predictions.csv'))
    classification=pd.DataFrame(rows);classification.to_csv(base/'classification_comparison.csv',index=False)
    seq=predictions['V2 foveal sequence + fusion'];paired={}
    for name,data in predictions.items():
        if name!='V2 foveal sequence + fusion':paired[name]=cluster_bootstrap(seq,data,1000,42)
    (base/'paired_auroc_difference.json').write_text(json.dumps(paired,indent=2))
    report='# Spatial V2: 공유 위치 지도·패치 feedback·global/local fusion\n\n'
    report+='CBIS-DDSM train376 / validation93, seed42 개발 결과다. 공간 pretraining은 같은 train 환자의 미주석36장을 추가해412장으로 진행했다. test131장과 INbreast quarantine은 사용하지 않았다.\n\n'
    report+='## 구현\n\n- 사전학습 ResNet50, 위치별 공유 spatial head와 Fourier 좌표 embedding, 128차원 GRU controller. 이전 cell ID만 사용하던 모델을 교체했다.\n- 원시 gaze로 공간 지도를 먼저 학습한 뒤 순서별 위치를 학습한다. Gaussian sigma96px와 one-hot target을 비교한다.\n- FPN은 stride8/16/32 특징을 결합한다. 후보 grid24×48 및 global 입력256×512는 유지해 해상도 변경 효과를 섞지 않았다.\n- foveal 조건은 이전 위치의 원본224×224 crop 특징을 next-step memory에 입력한다. global/local crop은 trajectory 내부에서 같은 ResNet50을 공유한다. GRU를 사용한 간소화된 시각 feedback 모델이며 HAT의 완전한 재현이 아니다.\n- 각 단계 첫2 epochs는 head를 학습하고 이후 ResNet layer4를 fine-tuning한다. 분류에서는 ResNet50 전체를 fine-tuning하며 BN running statistics만 유지한다. 입력 픽셀만 캐시하고 학습 중 CNN 특징을 다시 계산한다.\n- 생성은 estimated breast mask + hard112 + sample, K16이다. classifier training 경로는 fold별 gaze pretraining부터 해당 환자를 제외해 생성했다.\n\n'
    report+='## 경로 비교\n\n| 모델 | Raw next-step NLL ↓ | Train prior 대비 IG(bits) ↑ | Any-lesion hit | DTW ↓ |\n|---|---:|---:|---:|---:|\n'
    for _,r in trajectory.iterrows():report+=f'| {r.model} | {r.raw_next_step_nll:.3f} | {r.information_gain_bits:.3f} | {r.any_lesion_hit:.1%} | {r.normalized_dtw:.4f} |\n'
    report+='\nIG는 같은 grid에서 raw conditional probability와 train-only 공간 prior를 비교한다. GT 이전 좌표를 제공하는 teacher-forcing 지표이며 mask/visit 억제 후 경로의 성능과 구분한다. 모든 생성 경로는16개 서로 다른 위치를 유지하고 추정 유방 밖 비율은0%였다.\n\n'
    report+='초기 FPN no-feedback은 batch8, foveal은batch2였으므로 그 차이를 feedback 효과로 바로 해석할 수 없다. no-feedback을batch2로 다시 학습한 matched 비교를 추가했다. NLL 약6.101→6.097로 차이는 작으며 현재 시각 feedback의 추가 효과는 강하게 입증되지 않았다. 공간 지도와 gaze pretraining 자체가 주요 개선 후보로 남는다.\n\n'
    report+='## 악성 분류\n\n| 모델 | AUROC | 환자 bootstrap 95% CI | AUPRC | sensitivity* |\n|---|---:|---|---:|---:|\n'
    for _,r in classification.iterrows():report+=f'| {r.model} | {r.auroc:.3f} | {r.ci_low:.3f}–{r.ci_high:.3f} | {r.auprc:.3f} | {r.sensitivity:.3f} |\n'
    report+='\n*specificity90% 이상을 목표로 같은 validation에서 고정한 threshold의 결과이며 독립 test 성능이 아니다.\n\n'
    report+='V2 sequence fusion − 각 대조군, paired 환자 bootstrap AUROC 차이95% CI:\n\n'
    for name,r in paired.items():report+=f'- {name}: {r["low"]:+.3f}–{r["high"]:+.3f}.\n'
    report+='\nFusion 모델은 global/local/fusion BCE의 평균을 사용한다. 같은 checkpoint의 global head는 local/auxiliary loss의 영향을 받았으므로 별도 global-only 학습 결과와 동일하지 않다. 강한 대조군으로 같은 encoder·head 초기화·split·미세 조정 설정의 global-only 학습을 추가했다. 초기 crop-cache iterator 차이가 head 초기화를 바꾸지 않도록 RNG를 맞췄다.\n\n'
    report+='## 해석과 한계\n\n병변 patch hit는 이전 Swin-B sample hard112의67.7%에서 새 모델 약89–95%로 높아졌다. 원문 box는 사후 평가에만 사용했다. 이 개선을 악성 진단 개선과 동일하게 해석하지 않는다.\n\n'
    report+='기존 patch-only0.693 대비 sequence fusion0.781은 수치상 높지만, matched global-only0.783과 비슷하다. 현재 결과는 trajectory의 추가 분류 가치를 입증하지 않는다. 같은 K16 patch budget의 image-only grid+fusion은0.796으로 가장 높았다. 정적 분산 patch 선택보다 gaze 순서가 낫다는 근거도 아직 없다. Gaussian/FPN/feedback/global 문맥·fine-tuning이 모두 다른 V1과의 차이를 특정 모듈의 인과 효과로 단정하지 않는다.\n\n'
    report+='각 모델의 checkpoint와threshold를 동일 validation으로 선택한 단일 seed 결과다. CI는 model selection bias를 보정하지 않는다. 순서 섞기, 더 넓은crop 및3 seeds는 후속 단계이며, test는 설정을 고정한 후 한 번 평가한다. gaze 기록의 timestamp/session 정보가 부족해 임상적 fixation duration/종료를 학습하지 않는다.\n\n'
    report+='실행 안내: `../../SPATIAL_V2.md`. 모델파일: `src/spatial.py`, 추론: `src/spatial_inference.py`, CLI: `scripts/spatial_pipeline.py`, `scripts/predict_spatial_v2.py`.\n'
    (base/'report.md').write_text(report)
    fig,axs=plt.subplots(1,2,figsize=(14,5))
    axs[0].bar(trajectory.model,trajectory.any_lesion_hit*100);axs[0].tick_params(axis='x',rotation=35);axs[0].set_ylabel('Any-lesion hit (%)')
    axs[1].bar(classification.model,classification.auroc);axs[1].errorbar(np.arange(len(classification)),classification.auroc,yerr=np.stack([classification.auroc-classification.ci_low,classification.ci_high-classification.auroc]),fmt='none',ecolor='black',capsize=3)
    axs[1].tick_params(axis='x',rotation=35);axs[1].set_ylabel('Validation AUROC');axs[1].set_ylim(.4,.95)
    fig.tight_layout();fig.savefig(base/'comparison.png',dpi=160);plt.close(fig)
    frame=pd.read_csv(ROOT/'manifests/trajectory.csv');examples=frame[frame.split=='val'].sort_values('image_id').iloc[np.linspace(0,92,3).astype(int)]
    paths={'static':base/'static_validation/predictions.json','sequence':base/'foveal_soft/validation/predictions.json'}
    data={name:json.loads(path.read_text()) for name,path in paths.items()}
    fig,axes=plt.subplots(3,2,figsize=(9,17))
    for row,(_,r) in enumerate(examples.iterrows()):
        with Image.open(ROOT/'data/image'/r.file_name) as image:background=np.asarray(image.convert('L').resize((300,600)))
        for col,name in enumerate(paths):
            ax=axes[row,col];ax.imshow(background,cmap='gray');points=np.array(data[name][str(r.image_id)])*[300,600]
            ax.plot(points[:,0],points[:,1],color='cyan',alpha=.5);ax.scatter(points[:,0],points[:,1],c=np.arange(len(points)),cmap='turbo',s=20);ax.axis('off');ax.set_title(f'{name} / image{r.image_id}')
    fig.tight_layout();fig.savefig(base/'examples.png',dpi=140);plt.close(fig)
