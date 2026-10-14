# Spatial V2 구현과 실험

문헌 조사에서 제안한 위치별 공유 예측, 이전 패치 feedback, global/local fusion을 구현하고 학습했다. [실험 보고서](runs/spatial_v2_seed42/report.md), [비교 그림](runs/spatial_v2_seed42/comparison.png), [단일 영상 예측](results/spatial_v2_example/trajectory.png)을 함께 제공한다.

## 구조

- ImageNet ResNet50 → stride8/16/32 FPN → 24×48 위치 지도. 전체 영상 입력은256×512이다.
- 공유 spatial head는 위치별 시각 특징과128차원 GRU query의 내적으로 다음 위치 확률을 계산한다. Fourier 좌표, 이전 좌표·이동량·step을 사용한다.
- 원시 gaze의 Gaussian 지도(sigma96 원본 픽셀)로 사전학습한 뒤 K16 순서별 위치를 학습한다. one-hot target 및 FPN 없는 모델도 비교했다.
- foveal 모델은 이전 위치의 원본224×224 crop을 같은 ResNet50으로 인코딩한다. 학습에서는 이전 GT crop, 추론에서는 이전 예측 crop을 사용한다. 첫 위치는 전체 영상 문맥에서 시작한다. HAT의 아이디어를 참고한 간소화 모델이며 원 논문의 재현은 아니다.
- 경로 생성은 영상에서 추정한 유방 mask와 hard112 방문 억제를 적용한다. 제약은 생성 단계이며, 중심점이 유방 안이어도224×224 패치 전체가 유방 안이라는 보장은 없다. 후보가 부족하면 거리 조건을 완화하되 유방 영역과 미방문 조건은 유지한다.
- 분류는 전체 영상 특징과16개 패치의 attention MIL 특징을 결합한다. global/local/fusion BCE를 평균한다. 분류 모델의 ResNet50은 전체 fine-tuning하며, trajectory encoder는 처음2 epochs 고정 후 layer4를 fine-tuning한다. BN running statistics는 고정한다. 학습 중 CNN 특징은 다시 계산한다.

주요 파일: [spatial.py](src/spatial.py), [spatial_inference.py](src/spatial_inference.py), [학습 CLI](scripts/spatial_pipeline.py), [실험 runner](scripts/run_spatial_v2.py).

## 데이터와 검증

CBIS-DDSM train376 / validation93을 사용했다. gaze 사전학습은 같은 train 환자의 미주석36장을 더한412장으로 수행했다. 분류 학습 경로는3-fold cross-fitting으로 만들었으며, 각 fold의 사전학습부터 held-out 환자를 제외했다. static/sequence 경로 파일은 각각 train376+validation93의469개 예측과 provenance를 포함한다. test131장과 환자 키가 미확인된 INbreast는 사용하지 않았다.

seed42의 개발 결과:

| 분류 모델 | Validation AUROC |
|---|---:|
| 기존 patch-only fine-tuning | 0.693 |
| 정적 gaze heatmap + fusion | 0.763 |
| 순차 gaze + fusion | 0.781 |
| 동일 초기화 global-only | 0.783 |
| 영상 기반 분산 grid + fusion | 0.796 |

V2와 기존 모델은 구조와 학습 조건이 달라 차이를 gaze 효과로 해석할 수 없다. 동일 조건의 global-only 및 grid 대조군을 보면 trajectory의 추가 분류 효과는 아직 확인되지 않았다. paired 환자 bootstrap 차이 CI도0을 포함한다. grid는 같은 K16, 후보 grid, 유방 mask,112px 거리 조건을 사용하며 gaze를 사용하지 않는다.

경로는16개 서로 다른 중심점을 생성했고, 추정 유방 밖 중심점 비율은0%였다. 병변 box hit는 약89–95%였지만 static heatmap이 가장 높았다. batch2를 맞춘 FPN의 NLL6.101과 foveal NLL6.097 차이는 작다. 이 NLL은 이전 GT를 제공하는 teacher-forcing 지표다. gaze timestamp/session 정보가 없어 임상적 fixation duration이나 자연스러운 종료는 학습하지 않는다.

29개 단위 테스트와 실제 단일 영상 추론을 통과했다. 모든 수치는 동일 validation에서 checkpoint를 선택한 단일 seed 결과이며 bootstrap CI는 선택 편향을 보정하지 않는다. 다음 실험은 순서 섞기, crop scale,3 seeds이다. 최종 test는 설정을 확정한 뒤 평가한다.

## 실행

상위 `code` 폴더에서 기존 학습 환경과 다운로드된 데이터로 실행한다. 전체 runner는 CUDA GPU2개를 사용하며 모델들을 다시 학습한다.

```bash
python gaze_mammo/scripts/spatial_pipeline.py cache --output gaze_mammo/runs/spatial_v2_cache
python gaze_mammo/scripts/run_spatial_v2.py
python gaze_mammo/scripts/run_spatial_grid_control.py
python gaze_mammo/scripts/report_spatial_v2.py
```

추가 matched batch2 경로 및 독립 global-only 대조군은 별도 CLI 실행 결과다.

```bash
python gaze_mammo/scripts/spatial_pipeline.py train \
  --phase sequence --fpn --sigma 96 --batch-size 2 --epochs 15 \
  --init gaze_mammo/runs/spatial_v2_seed42/pretrain_fpn \
  --output gaze_mammo/runs/spatial_v2_seed42/fpn_soft_batch2
python gaze_mammo/scripts/spatial_pipeline.py generate \
  --checkpoint gaze_mammo/runs/spatial_v2_seed42/fpn_soft_batch2 \
  --output gaze_mammo/runs/spatial_v2_seed42/fpn_soft_batch2/validation
python gaze_mammo/scripts/spatial_pipeline.py fusion \
  --head-mode global --epochs 12 \
  --features gaze_mammo/runs/spatial_v2_seed42/sequence_classification/features \
  --output gaze_mammo/runs/spatial_v2_seed42/global_only_matched
```
 설정은 각각 `runs/spatial_v2_seed42/fpn_soft_batch2`와 `global_only_matched`의 checkpoint 및 로그에 저장했다. global-only의 `--head-mode global`은 patch branch를 계산하지 않는다. 초기 crop-cache iterator의 RNG 소비 차이를 맞춰 fusion과 같은 encoder/head 초기화에서 비교한다.

원본 크기1500×3000 영상 하나로 경로와 악성 확률을 생성한다. gaze·박스·라벨을 추론 입력으로 사용하지 않는다.

```bash
python gaze_mammo/scripts/predict_spatial_v2.py \
  --checkpoint gaze_mammo/runs/spatial_v2_seed42/foveal_soft \
  --classifier gaze_mammo/runs/spatial_v2_seed42/sequence_classification/fusion \
  --image gaze_mammo/data/image/Mass-Training_P_00026_LEFT_CC_BENIGN.png \
  --output gaze_mammo/results/spatial_v2_example \
  --device cuda:0
```

`--image`는 실제 존재하는 PNG 경로로 지정한다. CPU는 `--device cpu`를 사용한다. 출력은 `prediction.json`, `trajectory.png`, `patches.png`, 개별224×224 패치16개다. 확률은 독립 test에서 검증·보정된 임상 확률이 아니다.
