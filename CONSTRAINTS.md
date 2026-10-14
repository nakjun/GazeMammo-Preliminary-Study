# 유방 영역과 방문 이력 기반 생성 제약

## 구현

`src/constraints.py`는 이미지로 추정한 유방 마스크와 autoregressive decoder의 다음 위치 확률에 제약을 적용한다. predictor checkpoint를 재학습하지 않는 생성 정책 실험이다.

- `none`: 기존 logits.
- `breast`: 추정 마스크 밖 grid 중심의 logit을 `-inf`로 차단.
- `hard`: breast 조건 + 방문한 grid 셀 금지 + 원본 PNG 좌표의 최소 거리 적용.
- `soft`: breast 조건 + 가장 가까운 방문 지점의 Gaussian penalty (기본 lambda=3).

마스크는 grayscale >10, 가장 큰 연결 성분, hole filling 및 closing으로 계산한다. 박스·시선·라벨은 사용하지 않는다. **임상적으로 검증된 유방 segmentation은 아니므로 보장은 추정 마스크 기준이다.** 경계와 관측 gaze 밖 비율은 mask audit와 고정 예시에서 검토한다.

Hard 제약에서 후보가 없어지면 거리 제한만 절반씩 완화한다. 유방 밖 또는 같은 grid 셀 선택은 허용하지 않는다. 모든 유효 셀을 방문하면 경로를 종료하고 valid mask로 padding을 제외한다. Soft 제약은 재방문을 완전히 금지하지 않는다.

기본 K=16, grid=24×48로 셀 중심 간 축 방향 간격은 원본에서 62.5픽셀이다. 거리 64/112/224픽셀은 물리적 mm가 아닌 제공 PNG 좌표 기준이다. 224×224 패치 중심만 제한하며 패치 전체가 유방 안에 있음을 보장하지 않는다.

## 생성 예시

```bash
python gaze_mammo/scripts/predict_trajectory.py --checkpoint gaze_mammo/runs/trajectory_large_seed42/swin_b --image gaze_mammo/data/image/IMAGE_NAME.png --constraint hard --radius 112 --strategy argmax --output gaze_mammo/results/constrained_example --device cuda:0
```

제약 경로는 원본 예측 중심을 그대로 crop한다. 기존 `none` 단일 예시 CLI는 종전 foreground projection을 유지한다. 아래 성능 비교에서는 정책 효과를 분리하기 위해 모든 조건의 **원본 예측 중심**을 사용하며 projection을 추가하지 않는다.

## 경로와 공간 성능 평가

```bash
python gaze_mammo/scripts/evaluate_constraints.py --all-models
python gaze_mammo/scripts/report_constraints.py
```

큰 checkpoint 4개, argmax/sample 2가지 생성, none/breast/hard64/hard112/hard224/soft112의 6가지 정책을 같은 validation 93장에서 비교한다. 실제 시선과 COCO 박스는 경로 생성 후 DTW와 병변 커버리지 평가에만 사용한다.

결과: `runs/trajectory_constraints_seed42/comparison.csv`, 모델별 per-image metrics/좌표 JSON/provenance/mask audit/예시 그림, `report.md`, `paired_geometry_ci.json`.

주요 지표:

- 추정 유방 밖 중심 비율과 서로 다른 위치 수.
- 최소 중심 거리, 거리 완화 단계 수, 쌍별 패치 겹침(intersection / patch area 평균).
- 유방 면적 커버리지와 패치의 평균 유방 포함률.
- 한 개 이상의 병변 박스에 겹치는 영상 비율, 병변별 hit 및 최선 단일 패치의 병변 면적 커버리지.
- 관측 저장 순서에 대한 normalized DTW. 실제 시선의 재방문을 막으면 DTW가 악화될 수 있으므로 분류용 탐색 정책과 원본 시선 재현 목표를 구분한다.

## 악성 분류 평가

```bash
python gaze_mammo/scripts/run_constraint_classification.py

# 같은 fold predictor들을 재사용해 argmax 조건도 비교
python gaze_mammo/scripts/run_constraint_classification.py --strategy argmax --reuse-folds-from gaze_mammo/runs/constraint_classification_seed42 --output gaze_mammo/runs/constraint_classification_argmax_seed42

python gaze_mammo/scripts/report_constraints.py
```

Swin-B + 큰 decoder를 train 내부 3 folds로 학습한다. 각 학습 환자는 자신을 보지 않은 predictor의 경로를 사용하고 validation은 full-train predictor를 사용한다. none/breast/hard112 모두 동일 환자 분할·patch 수·고정 ResNet50 encoder·attention MIL·seed42로 비교한다. 거리 112를 주 비교로 미리 고정하고 64/224는 공간 분석용 보조 조건으로 둔다.

샘플링 실험은 `runs/constraint_classification_seed42/`, argmax 실험은 `runs/constraint_classification_argmax_seed42/`에 저장한다. 결과는 validation AUROC/AUPRC, 환자 bootstrap CI 및 paired AUROC 차이 CI다. threshold와 checkpoint 선택에 validation을 사용했으므로 독립 test 결과가 아니며 선택 편향은 CI에 반영되지 않는다.

Feature extraction에는 `--keep-predicted-centers`를 사용한다. feature config의 `generation-policy`, `generation-radius`, `generation-strategy`는 단일 영상 분류에서도 같은 생성 방식을 복원하는 데 사용된다. provenance가 이 필드를 제공하면 feature extraction에서 설정 일치를 검사한다.

## 실행 범위

test 131장 및 INbreast quarantine은 사용하지 않는다. 이번 실험은 단일 seed의 개발 비교다. 다른 seed에서는 CF 경로와 full-validation 경로를 모두 동일 seed로 재생성해야 한다. 기본 cross-fitting 비교 runner는 기존 seed42 validation artifacts를 사용하므로 다른 seed 요청을 거부한다.

지금까지의 결과는 `runs/trajectory_constraints_seed42/report.md`에 갱신한다. 생성 위치 제약이 성공했다는 결과와 악성 분류가 개선됐다는 주장은 구분하고, CI가 0을 포함하면 분류 개선을 확정하지 않는다.
