> 2026-10-12 현재 우선순위는 **시선 예측 → 패치 추출 → Benign/Malignant 분류 성능 개선**이다. 최신 개발 명세는 [plan_gaze_patch_classification_1012.md](plan_gaze_patch_classification_1012.md), 구현 및 검증은 [GAZE_PATCH_CLASSIFICATION_1012.md](GAZE_PATCH_CLASSIFICATION_1012.md)를 따른다. 아래 2026-10-10 계획은 초기 연구 기록이며, 이전 NLL/병변 확보율 반복 목표는 중단 상태다. 시선 NLL·병변 확보율·분류 지표를 서로 다른 척도로 보고한다.

# Gaze trajectory 기반 패치 선택 및 유방촬영 악성 분류 연구 계획

작성일: 2026-10-10

## 1. 연구 목표와 범위

유방촬영 영상에서 시선 경로를 예측하고, 경로상의 지점을 중심으로 224×224 패치를 추출한 뒤, 패치 특징을 통합해 영상 단위 악성 확률을 예측한다. 최종 추론 입력은 영상 한 장이다. 실제 시선과 병변 박스는 추론에 제공하지 않는다.

첫 연구의 범위는 **주석이 있는 영상의 benign vs malignant 분류**다. 정상 선별 검사, 환자 단위 진단, 판독 시간 예측은 별도 검증이 필요한 확장 과제로 둔다.

```text
전체 mammogram → 영상 encoder → trajectory decoder → K개 패치 중심
                                                         ↓
                                             224×224 패치 추출
                                                         ↓
                                                patch encoder
                                                         ↓
                                         MIL / sequence aggregator
                                                         ↓
                                              malignant probability
```

## 2. 검증할 가설

| 가설 | 실험으로 확인할 내용 |
|---|---|
| H1: 관측 시선 위치는 유용한 패치를 선택한다 | 실제 gaze 패치가 동일 K의 무작위·규칙적 패치보다 분류 성능이 높은가 |
| H2: 영상에서 예측한 시선 위치도 유용하다 | 예측 gaze 패치가 기본 패치 선택보다 좋고, 관측 gaze 조건의 효과를 얼마나 유지하는가 |
| H3: 시선 순서는 위치 집합 이상의 정보를 제공한다 | 동일 패치 집합에서 원래 순서·순서 섞기·순서 없는 MIL을 비교한다 |
| H4: 시선 경로는 제한된 패치 예산에 유리하다 | K와 추론 비용에 따른 성능을 비교한다 |

실제 gaze는 관측 시선의 유용성을 확인하는 참조 조건이다. 이 조건의 성능이 모든 방법의 절대 상한이라는 가정은 하지 않는다. H3가 성립하지 않으면 결과를 위치 선택의 효과로 설명하고, 순차 탐색의 효과라고 주장하지 않는다.

## 3. 데이터와 라벨

원본: https://huggingface.co/datasets/yankon0408/GAA-DETR-breast

분석 근거: `eda/summary.json`, `eda/image_inventory.csv`, `eda/lesion_inventory.csv`. 다운로드 revision은 `provenance.json`에 기록되어 있다.

| 항목 | 현황 |
|---|---:|
| 영상 | 857장, 모두 1500×3000 |
| 출처 | CBIS-DDSM 664장, INbreast 193장 |
| 병변 박스 | 756개: benign 389개, malignant 367개 |
| 시선 샘플 | 422,926개, 영상당 중앙값 478개 |
| benign 박스만 있는 영상 | 324장 |
| malignant 박스가 있는 영상 | 335장: malignant만 323장 + 혼합 12장 |
| 미주석 영상 | 198장 |

분류 라벨은 COCO 병변 주석에서 만든다. malignant 병변이 하나라도 있으면 1, benign 병변만 있으면 0으로 한다. **파일명에 포함된 진단 문자열을 모델 입력으로 사용하지 않는다.** 이 라벨은 제공 주석에 따른 영상 라벨이며 독립적으로 확인한 환자 진단은 아니다.

미주석 영상은 정상/benign으로 취급하지 않는다. 첫 분류 실험은 659장으로 진행한다. 미주석 영상의 시선 학습 활용은 학습 환자에 속하는 영상만 허용하고, 별도 ablation으로 보고한다.

## 4. 데이터 검증과 분할: 먼저 완료할 단계

1. 원본 코드·수집 설명을 확인해 좌표 리스트가 시간 순서인지, 판독자·세션 기록이 병합되었는지 확인한다. 확인되지 않으면 명칭은 `stored-order coordinate sequence`로 제한한다. 타임스탬프 없는 좌표 군집을 검증된 fixation으로 부르지 않는다.
2. CBIS-DDSM의 P 번호를 원본 환자 메타데이터와 대조한다. INbreast 파일명 해시는 현재 **추정 그룹**이며 환자 ID로 확정되지 않았으므로 원본 메타데이터로 연결한다. INbreast는 그룹 확인 전 확정적인 환자 독립 평가에 포함하지 않는다.
3. RGB 671장과 grayscale 186장의 색상 채널을 확인하고 영상 입력을 통일한다. RGB 채널이 동일한지, 마킹이나 색상 정보가 포함되었는지도 검사한다. 전처리 통계는 학습 데이터로만 결정한다.
4. 시선은 각 샘플의 `(x/scale, y/scale)`로 복원한다. 범위 밖 331개는 제외하고, 제외 구간을 건너 연결하지 않는다. 필요하면 연속 유효 구간으로 나누어 학습한다.
5. 범위를 벗어난 박스 12개는 품질 로그를 유지하고 파생 주석에서 clipping한다. 부동소수점 경계 오차는 1e-6 픽셀 허용오차로 구분한다. 원본은 보존한다.
6. 바이트 수준 중복은 없지만, 동일 픽셀·근접 중복과 같은 환자의 다른 뷰를 추가 점검한다.

### 권장 평가 분할

- **주 평가:** CBIS-DDSM 파일명의 원본 Training/Test를 보존한다. Training 환자 안에서만 train/validation을 분리하고, Test는 최종 평가까지 잠근다. 두 집합의 환자 중복은 원본 메타데이터로 재확인한다. 분류 대상 수는 미주석 제외 후 별도로 기록한다.
- **출처 간 평가:** CBIS-DDSM에서 개발한 모델을 INbreast에 적용한다. 타깃 출처로 튜닝하지 않은 결과를 먼저 보고한다. 이 평가는 동일 gaze 수집 파이프라인을 공유하므로 완전히 독립된 임상 외부 검증으로 표현하지 않는다.
- **보조 평가:** 두 출처의 환자 키가 확정되면 출처·라벨 분포를 고려한 환자 단위 5-fold 평가를 추가한다. 주 평가와 결과를 구분한다.
- trajectory, heatmap, patch classifier가 모두 동일한 환자 분할을 사용한다. 평가 영상은 비지도 사전학습·시선 학습에도 사용하지 않는다.

분할 결과는 영상 ID, 환자 키, 출처, 라벨, split, fold를 포함한 고정 manifest로 저장한다. 라벨 균형을 맞추기 위해 환자 그룹을 분리하지 않는다.

## 5. 패치와 시퀀스 전처리

기본 조건은 원본 1500×3000 영상에서 **224×224 영역을 crop**하는 것이다. 경계에서는 명시적인 padding으로 패치 크기를 유지한다. 좌표는 영상 너비·높이로 정규화하고, encoder 입력이 resize/padding되면 그 좌표 변환을 정확히 적용한다.

- pilot: K=16, 이후 K∈{8,16,32} 비교.
- 반복되는 가까운 샘플은 연속 구간 내 거리 기준으로 정리한다. 모든 구간을 전역 dedup하면 재방문 정보가 사라지므로 처음에는 사용하지 않는다.
- 정리 후 긴 좌표열은 순서를 유지하는 균등 index 선택으로 최대 K개를 얻는다. 짧은 좌표열은 mask를 적용해 padding한다. 시선 기록 사이에 시간 간격이 같다는 가정은 하지 않는다.
- 원시 샘플 순서와 축약한 좌표열을 모두 보관한다. 학습 target과 분류 패치 중심은 축약 좌표열을 기본으로 한다.
- 근접점 제거 거리·K·최대 길이는 validation에서 결정하고 test에서는 고정한다. 실제/예측 gaze에 동일 규칙을 적용한다.
- 다중 크기 실험: 224 crop vs 448 crop을 224로 resize. 더 큰 문맥이 필요한지 확인한다. 이는 물리적 mm 단위가 아닌 전처리된 PNG의 픽셀 단위 비교다.
- 영상 회전·뒤집기 등 증강은 이미지·시선·박스에 동일 변환을 적용한다. crop 밖 시선은 명시적으로 처리한다.

고정 K 비교에서는 trajectory 모델이 같은 패치 예산의 경로를 생성하게 한다. 자연 길이와 종료 예측은 별도 실험으로 분리하여 K의 차이가 성능 비교를 혼동하지 않게 한다.

## 6. 모델 구성

### 6.1 관측 gaze 기반 분류 참조 모델

관측 gaze에서 뽑은 K개 패치를 사전학습 patch encoder로 처리한다. 처음에는 encoder를 고정하고 attention MIL aggregator 및 binary head만 학습한다. 안정적인 baseline을 얻은 후 encoder 일부를 미세 조정한다. 동일 backbone과 학습 조건을 기본 패치 선택 방법에도 적용한다.

MIL은 순서 없는 패치 특징 통합이다. sequence aggregator는 같은 특징에 순서 embedding을 붙인 작은 Transformer로 구성한다. 파라미터 수 차이를 기록하고, 같은 sequence 모델의 입력 순서를 섞는 실험도 수행한다.

### 6.2 영상 → trajectory 모델

- 사전학습 영상 encoder + 작은 autoregressive Transformer decoder.
- 영상 전체의 저해상도 특징에서 시작한다. 미세 병변이 사라지는 경우 높은 해상도 또는 다중 스케일 특징을 비교한다.
- BOS에서 첫 위치를 예측하고, 이전 위치와 영상 특징에서 다음 위치를 생성한다. 영상만으로 경로를 생성하는 조건에서는 실제 첫 시선 위치도 제공하지 않는다.
- 초기 구현은 2D 위치 grid의 확률분포와 cross-entropy를 사용한다. grid 양자화 오차를 보고하고, 필요하면 연속 offset 또는 좌표 혼합분포를 추가한다.
- teacher forcing으로 학습하고, free-running rollout으로 평가한다. 두 조건의 성능을 따로 보고한다.
- 샘플 순서는 position embedding으로 제공한다. 확대 배율은 초기 모델 입력에서 제외한다. 배율 공동 예측은 후속 실험으로 둔다.
- 시선 heatmap 예측 보조 손실은 단독 baseline과 비교한 후 추가한다. 병변 박스 보조 감독은 기본 모델에서 제외하고 독립 ablation으로 둔다.

### 6.3 예측 trajectory 기반 분류

trajectory predictor를 먼저 학습하고 고정한 뒤 예측 경로에서 패치를 추출해 classifier를 학습한다. 관측 gaze 패치로만 학습한 classifier를 예측 gaze로 평가하는 조건은 입력 분포 차이 분석용으로 구분한다.

권장 학습 입력은 **환자 단위 내부 cross-fitting으로 만든 예측 경로**다. 학습 환자를 내부 fold로 나누고 해당 환자를 보지 않은 predictor로 경로를 생성한다. validation/test 경로는 전체 training 환자로 학습한 predictor가 생성한다. 내부 predictor의 설정은 validation/test를 이용해 조정하지 않는다.

첫 단계는 분리 학습으로 원인을 분석한다. 정수 좌표 crop은 trajectory까지 gradient를 전달하지 않으므로, end-to-end 분류 미세 조정은 후속 단계에서 differentiable sampling과 원래 시선 감독을 함께 사용해 별도로 검증한다.

## 7. 단계별 실험과 진행 기준

| 단계 | 실행 내용 | 진행 판단 |
|---|---|---|
| A: 데이터 확정 | 순서/세션 확인, 환자 키, 라벨, 전처리, split manifest | 분할 누수와 타깃 정의가 해결되면 학습 시작 |
| B: 패치 선택 가치 | K=16에서 random/grid/실제 gaze + 동일 MIL, 전체 영상 분류 baseline | 실제 gaze 효과와 신뢰구간을 확인; 효과가 없으면 K·중복·문맥을 먼저 점검 |
| C: 시선 예측 | heatmap baseline 및 trajectory 모델, next-step와 rollout 평가 | 경로를 외우는 현상과 분포 붕괴가 없는지 확인 |
| D: 최종 파이프라인 | cross-fitted 예측 gaze 패치 + MIL | 동일 예산의 기본 선택과 분류 성능·비용 비교 |
| E: 순차 정보 | 동일 패치의 MIL/순서 Transformer/shuffled order | 위치 선택 효과와 순서 효과를 분리 |
| F: 확장 및 최종 평가 | K·crop 크기·출처 간 일반화, 고정 test 평가 | 주 가설을 중심으로 결과와 한계를 보고 |

초기 개발은 한 고정 split와 한 seed로 수행한다. 설정을 고정한 뒤 주요 비교를 3개 seed로 반복한다. 모든 조합의 대규모 탐색보다 단계별 근거에 따라 필요한 ablation을 진행한다.

## 8. 최소 비교 실험

| 방법 | 선택 신호 | 통합 | 목적 |
|---|---|---|---|
| 전체 영상 분류 | 전체 영상 | image head | 전체 영상 기준 성능·비용 |
| Random-K | 유방 영역 내 무작위 | MIL | 기본 선택 대비 |
| Grid-K | 유방 영역 내 규칙적 위치 | MIL | 공간 커버리지 대비 |
| Observed gaze-K | 실제 시선 | MIL | 시선 위치의 참조 효과 |
| Predicted heatmap-K | 예측 시선 밀도, 공간 중복 억제 | MIL | 정적 시선 분포 대비 |
| Predicted trajectory-K | 영상에서 생성한 경로 | MIL | 경로 기반 패치 선택 효과 |
| Predicted trajectory-K + sequence | 위와 같은 패치 | Transformer | 순서 활용 효과 |
| Shuffled trajectory-K + sequence | 같은 패치, 순서만 섞기 | 같은 Transformer | 순서의 직접 대조군 |

유방 영역 마스크는 영상으로만 생성하며 모든 패치 선택에 동일하게 적용한다. 단순 background 제거로 gaze 방법이 유리해지는 효과를 통제한다. random 방법은 반복 추출에 따른 변동도 보고한다. 선택 모델을 포함한 전체 파라미터·연산 비용을 기록한다.

## 9. 평가와 결과 해석

### 분류

- 주 지표: 영상 단위 AUROC. 보조 지표: AUPRC, sensitivity, specificity, balanced accuracy, calibration.
- threshold는 validation에서 고정한다. 예를 들어 validation specificity 90%를 목표로 threshold를 정하고, test에서 실제 specificity와 sensitivity를 함께 보고한다. test threshold를 역으로 튜닝하지 않는다.
- 환자 단위 cluster bootstrap으로 95% 신뢰구간과 방법 간 paired 차이를 계산한다. seed 변동과 표본 불확실성을 구분한다.
- CBIS-DDSM/INbreast 출처별 성능과 표본 수를 함께 보고한다. 한 환자의 여러 영상 때문에 생기는 상관을 고려한다.

### trajectory 및 패치 선택

- next-step 확률의 NLL과 정규화 위치 오차, 영상만으로 생성한 rollout의 공간 분포·순서 유사도를 구분한다.
- 길이 정규화 DTW와 grid 기반 sequence similarity를 보조 지표로 사용한다. 한 개의 관측 경로만으로 다른 타당한 경로를 완전히 평가할 수 없다는 한계를 명시한다.
- 실제 시간 정보가 필요한 fixation duration/속도 지표는 사용하지 않는다.
- 패치 중 하나라도 병변 박스와 겹치는 영상 비율, 병변별 patch hit rate, 유방 영역 커버리지, 패치 간 중복률을 측정한다. 단순 교집합과 병변 면적 커버리지 기준을 구분한다.
- 병변 박스는 위 사후 평가에만 사용하며 테스트 패치 선택에는 사용하지 않는다. 높은 lesion hit rate만으로 높은 분류 성능을 주장하지 않는다.

### 분석 그림

- 실제 vs 예측 시선 경로와 동일 K 패치 중심 비교.
- 선택 패치와 패치별 attention, 예측 악성 확률 표시.
- K 대비 AUROC 및 전체 추론 시간/메모리.
- 성공·실패 사례: 큰 병변, 작은 병변, 혼합 라벨, 다른 촬영 뷰, 출처 간 차이.
- MIL attention은 모델 가중치로 설명하며 인과적·임상적 설명으로 단정하지 않는다.

## 10. 주요 위험과 대응

| 위험 | 대응 |
|---|---|
| 857장 규모와 모델 과적합 | 사전학습 encoder, 작은 decoder, 초기 고정 학습, 환자 단위 validation |
| 저장 순서가 실제 연속 판독이 아닐 가능성 | 수집/병합 규칙 확인, 미확인 시 순차 해석 제한 |
| 인접 gaze의 중복 패치 | 순서 유지 근접점 정리, 중복률 보고, 동일 K 비교 |
| 원본 224 crop의 문맥 부족 | 448→224 비교, 병변 크기별 분석 |
| 관측 시선과 예측 시선의 차이 | 예측 경로 기반 학습, 내부 cross-fitting |
| 좌표 예측이 평균 위치로 붕괴 | 위치 분포 예측, rollout 시각화 및 다양성 분석 |
| 출처·영상 모드의 shortcut | 동일 전처리, 출처별 평가와 출처 간 검증 |
| 임상 목표 과장 | 주석 있는 영상의 악성 분류로 범위 명시, 정상 선별 성능으로 일반화하지 않음 |

## 11. 산출물과 첫 실행 목표

예정 산출물:

```text
gaze_mammo/
  plan.md
  manifests/                 # 환자 키, 라벨, split/fold
  configs/                   # 전처리·선택·모델·평가 설정
  src/                       # dataset, sampler, models, metrics
  scripts/                   # 분류/시선 학습 및 평가 진입점
  runs/                      # 설정, seed, 학습 로그, checkpoint
  results/                   # 영상별 예측, paired 비교, 그림
```

첫 실행 목표는 **환자 키와 분할을 확정하고, K=16의 random/grid/observed gaze + 동일 attention MIL을 비교하는 것**이다. 이 결과를 근거로 trajectory predictor 개발 규모와 순서 실험의 우선순위를 결정한다.

### 구현 진행 기록 (2026-10-10)

CBIS-DDSM 메타데이터와 환자 키를 대조했고, train 376 / validation 93 / test 131개의 주석 영상으로 분할했다. INbreast는 환자 키 미확인으로 quarantine 처리했다. K=16의 random/grid/observed 및 전체 영상 baseline을 학습했다. trajectory/heatmap과 cross-fitting 파이프라인도 구현하고 소규모 연결 검증을 수행했다. 구현과 실행 명령은 `IMPLEMENTATION.md`, 첫 validation 결과는 `results/baseline_report.md`에서 관리한다. 3 seeds, 최종 test 및 출처 간 평가는 아직 연구 후속 단계다.

추가 실행: trajectory predictor의 ResNet50 / DenseNet121 / EfficientNet-B0 / ConvNeXt-Tiny / Swin-T 고정 백본 비교를 전체 학습 376장과 validation 93장으로 완료했다. argmax 및 확률 샘플링 생성과 train-only prior를 비교했다. 현재 argmax의 반복 수렴 및 prior 대비 경로 개선 미확인이 관찰됐으므로, 아직 최종 백본을 결정하지 않는다. 설정·결과는 `TRAJECTORY.md`와 `runs/trajectory_backbones_seed42/report.md`에 기록한다. 3 seeds, 백본 미세 조정, 전체 규모 cross-fitting 분류 및 test 평가는 남아 있다.

큰 모델 확장: decoder 512차원·6층·8 heads·FFN 2048과 ResNet101 / ConvNeXt-Base / Swin-B를 구현했다. ResNet50 + 큰 decoder를 포함해 같은 train/validation에서 학습했다. 큰 모델의 생성과 cross-fitting은 checkpoint에 저장된 구조를 복원한다. 실행 안내는 `LARGE_MODELS.md`, 결과는 `runs/trajectory_large_seed42/report.md`에서 관리한다. 백본은 현재 고정이며, 모델 규모 증가가 경로 반복 수렴의 해결을 보장하지 않는다.

생성 제약 실험: 이미지 기반 추정 유방 mask와 hard/soft 방문 억제를 구현했다. 큰 모델 4종에서 argmax/sample × 6개 생성 정책을 val 93장으로 비교했다. Swin-B의 train 내부 3-fold predictor를 학습하고, 동일 checkpoint를 사용한 sample/argmax의 none/breast/hard112 경로로 각각 MIL 분류까지 비교했다. hard112는 반복 억제와 추정 유방 밖 이동 차단을 달성했으나 분류 AUROC 차이의 CI는 0을 포함한다. `CONSTRAINTS.md` 및 `runs/trajectory_constraints_seed42/report.md`에 결과를 기록한다. predictor를 제약 손실로 재학습한 결과는 아니며 별도 test 평가는 수행하지 않았다.

파인튜닝 실험: Swin-B 전체와 ResNet50+MIL 전체를 각 단계의 loss로 업데이트하도록 구현하고 full/CF predictor 및 4가지 분류 조합을 실행했다. backbone lr1e-5, decoder/MIL lr1e-4를 사용했다. full Swin-B와 CF 0/1은 초기 상태를, CF2는 epoch1을 선택했다. ResNet50만 파인튜닝한 경우 val AUROC0.693이었으나 paired CI가 0을 포함한다. `FINETUNING.md`와 `runs/finetuning_report_seed42/report.md`에 선택 epoch와 실제 가중치 변화를 기록한다. test 및 INbreast quarantine은 유지한다.

문헌 조사 후 다음 후보: Gazeformer/HAT/DeepGaze III, 원 데이터 GAA-DETR, GMIC/GLAM, SalTR, McGIP 및2026 DeepGaze3.5-VL 원고를 검토했다. 우선순위는 (1) 위치별 시각 특징과 query를 연결하는 공유 spatial head 및 Gaussian soft 감독, (2) 이전 고해상도 patch의 시각 feedback, (3) global/local 분류 fusion이다. 단계별 ablation과 spatial prior/heatmap+IOR 대조군으로 순서의 추가 가치를 검증한다. 후속 Spatial V2 학습과 대조 실험을 완료했다. 상세 근거·데이터 진단·실험 순서는 `literature_review.md`에서 관리한다.

Spatial V2 실행: ResNet50 공유 spatial head, Gaussian target, FPN, 이전 원본224×224 patch feedback, global/local fusion을 구현했다. gaze 지도 사전학습은 train412장, 순차 학습·분류는 train376 / val93장이다. classifier의 train 경로는3-fold에서 사전학습부터 held-out 환자를 제외했다. 경로6종과 분류 대조군을 비교했다. sequence fusion AUROC0.781, static fusion0.763, 동일 초기화 global-only0.783, image-only grid fusion0.796으로 trajectory의 추가 분류 가치는 확인되지 않았다. batch2를 맞춘 feedback ablation의 NLL 차이도6.101→6.097로 작다. 구현 안내는 `SPATIAL_V2.md`, 결과는 `runs/spatial_v2_seed42/report.md`이다. 다음 단계는 순서 섞기·crop scale·3 seeds의 사전 정의된 검증이며 최종 test는 여전히 미사용이다.

최고 설정 백본 비교: gaze를 사용하지 않는 분산 grid16개 + global/local fusion을 유지하고 DenseNet161 / EfficientNet-B5 / ConvNeXt-Base / Swin-B / ViT-B/16 전체 fine-tuning을 완료했다. 입력 크기·split·seed42·학습률·batch2×accumulation4·최대12 epochs를 유지했다. AUROC는 ResNet50 0.796, ViT-B/16 0.753, DenseNet161 0.736, ConvNeXt-Base 0.729, EfficientNet-B5 0.709, Swin-B 0.697이다. 모든 paired AUROC 차이 CI가0을 포함하며 최종 test는 미사용이다. DenseNet161의 AUPRC0.791이 가장 높았다. 설정은 `FUSION_BACKBONES.md`, 결과는 `runs/grid_fusion_backbones_seed42/report.md`이다.

백본 HPO 완료: 기존 학습 곡선의 일반화 저하를 진단하고, train의 환자 fold0으로 inner train257 / validation119를 분리했다. backbone LR1e-6/3e-6/1e-5 screening, minimal/regularized recipe 및 warm-up0 대조군을 비교한다. 내부 최고 설정과 epoch를 선택해 전체 train376장으로 refit하며 outer val93 조기 종료는 사용하지 않는다. 내부 실험30개와 전체 train refit을 완료했다. ConvNeXt-Base0.791, Swin-B0.723으로 수치상 개선됐고 DenseNet1610.665, EfficientNet-B50.697, ViT-B/160.721은 개선되지 않았다. 기존 동일 백본 대비 paired 차이 CI는 모두0을 포함한다. 최종 test는 유지한다. `FUSION_HPO.md`와 `runs/fusion_hpo_seed42/report.md`에 관리한다.

## 12. 참고 자료

- 데이터 카드: https://huggingface.co/datasets/yankon0408/GAA-DETR-breast
- GAA-DETR: https://papers.miccai.org/miccai-2025/0738-Paper1266.html — 시선 감독을 이용한 병변 검출 연구; 본 계획의 trajectory 생성 목표와 구분한다.
- Gazeformer: https://arxiv.org/abs/2303.15274 — Transformer 기반 scanpath 예측 참고. 일반 영상의 목표 지향 시선 예측 결과가 유방촬영에 그대로 적용된다는 가정은 하지 않는다.
