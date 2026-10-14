# 시선 예측 → 패치 추출 → 양성·악성 분류

2026-10-12. 사용자의 새 연구 방향에 맞춰 정리 후 작성한 개발 명세다. 이전 NLL/병변 확보율 반복 개선 목표는 중단 상태로 유지한다. 이번 단계는 파이프라인 개발과 기능 검증이며, 대규모 재학습이나 TEST 재선택은 수행하지 않는다.

## 근거와 주 가설

기존 global 분류 validation AUROC는 0.7586, GRU gaze 결합은 0.7331이었다. 중복 보정 평균 패치 + global residual은 단일 head 3 seed 평균 0.7727±0.0047이었다. 학습된 attention이 평균 pooling보다 우수하다는 근거는 없었다. 현재 고정 시선 모델은 TEST NLL 5.8878, 실제224px 크롭 병변50% 확보율74.43%를 보였다. 병변 확보는 보조 지표이며 분류 개선을 보장하지 않는다.

주 가설은 **예측 시선으로 선택한 패치가 전체 영상 단독 분류를 개선하고, 동일 패치 수·분류 백본·분류 head 조건에서 grid·proposal 기반 선택보다 양성·악성 분류 성능을 개선한다**는 것이다. 패치 수를 맞추는 비교는 patch arm끼리이며 global은 패치가 없는 기준점이다. 현재 ROI14+RAW2는 대부분 proposal이므로 순수 시선과 구분한다. Proposal/hybrid만 개선하면 시선 경로 자체의 효과로 주장하지 않는다.

## 고정 입력 및 데이터

- 시선 source: `runs/dual_target_1011/mammo_clip_unified_runtime/config.json`, semantic SHA `0bc7e224c6dafe36b8be8b12dbe3694af3b272f22cfedca3b39795d56522d239`. 전체 source/weights를 동결한다. 기존 hash 고정 파일은 수정하지 않는다.
- CBIS-DDSM train376장(benign187/malignant189), validation93장(46/47). 기존 환자 split 및 benign=0/malignant-or-mixed=1을 유지한다. unknown=-1 제외. INbreast quarantine 및 기존 TEST는 개발 cache/학습/선택에서 제외한다.
- TEST150은 이미 시선·병변 결과를 확인했으므로 새로운 미사용 TEST로 표현하지 않는다. 최종 일반화 주장은 별도 미사용 환자 cohort/외부 데이터에서 검증한다. 초기 개발은 반복 사용된 validation의 탐색적 결과다.
- 현재 시선 모델도 같은 train으로 학습됐으므로 frozen transfer이며 OOF/cross-fitting이 아니다. Nested source checkpoint의 train/selection 환자가 허용 train/val 밖이면 거부한다.
- GT gaze/병변 박스/진단 라벨/filename의 진단 문자열은 예측 시선·패치 선택의 입력으로 사용하지 않는다. 이번 첫 구현에서는 실제 시선 oracle arm을 구현하지 않는다. 필요하면 별도 연구용 reference로 추가하며 배포 성능과 분리한다.

## 동일 패치 예산 비교

| arm | 패치 선택 | 분류 입력 |
|---|---|---|
| global | 없음 | 전체 영상 |
| random | 영상에서 추정한 foreground grid 안에서 seed 고정 무작위16 | 전체 영상+패치 |
| grid | 기존 farthest-point foreground grid16 | 전체 영상+패치 |
| proposal | 원래 guided relevance+offset+NMS로16개, RAW gaze 선택 없음 | 전체 영상+패치 |
| gaze | 현재 RAW 확률로16단계 자율 생성, 별도 ROI 편향 없음 | 전체 영상+패치 |
| hybrid | 기존 ROI14+RAW2 그대로 | 전체 영상+패치 |

Primary K=16, crop224, 경로 생성 seed42와 image ID로 CPU RNG를 고정한다. 분류 head seed42/43/44는 경로 seed와 구분한다. K=4/8은 동일16개 cache의 prefix를 쓰는 사전 지정 보조 실험 옵션이며 본 실행을 자동 확장하지 않는다. 패치 수만 같다고 전체 추론 FLOPs가 같다고 주장하지 않는다. 시선 모델은 6개 전체 영상 encoder와 3개 과거 패치 encoder를 사용하므로 selector 시간·classification 특징 추출 시간·patch 수를 따로 기록한다.

시선 생성은 이전에 **예측한** 좌표와 그 위치에서 실제 추출한 과거 패치만 사용한다. 첫 단계는 empty prefix/과거 vector 없음. GT teacher forcing은 분류 cache에서 금지한다. Proposal16은 기존 후보 offset/relevance/NMS를 재사용하되 예약 수16으로 만든 별도 선택 정책이며 현재 source config를 덮어쓰지 않는다. Hybrid는 원래 unified rollout 그대로다.

## 분류 특징과 head

첫 단계는 현재 probability bundle의 원래 control visual encoder(1024차원)를 모든 arm에서 공통 분류 백본으로 사용한다. `classification_encoder(probability)`가 상위 independent crop bundle의 `visual.parent`를 따라 원래 `PatchBundle.encoders[0]`를 찾는다. 같은 encoder로 전체 영상과 실제224px 패치를 인코딩한다. 경로 생성에 필요한 나머지 과거 특징은 별도 처리한다. 선택 정책 차이 외에 arm별 분류 encoder나 해상도가 달라지지 않아야 한다.

각 classifier seed에서 새 global head를 동일 train376/val93 특징으로 학습한다. 기존 old-improved global checkpoint를 다른 encoder에 그대로 연결하지 않는다. 선정한 global head를 freeze한 뒤 모든 patch arm이 이를 공통 기준점으로 쓴다. 기존 `GazeDiagnosisHead`와 `GazeAttention1011`을 재사용하고 pinned source는 수정하지 않는다.

Primary pooling은 기존 증거가 가장 단순하게 지지한 **중복 보정 mean residual**이다. 비교 arm에서 head 구조를 동일하게 유지한다. 좌표가 같은 token은 총 가중치가 반복 횟수만으로 늘지 않도록 보정한다. 전체 영상 logit + alpha×local delta, delta0 초기화, all-invalid는 정확히 global fallback. 기존 image-level auxiliary BCE .25 + residual 제곱 .01을 재사용한다. Patch별 malignant pseudo-label은 부여하지 않는다.

학습 기본값: dim64, dropout.2, global dim128/dropout.3, AdamW lr3e-4/wd.01, batch32, 최대60 epoch, patience12, gradient clip1. Patch epoch0 global 동일 예측을 선택 후보에 포함한다. Best 선택은 validation AUROC, 동률이면 먼저 나온 epoch다. 기준점 보호가 TEST 향상을 보장하는 것은 아니다. Gated/contextual attention은 이후 별도 pooling ablation 옵션이며 첫 selector 비교에 섞지 않는다.

## 지표와 해석

- Primary AUROC, 함께 AUPRC/accuracy/balanced accuracy/sensitivity/specificity/F1/Brier/confusion matrix. Threshold=.5 고정, TEST threshold tuning 없음.
- 동일 환자의 모든 이미지를 포함하는 paired patient bootstrap 5,000회(seed42)로 arm−global, gaze−grid, gaze−proposal 차이와 CI를 보고한다. Validation selection bias를 제거한다고 주장하지 않는다.
- Seed별 결과와 단일 head mean/SD, 별도로 3-head probability ensemble을 구분한다. Smoke는 성능 근거가 아니다.
- 이미지별 좌표, 실제 crop rectangles, 선택 시간, 분류 probability, baseline/local correction을 기록한다. Attention이나 signed contribution은 악성 확률/병변 정답으로 표시하지 않는다.

## 후속 실험 순서

1. 이번 개발을 검증한 뒤 전체 train/val의 image-only cache를 한 번 추출하고, 고정된6arm×3head-seed 본 비교를 실행한다. 전체 과거 모델을 다시 학습하지 않는다.
2. 순수 gaze가 grid/proposal보다 낫지 않으면 선택 경로 기여를 주장하지 않는다. 우선 선택 실패와 global/local 기여를 진단한다. 의료 사전학습 patch encoder 또는 제한적인 patch encoder fine-tuning은 모든 patch arm에 같은 조건으로 적용하는 별도 단계다. 새 encoder를 gaze arm에만 적용해 selector 효과와 혼합하지 않는다.
3. Mean pooling 이후에만 learned attention/순서 기반 pooling을 별도 ablation으로 평가한다. Mean pooling은 순열 불변이므로 같은 패치를 섞는 실험에서 차이가 없어야 한다. 순서의 분류 기여는 순서 민감 head에서 검증한다.
4. 최종 설정을 고정한 후 미사용 환자 cohort에서 분류 AUROC 및 정확도 개선을 검증한다. 기존 TEST를 재사용하면 이미 관측된 cohort라는 사실을 명시하고 새로운 독립 검증이라고 부르지 않는다.

## 개발 산출물과 역할

1. Extractor agent: `src/gaze_patch_features_1012.py`, `scripts/cache_gaze_patch_features_1012.py`, `tests/test_gaze_patch_features_1012.py`. Pixel-only selectors/동일 encoder/cache/provenance/인구 검증. API `load_frozen_source(config_path, device) -> (probability,guided,offset,payload)`, `extract_image_features(bundle,image,image_id,seed=42,arms=...) -> dict`, `validate_development_population(frame,payload) -> selected_frame`. 출력은 `global_features[1024]`, arm별 `{features[16,1024], coordinates[16,2], rectangles[16,4], mask[16]}`와 timing/provenance. Cache schema `kind=gaze_patch_classification_features_1012`, `rows`, `tensors`(global_features, `{arm}_features/{arm}_coordinates/{arm}_mask/{arm}_rectangles`), `provenance`. Smoke는 metadata 순서 train/val 각 클래스2장(총8장), 전체469장과 명확히 구분한다.
2. Classifier agent: `scripts/train_gaze_patch_classifier_1012.py`, `scripts/predict_gaze_patch_classifier_1012.py`, `tests/test_gaze_patch_classifier_1012.py`, 필요하면 새 non-pinned helper module. Cache에 맞춰 global+5arm, 저장 시 head만, original source/encoder/global SHA 정합. CLI는 새 output만 허용하고 소스 cache를 복제하지 않는다. 이미지 단독 예측은 manifest/라벨/gaze/박스를 읽지 않는다. 분류 API는 extractor의 표준 출력만 소비한다.
3. Parent: 정리·계획 작성, 구현 리뷰, 독립 테스트, 실제 GPU8장 extract 및 CPU2epoch 기능 smoke, source unchanged 검증, README/실행 안내.

개발 agent는 사용자 지정 **GPT-6.1-Sol low**를 사용한다. 두 agent의 작업 파일은 분리한다. 이번 요청에서 본 실험의 전체 cache 추출/3seed 본학습은 자동 시작하지 않으며, 실행 가능한 명령과 smoke 결과를 남긴다.

## 필수 검증

- TEST/quarantine/unknown 및 환자 overlap 거부; nested source provenance 검사. Smoke fixture와 full population 엄격 구분.
- Label/GT gaze/GT box를 전달할 수 없는 extraction API. 같은 pixel/image ID/seed에서 label이나 파일 이름 문자열이 바뀌어도 selector 결과가 같아야 한다.
- RAW gaze future-free prefix/실제 과거 crop, 정확한224 rounding, proposal16/hybrid14+2 metadata 구분, deterministic CPU RNG, 16개 유효 token.
- 모든 arm에서 동일 분류 encoder; 전체 영상 feature arm 독립. Cache source/config/image SHA와 topology, shapes, finite values 검증.
- Mean residual permutation/duplicate 불변성, all-mask fallback, epoch0 exact global, freeze gradient/state, checkpoint roundtrip, mislabeled row/cache/provenance 거부.
- Classification metrics에는 양쪽 class 필요; seed/ensemble/paired patient bootstrap 구분. Train에는 val/test gradient 없음.
- 실제 smoke source restore + train4/val4 + 각 arm2epoch + pixel-only 단독 추론. 기존 classifier 회귀 테스트와 새 테스트를 실행한다.

MIL의 bag-label 기반 pooling 출발점은 [Ilse et al., ICML2018](https://proceedings.mlr.press/v80/ilse18a.html)이며, 현재 설계가 해당 논문 또는 다른 SOTA의 성능을 재현한다는 주장은 하지 않는다.
