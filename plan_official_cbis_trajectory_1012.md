# 새 trajectory 학습 → 전체 CBIS-DDSM 분류 → 공식 TEST 평가

2026-10-12. 사용자 요청에 따라 계획과 개발 뒤 tmux에서 본학습까지 실행한다. 백본 구조는 ConvNeXt-Base를 유지한다. 기존 모델·결과·source hash를 덮어쓰지 않는다. 이전 NLL/확보율 반복 목표는 재개하지 않는다.

실제 cohort 검증 및 개발 후 tmux `official_cbis_trajectory_1012`에서 실행을 시작했다. 확정 분류 train2037/val356장(1034/183환자), gaze train427/val79장이다. 신규 단위 검증20개와 실제8장 trajectory/classifier smoke가 통과했다. [실행 조건](runs/official_cbis_trajectory_1012/run_manifest.json), [실행 상태](runs/official_cbis_trajectory_1012/state.json), [cohort 감사](runs/official_cbis_trajectory_1012/audit.json), [실제 smoke 감사](runs/official_cbis_trajectory_1012/smoke/parent_verification.json)를 확인한다. `tmux attach -t official_cbis_trajectory_1012`의 monitor 창에서 로그를 볼 수 있다.

실행 단계는 gaze PNG canonical변환 → gaze 전체 특징 cache → baseline/improved trajectory 학습 → 전체 분류 특징 cache →15head학습 →freeze →공식TEST645장 전처리/추론/평가다. TRAIN full-DICOM 전처리는 CPU에서 trajectory 작업과 병행하고 분류 특징 추출 전에 완료 여부를 검사한다. 오류 발생 시 후속 단계를 중단한다. 최종 결과는 `runs/official_cbis_trajectory_1012/test_evaluation/`에 저장한다.

## 먼저 발견한 benchmark 제약

NAS의 TCIA metadata에는 full mammogram TRAIN2457장/1248환자, TEST645장/349환자가 있다. MorphHR 원문은 TRAIN2458장 및 TEST645장을 보고한다. 로컬 routing JSON의 `Calc-Training_P_00673_LEFT_MLO`는 metadata와 DICOM에 없어1장 차이가 난다. 누락 영상을 임의 생성하거나 다른 영상으로 대체하지 않는다.

공식 TRAIN/TEST는31개 P-number 환자가 겹친다. 현재 시선 source의 train/selection 환자도 일부 공식 TEST와 겹친다. 따라서 이미 학습한6expert trajectory와 old ROI checkpoint는 이번 benchmark 초기화에 사용할 수 없다. **공식 TEST645장은 그대로 유지하고, 그349환자의 TRAIN64장을 제외**한다. 사용 가능한 TRAIN은2393장/1217환자다. 이는 공식 TEST 영상 목록을 유지하면서 TRAIN을 더 엄격하게 정리한 별도 protocol이며 MorphHR와 train 조건까지 완전히 같다고 표현하지 않는다.

공식4개 pathology CSV를 TCIA에서 내려받아 URL/SHA를 저장했다. Benign 및 BENIGN_WITHOUT_CALLBACK은0, MALIGNANT는1이다. 같은 full-image에 여러 병변이 있으면 malignant가 하나라도 있으면1로 한다. Label은 ROI/filename의 진단 문자열에서 추론하지 않는다. ROI/cropped DICOM은 분류 입력으로 사용하지 않는다.

## 데이터 및 분할

1. `runs/official_cbis_trajectory_1012/metadata/`의 TCIA4CSV + NAS metadata.csv로 subject/full-series를 join한다. 누락·다중매칭·라벨 충돌 및 중복은 명시적 audit를 남긴다. 공식 TEST pixel/gaze는 cohort 감사나 학습에서 읽지 않는다.
2. TEST환자를 제외한 TRAIN의 환자를 pathology max로 stratify하고 seed42의85:15 train/validation을 고정한다. 전체 분류셋과 gaze subset이 **동일한 patient split**을 사용한다. 기존 gaze split을 강제로 재사용하지 않는다. 새 split에서는 시선 및 분류 head를 모두 새로 학습한다.
3. 시선 annotation이 있는 기존 GAA-CBIS PNG 중 공식 TRAIN환자로 확인된 행만 사용하고 새 split을 부여한다. INbreast/기존 Test filename/공식TEST환자는 제외한다. 시선 label 미확인 사례도 gaze loss에는 사용할 수 있지만 분류 label은 공식 pathology만 사용한다.
4. GAA-PNG 및 raw full-DICOM을 image-only foreground bounding box crop + aspect-preserving padding으로1500×3000 canonical canvas에 정규화한다. DICOM intensity percentile clipping 및 MONOCHROME1 inversion을 기록한다. 실제 gaze의 원래 PNG pixel좌표는 같은 crop/resize/pad affine로 변환한다. 학습과 추론 crop224는 canonical pixel 단위다. 병변 mask/box를 전처리에 사용하지 않는다. 전처리는 공식benchmark 논문과 다른 자체 protocol로 공개한다.
5. 원본 NAS는 읽기만 하고 변환 PNG/새 manifest/cache는 workspace 아래 새 경로에 저장한다. TRAIN/val만 먼저 변환하며 TEST 변환·특징 추출은 모든 선택이 끝난 후 한다.

## Trajectory를 새로 학습하는 방식

- ImageNet ConvNeXt-Base 원본 `/home/njsung/.cache/torch/hub/checkpoints/convnext_base-6075fbad.pth`를 초기화로 사용한다. 이를 기존 구조 `GazePyramid`로 복원하며 의료/gaze 학습 checkpoint를 초기화로 사용하지 않는다. ImageNet visual은 동결한다. 분류 arm 모두 이 same visual을 공유한다.
- 재사용 가능한 non-pinned architecture 함수는 `Trajectory1011`/`GazeNLLAdaptation1011`, grid24×48, fovealTrue/FPNTrue/motionTrue/spatial_historyTrue/history4/dim128, whole256×512, 실제 과거224crop1024vector다. 기존 코드와 hash 고정된 파일은 수정하지 않는다.
- 원래 sample order를 유지하고 min-distance32로 segments를 만든다. Canonical validation은 seed42 legacy16로 고정한다. Train에는 원래 chronology pool을 저장하고 window선택마다 실제 이전 point의 crop vector를 일치시킨다. 미래 crop 또는 현재 target crop은 decoder 입력으로 주지 않는다.
- 같은 visual과 초기 head(seed42)를 사용하는 두 방식만 사전 지정한다:
  - **baseline:** canonical legacy16, hard next-cell NLL, 최대40epochs/patience10.
  - **improved:** train-only mixed legacy/contiguous16, burn-in8, 같은 chronology에서 window augmentation; `.8 hardNLL + .2 Gaussian32px softCE`, 최대40epochs/patience10. Ground truth prefix는 gaze 학습/teacher likelihood에만 쓴다.
- AdamW head lr3e-4/wd.01, batch8, clip1. Source epoch0를 포함해 canonical validation hardNLL이 가장 낮은 checkpoint를 선택한다. 동률은 이른 epoch. NLL 개선을 보장하거나 분류 개선과 동일시하지 않는다.
- Image-only inference는 empty prefix에서16점 sampling(seed42+opaque numericimageID)하며, 실제 이전 예측점 crop을 feedback으로 사용한다. Teacher forcing으로 분류용 경로를 생성하지 않는다. 두 selected trajectory를 모두 동결한다.

## 더 큰 분류셋에서 학습

사용 가능한 공식 TRAIN2393장을 새 train/validation에 따라 활용한다. Gaze annotation이 없는 영상도 image-only trajectory를 생성해 분류 학습에 사용한다. 전체 global 특징 및224patch 특징은 동일한 원본ImageNet ConvNeXt-Base1024encoder로 추출한다.

| arm | 패치 입력 |
|---|---|
| global | 전체 영상만 |
| random | image-only foreground 후보에서16개 |
| grid | foreground farthest-point16개 |
| gaze_baseline | 새 baseline trajectory16개 |
| gaze_improved | 새 improved trajectory16개 |

오염된 old guided/ROI14 모델을 사용하는 proposal/hybrid arm은 이번 단계에서 제외한다. Primary K16/crop224/pathseed42, headseeds42/43/44. 각 seed의 global head를 새로 학습한 뒤 동일 global reference를 freeze하고 모든 patch arm에 mean residual head를 적용한다. 기존 `GazeDiagnosisHead`/`GazeAttention1011` helper를 재사용하되 새로운 cache/checkpoint 종류와 strict protocol을 사용한다. Global128/dropout.3, local64/dropout.2, BCE + local auxiliary.25 + residual square.01, AdamW3e-4/wd.01/batch32/max60/patience12, patch epoch0 global 후보를 유지한다.

Validation AUROC로 head epoch를 선택한다. Primary method는 사전 지정된gaze_improved이고 baseline trajectory/global/grid/random도 모두 같은 TEST에서 평가한다. 모델 family를 TEST 점수로 고르지 않는다. 단일head3seed mean/SD와3-head probability ensemble을 구분한다. Threshold=.5 고정, TTA 없음. Accuracy만 개선됐다고 trajectory 기여를 주장하지 않고 AUROC 및matched difference도 본다.

## 공식 TEST 평가

모든 trajectory/head/threshold/seed/protocol hash를 freeze한 뒤645개 full-DICOM을 같은 image-only 전처리로 변환한다. GT gaze/ROI를 읽지 않고 경로/특징을 생성해 모든 사전 지정 arm을1회 평가한다. Benign/Malignant pathology는 후처리 지표 계산에만 사용한다.

AUROC/AUPRC/accuracy/balanced accuracy/민감도/특이도/F1/Brier/confusionmatrix를 저장한다. 349환자 단위 paired bootstrap5000(seed42)로 gaze_improved−baseline/global/grid의 AUROC차이와95%CI를 구한다. TEST이미지subset은 과거 gaze/ROI평가에서 일부 관측됐으므로 전체 cohort를 완전히새미사용데이터라고 표현하지 않는다. 새 실행의 학습·선택에는 사용하지 않는다.

[MorphHR 원문](https://doi.org/10.1016/j.media.2022.102618)의 공식TEST 단일.7964/TTA.8187/ensemble+TTA.8313과 대조할 수 있다. 다만 TRAIN환자purge64/로컬누락1/전처리/초기화/3seed vs4model/TTA조건 차이를 같이 표시하고 동일protocol재현이나통계적우월성으로주장하지 않는다.

## 개발 역할과 실행 순서

개발은 사용자 지정 GPT-6.1-Sol low 서브에이전트에 위임한다. Parent는 cohort 결정·계획·리뷰·실제 smoke·tmux runner 및 실행 확인을 담당한다.

1. Data agent: `src/official_cbis_data_1012.py`, `scripts/prepare_official_cbis_1012.py`, tests. `build_cohort(metadata_dir, raw_root, output)`; protocol/trainval.csv/test.csv/gaze.csv; `canonical_image(image_or_dicom)->(PIL, affine_metadata)`; `transform_points(points, metadata)`; CLI cohort/convert --split trainval|test. Stable unique numericimageIDs, fullimagepath/sourceSN/pathologylabel/patient/newsplit 및 gaze source metadata.
2. Trajectory agent: `src/official_cbis_trajectory_1012.py`, `scripts/train_official_cbis_trajectory_1012.py`, tests. CLI cache/train; safe gaze payload matching allowed filenames만 parse; fresh pretrainedvisual; originalchronology pool/cache; baseline/improved selected compactheads; `load_selected_trajectory(checkpoint,device)`와pixel-only `rollout(model,image,image_id,seed42)` 제공. Train/val/test sourceprovenance와 backboneSHA검증.
3. Classifier agent: `src/official_cbis_classifier_1012.py`, `scripts/run_official_cbis_classifier_1012.py`, tests. CLI cache/train/freeze/evaluate; selected two trajectories 공통 frozenImageNetvisual,5arm×3head train, finalTEST automaticstage guard, compactheads와posthocbootstrapmetrics. Source/cache/protocol/image/SHA 및 family/seed/mask/GT-inputguard.

모든 output은 새 폴더로 만들고 partial cache는 state/row hashes를 남긴다. TRAIN/TEST overlap/unknownlabel/missingDICOM/affine좌표 오류/sourcechange 발생 시 dependentstage를 중단한다. TEST출력은최종freeze이전생성금지.

Durable tmux 실행: cohort audit → TRAIN/val conversion 및 gaze cache → 실제 small training smoke → fullbaseline/improvedtrajectory training → officialTRAIN classificationfeaturecache →15classifiers → freeze →645TEST conversion/cache → posthocmetrics/audit. 개발후검증된runner가후속단계를자동실행하고각stage로그와failure상태를저장한다. GPU0 trajectory/extraction, CPU cachedclassifiertraining; 다른 GPU세션은 변경하지 않는다. 작업목표는 학습및공식TEST평가실행이며10%개선달성까지무제한반복하지않는다.
