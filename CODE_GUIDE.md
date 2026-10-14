# 코드 읽기와 실험 재현 안내

처음 읽는다면 데이터 → 시선 경로 → 패치 특징 → 분류 → 평가 순서로 살펴보세요. 날짜·round가 붙은 파일은 실험 시점의 구현을 보존한 것이며, 모든 파일이 최신 실행에 필요한 것은 아닙니다.

## 1. 입력과 실제 크롭

[src/data.py](src/data.py)의 `patch_tensor`는 canonical 영상 좌표를 반올림한 뒤 정사각형 영역을 잘라 224 × 224로 resize하고 ImageNet 정규화를 적용합니다. 영상 바깥 부분은 PIL 크롭의 검은 패딩을 사용합니다. `crop_size`는 원본 영상에서 잘라낼 영역 크기입니다.

[scripts/prepare_official_cbis_preprocess_v2_1012.py](scripts/prepare_official_cbis_preprocess_v2_1012.py)는 공식 코호트 구성과 DICOM 전처리·시선 좌표 정합 단계의 진입점입니다. 데이터는 Git에 포함하지 않습니다.

## 2. 시선 모델과 특징 추출

[src/gaze_backbones.py](src/gaze_backbones.py)에서 시각 인코더를 확인할 수 있습니다. [시선 학습](scripts/train_gaze_lesion_trajectory_1013.py)과 [경로 예측](scripts/predict_gaze_extent_round12_1013.py)은 실제 시선 학습 감독과 영상 단독 예측을 구분합니다.

기본 파이프라인은 [특징 추출](src/gaze_patch_features_1012.py), [분류 모델](src/gaze_patch_classifier_1012.py), [분류 학습 CLI](scripts/train_gaze_patch_classifier_1012.py)로 연결됩니다. 사용 예시는 [기본 실행 문서](GAZE_PATCH_CLASSIFICATION_1012.md)에 있습니다. 해당 문서의 초기 smoke·이전 코호트와 README의 최신 코호트 결과는 구분해야 합니다.

## 3. 현재 비교 실험

| 실험 | 실행 진입점 | 무엇을 비교하는가 |
|---|---|---|
| Grid14 패치 개수 | [run_grid14_patchcount_ablation_1013.py](scripts/run_grid14_patchcount_ablation_1013.py) | 고정 Grid14에 시선/랜덤 패치를 추가 |
| Round18 | [compare_backbones_round18_1014.py](scripts/compare_backbones_round18_1014.py) | ConvNeXt / DenseNet / ResNet, 동일 패치 위치 |
| Round19 | [train_frozen_attention_round19_1014.py](scripts/train_frozen_attention_round19_1014.py) | 동결 global + 작은 시선 attention residual |
| Round20 추출 | [extract_gaze_context_round20_1014.py](scripts/extract_gaze_context_round20_1014.py) | 동일 중심, 원본 크롭 크기 448/896 |
| Round20 학습 | [train_gaze_context_round20_1014.py](scripts/train_gaze_context_round20_1014.py) | TRAIN 내부에서 224/448/896 설정 선택 |
| Round20 검증 | [audit_gaze_context_round20_1014.py](scripts/audit_gaze_context_round20_1014.py) | 예측 CSV 지표와 분리·출처 검증 |

Round20은 일시중지 상태입니다. CLI 안내를 읽는 것과 학습 스크립트를 실행하는 것을 구분하세요. 일부 round 스크립트는 인자 없이 고정 로컬 경로를 읽고 쓰며, 단순 실행만으로 학습이 시작됩니다.

## 4. 선행 산출물과 재개 조건

최신 실험은 다음 로컬 산출물에 의존합니다. GitHub 코드만 내려받으면 이 파일들은 존재하지 않습니다.

- `runs/official_cbis_preprocess_v2_1012/`: canonical 영상과 분리·좌표 정합 정보.
- `runs/official_cbis_v2_experiment_1012/development.pt`: 개발 코호트의 특징·행 메타데이터.
- `runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42/`: 선택 시선 모델.
- `runs/gaze_focus_auroc88_1013/round12_gaze_extent_pilot42/`: 영상 단독 OOF 시선 예측·패치 선택.
- `round09_spatial_mil_pilot42/global_only/`: TRAIN-inner / fullTRAIN global 모델.
- `round10_frozen_global_residual_pilot42/`: 두 global 모델의 문맥 캐시.
- `round18_backbone_comparison_pilot42/`: 고정 시선·랜덤 좌표와 기존 224 특징.
- `round20_gaze_context_pilot42/`: 중단된 추출의 원자적 기록.

원본 경로, 선행 checkpoint, 캐시와 source SHA가 일치해야 합니다. 변경된 코드에 이전 캐시를 그대로 연결하면 재현 검증에 실패할 수 있습니다. 정리한 과거 run은 재실행 전에 다시 생성해야 합니다.

## 5. 결과를 읽는 방법

공개 집계는 [reports/experiment_summary.json](reports/experiment_summary.json)에 있습니다. 로컬 결과의 주요 파일은 `final_report.json`, `summary.csv`, 각 비교군의 예측 CSV, 독립 감사 보고서입니다.

- `seed42` pilot과 seed42/43/44 평균을 구분합니다.
- `singleview`, 특징/문맥 통합, 확률 평균 통합은 서로 다른 평가입니다.
- 실제 유효 패치 수와 padding mask를 확인합니다.
- epoch·설정 선택이 TRAIN 내부에서 수행됐는지 확인합니다.
- 높은 시선 병변 확보율이 분류 이득을 의미하는지 별도 대조군으로 검증합니다.

## 6. Git 공개 범위

`.gitignore`는 `data/`, `runs/`, 환자 메타데이터, 모델 가중치, 대형 캐시, 로그, 로컬 환경과 다운로드한 참고 구현·논문을 제외합니다. `reports/`는 환자 ID나 영상별 예측을 포함하지 않는 집계 결과용입니다. 실제 업로드 전에 Git 포함 파일 목록을 확인해야 합니다.
