# 구현 및 실행 안내

현재 시선 기반 양성·악성 분류 파이프라인의 실행 명령과 개발 검증은 [GAZE_PATCH_CLASSIFICATION_1012.md](GAZE_PATCH_CLASSIFICATION_1012.md)를 따른다. 아래는 초기 구현 기록이다.

## 구현 범위

- COCO 라벨 생성, TCIA Subject ID를 이용한 CBIS 환자 키 대조, 환자 단위 validation/cross-fitting 분할.
- 시선 배율 복원, 연속 유효 구간 분리, 순서를 유지하는 근접점 제거 및 최대 K 선택.
- 영상으로만 생성한 foreground mask와 random/grid/observed/predicted 선택, 경계 padding, 224×224 패치.
- ImageNet 사전학습 ResNet50 고정 특징, attention MIL 및 순서 Transformer 분류.
- 영상 특징에 cross-attention하는 autoregressive grid trajectory 모델과 정적 heatmap 모델.
- 내부 cross-fitting 예측 경로 생성, 경로 provenance 검사, 예측 패치 기반 분류.
- validation threshold, 환자 cluster bootstrap, paired AUROC 비교, 사후 박스 커버리지 분석.

코드는 Python/PyTorch CLI다. 현재 기본 실험은 이미지 증강 없이 고정 encoder를 사용한다. GPU 1대를 사용하며 다른 GPU 프로세스는 변경하지 않는다.

## 데이터와 현재 확정된 조건

`manifests/audit.json`에 전처리와 분할 근거가 있다. CBIS 환자 키는 로컬 TCIA 메타데이터의 Subject ID에 대조했다. 해당 파일은 병리 라벨 재검증 자료가 아니며 분류 라벨은 COCO를 따른다.

CBIS 분류 대상 600장: train 376장(benign 187 / malignant 189), validation 93장(46 / 47), 원본 Test 131장(68 / 63). 동일 P 번호의 영상은 하나의 split에만 있다. 분할 seed는 42다.

INbreast 원본 CSV의 Patient ID는 removed이고 확인한 DICOM의 PatientID도 비어 있다. 59개 주석 영상을 포함한 INbreast 전체는 `quarantine`으로 보존한다. 파일명 해시를 확정된 환자 ID로 취급하지 않는다. 향후 환자 키를 제공하면 독립 분할 및 출처 간 평가를 추가할 수 있다.

모든 RGB 영상의 세 채널이 같음을 확인했다. 입력은 grayscale을 3채널로 복제해 ImageNet 정규화를 적용한다. foreground는 grayscale 값 >10인 후보 격자다. 실제/예측 gaze가 background를 가리키면 가장 가까운 foreground 후보로 투영한다. 이 변환을 적용한 패치 중심은 feature cache에 저장한다.

원본 저자 README는 시선이 bounding-box annotation 중 수집됐다고 설명한다: https://github.com/YanKong0408/GAA-DETR#dataset . 저장 순서와 세션 병합 규칙은 확인되지 않아 모델을 `stored-order` 예측으로 해석한다.

## 실행 환경

작업 디렉터리는 `/home/njsung/NJ_BK/code`를 기준으로 한다. Python은 torch, torchvision, numpy, pandas, Pillow, scikit-learn이 필요하다. 현재 실행 환경의 패키지 버전은 `requirements-training.txt`에 기록한다. ImageNet ResNet50 V2 가중치는 기존 torch cache를 사용한다.

```bash
python gaze_mammo/scripts/prepare.py
python -m unittest discover -s gaze_mammo/tests -v
python gaze_mammo/scripts/audit_similarity.py
python gaze_mammo/scripts/run_baselines.py --device cuda:0 --name baseline_seed42
```

baseline 실행은 `configs/pilot.json`을 읽고 전체 train/validation에 대해 random/grid/observed/whole 특징 추출과 분류 학습을 진행한다. 최종 test는 이 명령에서 읽지 않는다. 결과는 `runs/baseline_seed42/comparison.csv`에 저장한다. 단계별 validation 수치이며 독립 test 성능이 아니다.

### 개별 baseline과 사후 분석

```bash
python gaze_mammo/scripts/pipeline.py extract --method observed --k 16 --output gaze_mammo/runs/observed_features --device cuda:0
python gaze_mammo/scripts/pipeline.py train-classifier --features gaze_mammo/runs/observed_features --output gaze_mammo/runs/observed_mil --device cuda:0
python gaze_mammo/scripts/analyze_patches.py --features gaze_mammo/runs/observed_features --output gaze_mammo/results/observed_coverage
```

`--crop-size 448`는 원본 448×448 crop을 encoder 입력 224×224로 resize한다. `--k 8`, `--k 32`도 동일 코드로 비교한다. 기본 grid selector는 같은 foreground 후보에서 farthest-point sampling으로 커버리지를 확보하는 결정적 spatial baseline이다.

### trajectory / heatmap 학습 및 검증 생성

```bash
python gaze_mammo/scripts/pipeline.py train-gaze --method trajectory --k 16 --manifest gaze_mammo/manifests/trajectory.csv --output gaze_mammo/runs/trajectory --device cuda:0
python gaze_mammo/scripts/pipeline.py generate --checkpoint gaze_mammo/runs/trajectory --output gaze_mammo/runs/trajectory_val --splits val --with-gaze-evaluation --device cuda:0
```

`--method heatmap`으로 정적 위치 분포 baseline을 학습한다. 학습 target은 근접점을 제거한 가장 긴 연속 유효 구간에서 선택한 좌표다. 현재 모델은 **최대 K의 축약 경로**를 학습하고 추론 시 고정 K를 생성한다. 자연 길이·종료 토큰·체류 시간·확대 배율 예측은 구현 범위에 포함되지 않는다.

trajectory 위치 grid는 24×48로, 원본 영상에서 셀당 62.5×62.5 픽셀이다. 생성 좌표는 셀 중심이다. 자유 생성은 argmax를 기본으로 하고, heatmap은 공간 이웃을 억제하며 상위 K개 중심을 선택한다. 반복 좌표와 양자화 오차는 추후 연속 offset/확률 샘플링 ablation으로 검토한다.

`generate`는 기본적으로 이미지와 image ID만 로드한다. `--with-gaze-evaluation`을 지정했을 때만 실제 시선을 불러 사후 DTW 비교를 계산한다. 실제 좌표는 생성 모델에 입력하지 않는다.

### cross-fitting → 예측 패치 분류

```bash
python gaze_mammo/scripts/run_crossfit.py --method trajectory --k 16 --epochs 50 --output gaze_mammo/runs/crossfit_trajectory --device cuda:0
python gaze_mammo/scripts/pipeline.py extract --method trajectory --predictions gaze_mammo/runs/crossfit_trajectory/predictions.json --output gaze_mammo/runs/predicted_features --device cuda:0
python gaze_mammo/scripts/pipeline.py train-classifier --features gaze_mammo/runs/predicted_features --output gaze_mammo/runs/predicted_mil --device cuda:0
python gaze_mammo/scripts/pipeline.py train-classifier --features gaze_mammo/runs/predicted_features --aggregation sequence --output gaze_mammo/runs/predicted_sequence --device cuda:0
python gaze_mammo/scripts/pipeline.py train-classifier --features gaze_mammo/runs/predicted_features --aggregation shuffled --output gaze_mammo/runs/predicted_shuffled --device cuda:0
```

cross-fitting은 train 환자 내부의 3개 fold 각각을 제외한 predictor로 held-out train 환자 경로를 생성한다. 전체 train으로 학습한 predictor가 validation 경로를 생성한다. 예측 JSON 옆의 provenance에는 predictor 학습 환자와 생성 이미지 ID를 남긴다. feature extraction은 환자 중복 또는 누락된 provenance를 거부한다. fold 수를 바꾸면 prepare와 cross-fitting에 같은 값을 적용해야 한다.

순서 모델은 동일한 feature cache를 사용한다. `shuffled` 조건은 학습과 평가에서 valid 패치 순서를 섞는다. 추론 seed를 고정해 재현하며, 최종 연구에서는 여러 permutation의 결과도 확인한다.

### paired 비교

```bash
python gaze_mammo/scripts/pipeline.py compare --first gaze_mammo/runs/baseline_seed42/observed/validation_predictions.csv --second gaze_mammo/runs/baseline_seed42/random/validation_predictions.csv --output gaze_mammo/results/gaze_vs_random.json
```

두 CSV의 영상 ID·라벨·환자 키가 동일해야 한다. 환자를 재표집하고 방법 간 AUROC 차이의 95% 신뢰구간을 계산한다.

### 잠긴 test 평가

모델과 설정을 validation으로 결정한 뒤 수행한다. 기본 baseline도 test 특징 추출은 명시적 `--include-test`가 필요하다.

```bash
python gaze_mammo/scripts/pipeline.py extract --method observed --include-test --output gaze_mammo/runs/observed_with_test_features --device cuda:0
python gaze_mammo/scripts/pipeline.py evaluate --features gaze_mammo/runs/observed_with_test_features --checkpoint gaze_mammo/runs/baseline_seed42/observed --split test --final-test --output gaze_mammo/results/observed_test --device cuda:0
```

예측 경로 조건은 full-train predictor로 `generate --splits test --final-test`를 실행하고 기존 cross-fit/validation 예측과 provenance를 병합한 뒤 test 특징을 만든다. test에서는 threshold를 바꾸지 않으며 checkpoint 폴더의 validation threshold를 사용한다.

## 파일과 재현성

### 단일 영상 추론

```bash
python gaze_mammo/scripts/predict_image.py --image PATH_TO_IMAGE.png --gaze-checkpoint gaze_mammo/runs/crossfit_trajectory/full_train --classifier-checkpoint gaze_mammo/runs/predicted_mil --output gaze_mammo/results/image_prediction.json --device cuda:0
```

영상 픽셀만 입력하고 예측 경로, 실제 패치 중심, patch attention과 malignant 확률을 저장한다. 이 명령은 실제 시선·박스·라벨 파일을 읽지 않는다. 새 영상은 현재 데이터와 동일한 1500×3000 전처리를 요구한다. 형식이 다른 원본 의료 영상을 무조건 resize하여 적용하지 않는다.

### 전체 연결 검사

```bash
python gaze_mammo/scripts/smoke_integration.py
```

실제 데이터의 train 12장 / validation 8장을 사용해 CPU에서 1 epoch의 3-fold cross-fitting, 예측 패치 추출, MIL/sequence/shuffled 학습, 평가, 박스 커버리지까지 검사한다. 작은 실행에서 얻은 수치는 연구 성능으로 해석하지 않는다. 완료 표시는 `runs/integration_smoke/completed.json`이다.

픽셀 중복과 32×64 썸네일 상관 >0.995의 후보 검사는 `manifests/similarity_audit.json`에 저장한다. 현재 후보는 없지만 이 검사는 모든 유사 영상 중복의 부재를 보장하지 않는다.

- `manifests/all.csv`: 미주석·quarantine 포함 원본 전체 manifest.
- `manifests/classification.csv`, `trajectory.csv`: 학습용 라벨 및 분할.
- `manifests/gaze_sequences.json`, `instances_clipped.json`, `channel_audit.csv`: 파생 전처리와 품질 검사.
- feature cache: 특징·유효 mask·실제 사용 중심·영상 ID, 추출 설정과 manifest.
- 학습 결과: `best.pt`, `config.json`, `history.json`, validation 예측과 threshold.
- predictor 결과: 좌표 JSON, 학습 환자 provenance, 사후 경로 지표.
- 평가 결과: 확률·attention CSV, 환자 bootstrap 지표, paired 비교, patch coverage.

모든 명령은 seed와 manifest hash를 기록한다. 모델은 파일명·병변 박스·진단 문자열을 입력으로 사용하지 않는다. 현재 개발 실험은 seed 42이며 연구 최종 결과에는 주요 비교의 3 seeds와 독립 test 평가가 추가로 필요하다.
