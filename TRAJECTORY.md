# 영상 → gaze trajectory 모델

## 학습 목표

전체 유방촬영 영상을 입력으로 받아 최대 16개의 축약 시선 지점을 순서대로 생성한다. 타깃은 저장된 gaze 좌표열이며 각 점의 확대 배율을 복원한다. 시간 간격·체류 시간은 예측하지 않는다. 추론에서는 실제 시선이나 병변 박스를 입력하지 않는다.

```text
1500×3000 mammogram
    ↓ grayscale / ImageNet normalization / 256×512
사전학습 spatial backbone
    ↓ spatial feature tokens + 2D position encoding
128차원 Transformer decoder (2 layers, 4 heads)
    ↓ BOS → location₁ → location₂ → … → location₁₆
24×48 grid 확률분포 → 정규화 좌표
```

학습은 teacher forcing과 grid cross-entropy를 사용한다. target의 미래 위치를 참조하지 못하도록 causal mask를 적용한다. 패딩된 target은 loss에서 제외한다. BOS에서 첫 위치를 예측하며 실제 첫 좌표도 추론에 제공하지 않는다.

백본의 출력 채널 수를 자동으로 처리한다. CNN 출력은 NCHW, Swin 출력은 LayerNorm 후 NCHW로 변환해 같은 decoder 인터페이스를 사용한다. 기본 비교는 백본을 고정하고 projection·position·decoder·output head만 학습한다. 고정 특징을 메모리에 캐시하여 동일 encoder 계산을 매 epoch 반복하지 않는다.

## 백본 비교 조건

| 백본 | 유형 | 출력 채널 |
|---|---|---:|
| ResNet50 | CNN | 2048 |
| DenseNet121 | CNN | 1024 |
| EfficientNet-B0 | CNN | 1280 |
| ConvNeXt-Tiny | CNN | 768 |
| Swin-T | 계층적 Transformer | 768 |

모두 ImageNet 사전학습 가중치를 사용한다. CBIS-DDSM train 376장 / validation 93장, seed 42, 입력 256×512, K=16, grid=24×48, decoder dimension=128, batch=32, AdamW lr=0.001, 최대 50 epochs, validation loss patience=10으로 비교한다. 출력 projection의 입력 채널 차이로 전체 학습 파라미터 수는 약간 다르며 결과에 기록한다.

INbreast는 환자 ID가 미확인이라 quarantine 상태다. 원본 Test 131장은 이번 백본 비교에서 사용하지 않는다. 3 seeds와 백본 미세 조정은 후속 비교이며 이번 실행에 포함하지 않는다.

## 실행

```bash
# 다섯 모델 학습 + validation 생성 (두 GPU에 작업 분배)
python gaze_mammo/scripts/compare_trajectory_backbones.py --devices cuda:0 cuda:1

# 개별 모델
python gaze_mammo/scripts/pipeline.py train-gaze --method trajectory --backbone convnext_tiny --cache-visual --manifest gaze_mammo/manifests/trajectory.csv --k 16 --batch-size 32 --output gaze_mammo/runs/trajectory_convnext --device cuda:0

# 영상만으로 validation 경로 생성; 실제 시선은 사후 평가에만 사용
python gaze_mammo/scripts/pipeline.py generate --checkpoint gaze_mammo/runs/trajectory_convnext --manifest gaze_mammo/manifests/trajectory.csv --output gaze_mammo/runs/trajectory_convnext/validation --with-gaze-evaluation --device cuda:0

# 비교 표, 관측/예측 경로 그림 및 train-only prior 비교
python gaze_mammo/scripts/report_trajectories.py

# 동일 checkpoint의 확률 샘플링 생성 비교
python gaze_mammo/scripts/evaluate_trajectory_sampling.py --devices cuda:0 cuda:1
python gaze_mammo/scripts/report_trajectories.py
```

`runs/trajectory_backbones_seed42/`에 백본별 `best.pt`, config, history, run.log, training_summary, validation 좌표·지표를 저장한다. 모든 비교가 완료되면 `comparison.csv`를 생성한다. 보고서 생성 후 `report.md`, `trajectory_examples.png`, `rollout_comparison.png`도 저장한다.

## 평가 해석

- teacher-forced next-step NLL: 실제 이전 위치를 조건으로 다음 위치의 확률을 평가한다.
- teacher-forced 정규화 위치 오차: 같은 조건의 argmax 좌표 오차.
- free-running normalized DTW: 이미지와 BOS만으로 시작한 전체 생성 경로를 실제 축약 경로와 정렬해 비교한다.
- unique grid locations: 16개 예측 중 서로 다른 위치 수. 동일 지점 반복 수렴을 점검한다.
- center prior / train median trajectory: 학습 데이터에서만 만든 단순 prior와 비교한다. 시선이 주로 같은 영역에 모일 때 작은 오차를 얻는 효과를 구분한다.

낮은 teacher-forcing loss가 좋은 전체 rollout을 보장하지 않는다. 한 영상에는 관측 시선 경로가 하나뿐이므로 다른 타당한 경로가 과하게 벌점받을 수 있다. 시선 기록은 주석 작업 중 수집됐고 저장 순서·세션 병합 규칙은 미확인이다.

## 한 영상의 경로와 패치 출력

```bash
python gaze_mammo/scripts/predict_trajectory.py --checkpoint gaze_mammo/runs/trajectory_backbones_seed42/convnext_tiny --image gaze_mammo/data/image/IMAGE_NAME.png --output gaze_mammo/results/trajectory_example --device cuda:0
```

이미지 한 장만 로드하고 `prediction.json`, `trajectory.png`, `patches.png` 및 개별 224×224 패치를 생성한다. 경로 자체는 원본 예측을 표시하고, 패치 중심은 공통 foreground mask에 투영한 좌표를 별도로 저장한다. 모델은 현재 데이터의 1500×3000 전처리 형식을 요구한다.

기본 생성은 argmax다. `--strategy sample --temperature 1.0 --seed 42`로 확률 샘플링도 가능하다. 이는 다른 생성 조건이므로 기본 비교 수치와 혼합하지 않는다. 자연 길이 종료 토큰, 연속 좌표 offset, 배율 예측은 현재 구현에 포함되지 않는다.

## 첫 비교의 결론

다섯 백본 모두 전체 train/validation 학습 및 생성 평가를 완료했다. best validation loss는 모두 2 epoch 부근에서 선택됐고 patience에 따라 12 epochs에서 종료됐다. argmax는 평균 1.00–1.69개 위치로 반복 수렴했다. 샘플링은 평균 15.76–15.83개 위치를 만들지만, DTW 약 0.157–0.160으로 train median prior 0.123보다 컸다. 현재 결과로 백본을 확정하거나 patch selector의 우수성을 주장하지 않는다. 상세 표와 예시는 `runs/trajectory_backbones_seed42/report.md`에 있다.

## 패치 분류로 연결

백본을 선택한 뒤 아래와 같이 학습 환자 내부 cross-fitting을 수행한다. full-train predictor가 자기 학습 영상에 만든 경로를 classifier 학습에 사용하지 않는다.

```bash
python gaze_mammo/scripts/run_crossfit.py --method trajectory --backbone convnext_tiny --cache-visual --k 16 --epochs 50 --output gaze_mammo/runs/crossfit_convnext --device cuda:0
```

예측 좌표 JSON과 provenance를 기존 patch extraction/MIL pipeline에 연결한다. 세부 명령은 `IMPLEMENTATION.md`를 따른다.
