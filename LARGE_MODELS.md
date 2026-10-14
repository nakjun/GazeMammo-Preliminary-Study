# 큰 trajectory 모델

## 확장한 구성

| 항목 | 기존 기본 모델 | 큰 모델 |
|---|---|---|
| 백본 | ResNet50 / ConvNeXt-Tiny / Swin-T 등 | ResNet101 / ConvNeXt-Base / Swin-B |
| decoder dimension | 128 | 512 |
| decoder layers | 2 | 6 |
| attention heads | 4 | 8 |
| FFN dimension | 256 | 2048 |
| dropout | 0.1 | 0.2 |
| learning rate | 0.001 | 0.0003 |
| batch size | 32 | 16 |

비교에는 ResNet50 + 큰 decoder도 포함한다. 같은 큰 decoder 안에서 백본을 비교하며, 기존 작은 decoder 결과는 참고 자료로만 사용한다. optimizer 설정도 달라졌으므로 과거 실행과의 차이를 모델 크기 하나의 효과로 단정하지 않는다.

영상 입력 256×512, 위치 grid 24×48, K=16과 CBIS-DDSM 환자 분할(train 376 / validation 93)은 유지한다. 백본은 ImageNet 사전학습 가중치를 고정하고 trajectory decoder와 projection을 학습한다. 백본까지 미세 조정한 실험은 아니다. INbreast quarantine과 test 미사용 정책도 유지한다.

초기 실행은 최대 50 epochs, validation loss patience=10이다. 큰 모델을 사용해도 반복 좌표 생성이 해결된다고 가정하지 않으며, teacher-forced next-step 평가와 argmax/sample rollout을 구분해 보고한다.

## 전체 비교 실행

```bash
python gaze_mammo/scripts/compare_trajectory_backbones.py --config gaze_mammo/configs/trajectory_large.json --output gaze_mammo/runs/trajectory_large_seed42 --devices cuda:0 cuda:1
python gaze_mammo/scripts/evaluate_trajectory_sampling.py --runs gaze_mammo/runs/trajectory_large_seed42 --devices cuda:0 cuda:1
python gaze_mammo/scripts/report_trajectories.py --runs gaze_mammo/runs/trajectory_large_seed42
```

`runs/trajectory_large_seed42/`에 학습 checkpoint, 모델 구조·파라미터 수, 학습 로그, validation 좌표, 생성 방식별 비교표와 보고서를 저장한다. 기존 `trajectory_backbones_seed42/` 결과와 checkpoint는 보존한다.

## 개별 모델 및 추론

```bash
python gaze_mammo/scripts/pipeline.py train-gaze --method trajectory --backbone convnext_base --decoder-dim 512 --decoder-layers 6 --decoder-heads 8 --ff-multiplier 4 --dropout 0.2 --lr 0.0003 --batch-size 16 --cache-visual --manifest gaze_mammo/manifests/trajectory.csv --output gaze_mammo/runs/large_convnext --device cuda:0

python gaze_mammo/scripts/predict_trajectory.py --checkpoint gaze_mammo/runs/large_convnext --image gaze_mammo/data/image/IMAGE_NAME.png --output gaze_mammo/results/large_trajectory_example --device cuda:0
```

모델 구조는 checkpoint의 `model_config`에 저장한다. 생성·단일 영상 추론·cross-fitting 모두 해당 구조를 복원한다. 예전 checkpoint에 `model_config`가 없으면 기존 128차원·2층 기본 구조로 복원한다.

cross-fitting도 같은 decoder 옵션을 지원한다. 큰 full-train predictor가 자신의 학습 영상에 만든 경로를 classifier 학습에 사용하지 않는다.

```bash
python gaze_mammo/scripts/run_crossfit.py --backbone convnext_base --decoder-dim 512 --decoder-layers 6 --decoder-heads 8 --ff-multiplier 4 --dropout 0.2 --lr 0.0003 --batch-size 16 --cache-visual --output gaze_mammo/runs/large_crossfit --device cuda:0
```

cross-fitting의 `--lr`, `--batch-size`도 전체 비교와 같은 값으로 명시한다.
