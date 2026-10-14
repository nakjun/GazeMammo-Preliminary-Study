# 백본 파인튜닝

Swin-B trajectory predictor와 ResNet50 patch encoder를 각각 업데이트하는 코드를 구현하고 실험했다. 이전 checkpoint에서 같은 학습 환자만 사용해 시작한다. trajectory는 gaze CE, ResNet50 + MIL은 분류 BCE로 각각 학습하며 crop 좌표는 현재 미분하지 않는다.

## Swin-B + trajectory decoder

```bash
python gaze_mammo/scripts/pipeline.py train-gaze --method trajectory --backbone swin_b --decoder-dim 512 --decoder-layers 6 --decoder-heads 8 --ff-multiplier 4 --dropout 0.2 --finetune-backbone --backbone-lr 0.00001 --lr 0.0001 --batch-size 8 --epochs 20 --patience 6 --amp --cache-images --init-checkpoint gaze_mammo/runs/trajectory_large_seed42/swin_b --manifest gaze_mammo/manifests/trajectory.csv --output gaze_mammo/runs/trajectory_finetune_seed42/swin_b --device cuda:0

python gaze_mammo/scripts/evaluate_constraints.py --checkpoints gaze_mammo/runs/trajectory_finetune_seed42 --output gaze_mammo/runs/trajectory_finetune_constraints_seed42 --backbone swin_b --device cuda:0
```

Swin-B 전체 가중치에 gradient를 전달하며 약113.7M 파라미터를 업데이트한다. decoder와 backbone의 학습률을 분리한다. `--cache-images`는 입력 영상만 RAM에 저장하고 backbone 특징은 매 업데이트마다 다시 계산한다. `--finetune-backbone --cache-visual` 조합은 거부한다.

## 환자 단위 cross-fitting

```bash
python gaze_mammo/scripts/run_constraint_classification.py --finetune --validation-source gaze_mammo/runs/trajectory_finetune_constraints_seed42/swin_b --output gaze_mammo/runs/finetune_constraint_classification_seed42
```

각 fold는 대응하는 기존 frozen-fold checkpoint에서 초기화한다. 제외한 환자를 학습한 full-train checkpoint를 fold 초기화에 사용하지 않는다. initializer의 학습 환자가 새 train 집합의 부분집합인지 검사한다.

## ResNet50 + MIL 공동 파인튜닝

```bash
# 기존 frozen trajectory 패치로 학습
python gaze_mammo/scripts/finetune_patch_classifier.py --features gaze_mammo/runs/constraint_classification_seed42/hard112/features --init-classifier gaze_mammo/runs/constraint_classification_seed42/hard112/classifier --output gaze_mammo/runs/patch_finetune_frozen_trajectory_seed42 --device cuda:1

# 파인튜닝 predictor의 CF 패치로 학습
python gaze_mammo/scripts/finetune_patch_classifier.py --features gaze_mammo/runs/finetune_constraint_classification_seed42/hard112/features --init-classifier gaze_mammo/runs/finetune_constraint_classification_seed42/hard112/classifier --output gaze_mammo/runs/patch_finetune_finetuned_trajectory_seed42 --device cuda:1
```

모든 ResNet50/attention MIL 파라미터 약23.8M을 업데이트한다. BatchNorm running statistics만 유지하고 affine weight는 업데이트한다. lr(backbone)=1e-5, lr(MIL)=1e-4, microbatch2 × accumulation4, BF16 학습 및 FP32 검증을 사용한다. 입력 패치를 RAM에 저장하며 features는 저장하지 않는다. 새로운 `end_to_end` classifier checkpoint는 `predict_image.py`에서 복원할 수 있다.

## 결과와 선택 기준

```bash
python gaze_mammo/scripts/report_finetuning.py
```

`runs/finetuning_report_seed42/report.md`와 `comparison.csv`에 4가지 조합의 성능과 paired 환자 bootstrap CI를 기록했다. 검증은 기존 93장, 학습 376장, seed42, sample + breast mask + hard112, K16이다. test와 INbreast는 사용하지 않았다.

초기 상태(epoch0)를 평가하고, 검증이 악화되면 초기 checkpoint를 선택한다. `finetune_backbone=True` flag만으로 선택된 가중치가 바뀌었다고 주장하지 않는다. 로그에서 실제 first-weight 변화와 선택 epoch를 확인한다. 첫 full trajectory 실행에는 last 저장 기능이 추가되기 전이어서 `last.pt`가 없으며, 이후 CF와 patch 실행에는 `best.pt`/`last.pt`를 저장한다.

이번 full Swin-B와 CF folds 0/1은 초기 상태를 선택했고, CF fold2는 epoch1을 선택했다. ResNet50만 fine-tuning한 조합은 AUROC 0.666→0.693, 선택 epoch8이었다. 두 단계 fine-tuning 조합의 ResNet50은 초기 상태를 선택했다. paired 차이 CI가 0을 포함해 확정적 개선으로 해석하지 않는다.
