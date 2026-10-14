# 백본 성능 진단과 내부 분할 하이퍼파라미터 튜닝

기존 큰 모델의 train loss 감소와 validation AUROC 저하가 함께 관찰됐다. 모델별 fine-tuning 강도, 학습 기간과 일반화를 함께 비교했다. 작은 train376장/306환자는 원인 후보이지만, DenseNet161/EfficientNet-B5는 ResNet50과 비슷한 파라미터 수이므로 크기만으로 모든 차이를 설명할 수 없다. ImageNet 사전학습 recipe, 입력 크기, 유방촬영의 작은 병변과 패치 분류 특성도 후보다.

EfficientNet-B5의 [공식 사전학습 전처리](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.efficientnet_b5.html)는456 입력이다. 이번에는 기존 연구의224×224 patch 범위를 유지하므로 해상도 효과 자체는 검증하지 않는다. 모델별 과정이 민감할 수 있다는 배경 연구는 [How to Fine-Tune Vision Models with SGD](https://arxiv.org/abs/2211.09359)이며, 그 논문의 optimizer 결과를 현재 AdamW의 실패 원인으로 단정하지 않는다.

## 결과

내부 튜닝30개와 전체 train 재학습을 완료했다. [상세 보고서](runs/fusion_hpo_seed42/report.md)와 [학습 곡선](runs/fusion_hpo_seed42/original_learning_curves.png)을 제공한다.

| 백본 | 기존 AUROC | 내부 선택 refit AUROC | LR | warm-up | 총 epoch |
|---|---:|---:|---:|---:|---:|
| densenet161 | 0.736 | 0.665 | 1e-05 | 2 | 8 |
| efficientnet_b5 | 0.709 | 0.697 | 1e-05 | 2 | 8 |
| convnext_base | 0.729 | 0.791 | 1e-05 | 0 | 12 |
| swin_b | 0.697 | 0.723 | 1e-05 | 2 | 10 |
| vit_b_16 | 0.753 | 0.721 | 3e-06 | 2 | 10 |

ConvNeXt-Base와 Swin-B는 수치상 개선됐고 나머지3개는 개선되지 않았다. ResNet50 기존 기준0.796과 ConvNeXt-Base0.791은 비슷하지만 통계적 동등성을 입증한 것은 아니다. 모든 기존 동일 백본 대비 paired 차이 CI는0을 포함한다. 큰 모델 자체의 한계나 데이터 규모만을 원인으로 단정하지 않는다.

학습 영상257장의 선택 checkpoint에서 train AUROC0.933–0.974, 내부 새 환자 AUROC0.705–0.738의 일반화 차이가 확인됐다. 이 결과와 기존 학습 곡선은 과적합/환자 간 일반화 문제를 의심할 근거다. ConvNeXt의 개선은 첫 공통 설정을 모든 백본의 최적 설정으로 간주할 수 없다는 점도 보여준다.6-epoch screening과 단일 내부 fold 때문에 낮은lr 및 느린 학습 후보가 불리할 수 있다.

## 튜닝 범위

- DenseNet161, EfficientNet-B5, ConvNeXt-Base, Swin-B, ViT-B/16.
- 같은 image-only grid16개, 원본224×224 patch, 전체 영상512×256, 공유 backbone + global/local fusion.
- 기존 train의 환자 fold0을 내부 검증으로 사용: inner train257장/204환자, inner validation119장/102환자.
- 단계1: backbone lr1e-6/3e-6/1e-5를 각각6 epochs, head lr1e-4, head-only warm-up2 epochs.
- 단계2: 내부 최고lr에서 minimal과 regularized recipe를 최대12 epochs 비교. minimal은 wd1e-4/dropout0.1/고정lr/증강 없음, regularized는 wd1e-2/dropout0.3/일관된 수평 반전/cosine 마지막0.1배.
- warm-up0 minimal 대조군을 같은 내부 최고lr에서 최대12 epochs 비교한다. 이 후보는 epoch1부터, warm-up2 후보는 실제 backbone fine-tuning이 시작된 epoch3부터 best checkpoint를 선택한다. stage2 patience5, 최소6 epochs.
- 내부 최고 recipe와 epoch를 확정한 뒤 전체 train376장으로 처음부터 재학습. outer validation93장은 refit 마지막에 평가하며 epoch 선택과 조기 종료에 사용하지 않는다.
- batch2×accumulation4, AdamW, BF16 train/FP32 eval, gradient clip1, BN running statistics 고정, seed42.

외부 validation은 이전 연구에서 이미 반복 사용됐으므로 독립 test가 아니다. 최종 test131장과 INbreast는 미사용이다. 단일 내부 fold와6-epoch screening은 후보 순위를 불안정하게 하거나 낮은lr을 불리하게 평가할 수 있다. 정규화 recipe는 여러 설정의 묶음이므로 개별 기여를 분리하지 않는다.

## 실행과 산출물

```bash
python gaze_mammo/scripts/tune_fusion_backbones.py run \
  --output gaze_mammo/runs/fusion_hpo_reproduce_seed42
python gaze_mammo/scripts/report_fusion_hpo.py \
  --output gaze_mammo/runs/fusion_hpo_reproduce_seed42
```

두 GPU에서 진행한다. 입력 캐시는 자동 준비한다. 기존 결과와 섞이지 않도록 새 output 폴더를 지정한다. 입력 픽셀 캐시만 공유하며 CNN 특징은 매 학습 step 다시 계산한다. `runs/fusion_hpo_seed42`의 `protocol.json`, `gpu0.log`, `gpu1.log`, 각 백본의 `screen_lr*`, `refine_*`, `selected.json`, `selected_final.json`, `refit` 또는 `refit_no_warm`에 설정과 모델을 저장한다. 최종 비교는 `report.md`, `comparison.csv`, `paired_differences.json`, `comparison.png`다.

33개 단위 테스트 통과. 핵심 검증은 warm-up/cosine/refit 시점의 동일 LR, 전체 영상과 patch의 일관된 반전, 기존 backbone/trajectory/inference 호환성이다. 이번 튜닝에서 패치 범위와 위치는 변경하지 않았다.

확장 기록: 처음 두 warm-up refit의 outer 결과를 확인한 뒤 warm-up0 대조군을 추가한 탐색적 튜닝이다. 최종 recipe 선택은 추가 후보에서도 내부 점수만 사용한다. 최종 선택은 `selected_final.json`의 `refit_dir`가 가리키는 checkpoint를 사용한다. 기존 모델은 outer validation에서 checkpoint를 선택했으므로 새 내부 선택 refit과 비교할 때 선택 절차 차이를 고려한다.

최종 검증 완료:5개 모델 모두 pretrained 첫 convolution 가중치가 실제 업데이트됐고 train/held-out 환자 중복이 없었다. 같은 배치 크기의 영상 추론은 저장된 validation CSV를 재현했다. EfficientNet-B5와 ViT-B/16의 singleton 확률 차이는 TF32 convolution에서 발생했고 TF32를 끄면 각각 약3.6e-7/1.8e-7로 줄었다. 실제 차이와 BN 통계 확인은 `runs/fusion_hpo_seed42/verification.json`에 기록했다.
