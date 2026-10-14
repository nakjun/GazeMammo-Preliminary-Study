# 최고 grid + fusion 설정의 백본 비교

ResNet50 최고 설정(image-only 분산 grid + global/local fusion)을 유지하고 사용자가 지정한 DenseNet161, EfficientNet-B5, ConvNeXt-Base, Swin-B, ViT-B/16을 학습했다. 결과는 [보고서](runs/grid_fusion_backbones_seed42/report.md), [CSV](runs/grid_fusion_backbones_seed42/comparison.csv)에서 관리한다.

## 결과

| 백본 | Validation AUROC | AUPRC |
|---|---:|---:|
| resnet50 | 0.796 | 0.770 |
| vit_b_16 | 0.753 | 0.747 |
| densenet161 | 0.736 | 0.791 |
| convnext_base | 0.729 | 0.742 |
| efficientnet_b5 | 0.709 | 0.698 |
| swin_b | 0.697 | 0.709 |

AUROC 기준 기존 ResNet50이 최고이며, 새 백본 중에는 ViT-B/16이 가장 높았다. DenseNet161은 AUPRC와 specificity≥90%에서의 sensitivity가 수치상 가장 높았다. ResNet50 대비 paired AUROC 차이의95% CI는 모든 백본에서0을 포함한다. 단일 seed의 반복 사용된 validation 결과이므로 우열이 확정된 것은 아니다.

## 공통 조건

- CBIS train376 / validation93, seed42. 최종 test131 및 INbreast 미사용.
- 기존 grid16개 중심을 그대로 사용한다. 추정 유방 mask와112px 거리 제약, 원본224×224 patch, 전체 영상512×256 입력.
- 하나의 backbone으로 global과 local을 인코딩한다.128차원 projection, attention MIL, global/local/fusion BCE 평균.
- 백본 전체 fine-tuning, BN running statistics만 고정. backbone lr1e-5, head lr1e-4, AdamW weight decay1e-4.
- batch2×gradient accumulation4, BF16 학습/FP32 검증, gradient clipping1, 최대12 epochs, validation fusion AUROC 선택 및 patience4.
- torchvision ImageNet-1K 사전학습. 큰 백본의 사전학습 recipe 차이까지 포함하는 비교이며 architecture만의 효과를 분리하지 않는다.

EfficientNet-B5의 patch 입력도224이다. 원래 권장456 입력을 사용하는 모델별 최적 해상도 비교는 후속 실험이다. ViT-B는 patch size16으로,224×224 patch에서는 원래14×14 위치 임베딩을 유지하고 전체 영상에서는32×16으로 bicubic 보간한다. CLS 특징을 사용하며 위치 임베딩도 fine-tuning한다. ConvNeXt/Swin의 원래 최종 normalization을 보존한다.

실행 세부 조건은 `runs/grid_fusion_backbones_seed42/protocol.json` 및 각 백본 폴더의 `config.json`, `model_info.json`, `history.json`, `timing.json`에 저장한다. `best.pt`와 `last.pt`는 백본 이름을 포함한다. 기존 ResNet50 checkpoint도 계속 읽을 수 있다.

## 실행

```bash
python gaze_mammo/scripts/compare_fusion_backbones.py
python gaze_mammo/scripts/report_fusion_backbones.py
```

runner는 GPU2개를 사용한다. 백본 하나만 실행하려면:

```bash
python gaze_mammo/scripts/spatial_pipeline.py fusion \
  --features gaze_mammo/runs/spatial_v2_seed42/grid_control/features \
  --backbone vit_b_16 --epochs 12 --patience 4 \
  --output gaze_mammo/runs/grid_fusion_backbones_seed42/vit_b_16 \
  --device cuda:0
```

영상 하나의 grid 패치와 악성 확률 추론:

```bash
python gaze_mammo/scripts/predict_grid_fusion.py \
  --checkpoint gaze_mammo/runs/grid_fusion_backbones_seed42/vit_b_16 \
  --image gaze_mammo/data/image/Mass-Training_P_00026_LEFT_CC_BENIGN.png \
  --output gaze_mammo/results/grid_vit_single_image.json
```

입력은 기존 전처리에 맞춘1500×3000 PNG이며 gaze·병변 box·진단 label은 추론에 사용하지 않는다. 모델 확률은 독립 test로 보정된 임상 확률이 아니다.

## 해석

이 실험은 gaze trajectory가 아닌, 현재 가장 높은 검증 AUROC의 분산 grid 설정에서 encoder를 비교한다. 단일 seed 및 동일 validation을 반복해 checkpoint와 backbone을 선택하므로 결과에 선택 편향이 있다. paired 환자 bootstrap CI와 함께 비교하고 여러 seed로 후보를 재검증한 뒤 최종 test를 평가한다.

검증: 단위 테스트31개 통과,5개 백본의 forward/backward 및 checkpoint 단일 영상 추론 확인. 실제 pretrained convolution 가중치가 업데이트됐고 BN running statistics는 그대로 유지됐다. train306 환자와 held-out 환자는 겹치지 않는다. 상세 확인은 `scripts/audit_fusion_backbones.py`와 `runs/grid_fusion_backbones_seed42/finetuning_audit.json`에 기록한다.
