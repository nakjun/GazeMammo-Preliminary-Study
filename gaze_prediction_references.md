# Gaze / scanpath 예측 연구와 목표 비교

2026-10-10 확인. 최우선 과제는 영상만으로 다음 gaze 좌표와 순서 있는 경로를 예측하는 것이다. GAA-DETR 병변 검출 AP나 grid 패치 악성 분류 AUROC는 이 목표의 성능으로 사용하지 않는다. `/goal`은 active이며 아직 논문 수준 달성을 확인하지 않았다.

## 직접적인 경로 예측 참고

| 연구 | 과제 | 비교에 사용할 요소 | 현재 데이터와의 차이 |
|---|---|---|---|
| [ChestSearch / GazeSearch, WACV2025](https://arxiv.org/html/2411.05780v1) | 특정 finding을 찾는 CXR scanpath | 의료영상 pretrained ResNet50, 이전 fixation 위치의 spatial indexing, Transformer decoder | CXR·finding 조건·duration을 사용한다. mammography와 보고서/시간이 없는 조건으로 adaptation하면 원 실험의 정확한 재현과 구분해야 한다. |
| [HAT, CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Unifying_Top-down_and_Bottom-up_Scanpath_Prediction_Using_Transformers_CVPR_2024_paper.html) | visual search와 free-viewing의 순차 위치 예측 | foveated working memory, 순차 dense heatmap | 자연영상, COCO panoptic segmentation pretrained encoder/decoder. 우리 GRU 기반 Spatial V2는 HAT의 재현이 아니다. |
| [MedGaze, Scientific Reports2025](https://doi.org/10.1038/s41598-025-97935-y) | CXR fixation 좌표·duration 예측 | 영상/보고서 사전학습과 gaze 학습의 단계적 구조 | 원 학습에서 radiology report와 duration 감독을 사용한다. 현재 breast CSV만으로 원 multimodal 학습·평가의 모든 부분을 동일하게 재현할 수 없다. |
| [DeepGaze III, JOV2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9055565/) | natural-image free-viewing scanpath | 영상 saliency와 fixation history의 조건부 확률; likelihood/IG/cAUC/NSS 평가 | 자연영상, center-bias baseline 및 데이터가 달라 발표 IG/LL 수치를 그대로 breast 목표값으로 쓰지 않는다. |
| [Gazeformer, CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Mondal_Gazeformer_Scalable_Effective_and_Fast_Prediction_of_Goal-Directed_Human_Attention_CVPR_2023_paper.html) | 목표 지향 scanpath | Transformer로 좌표·유효 여부 예측 | task encoding과 데이터가 다르다. 좌표 head의 비교 기준으로 활용한다. |

공식 코드: [ChestSearch](https://github.com/UARK-AICV/GazeSearch) · [HAT](https://github.com/cvlab-stonybrook/HAT) · [MedGaze](https://github.com/a04101999/MedGaze).

ChestSearch 논문의 같은 GazeSearch benchmark에서 ScanMatch(no duration)는 ChestSearch0.3321, HAT0.3120이다. 이 수치는 CXR benchmark 결과이지 현재 breast 데이터의 달성 기준이 아니다. 원 논문 수준을 목표로 하려면 같은 데이터·전처리·평가 조건에서 해당 방법을 재현한 후 비교해야 한다.

## 유방촬영에 직접 적용된 saliency 연구

[Predicting Radiologists' Gaze with Computational Saliency Models in Mammogram Reading, IEEE TMM26 (2024; online2023)](https://orca.cardiff.ac.uk/id/eprint/158274/)는196개 mammogram에서 saliency 방법을 비교하고, 자연영상 saliency 사전학습 후 mammogram fine-tuning과 multi-stream encoder를 사용한다. 우리 영상 modality와 직접 관련되지만 시간 순서 있는 scanpath 대신 공간 saliency map을 예측하는 연구다. 정적 saliency 기준 모델의 근거로 사용하고, 경로 예측 목표와 구분한다.

## 목표를 측정하는 방법

우선 비교 후보는 HAT와 ChestSearch다. mammography adaptation을 진행할 때 공식 방법과 달라지는 입력·시작 위치·길이·사전학습·duration 처리를 공개한다. 현재 자체 모델이나 간소화 변형을 원 논문의 정확한 재현이라고 부르지 않는다.

- 조건부 분포: held-out next-step NLL↓, train-only spatial prior 대비 IG↑, conditional AUC/NSS↑.
- 실제 생성 경로: DTW↓ 및 duration 없는 ScanMatch/MultiMatch geometry. 확률 샘플링은 여러 seed로 평가한다.
- 전처리와 후보 좌표 공간을 고정한 동일 데이터 비교. 다른 데이터의 논문 숫자와 직접 성능 동등성을 주장하지 않는다.
- bbox는 경로 예측 입력에 사용하지 않는다. 병변 hit는 보조 분석이다.
- 학습/validation에서 반복하고 최종 test는 설정 고정 후 평가한다.

## 현재 문제와 진행

기존 spatial 모델의 best raw NLL6.097, sample hard112 DTW0.137은 같은 데이터 static heatmap DTW0.135보다 우월하지 않았다. 현재 모델은 ImageNet encoder와 작은 GRU이며, 논문들의 gaze/medical-task pretrained 구조와 차이가 크다. 백본 규모 증가만으로 논문 수준이 확보된다고 가정하지 않는다.

validation93장의1395개 다음 지점 중354개(25.4%)는 실제 앞선 gaze의112px 이내로 돌아간다.63개(4.5%)는 앞서 방문한24×48 cell과 같다. `literature/gaze_constraint_audit.json`에 기록했다. 강한 중복 억제는 실제 gaze 동작을 제거할 수 있으므로 raw gaze 예측과 downstream patch 중복 억제 정책의 평가를 분리한다. 25.4%는 원 좌표 거리 기준이며 grid center 양자화 후 exact 금지 비율과 동일한 것은 아니다.

새 gaze 개선: train412장의 gaze 전체 사용, 조건부 Gaussian-mixture motion bias, spatial prior 보조 loss, raw NLL 기준 checkpoint 선택, gaze encoder 내부의 ResNet50/DenseNet161/EfficientNet-B5/ConvNeXt-Base/Swin-B/ViT-B 지원. ViT는 intermediate token level을 사용하며CNN의 stride8/16/32 pyramid와 동일한 구조라고 부르지 않는다. classifier 가중치를 초기화에 사용하지 않는다.

첫 ResNet50 motion 모델의 validation raw NLL5.950으로 수치상 낮아졌다. 생성 경로와 반복 seed 평가는 아직 진행 중이므로 목표 달성으로 판정하지 않는다. ConvNeXt-Base도 gaze density 사전학습을 완료하고 순차 gaze 학습을 시작했다. 새 기능을 포함한36개 단위 테스트와5종 gaze encoder의 실제 forward/backward를 통과했다.
