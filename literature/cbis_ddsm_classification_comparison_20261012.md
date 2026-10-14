# CBIS-DDSM 양성·악성 분류 문헌 비교

2026-10-12. GAA-DETR 탐지 AP 비교와 분리하고, 마지막 완료된 분류 실험의 결과만 사용했다. 새 학습이나 INbreast 평가를 수행하지 않았다. 아래는 각 논문이 보고한 수치의 문헌 대조이며 동일 split 재현 비교가 아니다.

| 연구/방법 | AUROC | 평가 조건 |
|---|---:|---|
| Shen et al., Scientific Reports2019, best single | .88 | 전체 영상 분류, 논문의 custom holdout |
| Shen et al.,4-model averaging | .91 | 위 custom holdout, ensemble |
| Wei et al., Medical Image Analysis2022, MorphHR single/1-crop | .7964 | 공식 CBIS-DDSM TEST645장 |
| MorphHR single/10-crop TTA | .8187 | 같은 공식 TEST |
| MorphHR4-model ensemble/10-crop TTA | .8313 | 같은 공식 TEST |
| Petrini & Kim,2026, EfficientNet-B3 PBC | .8325 | Table7, single-view/no TTA, best test result |
| 같은 PBC+TTA | .8343 | Table7, single-view/TTA |
| 현재 순수 시선, 단일 head3seed 평균 | .75316±.01309 | CBIS-DDSM gaze subset, validation93장/77환자 |
| 현재 순수 시선,3-head probability ensemble | .76503 | 같은 validation93장 |

현재 단일 head 평균 accuracy67.03%, ensemble accuracy66.67%다. 비교한 주요 원문의 해당 요약 표에는 대응 accuracy가 없어 AUROC 위주로 대조했다. 다른 논문의 sensitivity/specificity를 이용해 accuracy를 임의 계산하지 않았다.

MorphHR 공식 TEST 기준 문헌 차이는 단일 모델(1-crop)과 우리 단일 head 평균에서−.04324, ensemble+TTA와 우리 ensemble에서−.06627이다. TTA 유무, 서로 다른 train 규모와 split, 검증/TEST의 차이가 있으므로 paired effect나 통계적 우열로 해석할 수 없다. 현재 숫자로 기존 연구보다 높다고 주장할 근거가 없다.

## 주요 조건과 원문

1. [Shen et al., Deep Learning to Improve Breast Cancer Detection on Screening Mammography,2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6715802/): 발표 원문의 abstract는 CBIS-DDSM per-image AUC .88(single)/.91(four-model averaging)를 보고한다.2017 arXiv 초기 버전은 three-model ensemble이므로2019 결과와 모델 개수를 혼동하지 않았다. MorphHR 원문은 Shen의 custom split 수치와 공식 TEST 재평가 약.75를 구분한다.
2. [Wei et al., Beyond fine-tuning: Classifying high resolution mammograms using function-preserving transformations,2022](https://api.repository.cam.ac.uk/server/api/core/bitstreams/999ca522-f2fc-4a2a-85ac-d10ee6889f90/content), DOI[10.1016/j.media.2022.102618](https://doi.org/10.1016/j.media.2022.102618): Table1/2를 직접 확인했다. Dataset은 train2097/val361/test645장이고, 각1061/187/349명의 환자를 보고한다. Train에서 patient-level85:15 validation을 만든다. 단일 view를 독립 영상으로 분류한다. Val AUROC .9182/.9348/.9427과 공식 TEST .7964/.8187/.8313은 구분해야 한다.
3. [Petrini & Kim, A Comprehensive Study of Transfer Learning, Resolution Reduction, and Multi-View Classification for Breast Cancer Detection in Mammograms,2026](https://onlinelibrary.wiley.com/doi/full/10.1002/ima.70436): Table5/7에서 PBC EfficientNet-B3의 반복실험 test mean .8114±.0153, best test .8325, TTA .8343을 확인했다. 표의 best-test 수치를 우리3seed mean과 동일한 추정량으로 취급하지 않는다. Multi-view .8658은 입력과 평가 단위가 달라 주 비교 표에서 제외했다.

## 확인했지만 핵심 benchmark로 채택하지 않은 사례

[Dai et al., X-Pruning, QIMS2026](https://qims.amegroups.org/article/view/154958/html)는 양성·악성 분류 및 patient-level5fold CV를 기술한다. Table1의 CBIS-DDSM 값은 AUC .790, accuracy82.4±1.1%다. 그러나 abstract는 AUC .792, 결과 본문은 .775를 써 내부 수치가 일치하지 않는다. Dual-view 입력이며 우리 single-view와도 다르므로 해당 수치를 우월성 주장의 핵심 기준으로 사용하지 않았다.

원문에 나타난 단순 baseline의 수치보다 높다는 것만으로 해당 논문의 제안 방법을 이겼다고 주장하지 않는다. 병변 ROI 제공 모델, whole-image 모델, multi-view 모델은 따로 구분해야 한다.

## 현재 가능한 주장

우리의 환자 분리 CBIS-DDSM gaze subset validation에서 순수 시선 결합이 global 기준점보다3seed 평균 AUROC와 accuracy에서 높았다. Gaze ensemble−global AUROC 차이 .02081의 patient-bootstrap95%CI는[−.06900,+.11539]로0을 포함한다. 기존 연구 대비 우월성이나 독립 TEST 일반화는 아직 확인되지 않았다.

정식 정량 비교에는 같은 원본 영상 cohort/환자 split/라벨/전처리/입력 단위에서 비교 방법을 재현해야 한다. 현재 classifier에는 공식 CBIS-DDSM 전체 TEST 결과가 없고, 이전 gaze/ROI TEST 평가를 분류 TEST 성능으로 대체할 수 없다.

현재 원시 결과: [report.json](../runs/gaze_patch_classification_1012_full/report.json), [독립 결과 감사](../runs/gaze_patch_classification_1012_full/parent_result_audit.json).
