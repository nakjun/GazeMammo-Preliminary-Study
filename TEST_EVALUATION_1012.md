# 고정 모델 CBIS-DDSM TEST 평가

2026-10-12, 현재 설정 `0bc7e224c6dafe36b8be8b12dbe3694af3b272f22cfedca3b39795d56522d239`를 추가 학습이나 TEST 튜닝 없이 평가했다. 반복 개선 목표는 중단 상태다. INbreast는 제외했다. 환자 분리된 TEST 150장/124명에 대해 2,400개 시선 예측과 3개 시드(42/43/44)의 450개 이미지 기반 탐색 경로를 생성했다. 각 경로는 실제 224px 크롭 16개다.

시선 지표는 150장 전체, 병변 지표는 주석이 있는 131장/158개 병변에서 계산한다. 주석이 없는 19장은 음성이 아닌 주석 미확인 사례이므로 병변 지표에서 제외한다.

| 주요 지표 | TEST |
|---|---:|
| 시선 NLL ↓ | 5.88783 |
| 다음 시선 위치 conditional AUC ↑ | 0.86888 |
| conditional NSS ↑ | 2.19445 |
| TRAIN 공간 사전분포 대비 IG (bits) ↑ | 1.13561 |
| 균등분포 대비 LL (bits) ↑ | 1.67559 |
| DTW ↓ (단위 정사각형 좌표, 길이 정규화) | 0.14187 |
| ScanMatch ↑ (duration 없음) | 0.17023 |
| MultiMatch vector / direction ↑ | 0.91152 / 0.72794 |
| MultiMatch length / position ↑ | 0.90090 / 0.82337 |
| 병변 면적 50% 확보율 @16, 이미지 평균 ↑ | 74.43% |
| 병변 면적 50% 확보율 @16, 병변 평균 ↑ | 77.85% |
| 병변 면적 80% 확보율 @16, 이미지 평균 ↑ | 55.34% |
| 클래스 구분 없는 제안 AP ↑ | 0.02485 |
| 클래스 구분 없는 제안 AP50 ↑ | 0.11531 |
| 클래스 구분 없는 제안 AP75 ↑ | 0.00188 |
| 클래스 구분 없는 제안 AR10 / AR100 ↑ | 0.08291 / 0.08291 |

병변 확보율은 한 크롭이 포함한 GT 박스 면적의 비율로 정의한다. 여러 크롭의 합집합이나 확보 가능한 면적으로 정규화한 값이 아니다. AP는 박스 IoU .50:.05:.95와 101개 recall 지점을 사용하는 COCO 방식이다. 이미지 경계 밖 크롭 부분은 AP 계산에서 잘라낸다. 모든 실제 크롭의 점수는 고정된 guided relevance의 원래 anchor 값이며, 악성 확률이나 시선 확률이 아니다. 출력 박스는 최대 16개이므로 표준 COCO maxDets=100에서도 AR100=AR16이다.

Validation의 NLL 5.76627 / 확보율 85.48%와 구분해야 한다. TEST에서는 NLL ≤5.3 및 확보율 ≥85% 목표를 충족하지 못했다. 큰 크롭이 작은 병변 전체를 포함해도 IoU는 낮을 수 있으므로 확보율과 AP를 같은 척도로 해석하지 않는다.

## 문헌 지표와 비교 조건

[GAA-DETR MICCAI 2025 원문](https://papers.miccai.org/miccai-2025/paper/1266_paper.pdf)은 AP/AP50/AP75/AR10/AR100을 보고한다. 해당 breast 실험은 CBIS-DDSM과 INbreast를 함께 사용한 무작위 분할과 양성·악성 클래스 구분 탐지다. 이 평가의 CBIS-DDSM 단독 환자 분할 및 클래스 구분 없는 크롭 제안과 직접적인 SOTA 수치 비교는 성립하지 않는다. 같은 지표 정의를 채택했지만 동일 벤치마크 재현 결과는 아니다.

[DeepGaze III](https://pmc.ncbi.nlm.nih.gov/articles/PMC9055565/)의 조건부 시선 평가 방식에 맞춰 관측된 과거 시선만 사용해 다음 시선 분포를 평가한다. AUC는 24×48 격자의 다음 시선 위치와 나머지 1,151개 위치의 순위를 비교하며 암 분류 AUC가 아니다. IG baseline은 TRAIN 412장의 canonical 시선 histogram에 Dirichlet-1을 적용했고 TEST에서는 맞추지 않았다.

경로 유사도는 duration 없는 ScanMatch와 MultiMatch의 네 공간 차원, 기존 repository의 단위 정사각형 DTW를 보고한다. 측정된 체류시간이 없으므로 시간 지표는 산출하지 않는다. 현재 모델에는 악성 분류 head가 없어 분류 AUROC/AUPRC와 클래스 구분 탐지 AP를 산출하지 않는다. [문헌 조사 상세](runs/frozen_test_1012/metric_research/report.md)에 논문 수치와 적용 제한을 기록했다.

## 결과 및 재현

- [전체 지표와 환자 단위 95% 신뢰구간](runs/frozen_current_test_1012/metrics/metrics.json)
- [경로 유사도를 포함한 전체 표](runs/frozen_current_test_1012/metrics/report.md)
- [이미지별 지표](runs/frozen_current_test_1012/metrics/per_image.csv), [병변별 지표](runs/frozen_current_test_1012/metrics/per_lesion.csv)
- [평가 조건](runs/frozen_current_test_1012/protocol.json), [지표 조건](runs/frozen_current_test_1012/metrics_protocol.json)
- [모델·소스·cohort 무결성 검사](runs/frozen_current_test_1012/frozen_source_cohort_audit.json)
- [최종 지표 독립 검증](runs/frozen_current_test_1012/independent_terminal_metrics_audit.json)

체크포인트 9개 및 hash로 고정된 소스 파일 35개가 평가 전후 동일하다. 독립 검증으로 확률 정규화, 시선 미래 정보 차단, 실제 크롭 기하, 환자 분리, 450개 경로와 주석 분모를 확인했다. 지표 구현 테스트는 완전 일치·미탐지·점수 순위·공식 COCO 누적 계산과의 일치 등을 포함한다.

추론 스크립트: `scripts/evaluate_frozen_current_test_1012.py` (freeze → smoke → nll / generate). 후처리:

```bash
python scripts/test_metrics_1012.py --traces runs/frozen_current_test_1012/generation/traces.jsonl --teacher runs/frozen_current_test_1012/nll/raw_probabilities.pt --output NEW_OUTPUT --ap-bootstrap 2000
python scripts/audit_frozen_current_test_1012.py
python scripts/audit_frozen_test_metrics_1012.py
```

후처리는 추론을 다시 실행하지 않는다. 신뢰구간은 이미지별 시드 평균 후 환자 단위 재표집으로 산출한다(일반 지표 10,000회, AP/AR 2,000회, seed 42). ScanMatch의 공간 비교 행렬을 경로마다 구성하는 현재 고정 구현은 CPU 후처리에 약 20분 이상이 걸릴 수 있다.
