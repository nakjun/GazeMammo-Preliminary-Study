# Proposal 후보를 실제 경로 방문으로 연결하는 실험

2026년 10월 11일. GPT-6.1-Sol low 서브에이전트가 개발과 실험을 담당한다. 기존 모델과 결과는 보존하고 별도 모듈과 run에 저장한다.

핵심 질문은 **영상에서 발견한 병변 후보가 왜 16개 패치 선택으로 이어지지 않는가**다. 먼저 같은 checkpoint에서 선택 방식과 geometry 억제를 분리하고, 다음으로 실제 224px 패치의 병변 포착 능력을 예측하는 head를 학습한다. 사람 시선 예측과 병변 탐색의 성능을 함께 보고한다. 병변을 더 방문했다는 이유만으로 사람 시선 예측이 개선됐다고 판정하지 않는다.

## 현재 실험에서 확인한 사실

[combined 전체 학습](runs/guided_1011_combined_seed42/summary.json)은 15epoch, train412/validation93, 6180update로 완료했다. Best는 epoch11이며 guided legacy16 validation NLL은 5.907845다. Train376장은 박스가 있고 36장은 박스 감독을 생략했다. 기존 guided runner의 seed 인자는 데이터 순서와 rollout을 제어하지만 신규 proposal 초기화 전 `torch.manual_seed`가 없다. 후속 학습에서는 Python/NumPy/PyTorch seed를 모두 설정하고 초기 state hash도 기록한다.

[case73](runs/guided_1011_combined_seed42/best_predict_case73/trajectory.png)은 오른쪽 위 GT 병변에 proposal이 반응했지만 생성 패치는 병변을 방문하지 않았다. 저장된 JSON과 실제 geometry를 이용한 계산은 다음과 같다. Cell 좌표는 0부터 센다.

| 항목 | 관측값 |
|---|---:|
| GT와 겹치는 proposal cell 위치 | x23 y3 |
| 해당 proposal sigmoid score | 0.361206 |
| 전체1152cell 중 score 순위 | 3위 |
| 해당 cell의 224px crop GT box coverage | 100% |
| 해당 crop foreground 비율 | 63.84% |
| proposal guidance 가산값 | +0.722412 |
| 배경 및 경계 guidance 가산값 | −1.326994 |
| 합산 guidance | −0.604582 |
| 첫 step의 coverage50% 이상 후보 확률 raw | 0.078948% |
| 첫 step의 coverage50% 이상 후보 확률 guided | 0.060539% |

이 위치는 영상 오른쪽 위에 가까워 patch에 배경이 포함된다. 현재 geometry 항은 병변을 충분히 포함하는 crop도 억제한다. 위 확률은 **첫 step의 조건부 확률 질량**이며 16step의 방문 확률이 아니다. 개별 score는 box occupancy의 balanced BCE 출력이므로 악성 확률이나 보정된 병변 확률로 해석하지 않는다.

현재 `2R` guidance만으로 가능한 후보 간 odds 배율은 최대 exp(2) 약7.39이며, 해당 후보와 R0 후보 간 배율은 약2.06이다. 기존 raw 분포에서 확률이 매우 낮으면 이 방식만으로 경로 선택에 충분한 질량을 배정하기 어렵다는 가설을 검증한다. 다른 proposal peak도 더 강하므로 top1을 방문시키는 것만으로 case73 병변 방문을 보장하지 않는다.

## 1차 실험은 checkpoint를 고정한 선택 방식 비교

모든 군은 같은 `guided_1011_combined_seed42/best.pt`, 같은16step, 224px crop, 24×48grid, 같은 validation93장과 동일 image별 CPU sampling generator를 사용한다. Generation seeds42/43/44와 argmax를 분리한다. 박스·gaze·진단 라벨은 생성이 끝난 뒤 evaluator만 읽는다. Decoder hidden과 이전 crop은 실제 선택된 history로 갱신한다.

| 군 | 경로 선택 | 검증 질문 |
|---|---|---|
| P0 | 기존 combined 그대로 | 재현 기준 |
| P1 | raw +2R에서 geometry bias만 제거 | geometry가 후보 방문을 막는가 |
| P2 | raw +6R에 기존 geometry 유지 | 가산 계수 증가만으로 충분한가 |
| P3 | 기존 q와 proposal 분포를 확률 혼합하고 ROI에도 기존 geometry 적용 | proposal에 별도 질량을 주면 방문하는가 |
| P4 | P3에서 ROI 분포의 배경계수만0.5로 낮추고 edge 항 제거 | ROI 주변부 페널티 완화 효과가 추가되는가 |
| P5 | 서로 다른 proposal peak crop3개를 먼저 선택하고 나머지13개는 기존 conditional q로 생성 | proposal이 복구 가능한 병변 후보를 제공하는가 |

P5는 영상에서 예측한 score만 사용하며, crop IoU0.3보다 큰 후보를 순차 제외한다. 병변 GT를 이용하는 oracle은 아니다. 조직의 강한 다른 peak도 선택할 수 있으며, 고정3개 순서를 강제하므로 사람이 보는 순서의 개선으로 주장하지 않는다. 다른 군과 같은16개 patch 예산을 유지한다.

P3와 P4는 아래 정의를 공유한다. 기본 α=0.125, τ=0.5, ε=1e−6이며 검증된 최적값이 아니다. 첫 실험에서는 repeat 항을 새로 넣지 않아 혼합과 geometry의 효과를 분리한다.

```text
p_roi(c) = softmax(log(R(c)+ε)/τ − γ_bg C_bg(c) − γ_edge C_edge(c)(1−R(c)))
q_mix(c) = (1−α) q_original(c) + α p_roi(c)
P3: γ_bg=3, γ_edge=0.5
P4: γ_bg=0.5, γ_edge=0
```

이 혼합은 q_original이 매우 작은 위치에도 proposal 분포의 질량을 더할 수 있다. α0.125는16step에서 ROI 분기의 기대 사용량2회에 해당하지만, 실제 병변 방문이나 서로 다른 후보2개 방문을 보장하지 않는다. Argmax에서는 혼합 질량이 생겨도 최댓값 후보가 그대로일 수 있으므로 sampling 결과와 구분한다. α0 대조에서 원 경로의 완전 재현을 검사한다.

각 step의 raw 분포, proposal score, geometry bias, ROI 분포, 최종 q, 선택 crop을 저장한다. 같은 기존 prefix에서 분포를 바꾼 counterfactual은 분포의 직접 효과를 보여준다. 실제 변경된 prefix로 생성한 경로는 누적 효과를 보여준다. 두 분석을 섞지 않는다.

## 2차 실험은 패치 포착 능력 학습

현재 proposal target은 작은 grid cell의 박스 점유도이고 실제 action은224px crop의 선택이다. 새로운 utility head는 **해당 위치에서 crop을 뽑으면 표시된 병변을 얼마나 덮을 수 있는가**를 예측한다. 기존 occupancy head는 남겨두어 두 target을 비교한다.

Train box m과 후보 c에 대해:

```text
H_m(c) = area(crop(c) ∩ box_m) / area(box_m)
Hnorm_m(c) = H_m(c) / max_c H_m(c)
U(c) = max_m Hnorm_m(c)
```

큰 병변은224px crop으로 전부 포함할 수 없으므로 실제 coverage와 달성 가능한 최대값을 함께 기록한다. 여러 병변의 방문 평가는 각 box를 따로 수행하며 max target만으로 작은 병변을 모두 방문하도록 학습했다고 주장하지 않는다.

첫 학습은 기존 guided network 전체를 고정한 memory keys에서 별도 `Linear(D,1)` 또는 작은 MLP utility head만15epoch 학습한다. 별도 head의 효과를 projection/decoder 변화와 분리하며 train-only feature cache를 쓴다. 이미지별로 U>0와 U=0의 MSE를 각각 평균해 절반씩 더한다. 없는 class 항은 생략한다. 박스 없는36장은 전체 utility loss를 생략하고, 박스 밖을 임상적 정상 조직으로 단정하지 않는다. 표적은 표시된 박스의 포착 능력이다.

Training seeds42/43/44를 초기화 전 설정한다. 일차 pilot은 seed42로 진행하고15epoch의 last를 사전에 선택하여 validation ROI를 보고 epoch를 다시 고르지 않는다. Proposal source만 occupancy에서 utility로 바꾸어 P3/P4/P5를 비교한다. Utility score도 악성 확률은 아니다. 동일 head architecture와 MSE로 occupancy target을 학습하는 대조를 추가하면 loss 형식과 target 변화의 효과를 더 분리할 수 있다.

1차에서 proposal topK recall 자체가 낮으면 mixture만 확대하지 않는다. Utility head에서도 recall이 낮을 때에 한해 후속 adapter, multi-scale feature 또는 backbone fine-tuning을 검토한다.

## 후속 경로 학습의 조건

고정 checkpoint와 utility head 실험에서 proposal recall은 충분한데 actual visit만 낮은 현상이 반복될 경우 decoder adapter를 학습한다. 원 human-gaze branch를 별도 고정하거나 보존하고 ROI branch를 추가하여 shared projection 변화가 원 proposal을 망가뜨리지 않게 한다. 현재 behavior mixture의 global 사용률이 약98%였던 기록을 고려하여 처음에는 고정 α를 사용하고 gate 학습은 별도 ablation으로 둔다.

기존 visit loss는 여러 step의 넓은 확률 질량만으로 낮아질 수 있다. Train의 실제 sampled coverage와 surrogate를 동시에 기록하고, 둘의 불일치가 확인될 때 아래 집중 항을 단독 변경한다.

```text
v_tm = Σ_c q_t(c) 1[Hnorm_m(c) ≥ 0.5]
L_focus = mean_m { −log(ε + max_t v_tm) }
L_visit_new = L_visit_old + 0.1 L_focus
```

이 항은 적어도 한 step에 충분한 coverage 후보 질량을 집중시키며 max는 거의 모든 곳에서 미분 가능하다. Discrete sampling과 crop history는 여전히 detach되므로 실제 hit 확률의 정확한 gradient가 아니다. 한 곳에 경로가 몰릴 가능성을 검사하고, 계수0.1도 pilot 값으로 명시한다. Geometry 완화·target 변경·이 loss 변경을 한 번에 넣은 군만으로 각각의 효과를 주장하지 않는다.

## 평가와 판정

Proposal 감지, 실제 방문, 사람 시선 모방의 세 가지를 분리한다.

| 평가 대상 | 주요 지표 |
|---|---|
| Proposal | image-only distinct crop top1/3/5의 lesion recall50 및 recall80, 순위와 coverage |
| 실제 경로 | 1/4/8/16step lesion recall50/80, first sufficient hit step, 실제 최대 box coverage, any-overlap |
| 연결 실패 | proposal top3에서 포착 가능한 lesion 중 generated path가 놓친 비율과 denominator |
| 사람 gaze | 동일 legacy16 conditional NLL와 DTW, 가능하면 contiguous 조건을 별도 보고 |
| 패치 낭비와 다양성 | foreground/inbounds, unique 위치수, 과거 crop 최대IoU, 높은 배경 비율의 patch 비율 |

Recall50/80의 주 지표는 Hnorm≥0.5/0.8이고 실제 H≥0.5/0.8도 따로 기록한다. 박스가 없는 영상은 lesion recall denominator에서 제외한다. Proposal rank의 spatial suppression와 crop center 반올림을 고정한다. 환자 단위 paired bootstrap에서는 image별 generation seeds를 먼저 평균하고 같은 환자의 이미지를 함께 재표집한다. Training seed의 변동과 generation seed의 변동을 구분한다.

73번은 원인 발견 사례이며 설정 선택의 단독 기준으로 쓰지 않는다. Validation93장 전체와 주변부 병변/내부 병변, 작은/큰 박스, benign/malignant를 사후 분석한다. Boundary 그룹은 GT box center의 추정 foreground 경계 거리112px 이하로 사전에 정의하고, mask invalid/중심 outside는 별도 그룹으로 둔다. 이 그룹 정의를 generation에 전달하지 않는다.

Pilot 진입 기준은 P0를 기준으로 lesion recall50이5percentage points 이상 증가하면서 평균 foreground가2percentage points 이상 떨어지지 않고 DTW가0.005 이상 악화되지 않는 것이다. 이 수치는 공학적 탐색 기준이며 통계적 입증 기준이 아니다. Paired CI와 표본수도 함께 보고한다. Human gaze 개선 주장은 NLL와 실제 경로 지표의 개선을 별도로 요구한다. ROI만 개선하면 병변 탐색용 policy로 보고한다. 기준을 통과하는 군이 없으면 그 결과를 기록하고 원인별 다음 실험을 선택한다.

이 checkpoint는 이미 train412장에 학습되었고 기존 outer validation을 이용해 선택됐다. Train 일부를 나중에 떼어내도 이 checkpoint에 대해 독립 inner holdout이 되지 않는다. 현재93장은 반복 열람한 exploratory validation이다. 독립 model selection을 원하면 해당 inner 환자를 처음부터 제외한 초기 checkpoint부터 다시 학습해야 한다. 최종 test와 INbreast quarantine은 이번 pilot에서 읽거나 평가하지 않는다. 최종 설정과 provenance를 고정한 뒤 독립 test를 평가한다.

## 개발과 실행 순서

1. 신규 policy 모듈과 별도 posthoc evaluator를 개발한다. 확률 정규화, alpha0 재현, 미래 gaze/GT 입력 차단, compact checkpoint restore, 실제 geometry와224px crop 일치를 검증한다.
2. Case64/73의 실데이터 smoke와 P0 원 분포 재현을 확인한다. CUDA와 데이터 접근을 확인한다.
3. P0부터P5까지 validation93장 × generation3seed의 inference 비교를 실행한다. Argmax는 별도 실험으로 보고한다.
4. Train376장 utility head15epoch를 학습하고 occupancy source와 동일 policy에서 비교한다. Training seed 추가 및 후속 decoder adaptation은 결과에 따라 이어간다.
5. Head 학습·generation·사후평가의 입력과 checkpoint hash를 저장하고 report에 완료 범위를 명시한다. 특징 재추출과 classifier 재학습 전에는 기존 CXR AUROC를 새 경로의 성과로 인용하지 않는다.

장시간 작업은 durable runner에서 stage별 상태와 return code를 기록한다. GPU 작업과 보고서 생성을 모두 완료해야 state를 complete로 표시한다. 기존 checkpoint를 덮어쓰지 않는다.

## 참고 연구의 적용 범위

[HAT CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Unifying_Top-down_and_Bottom-up_Scanpath_Prediction_Using_Transformers_CVPR_2024_paper.html)은 foveated representation과 sequential dense prediction을 사용한다. Static proposal의 위치 반응과 순차 경로 선택을 함께 평가하는 구조적 참고이며, 여기의 mixture 식이나 계수를 검증한 연구는 아니다.

[GazeSearch와 ChestSearch](https://arxiv.org/html/2411.05780v1)는 특정 finding 탐색을 위한 gaze 정제와 조건부 경로 예측을 다룬다. 사람의 자연 관측 경로와 finding 탐색 목적을 구분해야 한다는 근거로 참고한다. 현재 breast CSV를 GT box로 재구성해 평가 정답을 유리하게 바꾸거나 원 benchmark 수치와 직접 비교하지 않는다.
