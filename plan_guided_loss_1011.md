# 배경·경계 억제와 병변 후보 방문을 위한 trajectory loss

2026-10-11. 설계: 주 에이전트. 구현: 기존 sol-low 서브에이전트 `gaze_1011_development`. 기존 `trajectory_1011` 모델·runner는 보존하고 별도 모듈로 확장한다. 이 문서는 설계이며, 아래 계수는 검증된 최적값이 아니다.

## 목적과 supervision의 변화

현재 모델은 실제 시선을 모방하지만, 예측 패치가 검은 배경이나 가장자리에 놓이거나 같은 위치를 반복한다. 고정 validation case73에서는 gaze16와 grid16 모두 병변 박스를 놓쳤다. 분류 정답만으로 경로의 병변 포착 능력을 판단할 수 없다.

목적은 사람의 시선 모방을 유지하면서 **영상에서 예측한 병변 후보를 여러 step 안에 방문하는 경로**를 학습하는 것이다. 밝거나 치밀하다는 이유만으로 조직을 억제하지 않는다. 유방 경계 부근에도 병변이 있으므로 경계를 hard mask로 금지하지 않는다.

기존 gaze-only 학습에 **train의 benign 및 malignant 병변 박스**를 추가한다. 따라서 기존 모델과 supervision이 다른 실험이다. 박스가 없는 영상은 음성으로 간주하지 않고 proposal/visit loss를 생략한다. 추론 입력은 영상과 생성 설정뿐이다. validation/test 박스·진단 라벨·관측 gaze는 추론에 전달하지 않는다. Train annotation loader는 image/patient allowlist를 강제한다. Validation 박스와 gaze는 경로 생성 이후 별도 평가에만 사용한다.

## 1. 영상에서 계산하는 geometry와 병변 후보

기존 checkpoint `runs/full_1011_seed42/improved/best.pt`에서 시작한다. Visual backbone은 고정하고 memory projection, decoder와 새 `Linear(D,1)` proposal head를 학습한다. Proposal은 `memory['keys']`의 24×48 cell마다 logit을 출력한다.

기존 foreground 추정(밝기 threshold10, 최대 연결 성분, hole filling/closing)은 **근사 유방 마스크**로만 사용한다. 이는 병변 segmentation이나 검증된 artifact 검출기가 아니다. 계산 실패/빈 mask이면 geometry 항과 bias를 0으로 하고 해당 비율을 기록한다. 원본 1500×3000 좌표에서 실제 224×224 PIL crop과 동일한 중심 반올림을 사용한다.

각 후보 cell c에 대해:

- F(c): padding을 포함한 crop 면적 중 foreground 면적 비율.
- A(c): crop 면적 중 영상 내부 면적 비율. 평가용으로 별도 기록한다.
- D(c): 중심의 foreground 내부 경계 거리. 영상 외곽도 경계다.
- C_bg(c) = 1 − F(c).
- C_edge(c) = 1[중심이 foreground] × exp(−D(c)/112).
- R(c) = stop_gradient(sigmoid(proposal_logit(c))).

R은 배경 loss를 직접 줄이기 위해 proposal이 모두 1로 치우치는 지름길을 막도록 guidance/geometry/repeat에서 detach한다. Proposal 자체는 아래 박스 감독 loss로 학습한다. 공유 memory에 대한 trajectory gradient는 유지한다.

## 2. 학습·추론에 공통으로 쓰는 guided distribution

기존 decoder의 raw log probability를 ℓ_t(c)라고 할 때:

`q_t = softmax(ℓ_t + ρ [2 R − 3 C_bg − 0.5 C_edge (1−R)])`

병변 후보가 높은 곳은 약한 edge cost를 완화한다. 배경 항은 남기지만 hard exclusion은 하지 않는다. 모델이 주변부 병변을 선택할 가능성을 보존한다. Geometry 추정 실패 시 해당 두 geometry bias만 끈다. ρ는 학습 checkpoint에 저장하여 복원한 추론에서도 같은 값을 쓴다. Raw ℓ와 guided q를 함께 기록한다.

첫 2 epoch는 ρ=0, 다음 3 epoch 동안 1까지 선형 증가시킨다. Proposal을 먼저 안정시키고 강한 경로 유도를 늦춘다. Smoke에서는 ρ=1을 강제해 모든 항의 gradient를 확인한다. 작은 수치 오차 외에는 train/inference guided-step 정의가 같아야 한다.

## 3. Loss

### 기존 gaze 모방과 proposal 학습

`L_gaze = 0.5 hard-NLL(q, observed gaze) + 0.5 Gaussian-CE(q, observed gaze; σ=64px)`

Teacher forcing에서는 유효한 observed prefix만 사용하고 padding을 loss에서 제외한다. 기존 prior head에는 train gaze density에 대한 `L_prior`를 유지한다.

Proposal target y(c)는 train의 모든 benign/malignant 박스 **합집합과 grid cell이 겹치는 면적 비율**이다. 중심이 box 안인지 여부로 이산화하지 않으므로 작은 병변에도 양의 질량이 생긴다. 겹치는 박스를 중복 가산하지 않는다.

`L_prop = 0.5 [−Σ y log(sigmoid(z))/Σy − Σ (1−y) log(1−sigmoid(z))/Σ(1−y)]`

영상별로 정규화하고 존재하지 않는 class 항은 생략한다. Box 내부를 병변 조직 전체로 단정하는 segmentation loss가 아니라 약한 박스 점유도 감독이다. 박스가 없는 경우 이 항 전체를 생략한다.

### 예측 history를 사용하는 free rollout

16 step 동안 q에서 좌표를 sampling하고, 이전 **예측 좌표**의 crop만 다음 step에 넣는다. Sampling한 index/좌표와 crop 위치는 detach하지만 q와 recurrent hidden의 계산 그래프는 유지한다. 아래 loss가 이미 잘못 고른 현재 좌표에만 작용하지 않고 다음 분포를 개선하도록 한다.

이는 discrete action의 gradient를 완전히 추정하는 RL이 아니다. Crop 선택 및 history sampling을 통한 gradient를 생략한 **편향된 미분 가능 surrogate**이며 실제 생성 결과로 별도 검증해야 한다.

`L_geo = mean_t Σ_c q_t(c) [C_bg(c) + 0.25 C_edge(c)(1−R(c))]`

중심만 foreground인지 검사하는 대신 patch 전체의 배경 비율을 줄인다. 실제 병변 방문을 위해 각 train box m에 대해:

`H_m(c) = area(crop(c) ∩ box_m) / area(box_m)`

이를 해당 box의 grid상 최대 달성값으로 나눠 최선의 후보를 1로 만든다. 큰 box를 작은 patch로 전부 포함할 수 없는 상황을 고려한다. 아주 조금만 겹쳐도 성공 처리하는 기존 hit 지표보다 엄격한 연속 target이다.

`u_tm = Σ_c q_t(c) H_m(c)`

`L_visit = mean_m { −log[1 − Π_t (1−u_tm)] }`

여러 step 중 적어도 한 번 병변을 충분히 방문하는 방향으로 학습한다. 모든 step을 병변 중앙에 고정시키는 loss를 피한다. 여러 병변은 각각 항을 가진다. `log1p`/`expm1`과 수치 안정화를 사용한다. History에 조건부인 q의 곱이므로 위 식은 **coverage surrogate이지 실제 rollout hit 확률이 아니다**.

반복 억제는 약하게 적용한다:

`L_repeat = mean_(t>0) Σ_c q_t(c) [1−0.75R(c)] max_(s<t) IoU(crop(c), crop(sample_s))`

첫 step의 반복 비용은 0이다. 병변 후보의 재방문 비용을 낮추며 돌아보기를 금지하지 않는다. Padding된 step 및 invalid geometry를 올바르게 제외한다.

### 기본 조합

`L = L_gaze + 0.25 L_prior + 0.5 L_prop + ρ [0.5 L_geo + 0.5 L_visit + 0.05 L_repeat]`

모든 항은 유효한 영상/step/box에 대해 평균을 내고 개별 값을 로그로 남긴다. Loss의 감소만으로 병변 포착 또는 분류 향상을 주장하지 않는다.

## 4. 동일 초기화로 비교할 ablation

| Variant | 추가 감독 및 guidance |
|---|---|
| gaze_only | 기존 gaze + prior만; 모든 새 bias/loss 비활성 |
| geometry | geometry bias와 L_geo만 추가; proposal/visit/repeat 없음 |
| lesion | proposal bias, L_prop, L_visit만 추가; geometry/repeat 없음 |
| combined | 전체 조합과 약한 repeat |

Combined의 repeat 효과는 이후 combined-no-repeat를 추가해야 분리할 수 있다. 이번 4군만으로 각 항의 독립 기여를 모두 주장하지 않는다. Decoder 학습 조건, frozen backbone, train/val split, 초기 checkpoint, epoch budget와 생성 seed를 맞춘다.

## 5. 검증과 시각화

1. 작은 실제 데이터 smoke: train2–4/val2, 1 epoch, ρ=1. Checkpoint 복원 후 이미지 단독 생성까지 검사한다. 이는 성능 검증이 아니다.
2. 단위/gradient 검증: 배경 고비용 후보 확률 감소, 병변 coverage 증가 방향, padding 무시, box 없음, 빈 mask, 주변부 병변을 선택할 유한 확률, 긴 sequence의 안정성, proposal detach, compact checkpoint 복원 일치.
3. 전체 학습 시 **ρ=1이 된 epoch부터** validation guided gaze NLL로 best checkpoint를 선택한다. ρ=0 warmup을 포함하면 guidance가 꺼진 모델이 최종 best가 될 수 있기 때문이다. Warmup의 last는 저장하지만 best 비교에서 제외한다. 기본 schedule에서는 최소5epoch를 요구한다. 명시적인 force-rho 실험은 지정된 강도에서만 선택하고 해당 값을 기록한다. 이 기준이 병변 방문 최적점과 다를 수 있으므로 동일 observed history의 raw NLL 및 자유 생성 지표를 별도로 보고한다. ROI 지표를 보고나서 선택 기준을 바꾸면 exploratory 실험으로 명시한다.
4. 자유 생성 sampling seeds42/43/44와 argmax에서 평균 F/A, 배경 crop 비율, 경계 거리, 고유 patch 수/반복 IoU, 병변별 최대 patch coverage와 방문율, 관측 gaze에 대한 DTW를 함께 측정한다. 약한 any-overlap와 충분한 coverage를 구분한다.
5. 고정 validation case64/73을 정기 출력한다. 이미지·proposal heatmap·raw/guided 분포·예측 trajectory를 보여주고 GT gaze/box는 생성이 끝난 뒤 별도 패널로 비교한다. Geometry mask 오검출/실패 사례도 확인한다.
6. 이후 분류 이점을 평가하려면 변경된 gaze에서 패치 특징을 **다시 추출**하고 classifier를 다시 학습해야 한다. 기존 feature cache나 CXR attention AUROC는 새 경로의 결과가 아니다. 반복 사용한 validation과 최종 독립 test를 구분한다.

## 6. 구현·저장 범위

신규 `src/guided_trajectory_1011.py`, `scripts/guided_trajectory_1011.py`, 전용 테스트를 서브에이전트에게 맡긴다. 기존 모델은 그대로 보존한다. Frozen visual encoder를 매번 중복 저장하지 않고 base checkpoint의 절대 경로/SHA256과 nonvisual state를 compact checkpoint에 저장한다. 복원 시 SHA를 검증한다. 따라서 base checkpoint는 반드시 보존해야 한다.

이번 요청에서는 구현, 검증, 작은 실제 데이터 smoke와 전체 학습 명령을 준비한다. 전체 학습이나 downstream 분류 개선은 완료한 것으로 표시하지 않는다. 구현 검증 결과는 별도 run 보고서에 기록한다.

## 근거와 적용 범위

- [Kervadec et al., Boundary loss, MIDL 2019](https://proceedings.mlr.press/v102/kervadec19a.html): segmentation에서 distance field와 예측 확률의 적분으로 경계 정보를 미분 가능하게 사용하는 원리를 참고했다. 여기의 crop 배경 비용/약한 내부 경계 비용은 gaze 문제에 맞춘 별도 설계이며 논문의 재현이 아니다.
- [Localization supervision of chest x-ray classifiers using label-specific eye-tracking annotation, 2023](https://www.frontiersin.org/journals/radiology/articles/10.3389/fradi.2023.1088068/full): 시선 기반 localization 감독과 분류 성능을 구분하여 평가할 필요가 있다. 본 설계의 계수, mask, visit surrogate를 검증해주는 근거로 해석하지 않는다.

치밀한 조직/visual artifact에 대한 의심은 밝기만으로 구별되지 않는다. 이 설계는 배경과 경계의 낭비를 줄이는 첫 단계다. 병변 후보 head가 artifact에 잘못 반응하는지 시각화와 별도 평가로 확인해야 한다.

## 이번 runs 정리 완료

오래된 모델·캐시 `.pt` 79개를 삭제하고 smoke 폴더 21개를 정리했다. Smoke의 비모델 산출물은 [압축 보관본](archive/smoke_summaries_20261011.zip)에 검증 후 보관했다. 다른 과거 실험의 결과·로그·시각화는 원래 경로에 남겼다. 순 확보량은 **16.286 GiB**다. 이전 checkpoint 정리의 12.194 GiB와 별도 작업이다.

현재 full gaze/classification/attention/CXR 결과와 의존성, gaze 문헌 기준 모델, 최고 ResNet grid 분류 기준(`spatial_v2_seed42/grid_control`의 fusion 및 features), 선택 ConvNeXt HPO refit, spatial 초기화 모델을 보존했다. 보존 파일 42개의 삭제 전후 SHA256 일치를 확인했다. 삭제된 과거 checkpoint/캐시를 요구하는 실행은 재생성이 필요하다. [상세 삭제·보존 목록](runs_cleanup_20261011.json), [안내](runs/CLEANUP_20261011.md).

## 구현 검증 완료 — 전체 성능 실험은 미실행

서브에이전트가 [loss/model 모듈](src/guided_trajectory_1011.py), [학습·추론 CLI](scripts/guided_trajectory_1011.py), [전용 테스트](tests/test_guided_trajectory_1011.py)를 구현했다. 신규10개를 포함한 관련80개 회귀 테스트가 통과했다. Proposal 계수, gradient 방향, 미래 gaze 차단, proposal detach, 박스 allowlist, 빈 mask/전체 padding, frozen backbone, compact restore를 검증했다. 기본 schedule의 best 선택은 ρ=1 이후로 제한했다.

[최종 smoke](runs/guided_1011_verified_final/summary.json)는 train3/val2, 1epoch, 16step, force-rho1에서 실제 gradient update3회를 완료했다. Compact best checkpoint는 약2.505MiB이며 visual state를 중복 저장하지 않는다. 복원 후 [case64 이미지 단독 추론](runs/guided_1011_verified_final/predict_case64/prediction.json)의 16개 유효 좌표 및 raw/guided 분포 정규화를 주 에이전트도 확인했다. 동일 observed history에서 raw NLL6.5093, guided NLL6.6543이며, 표본2장 smoke 수치로 성능을 평가하지 않는다.

시각점검: [case64](runs/guided_1011_verified_final/visual_checks/epoch_01/case_64/trajectory.png), [case73](runs/guided_1011_verified_final/visual_checks/epoch_01/case_73/trajectory.png). 주 에이전트가 case73을 직접 확인했다. 경로는 주로 조직 내부에 놓였지만 proposal은 넓게 분포한다. 아직 병변 localization이나 benign/malignant 분류 향상을 입증하지 않았다. 현재 주기적 시각화는 영상·proposal·raw/guided 분포이며 GT overlay는 포함하지 않는다. Validation ROI/DTW/3-seed 전체 비교와 downstream 재학습은 후속 검증으로 남겨둔다.

전체 train/val을 사용하는 실행 예시(이번에는 실행하지 않음, 프로젝트 폴더에서):

```bash
python scripts/guided_trajectory_1011.py train \
  --ablation combined --epochs 15 --device cuda:0 --seed 42 \
  --visual-every 3 --output runs/guided_1011_combined_seed42
```

동일 설정에서 `--ablation gaze_only`, `geometry`, `lesion`과 각각 다른 output으로 비교한다. 기본 base checkpoint는 보존된 full_1011 improved/best다. 각 run은 새 output 경로를 요구한다.

## 후속 요청: 서브에이전트 전체 학습 실행

2026-10-11 04:32 UTC, 사용자의 추가 요청에 따라 기존 sol-low 서브에이전트가 combined 전체 학습을 시작했다. 위의 “전체 성능 실험 미실행”은 구현 검증 직후 상태이며, 이 절이 후속 실행 기록이다. 현재는 학습 진행 중으로 완료나 성능 개선을 의미하지 않는다.

- 전체 train412/validation93, 15epoch, seed42, GPU0(RTX5880 48GB), frozen visual backbone.
- Train/validation limit0, mixed gaze, free rollout16step, 기본 rho warmup/ramp, rho1 이후 best 선택.
- tmux `guided_full_1011`; 시작 시 wrapper PID1404092, training PID1404094. 이후 실제 상태는 JSON을 기준으로 확인한다.
- [실행 상태](runs/guided_1011_combined_seed42_monitor/status.json), [학습 로그](runs/guided_1011_combined_seed42_monitor/train.log), [설정](runs/guided_1011_combined_seed42/config.json).
- 3epoch마다 및 마지막 epoch에 고정 validation64/73 시각화. 정상 학습 종료 후 best checkpoint의 이미지 단독 추론 및 GT 박스와의 사후 비교를 자동 실행한다.

대화 종료와 무관하게 tmux에서 학습·후처리를 계속한다. 실행 wrapper는 [run_guided_full_1011.py](scripts/run_guided_full_1011.py)다. 다른 ablation 및 downstream classifier 재학습은 이번 실행에 포함하지 않는다.

초기 준비가 끝난 뒤 주 에이전트가 학습 로그의 epoch1, gradient update60회 및 유한한 loss를 직접 확인했다. 정상 시작 확인이며 전체15epoch 완료 판정은 아니다.
