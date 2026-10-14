# 16개 gaze 패치의 attention 재설계

작성: 부모 에이전트, 2026-10-11. 분석과 설계는 부모가 수행하고 아래 명세의 구현은 서브에이전트에 위임한다.

## 판단 근거

기존 head는 `patch projection + 좌표 → GRU → attention → global concat`이다. GRU의 각 출력은 해당 패치 단독 특징이 아니라 이전 패치를 포함한 누적 상태이므로 attention weight를 패치 하나의 중요도로 읽기 어렵다. 또한 attention score가 전체 영상 특징을 조건으로 받지 않는다.

`runs/classification_1011_seed42/attention_diagnosis.json`에서 실제 선택 checkpoint를 검사했다.

| 진단 | train376 | validation93 |
|---|---:|---:|
| 원본 패치 특징 간 평균 cosine | 0.8147 | 0.8243 |
| GRU 상태 간 평균 cosine | 0.9512 | 0.9508 |
| attention entropy로 계산한 유효 패치 수 exp(H) | 15.9951 /16 | 15.9953 /16 |
| attention을 균등하게 바꿀 때 평균 절대 확률 변화 | 0.000312 | 0.000290 |
| 학습된 attention AUROC | 0.7433 | 0.7331 |
| 균등 attention으로 추론한 AUROC | 0.7430 | 0.7317 |
| 역순 패치로 추론한 AUROC | 0.7422 | 0.7368 |
| 고유 패치 중심 수 평균 | 15.1968 | 15.1935 |

검사는 추론 시 intervention이며 균등/역순 모델을 재학습한 비교가 아니다. GRU 후 특징 유사성이 높고 attention이 실제 예측을 거의 바꾸지 않는다는 사실은 확인됐지만, 이것만으로 성능 저하의 단일 원인을 확정할 수 없다. 전체 영상 단독 AUROC0.7586에 비해 기존 gaze 결합은0.7331이었다.

추가 geometry 검사에서224px crop16개 영역의 합집합은 train 평균 영상면적12.32%, validation12.12%였다. 이는 영상 canvas 기준으로, 유방조직/병변 coverage가 아니다. patch-only 진단을 강제할 근거가 없으므로 global 판단을 보존한다. 결과는 `patch_geometry_diagnosis.json`에 저장했다.

## 결론

첫 구현은 **전체 영상 조건부 gated MIL attention + global logit의 residual 보정**으로 한다. 분류 attention의 value는 패치 자체의 임베딩으로 두며 GRU 상태를 사용하지 않는다. 순서는 gaze 생성에는 필요하지만, 현재 근거로 분류에도 반드시 필요한지는 확인되지 않았으므로 분류 pooling은 순열 불변으로 둔다.

단순히 softmax 온도를 낮추거나 entropy를 강제로 줄이는 방식은 채택하지 않는다. 더 뾰족한 attention 자체가 더 정확한 분류나 올바른 병변 위치를 뜻하지 않는다. 깊은 Transformer, 복수 class query, patch pseudo-label clustering도 현재376장 자료에 비해 가정과 복잡성이 커서 첫 구현에서는 제외한다.

## 구현 명세

기존 `features.pt`를 재사용한다. 동일한 gaze epoch12, 16개 좌표, 1024차원 patch/global 특징을 고정한다. 기존 코드와 체크포인트는 그대로 재현 가능하게 유지하고 새 module/runner를 만든다.

1. **global 기준점 보존:** 기존 `classification_1011_seed42/global/best.pt`를 복원하고 모든 파라미터를 freeze한다. `.train()` 호출 후에도 이 submodule은 eval이어야 한다. global projection 출력 g(128차원)와 global logit l_g를 사용한다.
2. **패치별 value:** `z_i = GELU(Linear(LayerNorm(x_i)))`, dim64. pooling value는 z_i다. 좌표/시간 embedding이나 GRU로 value를 혼합하지 않는다.
3. **조건부 gated attention:** `s_i = wᵀ[tanh(V z_i + U g) ⊙ sigmoid(W z_i)]`. softmax 온도는1이다. attention dim64. 문맥 제거 ablation에서는 Ug만 제외한다.
4. **정확히 같은 crop의 중복 보정:** 동일 normalized 중심(허용오차1e-6)의 유효 token 개수를 m_i로 세고 `a_i = masked_softmax(s_i - log(m_i))`로 계산한다. 같은 위치가 반복됐다는 이유만으로 전체 가중치가 커지는 것을 막는다. 동일 중심의 crop feature가 동일하다는 현재 cache 조건을 가정한다. 근접 패치는 병변과 주변 조직을 함께 담을 수 있으므로 강제로 제거하지 않는다. IoU>0.5인 패치 쌍은 validation에서3.43%라 첫 단계에서는 exact duplicate만 다룬다.
5. **global 보정:** `z = Σ a_i z_i`, `alpha = sigmoid(Linear([g,z]))`, `delta = Linear(z)`, 최종 `l = l_g + alpha * delta`. gate는 초기 weight0/bias logit(0.1), delta head는 weight/bias 모두0으로 초기화한다. 따라서 학습 전 예측은 global baseline과 정확히 같다. 학습 후 성능을 보장하는 구조는 아니다.
6. **학습 신호:** `BCE(l,y) + 0.25 BCE(local_head(z),y) + 0.01 mean((alpha*delta)^2)`. local auxiliary는 영상/bag 단위이며 개별 patch에 malignant 정답을 부여하지 않는다. 패치 집합이 병변을 놓칠 가능성이 있으므로 보조 손실은 주 손실보다 작게 둔다. dim64, dropout0.2(공통 pooled representation에 적용), AdamW lr3e-4/wd0.01, batch32, 최대60epoch, patience12. 이번 실험 중 값을 재튜닝하지 않는다.
7. **mask/fallback:** padding/제거 patch에 attention0. 유효 patch가 하나도 없으면 attention0, alpha0, local evidence0, 최종 logit=l_g. NaN이 없어야 한다. patch-level 진단과 permutation test는 eval 모드에서 수행한다.

## 미리 정한 비교

같은 source cache와 global checkpoint로 아래 네 variant를 classifier training seeds42/43/44 각각 학습한다. gaze 생성 seed는42로 고정이며 source gaze encoder의 training seed도42다. 전체 파이프라인3seed 재학습이라고 부르지 않는다.

| variant | patches | pooling | global conditioning | residual |
|---|---|---|---|---|
| mean_residual | gaze16 | score0, 중복 보정 평균 | 없음 | 동일 |
| gated_residual | gaze16 | 학습 gated attention | 없음 | 동일 |
| contextual_residual | gaze16 | 학습 gated attention | 있음 | 동일 |
| grid_contextual_residual | grid16 | 학습 gated attention | 있음 | 동일 |

기존 global(0.7586) 및 GRU gaze(0.7331)는 저장된 동일93장 결과를 reference로 둔다. 구조 변경 효과는 mean/gated/contextual 간 비교로 분리한다. seed별 best AUROC 선택과 함께 epoch0 global checkpoint도 유효 후보로 저장한다. best가 epoch0이면 패치 보정을 선택하지 않았음을 명확히 보고한다. epoch0 보존은 validation 기준 보호장치이며 일반화 향상 보장이 아니다.

epoch0 모델의 attention은 학습되지 않은 초기값이므로 시각화와 보고서에 그 사실을 표시한다. all-invalid bag은 local auxiliary loss에서도 제외한다.

AUROC/AUPRC/Brier/threshold0.5 민감도·특이도, seed별 결과와 mean/std를 보고한다. 기존 분류 실험과 같이 patient bootstrap1000회를 사용한다. 개별 seed와3head-seed 확률 평균의 patient bootstrap을 구분하고, ensemble 이득을 단일 head의 이득으로 서술하지 않는다. source 환자 provenance, cache hash, global/gaze checkpoint hash를 저장하고 train/val 환자·image ID 정합과 test 미사용을 검사한다. train cache는 gaze pretraining의 in-sample transfer이며 OOF가 아니다. validation의 반복 사용과 model 선택 편향은 계속 존재한다.

## attention 자체를 검증하는 방법

- entropy/exp(H), 최대 weight, top4 mass, alpha와 residual logit 크기를 image별 기록한다. 낮은 entropy 자체를 성공 기준으로 삼지 않는다.
- attention은 normalized importance이지 malignancy probability가 아니다. `alpha * a_i * delta_weightᵀ z_i`의 signed contribution도 기록하고 residual bias 항을 별도 저장한다. 합계가 residual logit과 일치하는지 검사한다. gate를 고정한 분해이며 인과 설명은 아니다.
- 동일 patch/coordinate/mask를 함께 섞었을 때 fused logit은 같아야 한다.
- duplicate 하나를 추가해도 해당 crop group의 총 weight와 logit이 유지돼야 한다(동일 feature 조건, eval).
- 모든 패치를 마스킹하면 정확히 global baseline으로 돌아와야 한다.
- 선택 모델에서 top1/top4 제거와 고정 seed random1/random4 제거를 비교한다. 마스킹 후 attention·gate를 다시 계산한다. 원래 예측 class 방향의 logit 감소를 기록하되, attention의 인과적 정답이나 ROI 정확도로 해석하지 않는다.
- 고정 validation image64는 GT benign인 케이스로 이전 그림과 같은 원본을 사용한다. 16개 실제 crop gallery에 attention, signed contribution, gaze step을 표시하고 전체 영상 overlay도 저장한다. GT 라벨은 예측 후 표시/검증에만 쓴다. 독립 이미지 추론도 cached evaluation과 맞는지 확인한다.

## 구현 책임과 저장 공간

부모: 위 진단·설계, 구현 리뷰, 결과 해석. 서브에이전트: 새 모델/학습·추론 runner/검증 테스트/attention 시각화 구현. 기존 결과를 덮어쓰지 않고 작은 head checkpoint만 저장한다. feature cache와 대형 gaze checkpoint를 복제하지 않는다. CPU cached-feature 학습부터 사용하여 불필요한 GPU/파일 점유를 줄인다.

필수 테스트: 초기 global 출력 일치, frozen global 파라미터/BN·dropout 상태, 유한한 loss/gradient(초기 residual0에서 auxiliary gradient 포함), mask/all-invalid, permutation, duplicates, signed decomposition, checkpoint round-trip, provenance mismatch 거부, image-only inference. 신규 테스트 후 전체 regression을 실행한다.

## 근거 문헌과 적용 범위

- [Ilse et al., Attention-based Deep Multiple Instance Learning, ICML2018](https://proceedings.mlr.press/v80/ilse18a.html): bag label로 permutation-invariant attention pooling을 학습하는 기본 접근. gated MIL의 출발점으로 사용한다.
- [Li et al., DSMIL, CVPR2021](https://arxiv.org/abs/2011.08939): instance/bag 정보를 결합하는 MIL 연구. 현재 구현은 DSMIL 재현이 아니며 query를 전체 영상에서 얻는 자체 설계다.

이 논문들의 성능이 현재 유방촬영 gaze patch에 그대로 적용된다고 가정하지 않는다. global-conditioned scoring, exact-duplicate 보정과 residual 결합은 현재 관찰을 바탕으로 정한 검증 대상 설계다.


## 구현·전체 실험 완료 (2026-10-11)

부모가 위 설계를 작성했고 기존 sol-low 서브에이전트가 `src/gaze_attention_1011.py`, `scripts/train_attention_1011.py`, `scripts/predict_attention_1011.py`, `tests/test_gaze_attention_1011.py`를 구현했다. 부모가 검토 후 본학습을 실행했다. 신규6개를 포함해 전체58개 테스트 통과.12개 모델의 전체 train376/val93 실험을220.4초에 완료했고 checkpoint는각328–360KiB만 추가됐다.

단일 head3seed 평균 AUROC는 mean residual0.7727±0.0047, gated0.7715±0.0009, contextual0.7706±0.0000, grid contextual0.7661±0.0025다. 기존 global0.7586/GRU gaze0.7331보다 수치상 높지만 global대비 CI는0을 포함하며, 학습된 attention이 평균 pooling보다 우월하다고 확인되지 않았다. 낮은 entropy가 목적이 아니라 분류에 유용한 patch 근거를 안정적으로 더하는 것이 목적이라는 판단을 유지한다.

현재 기준점은 중복 보정 평균+global residual로, learned gated/conditional attention은 실험 후보로 보존한다. 전체 영상 문맥을 score에 넣는 효과는 아직 입증되지 않았다. 순서/중복 불변성 및 global fallback은 부모가 실제93장에서도 독립 검증했다. 고정case64에16개 patch의 attention와signed contribution을 시각화했고 실제 이미지 추론도 캐시와 일치했다.

- [부모의 결과 해석·재현 명령](runs/attention_1011_full/parent_review.md)
- [전체 수치·신뢰구간](runs/attention_1011_full/report.md)
- [16개 패치의 attention와 악성 점수 기여](runs/attention_1011_full/case64_contextual_seed42/crop_gallery.png)
- [원영상 overlay](runs/attention_1011_full/case64_contextual_seed42/overlay.png)
