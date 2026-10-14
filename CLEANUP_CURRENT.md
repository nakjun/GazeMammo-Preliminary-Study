# 현재 설정 보존 및 정리

## 2026-10-14 Git 업로드 전 종료 runs 디렉터리 정리

최신 연구와 무관한 종료 run 디렉터리 **30개**, 파일 **1,094개**를 정리했다. 삭제 전 JSON/CSV/MD/config/source/log 등 작은 증거 **521개**를 `archive/retired_runs_20261014.tar.gz`로 보관하고, archive에서 각 파일 SHA256을 재검증했다. 큰 tensor·종료 모델/optimizer·생성 이미지 등은 archive에 포함하지 않았다. 디렉터리 삭제량은 **2.88GiB**(3,096,969,276 bytes), archive를 제외한 순 확보량은 **2.88GiB**(3,094,409,899 bytes)이며 `runs`는 **21.71GiB**다.

현행 round03/06/08/09/10/12/18/20, v2 canonical·experiment, 최고 Grid14+Gaze4 3seed·시각화와 diagnosis selector/selector extension/quota의 원래 lineage, 선택 trajectory, pinned runtime·공개 weights 및 frozen TEST 근거는 유지했다. 보호 metadata가 참조하는 후보38개는 제거하지 않았다. 원본 가능성이 있는 canonicalv1 PNG도 보존했다. 남긴 run 파일 **69,150개**의 존재·크기를 확인하고 **69,102개**의 SHA256을 비교했으며 오류는 **0건**이다.

종료 run의 고정 경로를 사용하는 과거 scripts는 해당 실험을 재생성해야 한다. 아래 archive의 압축을 풀면 작은 증거만 복원되며, 삭제한 checkpoint/tensor/이미지는 복원되지 않는다. 이전 대형 cache 정리 원장은 그 시점의 보호 검증 기록이며, 이번에 종료 runs로 추가 정리한 파일은 별도 원장에 명시했다.

- [종료 run 삭제·보관·보호 참조 제외·검증 원장](cleanup_retired_runs_20261014.json)
- 증거 복원: `tar -xzf archive/retired_runs_20261014.tar.gz`

## 2026-10-14 완료 실험 중간 캐시 정리

완료된 과거 실험의 재생성 가능한 tensor 캐시 **12,472개**, **21.06GiB**(22,611,168,675 bytes)를 삭제했다. `runs`의 실제 파일 크기는 정리 후 **24.59GiB**다. 삭제 대상은 round9 boundary adaptation/round6 multiscale feature records, round05/07/17 boundary records, round17 restored frozen maps, 구 official trajectory gaze cache의 이미지별 tensor다. 숫자 image ID cache만 제거하고 index·manifest·JSON·CSV·감사·source·config 및 모델 checkpoint를 남겼다.

중단된 round20 전체(원자적으로 저장된346개 record 포함), round08/09/10/12/18 전체, v2 experiment 및 canonicalv2, 원본 `data`·환자 manifests·문헌·최고 Grid14+Gaze4 3seed 모델과 예측, 선택 trajectory 모델, pinned runtime 및 공개 가중치는 보존했다. optimizer/recovery도 현행 의존 가능성을 고려해 보수적으로 보존했다. 기존 official trajectory canonicalv1은 metadata의 원본 raw_path가 현재 존재하지 않아 PNG와 sidecar를 모두 보존했다.

삭제 후 보호파일 **72,478개**의 존재·크기를 확인하고, 그중 **72,405개**의 SHA256을 삭제 전과 비교했다. 누락·크기 변경·SHA 변경은 **0건**, 삭제 목록의 잔존 파일은 **0개**다. 현재 실행 담당과 제거 후보의 round20 비의존성을 확인했으며 새 학습·추출은 시작하지 않았다.

과거 실험 재학습·특징 재추출·구 TEST 파이프라인 재실행에는 삭제한 캐시를 다시 생성해야 한다. 보존한 report/CSV/감사와 선택 checkpoint는 그대로 사용할 수 있지만, 일부 과거 cache SHA 확인·학습 resume·cache 기반 추론은 재생성 전에는 실행할 수 없다. 삭제 원장에는 각 파일 path·bytes·SHA·사유·재생성 source와 보호파일 검증을 기록했다.

- [정확한 삭제 목록과 보호파일 검증](cleanup_completed_caches_20261014.json)

## 2026-10-12 분류 연구로 전환한 정리

새 방향은 동결된 시선 예측 → 패치 추출 → 양성·악성 분류다. 이전 시선 모델 재학습에만 쓰는 캐시와 미사용 smoke tensor 4,286개를 삭제해 **12.87GiB**를 추가 확보했다. `runs`는 약18GiB에서5.0GiB로 줄었다. 현재 runtime checkpoint·공개 encoder 가중치·원본 데이터·분류/attention 비교 결과·TEST 평가 증거는 유지했다. 주요 training cache의 provenance/config/history는 남겼지만 이전 gaze/ROI 모델 재학습에는 해당 cache를 재생성해야 한다.

사용하지 않는 HAT 실험 소스22개는 SHA 검증 후 `archive/classification_scope_retired_hat_20261012.tar.gz`에 보관했다. `literature/code/HAT/common`의 ScanMatch/MultiMatch 구현은 TEST 후처리 의존성이므로 그대로 유지했다. CXR 이름의 기존 attention 실험도 유방 분류 비교군이라 보존했다.

- [정확한 삭제 파일·bytes·SHA 및 보관 목록](cleanup_classification_scope_20261012.json)
- [정리 후 current 모델 checkpoint9개/source35개 무결성 검사](runs/frozen_current_test_1012/frozen_source_cohort_audit.json)
- [새 개발 계획](plan_gaze_patch_classification_1012.md)

복원 명령: `tar -xzf archive/classification_scope_retired_hat_20261012.tar.gz` (소스 복원이며 삭제한 training tensor는 복원하지 않는다).

## 2026-10-11 이전 정리 기록

현재 보존한 모델은 `runs/dual_target_1011/mammo_clip_unified_runtime/config.json`의 설정입니다. 동일 validation 93장, 3개 seed의 279개 경로에서 NLL **5.766271642459336**, 실제 224px crop 기준 병변 확보율 **85.4838709677%**가 검증됐습니다. 반복 실험 목표는 중단 상태로 유지합니다.

## 보존

- 원본 데이터, 환자 분리 manifest, 문헌 및 결과 보고서.
- 현재 모델과 여섯 단계의 상위 설정, 선택된 checkpoint, 공개 사전학습 가중치, 해시 검증 대상 소스.
- 현재 평가의 확률 캐시·93장 NLL·279개 경로·병변별 결과·독립 감사.
- 원본 ROI14의 비교 좌표와 주요 학습 캐시의 provenance.

## 제거·보관

사용하지 않는 대형 tensor와 과거 전체 확률 trace는 삭제했습니다. 삭제 파일과 용량은 [cleanup_20261011_final.json](cleanup_20261011_final.json)에 기록했습니다. 과거 보고서·개별 경로 요약은 남겼으며, 제거된 과거 실험 캐시는 재실행할 때 재생성해야 합니다.

대형 파일 2,994개를 제거해 **99.04GiB**를 확보했고, `runs`는 약 **117GiB → 18GiB**로 줄었습니다. 현재 모델의 주요 학습 캐시와 원본 데이터는 보존했습니다.

닫힌 실패 실험 및 미실행 보정 실험의 코드 42개는 해시를 확인한 압축 파일로 보관했습니다. 다른 분류·CXR 연구 코드와 현재 해시 검증 대상 코드는 유지했습니다.

- 코드 보관: `archive/obsolete_experiments_20261011.tar.gz`
- 파일 목록·SHA: [보관 manifest](archive/obsolete_experiments_20261011.json)

루트에 흩어져 있던 로그 70개는 `archive/root_logs_20261011.tar.gz`로 보관했습니다. 현재 평가 폴더 안의 로그와 감사 증거는 그대로 유지합니다.

보관한 코드가 다시 필요하면 저장소 루트에서 복원할 수 있습니다.

```bash
tar -xzf archive/obsolete_experiments_20261011.tar.gz
```

정리 후 현재 모델·원본 ROI 인코더·offset head의 전체 CPU 복원과 동일 성능 감사가 통과했습니다. 신규 학습이나 성능 실험은 실행하지 않았습니다.
