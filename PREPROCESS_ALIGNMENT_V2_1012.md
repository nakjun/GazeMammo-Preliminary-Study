# CBIS-DDSM 전처리 및 시선 좌표 정합 v2

2026-10-12. 시선 학습용 GAA-PNG와 분류용 full-DICOM의 위치·크기·명암 차이를 수정하는 작업이다. 기존 공식 benchmark의 코드·checkpoint·cache·결과는 그대로 보존했다. 백본 교체나 새 모델 학습은 수행하지 않는다.

후속 사용자 요청에 따라 v2 모델 재학습을 별도로 시작했다. 전처리 수정 단계 자체와 구분하며, 새 실험은 [study_plan.md](runs/official_cbis_v2_experiment_1012/study_plan.md), [실행 상태](runs/official_cbis_v2_experiment_1012/run_manifest.json)를 따른다. tmux `official_cbis_v2_experiment_1012`에서 새 gaze cache → 두 trajectory →15분류기 →새 freeze →v2 공식TEST645장 처리/평가 →독립 감사와v1/v2 비교를 자동 수행한다. 과거 모델이나 cache를 초기화로 재사용하지 않는다.

## 바뀐 데이터 흐름

기존에는 GAA-PNG와 RAW-DICOM을 각각 전처리한 다른 이미지에서 trajectory와 분류 특징을 학습했다. v2에서는 **trajectory와 분류가 같은 RAW-DICOM canonical 픽셀을 사용**한다. GAA-PNG는 기존에 기록된 시선 좌표를 옮기는 영상 정합에만 사용한다.

1. Full-DICOM의 modality rescale 및 MONOCHROME1 반전 후 image-only 유방 영역을 찾는다.
2. 연결된 얇은 필름 테두리·분리된 글자 등을 억제한다. 최대1024 해상도에서 disk opening radius18과 제한된 dilation을 적용한다. radius는 metadata 순서 TRAIN6장으로 정했고 TEST에서 선택하지 않았다.
3. 유방 mask 안의 .5/99.5 percentile로 명암을 정규화하고 mask 밖은 검정으로 만든다. 방향은 원본 그대로 유지한다. Aspect를 유지해1500×3000 canvas에 배치하고 원본 pixel-center→canonical affine와 실제 mask를 저장한다.
4. TRAIN/validation의 대응 GAA-PNG와 RAW-canonical을 image-only affine registration으로 연결한다. Anisotropic scale/translation 및 명시적 horizontal-reflection 후보를 검증하고 ECC로 내부 구조를 맞춘다. Gaze/진단 label/병변 bbox/mask는 정합 계산에 넣지 않는다.
5. 원본 GAA 좌표→GAA canonical→RAW canonical affine를 합성한다. Gaze image alias는 해당 RAW canonical과 **같은 SHA의 픽셀**이며, 별도 sidecar에는 GAA 원본 좌표를 RAW로 옮기는 affine가 들어간다. 기존 read-only `transform_points`도 이 합성 affine를 읽을 수 있다.
6. 불확실한 정합은 identity로 대체하지 않는다. 실패 기록을 남기고 해당 gaze annotation만 새 gaze manifest에서 제외한다. 분류 train/validation2393장의 환자·영상·라벨 목록은 바꾸지 않는다.

등록 quality gate는 maskDice≥.70 및 silhouette 경계를 제외한 내부 edge correlation≥.12다. 이는 고정된 heuristic gate이며 임상적 정확도나 보정된 confidence가 아니다. 경계성 사례와 residual film artifact는 QA overlay에서 검토해야 한다. 원본 또는 변환 좌표가 canvas 밖이면 유효 시선으로 간주하지 않는다.

## 검증과 실제 산출물

전체 재생성과 부모의 독립 검사가 완료됐다. RAW train/validation2393장 모두 변환됐으며 분류 cohort/라벨/환자 split은 그대로다. 시선506쌍 중497개(train420/val77)가 통과했고9개는 정합 실패로 제외했다. 통과한497개 모두 RAW/시선 alias의 pixel SHA가 같고 합성 좌표 변환·원래 invalid sample 보존을 확인했다. 유효한 원본 시선 좌표의 유지율 median은1.0이다.

통과 사례에서 내부 edge correlation median은 train .30635→.96737, validation .32261→.97044였으며, 정합 후 maskDice median은 각각 .98733/.98690이다. 실패9개를 제외한 지표이며, 모든506개 정합이 성공했다고 표현하지 않는다. [전체 부모 검증](runs/official_cbis_preprocess_v2_1012/parent_verification.json)과 [새 protocol](runs/official_cbis_preprocess_v2_1012/protocol.json)에 정확한 분모·source SHA를 기록했다. 모델 재학습과 v2 TEST pixel 처리는 수행하지 않았다.

신규 단위 검증20개가 통과했다. 연결된 필름 테두리/문자, 유방 외곽 보존, MONOCHROME1, half-pixel affine/inverse, scale/translation/reflection, NaN/원본 및 변환 out-of-bounds, 무관한 영상 정합 거부, RAW/gaze alias SHA일치, TEST 차단을 확인했다.

Metadata 순서 TRAIN6/VAL6 paired QA에서 TRAIN6개는 모두 통과하고 VAL은5개 통과·1개 실패했다. 내부 edge correlation median은 TRAIN .300→.967, VAL .351→.916으로 개선됐다. 이는 영상 대응 정도이며 분류 성능 개선 수치가 아니다.

- [12쌍 QA 수치](runs/official_cbis_preprocess_v2_1012_qa/metrics.json)
- [정합 전후와 overlay](runs/official_cbis_preprocess_v2_1012_qa/contact_sheet.png)
- [전체 데이터 재생성 상태](runs/official_cbis_preprocess_v2_1012/state.json)
- 전체 RAW 변환: `runs/official_cbis_preprocess_v2_1012/conversion_trainval.json`
- 전체 시선 정합: `runs/official_cbis_preprocess_v2_1012/registration_report.json`
- 완료 후 새 manifest/protocol: `runs/official_cbis_preprocess_v2_1012/{trainval,test,gaze}.csv`, `protocol.json`

전체2393장 RAW를8 CPU worker로 변환한 뒤506개 train/validation 시선 쌍을 검증한다. 공식 TEST645장의 routing 목록만 보존하고 이번 작업에서는 TEST pixel을 다시 처리하지 않는다. v2 TEST 변환은 새 v2 protocol에 맞는 model freeze와 명시적 실행 flag가 있어야 가능하다.

## 실행 및 기존 cache와의 관계

```bash
python scripts/prepare_official_cbis_preprocess_v2_1012.py cohort
python scripts/prepare_official_cbis_preprocess_v2_1012.py converttrainval --workers 8
python scripts/prepare_official_cbis_preprocess_v2_1012.py register
python scripts/prepare_official_cbis_preprocess_v2_1012.py qa
python -m unittest tests.test_official_cbis_preprocess_v2_1012 tests.test_official_cbis_registration_v2_1012 tests.test_official_cbis_preprocess_dataset_v2_1012
```

tmux `cbis_preprocess_v2_1012`에서 실행하며 monitor 창에 로그를 표시한다. `pipeline.log`, 각 stage log, progress JSONL, source/image/mask SHA를 남긴다. 일부만 처리한 `--limit` 실행은 full protocol을 publish하지 않는다. 분류 RAW 변환 실패는 기록 후 중단하며 분류 영상을 임의로 삭제하지 않는다.

기존 v1 학습용 visual cache와 trajectory/classifier checkpoint는 새 픽셀·좌표에 재사용하지 않는다. 이후 v2에서 학습하려면 새 protocol로 cache와 모델을 생성해야 한다. 새 protocol은 기존 read-only cohort validator를 위해 동일한 cohort kind를 유지하되 `preprocessing_version`과 source/manifest/image SHA가 다르다. **v1 `prepare_official_cbis_1012.py convert` 대신 반드시 v2 전처리 CLI를 사용**한다. 기존 freeze.json도 v2 TEST 처리에 재사용할 수 없다.
