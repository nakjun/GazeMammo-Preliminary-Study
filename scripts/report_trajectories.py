"""Compare learned trajectories against simple train-only population priors."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image
from src.data import ROOT
from src.metrics import dtw_distance

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',default=str(ROOT/'runs/trajectory_backbones_seed42'));a=p.parse_args()
    base=Path(a.runs);c=json.loads((base/'experiment_config.json').read_text());d=pd.read_csv(base/'comparison.csv')
    manifest=pd.read_csv(ROOT/'manifests/trajectory.csv');targets=json.loads((ROOT/'manifests/gaze_sequences.json').read_text())
    def target(image_id):return np.array(max(targets[str(image_id)]['segments'],key=len))
    train=manifest[manifest.split=='train'];val=manifest[manifest.split=='val']
    bank=[target(i) for i in train.image_id]
    median=np.array([np.median([x[min(t,len(x)-1)] for x in bank],axis=0) for t in range(c['k'])])
    grid=np.array([c['grid_x'],c['grid_y']]);median=(np.floor(median*grid)+.5)/grid
    prior_rows=[]
    for name,trajectory in [('center_prior',np.full((c['k'],2),.5)),('train_median_prior',median)]:
        errors=[dtw_distance(target(i),trajectory) for i in val.image_id]
        prior_rows.append(dict(backbone=name,normalized_dtw=float(np.mean(errors)),unique_grid_locations=len(np.unique(trajectory,axis=0))))
    (base/'prior_metrics.json').write_text(json.dumps(prior_rows,indent=2))
    results=pd.concat([d,pd.DataFrame(prior_rows)],ignore_index=True)
    results.to_csv(base/'comparison_with_priors.csv',index=False)
    fig,ax=plt.subplots(figsize=(10,5));ax.bar(results.backbone,results.normalized_dtw);ax.tick_params(axis='x',rotation=25)
    ax.set(ylabel='Mean normalized DTW (lower is better)',title='Image-only rollout on 93 validation mammograms')
    fig.tight_layout();fig.savefig(base/'rollout_comparison.png',dpi=160);plt.close(fig)
    # Fixed examples selected by image ID, not by model performance.
    examples=val.sort_values('image_id').iloc[np.linspace(0,len(val)-1,3).astype(int)]
    predictions={name:json.loads((base/name/'validation/predictions.json').read_text()) for name in c['backbones']}
    fig,axs=plt.subplots(3,1+len(c['backbones']),figsize=(18,18))
    for row,(_,r) in enumerate(examples.iterrows()):
        with Image.open(ROOT/'data/image'/r.file_name) as im:image=np.asarray(im.convert('L').resize((300,600)))
        paths=[target(r.image_id)]+[np.array(predictions[name][str(r.image_id)]) for name in c['backbones']]
        for col,(name,path) in enumerate(zip(['Observed stored order']+c['backbones'],paths)):
            ax=axs[row,col];ax.imshow(image,cmap='gray');points=path*[300,600]
            line=LineCollection(np.stack([points[:-1],points[1:]],1),cmap='turbo',linewidths=1.3);line.set_array(np.arange(len(points)-1));ax.add_collection(line)
            ax.scatter(points[:,0],points[:,1],c=np.arange(len(points)),cmap='turbo',s=14)
            ax.set_title(f'{name}\nimage {r.image_id}');ax.axis('off')
    fig.tight_layout();fig.savefig(base/'trajectory_examples.png',dpi=140);plt.close(fig)
    report='# 영상 → gaze trajectory: 백본 비교\n\n'
    report+=f'CBIS-DDSM 학습 376장 / 환자 독립 validation 93장, seed {c["seed"]}. ImageNet 사전학습 백본은 고정했다. 영상 입력은 256×512이며 모든 모델에 동일한 {c.get("decoder_dim",128)}차원, {c.get("decoder_layers",2)}층 autoregressive Transformer decoder를 사용했다. Attention heads={c.get("decoder_heads",4)}, FF multiplier={c.get("ff_multiplier",2)}, dropout={c.get("dropout",.1)}, batch={c["batch_size"]}, lr={c["lr"]}. 위치 grid {c["grid_x"]}×{c["grid_y"]}, 최대 {c["k"]}개 축약 시선 지점을 학습하며 argmax로 {c["k"]}개 좌표를 생성한다. 최초 위치도 영상만으로 예측한다. 테스트 세트는 사용하지 않았다.\n\n'
    report+='| 백본 | Next-step NLL ↓ | Next-step 위치 오차 ↓ | Rollout DTW ↓ | 평균 서로 다른 위치 수 /16 | 학습 epochs |\n|---|---:|---:|---:|---:|---:|\n'
    for _,r in d.iterrows():report+=f'| {r.backbone} | {r.teacher_forced_next_step_nll:.3f} | {r.teacher_forced_normalized_location_error:.3f} | {r.normalized_dtw:.3f} | {r.unique_grid_locations:.2f} | {int(r.epochs)} |\n'
    if d.unique_grid_locations.mean()<4:
        report+=f'\n**현재 baseline의 제한:** argmax 생성의 평균 서로 다른 위치 수는 백본별 {d.unique_grid_locations.min():.2f}–{d.unique_grid_locations.max():.2f}개로 반복 수렴한다. 유용한 16개 패치 탐색 경로를 생성했다고 판단할 수 없다.\n'
    report+='\n모델 크기:\n\n| 백본 | 전체 파라미터 (M) | 학습 파라미터 (M) |\n|---|---:|---:|\n'
    for _,r in d.iterrows():report+=f'| {r.backbone} | {r.total_parameters/1e6:.2f} | {r.trainable_parameters/1e6:.2f} |\n'
    report+='\n단순 prior 비교:\n\n'
    for r in prior_rows:report+=f'- {r["backbone"]}: DTW {r["normalized_dtw"]:.3f}, 서로 다른 위치 {r["unique_grid_locations"]}/16.\n'
    if (base/'sampling_comparison.csv').exists():
        sampling=pd.read_csv(base/'sampling_comparison.csv')
        report+='\n동일 checkpoint의 확률 샘플링(temperature=1, seed=42), 별도 생성 조건:\n\n| 백본 | Rollout DTW ↓ | 평균 서로 다른 위치 수 /16 |\n|---|---:|---:|\n'
        for _,r in sampling.iterrows():report+=f'| {r.backbone} | {r.normalized_dtw:.3f} | {r.unique_grid_locations:.2f} |\n'
        report+='\n샘플링은 반복 수렴을 완화할 수 있지만, 경로의 순서가 실제 판독을 재현한다는 근거는 아니다. 여러 seeds에서 생성 변동과 공간·순서 품질을 추가 검증해야 한다.\n'
    report+='\nNLL과 next-step 위치 오차는 실제 이전 좌표를 입력하는 teacher-forcing 조건이다. DTW는 실제 좌표를 입력하지 않는 전체 rollout 조건이며 두 조건을 혼동하지 않는다. DTW는 좌표를 너비·높이로 정규화한 후 누적 정렬 거리를 두 시퀀스 길이의 합으로 나눈 값이다. 서로 다른 위치 수가 적으면 반복 좌표로 수렴한 결과일 수 있으며, 작은 DTW만으로 유용한 patch selector라고 판단하지 않는다.\n\n'
    best=d.sort_values('normalized_dtw').iloc[0]
    report+=f'이번 validation에서 learned model의 DTW 최저 백본은 **{best.backbone}** ({best.normalized_dtw:.3f})이다. 단일 seed와 validation 선택 결과이므로 확정적인 백본 우열로 해석하지 않는다. 가장 강한 prior의 DTW는 {min(x["normalized_dtw"] for x in prior_rows):.3f}이다.\n\n'
    if float(d.normalized_dtw.min())>=min(x['normalized_dtw'] for x in prior_rows):
        report+='이번 수치에서는 learned model이 train median prior보다 낮은 DTW를 얻지 못했다. 이 결과는 통계적 유의성 검정 결과가 아니다. 샘플링의 높은 위치 다양성도 순차 예측 품질을 입증하지 않는다. 다음 개발에서는 좌표 이웃 관계를 반영하는 입력 표현과 공간적으로 부드러운 감독, 실제 이전 좌표와 생성 좌표의 차이, 백본 미세 조정을 분리해 검증한다.\n\n'
    report+='전처리는 저장 순서를 유지하는 거리 32픽셀 근접점 제거 후 구간별 최대 K 선택이다. 여러 유효 구간이면 보존된 길이가 가장 긴 구간을 사용하고 동률은 먼저 나타난 구간을 사용한다. 시선 수집은 병변 박스 주석 중 이뤄졌으며 시간 순서·세션 병합 규칙은 미확인이다. 따라서 현재 목표는 stored-order sequence prediction이다.\n\n'
    report+='`comparison.csv`, `comparison_with_priors.csv`, `rollout_comparison.png`, `trajectory_examples.png`와 백본별 checkpoint/학습 로그/validation 좌표 및 지표를 저장했다. 입력만으로 예측한 trajectory와 224×224 패치를 시각화하는 CLI는 `scripts/predict_trajectory.py`다. 최종 patch 분류에 사용할 학습 경로는 이 full-train checkpoint로 생성하지 않고 내부 cross-fitting으로 생성해야 한다.\n'
    (base/'report.md').write_text(report)
