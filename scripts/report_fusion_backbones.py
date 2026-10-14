"""Validation-only report for matched grid-fusion backbone experiments."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/gaze_mammo_mpl')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.data import ROOT
from src.metrics import cluster_bootstrap

if __name__=='__main__':
    base=ROOT/'runs/grid_fusion_backbones_seed42'
    paths={'resnet50':ROOT/'runs/spatial_v2_seed42/grid_control/fusion',**{name:base/name for name in ['densenet161','efficientnet_b5','convnext_base','swin_b','vit_b_16']}}
    rows=[];predictions={}
    for name,path in paths.items():
        if not (path/'metrics.json').exists():continue
        obj=json.loads((path/'metrics.json').read_text());result=obj['heads']['fusion_logits']
        info=json.loads((path/'model_info.json').read_text()) if (path/'model_info.json').exists() else {'parameters':24041412}
        timing=json.loads((path/'timing.json').read_text()) if (path/'timing.json').exists() else {}
        rows.append(dict(backbone=name,best_epoch=obj['best_epoch'],**result['metrics'],ci_low=result['auroc_ci']['low'],ci_high=result['auroc_ci']['high'],parameters=info['parameters'],seconds=timing.get('seconds')))
        predictions[name]=pd.read_csv(path/'fusion_logits_predictions.csv')
    frame=pd.DataFrame(rows).sort_values('auroc',ascending=False);frame.to_csv(base/'comparison.csv',index=False)
    reference=predictions['resnet50'][['image_id','patient_key','label']].sort_values('image_id').reset_index(drop=True)
    for name,pred in predictions.items():
        pd.testing.assert_frame_equal(pred[['image_id','patient_key','label']].sort_values('image_id').reset_index(drop=True),reference)
        if len(pred)!=93 or not np.isfinite(pred.probability).all():raise ValueError(f'Invalid validation predictions: {name}')
    paired={name:cluster_bootstrap(pred,predictions['resnet50'],1000,42) for name,pred in predictions.items() if name!='resnet50'}
    (base/'paired_vs_resnet50.json').write_text(json.dumps(paired,indent=2))
    text='# Grid + global/local fusion 백본 비교\n\n'
    text+='CBIS train376 / validation93, seed42 개발 결과다. 최종 test131 및 INbreast는 사용하지 않았다. 이전 실험 최고 설정인 image-only 분산 grid16개,224×224 patch,512×256 전체 영상, 같은 유방 mask/112px 거리 조건을 사용했다. 경로를 다시 학습하지 않았으며 모든 백본에 같은 patch 중심을 제공했다.\n\n'
    text+='전체 encoder fine-tuning, BN running statistics 고정,128차원 global/MIL projection 및 global/local/fusion BCE 평균을 유지했다. backbone lr1e-5, head lr1e-4, AdamW weight decay1e-4, batch2×accumulation4, 최대12 epochs, validation fusion AUROC 기준 patience4이다. 초기 가중치의 구조·차원이 달라 초기화가 완전히 동일하다는 의미는 아니다.\n\n'
    text+='모든 백본은 torchvision ImageNet-1K 사전학습 가중치를 사용한다. EfficientNet-B5도 입력 크기를224 patch로 맞췄으므로 원래 권장456 입력을 쓴 결과가 아니다. ViT-B는ViT-B/16이며 전체 영상용14×14 위치 임베딩을32×16으로 bicubic 보간하고 CLS 특징을 사용한다. ConvNeXt는 평균 풀링 후 pretrained LayerNorm, Swin은 pretrained LayerNorm 후 평균 풀링을 유지한다. 전처리는 기존 ImageNet 정규화를 유지했다.\n\n'
    text+='| 백본 | 파라미터(M) | 선택 epoch | AUROC | 환자 bootstrap95% CI | AUPRC | Sensitivity* |\n|---|---:|---:|---:|---|---:|---:|\n'
    for r in frame.itertuples():text+=f'| {r.backbone} | {r.parameters/1e6:.1f} | {r.best_epoch} | {r.auroc:.3f} | {r.ci_low:.3f}–{r.ci_high:.3f} | {r.auprc:.3f} | {r.sensitivity:.3f} |\n'
    text+='\n*같은 validation에서 specificity≥90% threshold를 선택했다.\n\nResNet50 대비 paired 환자 bootstrap AUROC 차이95% CI:\n\n'
    for name,value in paired.items():text+=f'- {name}: {value["low"]:+.3f}–{value["high"]:+.3f}.\n'
    text+='\n## 해석\n\n'
    best=frame.iloc[0];text+=f'현재 validation 최고는 {best.backbone}, AUROC {best.auroc:.3f}이다. '
    if paired and all(value['low']<=0<=value['high'] for value in paired.values()):text+='모든 ResNet50 대비 paired AUROC 차이 CI는0을 포함한다. '
    ap_best=frame.sort_values('auprc',ascending=False).iloc[0]
    text+=f'AUPRC 기준으로는 {ap_best.backbone}의 {ap_best.auprc:.3f}이 가장 높다. '
    if best.backbone!='resnet50':
        interval=paired[best.backbone];text+='ResNet50 대비 paired CI가0을 포함하므로 우월성이 확인된 것은 아니다. ' if interval['low']<=0<=interval['high'] else 'ResNet50 대비 paired CI는0을 포함하지 않는다. '
    text+='이 결과는 단일 seed에서 같은 validation으로 checkpoint와 백본을 선택한 개발 비교다. bootstrap은 모델 선택 편향을 보정하지 않으며 사전학습과 architecture 효과도 분리하지 않는다. 여러 seed 재검증 후 설정을 고정하고 최종 test를 평가해야 한다. gaze trajectory 효과를 검증하는 실험과는 별개다.\n\n'
    text+='실행: `python gaze_mammo/scripts/compare_fusion_backbones.py`. 모델과 추론의 backbone 정보는 checkpoint의 `model_config`에 저장했다.\n'
    (base/'report.md').write_text(text)
    fig,ax=plt.subplots(figsize=(9,5));ax.bar(frame.backbone,frame.auroc);ax.errorbar(np.arange(len(frame)),frame.auroc,yerr=np.stack([frame.auroc-frame.ci_low,frame.ci_high-frame.auroc]),fmt='none',ecolor='black',capsize=4);ax.set_ylim(.4,1);ax.set_ylabel('Validation AUROC');fig.tight_layout();fig.savefig(base/'comparison.png',dpi=160);plt.close(fig)
