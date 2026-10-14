"""Report inner-selected HPO refits, with paired outer-validation comparisons."""
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
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'runs/fusion_hpo_seed42'));args=parser.parse_args()
    base=Path(args.output);original=ROOT/'runs/grid_fusion_backbones_seed42'
    names=['densenet161','efficientnet_b5','convnext_base','swin_b','vit_b_16']
    resnet=pd.read_csv(ROOT/'runs/spatial_v2_seed42/grid_control/fusion/fusion_logits_predictions.csv')
    rows=[];paired={};trials=[]
    for name in names:
        for path in (base/name).glob('*/metrics.json'):
            if path.parent.name.startswith('refit'):continue
            obj=json.loads(path.read_text());cfg=json.loads((path.parent/'config.json').read_text());cfg.pop('backbone',None)
            trials.append(dict(backbone=name,trial=path.parent.name,inner_auroc=obj['heads']['fusion_logits']['metrics']['auroc'],epoch=obj['best_epoch'],train_auroc=obj['train_metrics']['auroc'],**cfg))
        selected_path=base/name/('selected_final.json' if (base/name/'selected_final.json').exists() else 'selected.json')
        if not selected_path.exists():continue
        selected=json.loads(selected_path.read_text());path=base/name/selected.get('refit_dir','refit')
        if not (path/'metrics.json').exists():continue
        selected=json.loads(selected_path.read_text());cfg=selected['config'];obj=json.loads((path/'metrics.json').read_text());before=json.loads((original/name/'metrics.json').read_text())
        current=obj['heads']['fusion_logits'];old=before['heads']['fusion_logits']
        refined=json.loads((base/name/selected.get('trial','refine_regularized' if cfg['augmentation'] else 'refine_minimal')/'metrics.json').read_text())
        assert selected['epoch']>cfg['warmup']
        pred=pd.read_csv(path/'fusion_logits_predictions.csv');previous=pd.read_csv(original/name/'fusion_logits_predictions.csv')
        columns=['image_id','patient_key','label']
        for reference in [previous,resnet]:pd.testing.assert_frame_equal(pred[columns].sort_values('image_id').reset_index(drop=True),reference[columns].sort_values('image_id').reset_index(drop=True))
        paired[name]=dict(vs_original=cluster_bootstrap(pred,previous,1000,42),vs_resnet50=cluster_bootstrap(pred,resnet,1000,42))
        rows.append(dict(backbone=name,original_auroc=old['metrics']['auroc'],tuned_auroc=current['metrics']['auroc'],difference=current['metrics']['auroc']-old['metrics']['auroc'],
            tuned_auprc=current['metrics']['auprc'],ci_low=current['auroc_ci']['low'],ci_high=current['auroc_ci']['high'],original_ci_low=old['auroc_ci']['low'],original_ci_high=old['auroc_ci']['high'],inner_auroc=selected['inner_auroc'],selected_epoch=selected['epoch'],
            train_auroc=obj['train_metrics']['auroc'],inner_train_auroc=refined['train_metrics']['auroc'],**cfg))
    frame=pd.DataFrame(rows);pd.DataFrame(trials).to_csv(base/'inner_trials.csv',index=False)
    if not rows:raise ValueError('No completed refits')
    frame.to_csv(base/'comparison.csv',index=False);(base/'paired_differences.json').write_text(json.dumps(paired,indent=2))
    text='# Global/local fusion: 원인 진단과 내부 분할 HPO\n\n'
    text+='## 기존 결과가 낮아진 이유\n\n'
    text+='관찰된 학습 곡선에서 ViT-B/16은 train loss0.692→0.391, validation AUROC0.753→0.703으로 변했다. EfficientNet-B5도 loss0.698→0.517 동안 AUROC 최고0.709에서 마지막0.659로 내려갔다. 단순한 학습 시간 부족보다 일반화 저하를 의심할 근거다. 단, 기존 로그에는 train AUROC가 없으므로 이 loss만으로 과적합을 완전히 확정하지 않는다.\n\n'
    text+='원본 train은376장/306환자다. 큰 모델은약86–88M parameters를 무작위 분류 head와 함께 첫 epoch부터 업데이트했다. head warm-up·scheduler·augmentation 없이 모든 백본에 같은lr1e-5를 적용했다. 데이터 규모 대비 자유도와 학습 설정 불일치가 원인 후보이며, 모델 구조가 열등하다는 근거는 아니다.\n\n'
    text+='EfficientNet-B5는 [공식 torchvision 설정](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.efficientnet_b5.html)에서456 입력을 사용한다. 기존224 입력과의 차이가 원인일 가능성은 있지만 이번 튜닝에서는 공간 범위와 입력을 고정했다. 해상도 효과는 검증하지 않았다. fine-tuning 과정의 민감성에 대한 참고는 [How to Fine-Tune Vision Models with SGD](https://arxiv.org/abs/2211.09359)이며, 해당 연구의 optimizer 비교가 현재 AdamW 실험의 원인을 직접 증명하지는 않는다.\n\n'
    text+='## 튜닝 절차\n\n'
    text+='기존 train의 patient crossfit_fold0을 내부 검증으로 사용했다. inner train257장/204환자, inner validation119장/102환자이며 환자 중복이 없다. outer validation93장/77환자 및 최종 test는 hyperparameter와 epoch 선택에 사용하지 않았다. outer validation은 이전 실험에서 이미 반복 사용됐으므로 독립 test가 아니다.\n\n'
    text+='각 백본의 backbone lr1e-6/3e-6/1e-5를6 epochs로 비교했다. head lr1e-4, warm-up2 epochs, batch2×accumulation4를 유지했다. warm-up에는 encoder가 일시 고정되며 이후 전체 encoder를 fine-tuning한다. 이 후보의 checkpoint는 실제 fine-tuning이 시작된 epoch3부터 선택한다. warm-up0 대조군은 epoch1부터 실제 fine-tuning한다.\n\n'
    text+='내부 검증 최고lr에서 minimal(weight decay1e-4, dropout0.1, 증강 없음, 고정lr)과 regularized(weight decay1e-2, dropout0.3, 일관된 수평 반전, cosinelr 마지막0.1배)를 최대12 epochs로 비교했다. warm-up을 일괄 적용하는 가정도 검증하려고 같은lr의 warm-up0 minimal 대조군을 추가했다. patience5이며 최소6 epochs까지 진행했다. 이후 내부 최고 recipe와 epoch를 확정하고 train376장으로 처음부터 재학습했다. refit 중 outer validation 조기 종료를 적용하지 않고 마지막에 평가했다. warm-up 단계의 첫 두 refit 결과를 관찰한 뒤 대조군을 확장한 탐색적 실험이며, 외부 결과로 최종 recipe를 선택하지 않았다. 묶음 recipe에서는 개별 모듈의 인과 효과를 분리하지 않는다.\n\n'
    text+='## Outer validation 비교\n\n| 백본 | 기존 AUROC | 튜닝 AUROC | 변화 | AUPRC | 선택 epoch | warm-up | backbone LR | recipe |\n|---|---:|---:|---:|---:|---:|---:|---:|---|\n'
    for r in frame.itertuples():text+=f'| {r.backbone} | {r.original_auroc:.3f} | {r.tuned_auroc:.3f} | {r.difference:+.3f} | {r.tuned_auprc:.3f} | {r.selected_epoch} | {r.warmup} | {r.backbone_lr:g} | {"regularized" if r.augmentation else "minimal"} |\n'
    text+='\n기존 모델은 outer validation에서 checkpoint를 선택했고, 이번 refit은 inner 선택 epoch를 사용했다. 선택 절차가 달라 수치 차이를 순수한 hyperparameter 효과로 단정하지 않는다. 기존 ResNet50 기준 AUROC0.796. 차이95% CI는 paired 환자 bootstrap1000회로 계산했다:\n\n'
    for name,obj in paired.items():
        a=obj['vs_original'];b=obj['vs_resnet50'];text+=f'- {name}: 기존 동일 백본 대비 {a["low"]:+.3f}–{a["high"]:+.3f}; ResNet50 대비 {b["low"]:+.3f}–{b["high"]:+.3f}.\n'
    text+='\n선택된 내부 checkpoint의 일반화 차이:\n\n| 백본 | Inner train AUROC | Inner validation AUROC |\n|---|---:|---:|\n'
    for r in frame.itertuples():text+=f'| {r.backbone} | {r.inner_train_auroc:.3f} | {r.inner_auroc:.3f} |\n'
    text+='\n## 해석과 한계\n\n'
    positive=int((frame.difference>0).sum());text+=f'완료된 {len(frame)}개 백본 중 {positive}개가 기존 동일 백본보다 outer AUROC가 수치상 높았다. '
    best=frame.sort_values('tuned_auroc',ascending=False).iloc[0];text+=f'튜닝 백본 중 최고는 {best.backbone}의 {best.tuned_auroc:.3f}이다. '
    text+='내부 단일 fold와 단일 seed의 제한된 탐색이며,6-epoch screening은 느린 학습률을 불리하게 평가할 수 있다. 수평 반전·정규화·cosine은 묶음 비교라 개별 기여를 알 수 없다. 데이터 증량과456 입력, layer-wise LR, patient3-fold 재검증은 별도 실험이다. 현재 결과로 작은 데이터 규모가 유일한 원인이라고 단정하거나 구조의 우열을 확정하지 않는다. 최종 test131장과 INbreast는 미사용이다.\n\n'
    text+='스크립트: `scripts/tune_fusion_backbones.py`, 최종 선택 기록: 각 백본 `selected_final.json`의 `refit_dir`, checkpoint: 해당 폴더의 `best.pt`. warm-up 후보만 선택한 중간 기록은 `selected.json`이다.\n'
    (base/'report.md').write_text(text)
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(frame));ax.bar(x-.18,frame.original_auroc,.36,label='Original (outer-selected)');ax.bar(x+.18,frame.tuned_auroc,.36,label='HPO (inner-selected)')
    ax.errorbar(x-.18,frame.original_auroc,yerr=np.stack([frame.original_auroc-frame.original_ci_low,frame.original_ci_high-frame.original_auroc]),fmt='none',ecolor='black',capsize=3)
    ax.errorbar(x+.18,frame.tuned_auroc,yerr=np.stack([frame.tuned_auroc-frame.ci_low,frame.ci_high-frame.tuned_auroc]),fmt='none',ecolor='black',capsize=3)
    ax.axhline(.7955596669750231,color='black',linestyle='--',label='ResNet50 original');ax.set_xticks(x,frame.backbone);ax.set_ylim(.4,1);ax.set_ylabel('Outer validation AUROC / patient bootstrap 95% CI');ax.legend();fig.tight_layout();fig.savefig(base/'comparison.png',dpi=160);plt.close(fig)
