"""Reproducible full-dataset EDA. Run: python gaze_mammo/run_eda.py."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/gaze_mammo_mpl')
import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent
DATA, OUT = ROOT / 'data', ROOT / 'eda'
OUT.mkdir(exist_ok=True)
coco = json.loads((DATA / 'annotations/instances_all.json').read_text())
cats = {c['id']: c['name'] for c in coco['categories']}
boxes = defaultdict(list)
for a in coco['annotations']:
    boxes[a['image_id']].append(a)
gazes, gaze_errors, duplicate_gaze = {}, [], []
for lineno, line in enumerate((DATA / 'gaze_data/merged_gaze_data.csv').read_text().splitlines(), 1):
    try:
        name, payload = line.split(';', 1)
        if name in gazes:
            duplicate_gaze.append(name)
        gazes[name] = ast.literal_eval(payload)
    except Exception as e:
        gaze_errors.append({'line': lineno, 'error': str(e)})

def identity(name):
    m = re.match(r'Mass-(Training|Test)_P_(\d+)_(LEFT|RIGHT)_(CC|MLO)_', name)
    if m:
        return 'CBIS-DDSM', 'CBIS_' + m[2], m[3], m[4], m[1]
    m = re.match(r'\d+_([a-f0-9]+)_MG_([LR])_(\w+)_ANON', name)
    if m:
        return 'INbreast', 'IN_' + m[1], m[2], m[3], 'unspecified'
    return 'unknown', name, 'unknown', 'unknown', 'unspecified'

def analyze(im):
    name, w, h = im['file_name'], im['width'], im['height']
    src, group, side, view, split = identity(name)
    r = dict(image_id=im['id'], file_name=name, source=src, inferred_group=group,
             laterality=side, view=view, filename_split=split, annotation_width=w,
             annotation_height=h, n_boxes=len(boxes[im['id']]))
    labels = {cats[a['category_id']] for a in boxes[im['id']]}
    r['image_label'] = '+'.join(sorted(labels)) if labels else 'UNANNOTATED'
    path = DATA / 'image' / name
    try:
        r['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            image.load()
            r.update(width=image.width, height=image.height, mode=image.mode, bytes=path.stat().st_size,
                     dimension_mismatch=(image.size != (w, h)))
            arr = np.asarray(image)
            r.update(pixel_min=int(arr.min()), pixel_max=int(arr.max()),
                     pixel_mean=float(arr.mean()), pixel_std=float(arr.std()),
                     zero_fraction=float((arr == 0).mean()))
            r['decode_error'] = ''
    except Exception as e:
        r['decode_error'] = str(e)
    points, scales, invalid = [], [], 0
    for g in gazes.get(name, []):
        try:
            xy, s = np.asarray(g[0], dtype=float), float(g[1])
            if xy.shape != (2,) or not np.isfinite(xy).all() or not np.isfinite(s) or s <= 0:
                raise ValueError('invalid coordinate or scale')
            points.append(xy / s)
            scales.append(s)
        except Exception:
            invalid += 1
    p = np.asarray(points).reshape(-1, 2)
    inside = (p[:, 0] >= 0) & (p[:, 0] < w) & (p[:, 1] >= 0) & (p[:, 1] < h)
    hit = np.zeros(len(p), dtype=bool)
    for a in boxes[im['id']]:
        x, y, bw, bh = a['bbox']
        hit |= inside & (p[:, 0] >= x) & (p[:, 0] < x+bw) & (p[:, 1] >= y) & (p[:, 1] < y+bh)
    r.update(gaze_present=name in gazes, n_gaze_raw=len(gazes.get(name, [])), n_gaze_valid=len(p),
             n_gaze_invalid=invalid, n_gaze_in_bounds=int(inside.sum()), n_gaze_out_of_bounds=int((~inside).sum()),
             n_gaze_in_lesion=int(hit.sum()), gaze_lesion_fraction=float(hit.sum()/inside.sum()) if inside.any() and r['n_boxes'] else np.nan,
             scale_min=min(scales) if scales else np.nan, scale_max=max(scales) if scales else np.nan)
    return r

with ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(analyze, coco['images']))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'image_inventory.csv', index=False)
lookup = {i['id']: i for i in coco['images']}
annrows = []
for a in coco['annotations']:
    im = lookup[a['image_id']]
    x,y,w,h = a['bbox']
    annrows.append(dict(annotation_id=a['id'], image_id=a['image_id'], file_name=im['file_name'],
                        category=cats[a['category_id']], x=x,y=y,width=w,height=h,
                        area_fraction=w*h/(im['width']*im['height']),
                        center_x=(x+w/2)/im['width'], center_y=(y+h/2)/im['height'],
                        invalid_bbox=w<=0 or h<=0 or x<0 or y<0 or x+w>im['width'] or y+h>im['height'],
                        area_mismatch=a.get('area') != w*h))
adf = pd.DataFrame(annrows)
adf.to_csv(OUT / 'lesion_inventory.csv', index=False)
duplicates = df[df.duplicated('sha256', keep=False)].sort_values('sha256')
duplicates.to_csv(OUT / 'exact_duplicate_images.csv', index=False)
adf[adf.invalid_bbox].to_csv(OUT / 'invalid_bboxes.csv', index=False)
overflow = np.maximum.reduce([-adf.x.to_numpy(), -adf.y.to_numpy(),
                              (adf.x+adf.width-1500).to_numpy(),
                              (adf.y+adf.height-3000).to_numpy()])
substantive_invalid = int(((overflow > 1e-6) | (adf.width <= 0) | (adf.height <= 0)).sum())
groups = df.groupby(['source','inferred_group']).agg(images=('image_id','size'), split_count=('filename_split','nunique')).reset_index()
groups.to_csv(OUT / 'inferred_groups.csv', index=False)
summary = dict(images=len(df), annotations=len(adf), source_counts=df.source.value_counts().to_dict(),
               image_label_counts=df.image_label.value_counts().to_dict(), lesion_label_counts=adf.category.value_counts().to_dict(),
               view_counts=df.view.value_counts().to_dict(), filename_split_counts=df.filename_split.value_counts().to_dict(),
               inferred_groups=df.groupby('source').inferred_group.nunique().to_dict(),
               group_crosses_filename_splits=int((groups.split_count > 1).sum()),
               image_bytes=int(df.bytes.sum()), corrupt_or_missing_images=int((df.decode_error != '').sum()),
               dimension_mismatches=int(df.dimension_mismatch.sum()), image_modes=df['mode'].value_counts().to_dict(),
               exact_duplicate_files=len(duplicates), exact_duplicate_groups=int(duplicates.sha256.nunique()),
               invalid_boxes=int(adf.invalid_bbox.sum()), area_mismatches=int(adf.area_mismatch.sum()),
               invalid_boxes_tolerance_1e6=substantive_invalid,
               missing_gaze=int((~df.gaze_present).sum()), empty_gaze=int((df.n_gaze_raw == 0).sum()),
               gaze_parse_errors=gaze_errors, duplicate_gaze_rows=duplicate_gaze,
               orphan_gaze_files=sorted(set(gazes)-set(df.file_name)),
               gaze_points=int(df.n_gaze_raw.sum()), invalid_gaze_points=int(df.n_gaze_invalid.sum()),
               out_of_bounds_gaze_points=int(df.n_gaze_out_of_bounds.sum()),
               gaze_count_statistics=df.n_gaze_raw.describe().to_dict(),
               bbox_area_fraction_statistics=adf.area_fraction.describe().to_dict(),
               gaze_lesion_fraction_statistics=df.gaze_lesion_fraction.describe().to_dict())
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
fig, axs = plt.subplots(2,3,figsize=(16,9))
df.image_label.value_counts().plot.bar(ax=axs[0,0],title='Image annotation labels',rot=15)
adf.category.value_counts().plot.bar(ax=axs[0,1],title='Lesion labels',rot=0)
df.groupby(['source','view']).size().plot.bar(ax=axs[0,2],title='Source / view',rot=25)
axs[1,0].hist(df.n_gaze_raw,bins=40); axs[1,0].set_title('Gaze samples per image')
axs[1,1].hist(adf.area_fraction*100,bins=40); axs[1,1].set_title('Lesion area / image (%)')
axs[1,2].hist(df.gaze_lesion_fraction.dropna(),bins=30); axs[1,2].set_title('In-bounds gaze fraction in lesion union')
fig.tight_layout();fig.savefig(OUT/'distributions.png',dpi=160);plt.close(fig)
fig, axs=plt.subplots(1,2,figsize=(10,6))
for ax, cat in zip(axs, cats.values()):
    a=adf[adf.category==cat];ax.scatter(a.center_x,a.center_y,s=8,alpha=.4);ax.set(xlim=(0,1),ylim=(1,0),title=cat,xlabel='Normalized x',ylabel='Normalized y')
fig.tight_layout();fig.savefig(OUT/'lesion_locations.png',dpi=160);plt.close(fig)
selected=[]
for src in sorted(df.source.unique()):
    for label in ['MALIGNANT','BENIGN','UNANNOTATED']:
        candidates=df[(df.source==src)&(df.image_label==label)]
        if len(candidates):selected.append(candidates.iloc[len(candidates)//2])
fig,axs=plt.subplots(2,3,figsize=(12,15));axs=axs.ravel()
for ax, (_,r) in zip(axs,enumerate(selected)):
    with Image.open(DATA/'image'/r.file_name) as im:ax.imshow(im,cmap='gray')
    p=np.array([np.array(g[0])/g[1] for g in gazes[r.file_name]])
    valid=(p[:,0]>=0)&(p[:,0]<r.width)&(p[:,1]>=0)&(p[:,1]<r.height)
    ax.scatter(p[valid,0],p[valid,1],s=2,c='cyan',alpha=.15)
    for a in boxes[r.image_id]:
        x,y,w,h=a['bbox'];ax.add_patch(Rectangle((x,y),w,h,fill=False,edgecolor='red' if a['category_id']==1 else 'lime',linewidth=1.5))
    ax.set_title(f'{r.source} / {r.image_label}\nimage_id={r.image_id}, gaze={r.n_gaze_raw}');ax.axis('off')
for ax in axs[len(selected):]:ax.axis('off')
fig.tight_layout();fig.savefig(OUT/'gaze_bbox_examples.png',dpi=140);plt.close(fig)
report = f'''# GAA-DETR-breast EDA

원본: https://huggingface.co/datasets/yankon0408/GAA-DETR-breast

## 구성 및 라벨
- 영상 {len(df):,}장, 병변 박스 {len(adf):,}개. 모든 실제 영상의 크기: {sorted(set(zip(df.width,df.height)))}.
- 영상 출처(파일명에서 추정): {summary['source_counts']}.
- 박스 라벨: {summary['lesion_label_counts']}; 영상 단위 라벨: {summary['image_label_counts']}.
- 영상당 박스 수: {df.n_boxes.value_counts().sort_index().to_dict()}.
- 촬영 뷰: {summary['view_counts']}. INbreast 파일명의 ML은 그대로 유지했으며 MLO로 임의 변환하지 않았다.
- 영상 모드: {summary['image_modes']}. 영상 파일 합계 {summary['image_bytes']/1024**3:.3f} GiB.

## 시선 데이터
- 헤더 없는 세미콜론 구분 파일: `filename;[[[x,y],scale], ...]`. `ast.literal_eval`로 읽었다.
- 각 샘플의 원영상 좌표는 `(x/scale, y/scale)`이다. 확대 배율은 영상 내에서도 달라지므로 샘플마다 복원했다.
- 전체 {summary['gaze_points']:,}개, 영상당 중앙값 {df.n_gaze_raw.median():.0f}, 범위 {df.n_gaze_raw.min()}–{df.n_gaze_raw.max()}개.
- 잘못된 샘플 {summary['invalid_gaze_points']}개; 복원 후 영상 밖 좌표 {summary['out_of_bounds_gaze_points']:,}개 ({df.n_gaze_out_of_bounds.sum()/df.n_gaze_valid.sum():.2%}). 음수 좌표도 제외했다.
- 박스가 있는 영상의 시선 중 병변 박스 합집합에 들어가는 비율(영상별, 영상 안 좌표를 분모로 사용): 중앙값 {df.gaze_lesion_fraction.median():.2%}, 평균 {df.gaze_lesion_fraction.mean():.2%}.
- 이 비율은 박스 크기·표본 반복·시선 수집 방식의 영향을 받는다. 타임스탬프나 판독자 식별자가 없어 체류 시간 또는 판독자별 통계로 해석하지 않는다.

## 품질 검사
- 누락/디코딩 오류 영상 {summary['corrupt_or_missing_images']}개, COCO 크기 불일치 {summary['dimension_mismatches']}개.
- 잘못된/영상 밖 박스 {summary['invalid_boxes']}개, 박스 면적 불일치 {summary['area_mismatches']}개.
- 위 박스 검사 중 1e-6 픽셀 수치 허용오차를 적용해도 남는 범위 이탈은 {substantive_invalid}개다. 최대 경계 이탈은 {max(0, float(overflow.max())):.2f} 픽셀. `invalid_bboxes.csv`에 따로 저장했다. 원본은 수정하지 않았으며 학습 전 경계 clipping을 검토해야 한다.
- 시선 누락 {summary['missing_gaze']}장, 빈 시선 {summary['empty_gaze']}장, 파싱 오류 {len(gaze_errors)}행, 중복 시선 행 {len(duplicate_gaze)}개.
- 바이트 수준 동일 영상 {summary['exact_duplicate_files']}장 / {summary['exact_duplicate_groups']}그룹. 픽셀 수준/유사 영상 중복은 이 검사로 확인하지 않았다.
- 병변 면적 비율 중앙값 {adf.area_fraction.median():.2%}, 최대 {adf.area_fraction.max():.2%}.

## 학습 시 고려사항
- 이 저장소에는 단일 COCO `instances_all.json`만 있다. Hugging Face의 자동 train 표시는 검증된 연구용 데이터 분할이 아니다.
- CBIS-DDSM 파일명에 있는 원본 분할: {summary['filename_split_counts']}. 출처별 추정 그룹 수: {summary['inferred_groups']}.
- CBIS의 P 번호와 INbreast 파일명의 해시를 그룹 키로 추정했다. INbreast 해시가 실제 환자 식별자인지는 제공 파일만으로 검증되지 않았으므로 원본 메타데이터 확인이 필요하다.
- 원본 파일명 분할을 가로지르는 추정 그룹 {summary['group_crosses_filename_splits']}개. 환자별 분할을 확정한 후 시선과 영상, 병변 라벨을 함께 묶어 분할해야 한다.
- 박스가 없는 영상은 `UNANNOTATED`로 표시했다. 데이터 카드에 일부 미세 소견은 미주석일 수 있다고 명시되어 있어 정상 판정이나 완전한 음성 라벨로 사용하면 안 된다.
- 양성/악성은 병변 단위이며, 혼합 라벨 영상은 별도로 유지한다. 시선 좌표의 공간적 일치를 임상적 판독 정확도로 해석하지 않는다.

## 결과 파일
- `image_inventory.csv`: 영상별 출처/추정 그룹/뷰/화소 통계/시선/박스 수/검사 결과.
- `lesion_inventory.csv`: 병변별 좌표/면적/정규화 중심/박스 검사.
- `summary.json`, `inferred_groups.csv`, `exact_duplicate_images.csv`: 집계 및 품질 결과.
- `distributions.png`, `lesion_locations.png`, `gaze_bbox_examples.png`: 분포 및 시선·박스 예시. 예시는 산점도이며 Gaussian heatmap이 아니다.

재실행: `python gaze_mammo/run_eda.py` (numpy, pandas, Pillow, matplotlib 필요).
'''
(OUT/'report.md').write_text(report)
import html
(OUT/'report.html').write_text('<meta charset="utf-8"><title>GAA-DETR-breast EDA</title><style>body{max-width:1100px;margin:40px auto;font-family:sans-serif}pre{white-space:pre-wrap;line-height:1.7}img{max-width:100%}</style><pre>'+html.escape(report)+'</pre>'+''.join(f'<img src="{f}">' for f in ['distributions.png','lesion_locations.png','gaze_bbox_examples.png']))
print(json.dumps(summary, indent=2, ensure_ascii=False))
