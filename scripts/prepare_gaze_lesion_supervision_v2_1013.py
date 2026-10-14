"""TRAIN-only COCO lesion boxes transformed by existing verified canonical affine."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from PIL import Image,ImageDraw
from src.official_cbis_classifier_1012 import digest

def main():
 out=Path('runs/gaze_focus_auroc88_1013/round06_lesion_trajectory_pilot42');out.mkdir(exist_ok=True);c=torch.load('runs/official_cbis_v2_experiment_1012/development.pt',weights_only=False,map_location='cpu');train={r['source_sn']:r for r in c['rows'] if r['split']=='train'};j=json.load(open('data/annotations/instances_all.json'));images={im['id']:im for im in j['images']};gaze=pd.read_csv('runs/official_cbis_preprocess_v2_1012/gaze.csv');byname={r.file_name:r for r in gaze[gaze.split=='train'].itertuples()};boxes={};excluded=[];accepted=[]
 for annotation in j['annotations']:
  im=images[annotation['image_id']];name=im['file_name'];sn=name.rsplit('_',1)[0];row=train.get(sn)
  if row is None:continue
  if not annotation.get('bbox'):continue
  meta_path=Path(f"runs/official_cbis_preprocess_v2_1012/canonical/gaze/{im['id']}.png.json")
  if name not in byname or not meta_path.exists():excluded.append(dict(source_id=im['id'],reason='no_registered_TRAIN_gaze_alias'));continue
  meta=json.load(open(meta_path));q=meta['registration']['quality'];
  if (not q.get('passed') or q.get('registration_confidence',0)<.5 or q.get('mask_dice',0)<.85) or meta['canonical_sha256']!=row['image_sha256'] or meta['registration']['image_id']!=row['image_id']:excluded.append(dict(source_id=im['id'],reason='registration_identity_or_quality_failed'));continue
  assert list(meta['original_size'])==[im['width'],im['height']] and digest(row['canonical_path'])==row['image_sha256'];a=np.asarray(meta['affine']);x,y,w,h=annotation['bbox'];corners=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]);mapped=corners@a[:,:2].T+a[:,2];lo=mapped.min(0);hi=mapped.max(0);cliplo=np.maximum(lo,0);cliphi=np.minimum(hi,[1500,3000]);origarea=np.prod(hi-lo);area=np.prod(np.maximum(cliphi-cliplo,0));ratio=area/max(origarea,1)
  if not np.isfinite(mapped).all() or area<=0 or ratio<.8:excluded.append(dict(source_id=im['id'],reason='outside_canvas_or_clip_area_below80percent'));continue
  box=np.r_[cliplo,cliphi]/[1500,3000,1500,3000];boxes.setdefault(str(row['image_id']),[]).append(box.tolist());accepted.append(dict(image_id=row['image_id'],source_id=im['id'],source_file=name,patient_key=row['patient_key'],raw_path=row['canonical_path'],source_box=annotation['bbox'],canonical_corners=mapped.tolist(),canonical_box=np.r_[cliplo,cliphi].tolist(),clip_area_ratio=float(ratio),confidence=q['registration_confidence'],metadata_sha256=digest(meta_path)))
 index=torch.load('runs/official_cbis_v2_experiment_1012/gaze_cache/index.pt',weights_only=False,map_location='cpu');records=[torch.load(p,weights_only=False,map_location='cpu') for p in index['records']];ids={str(r['image_id']) for r in records if r['split']=='train'};report=dict(state='complete',TRAIN_only=True,boxes=boxes,accepted=accepted,excluded=excluded,registered_bbox_images=len(boxes),with_teacher_record=len(set(boxes)&ids),teacher_train_images=len(ids),source_sha256=digest(__file__),annotation_sha256=digest('data/annotations/instances_all.json'));(out/'lesion_supervision_v2.json').write_text(json.dumps(report,indent=2))
 shown=[]
 for r in sorted(accepted,key=lambda r:(r['confidence'],r['image_id'])):
  if r['image_id'] in shown:continue
  source=Image.open(Path('data/image')/r['source_file']).convert('RGB');raw=Image.open(r['raw_path']).convert('RGB');draw=ImageDraw.Draw(source);x,y,w,h=r['source_box'];draw.rectangle([x,y,x+w,y+h],outline='orange',width=8);ImageDraw.Draw(raw).rectangle(r['canonical_box'],outline='orange',width=8);source.thumbnail((375,750));raw.thumbnail((375,750));panel=Image.new('RGB',(750,790),'white');panel.paste(source,(0,40));panel.paste(raw,(375,40));ImageDraw.Draw(panel).text((10,10),f"TRAIN id{r['image_id']}: source bbox -> registered model input; confidence {r['confidence']:.3f}",fill='black');panel.save(out/f"registration_v2_overlay_{r['image_id']}.png");shown.append(r['image_id'])
  if len(shown)==3:break
 print(json.dumps(dict(registered_bbox_images=len(boxes),teacher_overlap=len(set(boxes)&ids),excluded_annotations=len(excluded),overlay_ids=shown)),flush=True)
if __name__=='__main__':main()
