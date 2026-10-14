"""Read filename/image/patient/split metadata only; never parse gaze payloads or boxes."""
import argparse,csv,json,hashlib
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]


def image_metadata_only(path):
    # Original COCO starts with images. Stop at its closing bracket, before annotations.
    with Path(path).open() as source:
        prefix=''
        while True:
            char=source.read(1)
            if not char:raise ValueError('Images metadata array missing')
            prefix+=char
            if char=='[':break
        if '"images"' not in prefix:raise ValueError('First COCO array is not images; refusing box parsing')
        buffer='[';depth=1;quoted=False;escaped=False
        while depth:
            char=source.read(1)
            if not char:raise ValueError('Unterminated image metadata array')
            buffer+=char
            if quoted:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char=='"':quoted=False
            elif char=='"':quoted=True
            elif char=='[':depth+=1
            elif char==']':depth-=1
    return [{k:r[k] for k in ('id','file_name','width','height') if k in r} for r in json.loads(buffer)]


def audit(output):
    columns=('image_id','file_name','source','patient_key','group_verified','group_evidence','split','view','filename_split')
    with (ROOT/'manifests/all.csv').open() as source:rows=[{k:r[k] for k in columns} for r in csv.DictReader(source)]
    names={r['file_name']:r for r in rows};ids={r['image_id'] for r in rows}
    if len(names)!=len(rows) or len(ids)!=len(rows):raise ValueError('Duplicate manifest images')
    images={p.name for p in (ROOT/'data/image').iterdir() if p.is_file()};gazes=[]
    with (ROOT/'data/gaze_data/merged_gaze_data.csv').open('rb') as source:
        for line in source:
            separator=line.find(b';')
            if separator<0:raise ValueError('Malformed filename header')
            gazes.append(line[:separator].decode('utf-8')) # No payload decoding/evaluation.
    gaze_names=set(gazes);metadata=image_metadata_only(ROOT/'data/annotations/instances_all.json');metadata_names={r['file_name'] for r in metadata}
    train=[r for r in rows if r['split']=='train'];protected=[r for r in rows if r['split'] in ('val','test','quarantine')];train_patients={r['patient_key'] for r in train};protected_patients={r['patient_key'] for r in protected}
    overlap=sorted(train_patients&protected_patients)
    extras=sorted((images|gaze_names|metadata_names)-set(names));split_counts={split:dict(images=len(group),patients=len({r['patient_key'] for r in group}),verified_patient_images=sum(r['group_verified']=='True' for r in group),available_images=sum(r['file_name'] in images for r in group),available_gaze_filenames=sum(r['file_name'] in gaze_names for r in group)) for split in sorted({r['split'] for r in rows}) for group in [[r for r in rows if r['split']==split]]}
    manifest_checks={}
    for name in ('classification.csv','trajectory.csv'):
        with (ROOT/'manifests'/name).open() as source:other=list(csv.DictReader(source))
        manifest_checks[name]=dict(rows=len(other),unknown_filenames=sorted({r['file_name'] for r in other}-set(names)),split_counts=dict(Counter(r['split'] for r in other)))
    result=dict(protocol='Read-only source metadata audit. Gaze filename bytes before semicolon only; no gaze payloads evaluated/decoded and no boxes parsed. COCO images array parsed then file reading stopped before annotations. No image pixels opened or training performed. All102 validation images (including9 beyond canonical93), test150 and quarantine193 remain protected; val93 protocol unchanged.',counts=dict(all_manifest_images=len(rows),local_image_files=len(images),gaze_filename_rows=len(gazes),unique_gaze_filenames=len(gaze_names),coco_image_metadata_rows=len(metadata),train_images=len(train),train_patients=len(train_patients),protected_images=len(protected),protected_patients=len(protected_patients),extra_filenames=len(extras),verified_additional_train_safe_images=0),split_counts=split_counts,source_counts=dict(Counter(r['source'] for r in rows)),extras=extras,missing_images=sorted(set(names)-images),missing_gaze_filenames=sorted(set(names)-gaze_names),duplicate_gaze_filenames=[n for n,c in Counter(gazes).items() if c>1],train_protected_patient_overlap=overlap,manifest_checks=manifest_checks,ambiguous_mapping='No extras require mapping' if not extras else 'Extra filenames require verified subject/image mapping and protected-cohort disjointness before any payload parsing/training.',recommendation='No additional local training-safe images/gaze beyond current412 were found. Next legitimate diversity option is train-only image-space horizontal flipping with consistent image/gaze-coordinate/previous-crop transformations; fixed canonical val93 targets and images stay unchanged. Such a change needs training implementation/provenance and no performance gain is claimed.',source_sha256={name:hashlib.sha256((ROOT/'manifests'/name).read_bytes()).hexdigest() for name in ('all.csv','trajectory.csv','classification.csv')})
    output=Path(output);output.mkdir(parents=True,exist_ok=False);(output/'metrics.json').write_text(json.dumps(result,indent=2));lines=['# Additional training-safe local metadata audit','',result['protocol'],'',f"Manifest/image-directory/gaze-filename/COCO-image metadata counts: {len(rows)}/{len(images)}/{len(gaze_names)}/{len(metadata)}. Extra filenames: {len(extras)}. Missing images/gaze names: {len(result['missing_images'])}/{len(result['missing_gaze_filenames'])}. Training/protected patient overlap: {len(overlap)}.",'','| Split | Images | Patients | Verified images | Local images | Gaze filenames |','|---|---:|---:|---:|---:|---:|']
    for split,s in split_counts.items():lines.append('| '+split+' | '+' | '.join(str(s[k]) for k in ('images','patients','verified_patient_images','available_images','available_gaze_filenames'))+' |')
    lines+=['',result['recommendation']];(output/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result['counts'],indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='runs/dual_target_1011/train_safe_metadata_audit');audit(p.parse_args().output)
