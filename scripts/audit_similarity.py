"""Find pixel duplicates and thumbnail-near-duplicate candidates for manual review."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import json
import numpy as np
import pandas as pd
from PIL import Image
from src.data import ROOT

if __name__=='__main__':
    frame=pd.read_csv(ROOT/'manifests/all.csv');hashes=[];thumbs=[]
    for _,r in frame.iterrows():
        with Image.open(ROOT/'data/image'/r.file_name) as source:
            im=source.convert('L');a=np.asarray(im);hashes.append(hashlib.sha256(a.tobytes()).hexdigest())
            t=np.asarray(im.resize((32,64)),dtype=np.float32).flatten()/255
            thumbs.append((t-t.mean())/(t.std()+1e-6))
    frame['pixel_sha256']=hashes;duplicates=frame[frame.duplicated('pixel_sha256',keep=False)]
    duplicates.to_csv(ROOT/'manifests/pixel_duplicates.csv',index=False)
    t=np.array(thumbs);similarity=t@t.T/t.shape[1];pairs=[]
    for i,j in zip(*np.where(np.triu(similarity,k=1)>.995)):
        a,b=frame.iloc[i],frame.iloc[j]
        pairs.append(dict(image_id_a=int(a.image_id),image_id_b=int(b.image_id),thumbnail_correlation=float(similarity[i,j]),
                          same_patient_key=a.patient_key==b.patient_key,cross_split=a.split!=b.split,
                          pixel_equal=hashes[i]==hashes[j]))
    pd.DataFrame(pairs,columns=['image_id_a','image_id_b','thumbnail_correlation','same_patient_key','cross_split','pixel_equal']).to_csv(ROOT/'manifests/near_duplicate_candidates.csv',index=False)
    summary=dict(pixel_duplicate_files=len(duplicates),near_duplicate_candidates=len(pairs),cross_split_candidates=sum(x['cross_split'] for x in pairs),
                 note='Thumbnail correlation is a candidate screen, not proof of duplicate identity; review candidates manually')
    (ROOT/'manifests/similarity_audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
