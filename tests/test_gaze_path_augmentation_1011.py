import numpy as np
import torch
from src.gaze_path_augmentation_1011 import selected_indices,augmented_sample


def test_strata_endpoints_and_reproducibility():
    for length in (17,18,31,191,500):
        a=selected_indices(length,62,3);assert np.array_equal(a,selected_indices(length,62,3));assert a[0]==0 and a[-1]==length-1
        assert len(a)==16 and len(set(a))==16 and np.all(np.diff(a)>0)
        for index,stratum in zip(a[1:-1],np.array_split(np.arange(1,length-1),14)):assert index in stratum
    assert not np.array_equal(selected_indices(191,62,3),selected_indices(191,62,4))


def test_previous_crop_only_and_validation_guard():
    pool=dict(coords=torch.arange(40).reshape(20,2).float()/40,patch_vectors=torch.arange(20).float()[:,None].expand(-1,4),raw_indices=torch.arange(20))
    sample=augmented_sample(pool,62,1);indices=sample['pool_indices'];assert torch.equal(sample['vectors'][0],torch.zeros(4))
    assert torch.equal(sample['vectors'][1:,0],torch.tensor(indices[:-1]).float())
    try:augmented_sample(pool,64,1,split='val')
    except ValueError:pass
    else:raise AssertionError('Validation resampling accepted')


def test_short_path_masks_and_canonical_indices():
    pool=dict(coords=torch.rand(3,2),patch_vectors=torch.rand(3,4),raw_indices=torch.arange(3));sample=augmented_sample(pool,62,1)
    assert sample['mask'].sum()==3 and torch.equal(sample['vectors'][3:],torch.zeros(13,4))
    assert np.array_equal(selected_indices(191,62,1,augmented=False),np.linspace(0,190,16).round().astype(int))


def test_pool_source_file_and_mask_guards():
    import json,sys,tempfile
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from train_gaze_path_augmentation_1011 import load_pools
    from train_attention_1011 import digest
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);source=root/'source';source.mkdir();cache=root/'pool';cache.mkdir();source_header=dict(gaze_sha256='g',manifest_sha256='m',all_manifest_sha256='a');(source/'provenance.json').write_text(json.dumps(source_header))
        pool=dict(coords=torch.rand(33,2),patch_vectors=torch.rand(33,4),raw_indices=torch.arange(33),patient_key='train_patient',split='train',image_sha256='image')
        indices=selected_indices(33,62,0,augmented=False);pool['canonical_pool_indices']=indices.tolist();torch.save(pool,cache/'62.pt');sample=augmented_sample(pool,62,0,augmented=False)
        row=dict(image_id='62',patient_key='train_patient',split='train',coords=sample['coords'],vectors=sample['vectors'],mask=sample['mask'],image_sha256='image')
        header=dict(kind='gaze_stratum_pool_cache_1011',canonical_vector_validation=dict(passed=True,tolerance=0,max_abs_error=0),base_checkpoint_sha256='base',visual_state_sha256='visual',source_cache_provenance_sha256=digest(source/'provenance.json'),annotation_used=False,smoke=True,records=[dict(image_id=62,file_sha256=digest(cache/'62.pt'),canonical_raw_indices=pool['raw_indices'][indices].tolist())],**source_header)
        (cache/'provenance.json').write_text(json.dumps(header));(cache/'status.json').write_text(json.dumps(dict(state='complete')))
        assert 62 in load_pools(cache,[row],'base','visual',source,True)[0]
        changed=dict(row,mask=torch.zeros(16,dtype=torch.bool))
        try:load_pools(cache,[changed],'base','visual',source,True)
        except ValueError:pass
        else:raise AssertionError('Changed canonical mask accepted')
        header['records'][0]['file_sha256']='wrong';(cache/'provenance.json').write_text(json.dumps(header))
        try:load_pools(cache,[row],'base','visual',source,True)
        except ValueError:pass
        else:raise AssertionError('Changed pool file accepted')
