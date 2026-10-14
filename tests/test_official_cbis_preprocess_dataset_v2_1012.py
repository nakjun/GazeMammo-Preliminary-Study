"""Dataset v2 tests use image-only fixtures and never open TEST pixels."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from PIL import Image
from src.official_cbis_data_1012 import transform_points, sha256_file

spec = importlib.util.spec_from_file_location('prepare_v2', Path(__file__).parents[1]/'scripts/prepare_official_cbis_preprocess_v2_1012.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture_row(tmp_path):
    gaa, raw = tmp_path/'gaa.png', tmp_path/'raw.png'
    Image.new('L',(32,64),127).save(gaa)
    Image.new('L',(1500,3000),90).save(raw)
    Image.new('L',(1500,3000),255).save(str(raw)+'.mask.png')
    return dict(image_id=3, source_image_id=17, patient_key='p', split='train', label=1,
                source_path=str(gaa), canonical_path=str(raw),
                gaze_canonical_path=str(tmp_path/'canonical/gaze/17.png'))


def test_alias_and_composed_affine_old_reader(tmp_path, monkeypatch):
    row = fixture_row(tmp_path)
    monkeypatch.setattr(m,'canonical_image_v2',lambda _,**kw: (Image.new('L',(1500,3000)),dict(original_size=[32,64],affine=[[2,0,3],[0,3,4]],preprocessing=m.PREPROCESS_V2),Image.new('L',(1500,3000),255)))
    matrix = np.array([[1,0,7],[0,1,9],[0,0,1]],float)
    monkeypatch.setattr(m,'estimate_registration_v2',lambda *_,**kw: (matrix,dict(passed=True)))
    monkeypatch.setattr(m,'dependencies',lambda: [Path(__file__)]*3)
    audit = m.register_pair(row,tmp_path)
    target = Path(row['gaze_canonical_path'])
    meta = json.loads(Path(str(target)+'.json').read_text())
    assert sha256_file(target)==sha256_file(row['canonical_path'])
    np.testing.assert_allclose(transform_points([[1,2]],meta),[[12,19]])
    assert meta['original_size']==[32,64]
    assert audit['source_image_id']==17


def test_failed_registration_never_creates_identity_alias(tmp_path, monkeypatch):
    row = fixture_row(tmp_path)
    monkeypatch.setattr(m,'canonical_image_v2',lambda _,**kw: (Image.new('L',(1500,3000)),dict(original_size=[32,64],affine=[[1,0,0],[0,1,0]]),Image.new('L',(1500,3000),255)))
    monkeypatch.setattr(m,'estimate_registration_v2',lambda *_,**kw: (np.eye(3),dict(passed=False,failure_reason='bad_pair')))
    monkeypatch.setattr(m,'dependencies',lambda: [Path(__file__)]*3)
    audit = m.register_pair(row,tmp_path)
    assert audit['quality']['failure_reason']=='bad_pair'
    assert not Path(row['gaze_canonical_path']).exists()


def test_test_requires_new_freeze_before_payload_read(tmp_path, monkeypatch):
    monkeypatch.setattr(m,'check_state',lambda _: {})
    monkeypatch.setattr(m,'validate_protocol',lambda _: {})
    monkeypatch.setattr(m,'load_manifest',lambda _: (_ for _ in ()).throw(AssertionError('TEST payload was read')))
    try:
        m.convert(tmp_path,'test')
    except ValueError as exc:
        assert 'TEST requires' in str(exc)
    else:
        raise AssertionError('TEST conversion was permitted')


def test_failed_pair_exclusion_record_and_identity(tmp_path, monkeypatch):
    from src.official_cbis_data_1012 import _write_csv, _write_json
    row = fixture_row(tmp_path)
    _write_csv(tmp_path/'gaze_candidates.csv',[row])
    _write_json(tmp_path/'conversion_trainval.json',dict(complete=True))
    monkeypatch.setattr(m,'check_state',lambda _: {})
    monkeypatch.setattr(m,'register_pair',lambda *_: dict(image_id=row['image_id'],source_image_id=row['source_image_id'],quality=dict(passed=False,failure_reason='bad_pair')))
    try:
        m.register(tmp_path)
    except ValueError as exc:
        assert 'No reliable' in str(exc)
    else:
        raise AssertionError('Empty accepted cohort published')
    report=json.loads((tmp_path/'registration_report.json').read_text())
    assert report['original']==1 and report['rejected']==1
    assert report['failures']=={'bad_pair':1}
    assert report['pairs'][0]['image_id']==3
    assert not (tmp_path/'protocol.json').exists()


def test_cohort_preserves_identity_and_never_reads_test_pixels(tmp_path, monkeypatch):
    from src.official_cbis_data_1012 import _write_csv, _write_json
    source=tmp_path/'source'; source.mkdir()
    output=tmp_path/'v2'
    base=dict(image_id=11,patient_key='patient',patient_id='patient',split='val',label=1,
              source_sn='Training_example',raw_path=str(tmp_path/'unreadable_test_payload.dcm'),canonical_path='old')
    gaze=dict(base,source_image_id=8,source_path='unused',gaze_canonical_path='oldgaze')
    _write_csv(source/'trainval.csv',[base])
    _write_csv(source/'test.csv',[dict(base,image_id=12,patient_key='testpatient',split='test')])
    _write_csv(source/'gaze.csv',[gaze])
    _write_json(source/'protocol.json',dict(kind='official_cbis_1012_v1'))
    _write_json(source/'audit.json',{})
    monkeypatch.setattr(m,'validate_protocol',lambda _: dict(patient_splits={'patient':'val'}))
    monkeypatch.setattr(m,'dependencies',lambda: [Path(__file__)])
    monkeypatch.setattr(m,'canonical_image_v2',lambda *_: (_ for _ in ()).throw(AssertionError('Pixel read during cohort clone')))
    m.cohort(source,output)
    row=m.load_manifest(output/'trainval.csv')[0]
    for key in ('image_id','patient_key','patient_id','label','split','source_sn','raw_path'):
        assert row[key]==base[key]
    assert row['canonical_path']==str(output.resolve()/'canonical/raw/11.png')
    assert not (output/'protocol.json').exists()

# Keep this suite runnable without an optional pytest installation.
import tempfile
import unittest
from unittest.mock import patch

class _MonkeyPatch:
    def __init__(self): self.patches=[]
    def setattr(self,obj,name,value):
        item=patch.object(obj,name,value)
        item.start()
        self.patches.append(item)
    def close(self):
        for item in reversed(self.patches): item.stop()

class DatasetV2Tests(unittest.TestCase):
    def run_fixture(self,fn):
        monkeypatch=_MonkeyPatch()
        try:
            with tempfile.TemporaryDirectory() as folder:
                fn(Path(folder),monkeypatch)
        finally:
            monkeypatch.close()
    def test_cohort(self): self.run_fixture(test_cohort_preserves_identity_and_never_reads_test_pixels)
    def test_alias(self): self.run_fixture(test_alias_and_composed_affine_old_reader)
    def test_failure(self): self.run_fixture(test_failed_registration_never_creates_identity_alias)
    def test_test_guard(self): self.run_fixture(test_test_requires_new_freeze_before_payload_read)
    def test_exclusion(self): self.run_fixture(test_failed_pair_exclusion_record_and_identity)

if __name__=='__main__': unittest.main()
