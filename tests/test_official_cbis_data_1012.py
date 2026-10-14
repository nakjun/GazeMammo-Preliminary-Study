import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from src import official_cbis_data_1012 as data


class OfficialDataTests(unittest.TestCase):
    def test_pathology_and_patient_purge(self):
        self.assertEqual([data.pathology_label(x) for x in
                          ('BENIGN', 'BENIGN_WITHOUT_CALLBACK', 'MALIGNANT')], [0,0,1])
        self.assertEqual(max(map(data.pathology_label, ['BENIGN','MALIGNANT'])), 1)
        with self.assertRaises(ValueError):
            data.pathology_label('UNKNOWN')
        safe, removed = data.purge_test_patients(
            [{'patient_key':'A'}, {'patient_key':'B'}, {'patient_key':'B'}],
            [{'patient_key':'B'}])
        self.assertEqual(safe, [{'patient_key':'A'}])
        self.assertEqual(len(removed), 2)
        self.assertEqual(data.source_patient('Calc-Training_P_00001_LEFT_CC'), 'CBIS_00001')
        with self.assertRaises(ValueError):
            data.source_patient('Calc-Training_P_00001_LEFT_CC_1')

    def test_patient_stratification_is_deterministic_and_image_independent(self):
        rows = [dict(patient_key=f'p{i:03}', label=i%2) for i in range(40)]
        split = data.patient_split(rows)
        self.assertEqual(split, data.patient_split(list(reversed(rows))))
        self.assertEqual(split, data.patient_split(rows + [dict(patient_key='p000',label=0)]))
        self.assertEqual(list(split.values()).count('val'), 6)
        self.assertEqual(sum(rows[int(p[1:])]['label'] for p,s in split.items() if s == 'val'), 3)

    def test_half_pixel_affine_crop_padding_and_invalid_breaks(self):
        pixels = np.zeros((12,10), dtype=np.uint8)
        pixels[2:10,3:7] = 100
        image, meta = data.canonical_image(Image.fromarray(pixels))
        self.assertEqual(image.size, (1500,3000))
        self.assertEqual(meta['crop_bbox'], [3,2,7,10])
        # Four by eight crop exactly fills the canvas with scale375.
        transformed = data.transform_points([[3,2],[6,9],[np.nan,0],[-1,2],[10,2]], meta)
        np.testing.assert_allclose(transformed[:2], [[187,187],[1312,2812]])
        self.assertTrue(np.isnan(transformed[2:]).all())
        padded = np.zeros((4,4),dtype=np.uint8)
        padded[1:3,1:3] = 200
        _, m = data.canonical_image(Image.fromarray(padded))
        self.assertEqual(m['padding'], [0,750])
        np.testing.assert_allclose(data.transform_points([[1.5,1.5]],m),[[749.5,1499.5]])

    def test_resize_bankers_rounding(self):
        # Canvas (5,10), crop(4,5) gives height6.25->6; use width(4),height(6)
        # at scale1.25 gives7.5->8 (ties-to-even), rather than truncation.
        with patch.object(data, 'CANVAS', (5,10)):
            _, m = data.canonical_image(Image.fromarray(np.full((6,4),120,dtype=np.uint8)))
        self.assertEqual(m['resized_size'], [5,8])
        self.assertEqual(m['padding'], [0,1])

    def test_dicom_monochrome1_percentile_inversion(self):
        from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'full.dcm'
            fm = FileMetaDataset()
            fm.TransferSyntaxUID=ExplicitVRLittleEndian
            fm.MediaStorageSOPClassUID=SecondaryCaptureImageStorage
            fm.MediaStorageSOPInstanceUID=generate_uid()
            ds = FileDataset(str(p), {}, file_meta=fm, preamble=b'\0'*128)
            ds.Rows=20;ds.Columns=10;ds.SamplesPerPixel=1
            ds.PhotometricInterpretation='MONOCHROME1'
            ds.BitsAllocated=16;ds.BitsStored=16;ds.HighBit=15;ds.PixelRepresentation=0
            original=np.full((20,10),1000,dtype=np.uint16)
            original[3:18,2:8]=100
            ds.PixelData=original.tobytes();ds.save_as(p, enforce_file_format=True)
            image, meta=data.canonical_image(p)
            self.assertTrue(meta['monochrome1'])
            self.assertEqual(meta['crop_bbox'], [2,3,8,18])
            self.assertEqual(int(np.asarray(image)[1500,750]),255)

    def test_test_conversion_barrier_precedes_any_pixel_read(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'protocol.json').write_text('{}')
            with patch.object(data,'validate_protocol',return_value={}), \
                 patch.object(data,'canonical_image',side_effect=AssertionError('TEST pixel read')), \
                 patch.object(data,'load_manifest',side_effect=AssertionError('Manifest loaded after gate')):
                with self.assertRaisesRegex(ValueError, 'before classifier freeze'):
                    data.convert_manifest(root, 'test')
                (root/'freeze.json').write_text(json.dumps(dict(frozen=True,protocol_sha256='wrong')))
                with self.assertRaisesRegex(ValueError, 'freeze protocol mismatch'):
                    data.convert_manifest(root, 'test')

    def test_conversion_detects_source_or_output_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            source=Path(td)/'source.png'; target=Path(td)/'target.png'
            Image.fromarray(np.full((20,10),100,dtype=np.uint8)).save(source)
            first=data._convert_one(source,target)
            self.assertEqual(first,data._convert_one(source,target))
            Image.fromarray(np.full((20,10),130,dtype=np.uint8)).save(source)
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):
                data._convert_one(source,target)

    def test_metadata_join_aggregates_lesions_and_ignores_roi_pixels(self):
        def write(path, fields, rows):
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); raw=root/'raw';md=root/'metadata';out=root/'out'
            raw.mkdir();md.mkdir();(root/'data/image').mkdir(parents=True)
            subjects=['Mass-Training_P_00001_LEFT_CC','Mass-Training_P_00002_LEFT_CC',
                      'Mass-Training_P_00003_LEFT_CC','Mass-Test_P_00003_RIGHT_MLO']
            metadata=[]
            for sn in subjects:
                folder=raw/'CBIS-DDSM'/sn/'full';folder.mkdir(parents=True)
                (folder/'1-1.dcm').write_bytes(b'Not decoded in cohort audit')
                metadata.append({'Subject ID':sn,'Series Description':'full mammogram images',
                                 'File Location':str(folder.relative_to(raw))})
            metadata.append({'Subject ID':subjects[0]+'_1','Series Description':'ROI mask images',
                             'File Location':'CBIS-DDSM/nonexistent-ROI'})
            write(raw/'metadata.csv',['Subject ID','Series Description','File Location'],metadata)
            provenance=[]
            for category in ('mass','calc'):
                for split in ('train','test'):
                    name=f'{category}_case_description_{split}_set.csv';rows=[]
                    if category=='mass':
                        chosen=subjects[:3] if split=='train' else subjects[3:]
                        rows=[{'image file path':sn+'/uids/image.dcm',
                               'pathology':'BENIGN' if '00002' in sn else 'MALIGNANT'} for sn in chosen]
                        if split=='train':rows.append({'image file path':subjects[0]+'/uids/image.dcm','pathology':'BENIGN'})
                    write(md/name,['image file path','pathology'],rows)
                    provenance.append(dict(path=name,sha256=data.sha256_file(md/name),url='official-fixture'))
            (md/'download_provenance.json').write_text(json.dumps(provenance))
            fn=subjects[0]+'_UNKNOWN.png';(root/'data/image'/fn).write_bytes(b'Not decoded in cohort audit')
            manifest=root/'all.csv'
            write(manifest,['image_id','file_name','source','filename_split'],
                  [dict(image_id=99,file_name=fn,source='CBIS-DDSM',filename_split='Training')])
            with patch.object(data,'ROOT',root),patch.object(data,'patient_split',return_value={'CBIS_00001':'train','CBIS_00002':'val'}):
                data.build_cohort(md,raw,out,gaze_manifest=manifest,strict=False)
            train=data.load_manifest(out/'trainval.csv');gaze=data.load_manifest(out/'gaze.csv')
            self.assertEqual(len(train),2)
            self.assertEqual(gaze[0]['label'],1)
            self.assertEqual(gaze[0]['split'],'train')
            self.assertEqual(gaze[0]['source_image_id'],99)
            audit=json.loads((out/'audit.json').read_text())
            self.assertEqual(audit['purged_training_images'],1)
            self.assertEqual(audit['conflicting_lesion_labels'],[subjects[0]])
            data.validate_protocol(out)
            (raw/'metadata.csv').write_text('mutated')
            with self.assertRaisesRegex(ValueError,'source changed'):
                data.validate_protocol(out)


if __name__ == '__main__':
    unittest.main()
