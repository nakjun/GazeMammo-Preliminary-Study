import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd

spec=importlib.util.spec_from_file_location('report_full_1011',Path(__file__).resolve().parents[1]/'scripts/report_full_1011.py')
reporter=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporter)

class ReportTests(unittest.TestCase):
    def test_alignment_and_bootstrap(self):
        a=pd.DataFrame(dict(image_id=[2,1,3],patient_key=['x','x','y'],nll=[3.,2.,4.]))
        b=a.iloc[::-1].copy();b.nll-=.5
        pair=reporter.align(a,b,{1,2,3},'test');pair['patient_key']=pair.patient_key_left
        result=reporter.paired_bootstrap(pair,'nll_left','nll_right',10000)
        self.assertEqual(result['delta'],.5);self.assertEqual(result['ci95'],[.5,.5]);self.assertEqual(result['patients'],2)
        with self.assertRaises(ValueError):reporter.align(a,b.iloc[:2],{1,2,3},'test')
        with self.assertRaises(ValueError):reporter.frame_ids(pd.concat([a,a.iloc[:1]]),{1,2,3},'test')
        self.assertEqual(reporter.paired_bootstrap(pair,'nll_left','nll_right',10000),result)

    def synthetic(self,root):
        ids=list(range(93));patients={i:f'p{i//2}' for i in ids}
        def save(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj))
        save(root/'completion.json',dict(complete=True));save(root/'status.json',{name:dict(state='complete') for name in ('improved','control')})
        save(root/'experiment.json',dict(test_used=False,training_images=412,validation_images=93))
        for name,nll in [('improved',4.),('control',5.)]:
            config=dict(seed=42,epochs=15,limit=0,val_limit=0,mode='mixed' if name=='improved' else 'legacy16')
            save(root/name/'config.json',config)
            history=[dict(epoch=e,validation={mode:dict(raw_nll=nll+.01*(15-e),information_gain_bits=1.) for mode in ('legacy16','contiguous')}) for e in range(1,16)]
            save(root/name/'history.json',history)
            for file in ('best.pt','last.pt'):(root/name/file).write_bytes(b'synthetic checkpoint presence fixture')
            save(root/name/'timing.json',dict(seconds=100.,peak_cuda_memory_bytes=200))
            directory=root/f'{name}_eval'
            save(directory/'provenance.json',dict(independent_holdout=False,generated_image_ids=ids,checkpoint_sha256='fixture'))
            save(directory/'timing.json',dict(seconds=50.))
            for mode in ('legacy16','contiguous'):
                pd.DataFrame(dict(image_id=ids[::-1],patient_key=[patients[i] for i in ids[::-1]],raw_nll=nll,information_gain_bits=7-nll)).to_csv(directory/f'{mode}.csv',index=False)
                pd.DataFrame(dict(image_id=ids,gate_global=.5,gate_local=.3,gate_return=.2)).to_csv(directory/f'{mode}_steps.csv',index=False)
            for strategy,seed in [('sample',42),('sample',43),('sample',44),('argmax',42)]:
                directory=root/f'{name}_{strategy}_{seed}'
                save(directory/'config.json',dict(seed=seed,strategy=strategy,policy='none'))
                save(directory/'provenance.json',dict(generated_image_ids=ids))
                save(directory/'timing.json',dict(seconds=20.))
                save(directory/'predictions.json',{str(i):[[.5,.5]]*16 for i in ids})
                save(directory/'geometry.json',[dict(image_id=i,normalized_dtw=nll/10+(seed-42)*.01,return_rate_112px=nll/20) for i in ids[::-1]])
        baseline=root/'baseline.csv';pd.DataFrame(dict(image_id=ids,patient_key=[patients[i] for i in ids],raw_next_step_nll=6.)).to_csv(baseline,index=False)
        return ids,patients,baseline

    def test_full_synthetic_report_and_reject_incomplete(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ids,patients,baseline=self.synthetic(root)
            frame,ci=reporter.report(root,baseline,ids,patients)
            self.assertEqual(len(frame),10);self.assertAlmostEqual(ci['legacy16/raw_nll/improved_minus_control']['delta'],-1.)
            self.assertAlmostEqual(ci['sample_3seed_image_mean/normalized_dtw/improved_minus_control']['delta'],-.1)
            self.assertTrue((root/'report.md').exists());self.assertTrue((root/'paired_ci.json').exists())
            self.assertTrue((root/'learning_curves.png').exists())
            self.assertGreater((root/'learning_curves.png').stat().st_size,10000)
            with self.assertRaises(FileExistsError):reporter.report(root,baseline,ids,patients)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ids,patients,baseline=self.synthetic(root)
            path=root/'improved/history.json';path.write_text(json.dumps(json.loads(path.read_text())[:-1]))
            with self.assertRaisesRegex(ValueError,'complete 15 epochs'):reporter.report(root,baseline,ids,patients)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ids,patients,baseline=self.synthetic(root)
            path=root/'control_sample_43/predictions.json';points=json.loads(path.read_text());points.pop('1');path.write_text(json.dumps(points))
            with self.assertRaisesRegex(ValueError,'incomplete predictions'):reporter.report(root,baseline,ids,patients)

if __name__=='__main__':unittest.main()
