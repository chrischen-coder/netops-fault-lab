import copy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from netops_fault_lab import Case, NoiseModel, diagnose
from netops_fault_lab.cli import main
from netops_fault_lab.report import render_html
from netops_fault_lab.schema import parse_json, read_dataset, read_json

ROOT=Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.raw=read_json(ROOT/'examples/ambiguous.json')

    def test_rejects_ground_truth_at_inference_boundary(self):
        self.raw['truth']='core'
        with self.assertRaises(ValueError):
            Case.from_dict(self.raw)

    def test_rejects_path_not_in_graph(self):
        self.raw['observations'][0]['path']=['api','db']
        with self.assertRaises(ValueError):
            Case.from_dict(self.raw)

    def test_rejects_duplicate_probe_even_across_candidate_boundary(self):
        self.raw['available_probes'][0]['id']=self.raw['observations'][0]['id']
        with self.assertRaises(ValueError):
            Case.from_dict(self.raw)

    def test_rejects_loops_parallel_links_and_reserved_id(self):
        mutations=[{'id':'loop','source':'api','target':'api'},
                   {'id':'parallel','source':'r1','target':'api'},
                   {'id':'__healthy__','source':'new-a','target':'new-b'}]
        for item in mutations:
            raw=copy.deepcopy(self.raw)
            raw['links'].append(item)
            with self.subTest(link=item),self.assertRaises(ValueError):
                Case.from_dict(raw)

    def test_invalid_probability_values(self):
        for args in [(1,.1,.2),(.8,0,.2),(.1,.8,.2),(.8,.1,True),(.8,float('nan'),.2)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                NoiseModel(*args)

    def test_rejects_bad_json_constants_and_duplicate_keys(self):
        for text in ['{"cost":NaN}','{"id":"a","id":"b"}']:
            with self.subTest(text=text),self.assertRaises(ValueError):
                parse_json(text)

    def test_html_escapes_user_supplied_labels(self):
        self.raw['id']='<script>alert(1)</script>'
        case=Case.from_dict(self.raw)
        html=render_html(case,diagnose(case))
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertIn("default-src 'none'",html)

    def test_model_round_trip(self):
        model=NoiseModel()
        self.assertEqual(model,NoiseModel.from_dict(model.to_dict()))

    def test_cli_refuses_overwrite_and_reports_ambiguity_as_valid_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder)/'report.json'
            args=['diagnose',str(ROOT/'examples/ambiguous.json'),'--format','json','--output',str(output)]
            self.assertEqual(main(args),0)
            content=output.read_bytes()
            self.assertEqual(json.loads(content)['status'],'ambiguous')
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(args),2)
            self.assertEqual(content,output.read_bytes())

    def test_cli_invalid_input_has_no_partial_output(self):
        with tempfile.TemporaryDirectory() as folder:
            input_path=Path(folder)/'bad.json'
            output=Path(folder)/'output.json'
            input_path.write_text('{}')
            with redirect_stderr(io.StringIO()):
                result=main(['diagnose',str(input_path),'--output',str(output)])
            self.assertEqual(result,2)
            self.assertFalse(output.exists())

    def test_labeled_gzip_dataset_read_and_truth_validation(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'train.jsonl.gz'
            with gzip.open(path,'wt') as stream:
                stream.write(json.dumps({'case':self.raw,'truth':'core'})+'\n')
            rows=read_dataset(path)
            self.assertEqual(rows[0][1],'core')
            with gzip.open(path,'wt') as stream:
                stream.write(json.dumps({'case':self.raw,'truth':'invented'})+'\n')
            with self.assertRaises(ValueError):
                read_dataset(path)
