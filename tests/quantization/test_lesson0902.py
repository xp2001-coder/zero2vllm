import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.lesson0902 import main,run_experiments,check_torch_float8,make_passport
from zero2vllm.quantization import QuantizedModelPassport


class Lesson0902Tests(unittest.TestCase):
    def test_all_cases_and_observed_contracts(self):
        report,sym,asym=run_experiments()
        self.assertTrue(report['passed'],report['cases'])
        self.assertEqual(len(report['cases']),20)
        for name,t in [('symmetric',sym),('asymmetric',asym)]:
            p=make_passport(dict(report,course_source='test'),t,name)
            self.assertEqual(QuantizedModelPassport.from_dict(p.to_dict()).to_dict(),p.to_dict())
            self.assertEqual(p.sections['validation']['model_quality'].status,'unknown')
            self.assertEqual(p.sections['execution']['actual_kernel'].status,'not_applicable')

    def test_offline_cli_writes_finite_json_and_passports(self):
        with tempfile.TemporaryDirectory() as out,patch('builtins.print'):
            self.assertEqual(main(['--output-dir',out]),0)
            d=json.loads((Path(out)/'summary.json').read_text(encoding='utf8'))
            self.assertTrue(d['passed'])
            self.assertFalse(d['gpu_validated'])
            self.assertFalse(d['packed'])
            self.assertNotIn('torch_oracle',d)
            self.assertEqual(len(list(Path(out).glob('*passport.json'))),2)

    @unittest.skipUnless(importlib.util.find_spec('torch'),'Optional independent PyTorch oracle')
    def test_all_fp8_patterns_against_torch(self):
        report=check_torch_float8()
        self.assertTrue(report['passed'],report)
        self.assertEqual(sum(r['patterns'] for r in report['formats']),512)


if __name__=='__main__': unittest.main()
