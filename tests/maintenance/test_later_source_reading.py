import ast
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.lesson0301 import symbols
from scripts.experiments import lesson0304, lesson0305


class LaterSourceReadingTests(unittest.TestCase):
    def test_unrequested_property_setter_does_not_hide_target(self):
        tree=ast.parse('class A:\n @property\n def value(self): pass\n @value.setter\n def value(self,x): pass\n def run(self): return self.work()\n')
        self.assertIn('A.run',symbols(tree,required={'A.run'}))
        with self.assertRaises(ValueError):symbols(tree,required={'A.value'})

    def test_requested_duplicate_still_rejected(self):
        with self.assertRaises(ValueError):symbols(ast.parse('def run(): pass\ndef run(): pass'),required={'run'})

    def test_unrequested_overloads_do_not_hide_target(self):
        tree=ast.parse('@overload\ndef get(x:int): pass\n@overload\ndef get(x:str): pass\ndef get(x): pass\ndef run(): return get(1)')
        self.assertIn('run',symbols(tree,required={'run'}))
        with self.assertRaises(ValueError):symbols(tree)

    def test_default_lesson_clis_are_offline(self):
        for lesson in (lesson0304,lesson0305):
            with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()),patch('scripts.experiments.source_records.fetch_sources',side_effect=AssertionError('network forbidden')):
                self.assertEqual(lesson.main(['--output-dir',d]),0)
                self.assertTrue((Path(d)/'summary.json').exists())

    def test_missing_sources_fail_without_downloading(self):
        for lesson in (lesson0304,lesson0305):
            with tempfile.TemporaryDirectory() as d,contextlib.redirect_stdout(io.StringIO()),patch('scripts.experiments.source_records.fetch_sources',side_effect=AssertionError('network forbidden')):
                self.assertEqual(lesson.main(['--inspect-source','--source-root',d,'--output-dir',str(Path(d)/'out')]),1)


if __name__ == '__main__': unittest.main()
