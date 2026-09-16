import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.common import ROOT
from scripts.docs.course import project_files
from scripts.docs.layout import audit
from scripts.docs.catalog import load_catalog, load_translation


class LayoutTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for source in project_files(ROOT):
            target = self.root/source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    def test_current_tree_and_published_links_are_valid(self):
        report = audit(self.root)
        self.assertTrue(report['passed'], report['errors'])
        self.assertEqual(report['lessons'], 126)
        self.assertGreater(report['published_artifact_links'], 30)

    def test_empty_and_obsolete_directories_and_stray_files_are_rejected(self):
        (self.root/'configs/empty').mkdir()
        (self.root/'runs').mkdir()
        (self.root/'reports/stray.json').write_text('{}', encoding='utf-8')
        errors = audit(self.root)['errors']
        self.assertTrue(any('Empty directory: configs/empty' == e for e in errors))
        self.assertTrue(any('Obsolete directory remains: runs' == e for e in errors))
        self.assertTrue(any('Unclassified root file: reports/stray.json' == e for e in errors))

    def test_missing_published_artifact_is_rejected(self):
        summary = next((self.root/'docs/chapters/part-01-inference').rglob('summary.json'))
        value = json.loads(summary.read_text(encoding='utf-8'))
        value['cases'][0]['output'] = 'missing-case.json'
        summary.write_text(json.dumps(value), encoding='utf-8')
        self.assertTrue(any('Broken result link' in e for e in audit(self.root)['errors']))

    def test_split_catalog_preserves_all_translations(self):
        course, translation = load_catalog(self.root), load_translation(self.root)
        self.assertEqual(len(course['part_files']), 19)
        self.assertEqual({c['id'] for p in course['parts'] for c in p['chapters']}, set(translation['chapters']))
        raw = json.loads((self.root/'metadata/course/catalog.json').read_text(encoding='utf-8'))
        self.assertNotIn('parts', raw)

    def test_missing_duplicate_or_escaping_data_files_are_rejected(self):
        index = self.root/'metadata/course/catalog.json'
        original = json.loads(index.read_text(encoding='utf-8'))
        for value in ('parts/missing.json', '../../README.md', original['part_files'][1]):
            candidate = {**original, 'part_files': [value, *original['part_files'][1:]]}
            index.write_text(json.dumps(candidate), encoding='utf-8')
            with self.subTest(value=value), self.assertRaises(ValueError):
                load_catalog(self.root)


if __name__ == '__main__':
    unittest.main()
