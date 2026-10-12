"""Offline checks for source navigation; no vLLM installation or downloads."""
import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.experiments.lesson0301 import audit, call_sites, digest, fetch_sources, main, read_manifest, source_path, symbols


class SourceJourneyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = b'class Core:\n    async def step(self):\n        return self.worker.run()\n'
        self.path = self.root/'vllm/core.py'
        self.path.parent.mkdir()
        self.path.write_bytes(self.payload)
        self.manifest = {
            'repository': 'https://github.com/vllm-project/vllm', 'tag': 'test', 'commit': 'a'*40,
            'files': {'vllm/core.py': {'sha256_lf': digest(self.payload)}},
            'checkpoints': [{'id':'step', 'stage':'core', 'title':'one step', 'path':'vllm/core.py',
                             'symbol':'Core.step', 'calls':['self.worker.run']}], 'structures': []}

    def test_real_manifest_is_immutable_and_all_scopes_covered(self):
        manifest = read_manifest()
        self.assertEqual(manifest['commit'], 'ba41cc90e8ef7f236347b2f1599eec2cbb9e1f0d')
        self.assertEqual({v['stage'] for v in manifest['checkpoints']}, {'input','core','output','cancel'})

    def test_reads_source_without_executing_it(self):
        payload = b'raise RuntimeError("must not execute")\n' + self.payload
        self.path.write_bytes(payload)
        self.manifest['files']['vllm/core.py']['sha256_lf'] = digest(payload)
        report = audit(self.root, self.manifest)
        self.assertTrue(report['passed'])
        self.assertFalse(report['executed_upstream_code'])
        self.assertFalse(report['gpu_validated'])
        self.assertTrue(report['checkpoints'][0]['url'].endswith('#L3'))

    def test_crlf_does_not_change_content_identity(self):
        self.path.write_bytes(self.payload.replace(b'\n', b'\r\n'))
        self.assertTrue(audit(self.root, self.manifest)['passed'])

    def test_modified_source_fails_even_when_symbol_survives(self):
        self.path.write_bytes(self.payload + b'# local modification\n')
        report = audit(self.root, self.manifest)
        self.assertFalse(report['passed'])
        self.assertIn('Source hash mismatch', report['errors'][0])

    def test_missing_file_fails_with_fetch_hint(self):
        self.path.unlink()
        report = audit(self.root, self.manifest)
        self.assertFalse(report['passed'])
        self.assertIn('--fetch', report['errors'][0])

    def test_same_method_name_in_another_class_is_not_a_match(self):
        self.manifest['checkpoints'][0]['symbol'] = 'Other.step'
        self.assertFalse(audit(self.root, self.manifest)['passed'])

    def test_comments_and_strings_do_not_satisfy_calls(self):
        tree = ast.parse('def f():\n    # expected.run()\n    x="expected.run()"\n    actual.run()\n')
        sites = call_sites(symbols(tree)['f'])
        self.assertEqual([s['target'] for s in sites], ['actual.run'])

    def test_nested_function_and_decorator_are_not_body_calls(self):
        tree = ast.parse('@decorate()\nasync def f():\n    def helper():\n        inner.run()\n    outer.run()\n')
        indexed = symbols(tree)
        self.assertEqual([s['target'] for s in call_sites(indexed['f'])], ['outer.run'])
        self.assertEqual([s['target'] for s in call_sites(indexed['f.helper'])], ['inner.run'])

    def test_declared_message_fields_are_checked(self):
        self.manifest['structures'] = [{'path':'vllm/core.py', 'symbol':'Core', 'fields':['request_id']}]
        self.assertFalse(audit(self.root, self.manifest)['passed'])
        payload = self.payload.replace(b'class Core:\n', b'class Core:\n    request_id: str\n')
        self.path.write_bytes(payload)
        self.manifest['files']['vllm/core.py']['sha256_lf'] = digest(payload)
        self.assertTrue(audit(self.root, self.manifest)['passed'])

    def test_missing_call_fails_even_if_file_hash_matches(self):
        self.manifest['checkpoints'][0]['calls'] = ['self.worker.wrong']
        self.assertFalse(audit(self.root, self.manifest)['passed'])

    def test_source_paths_cannot_leave_the_selected_directory(self):
        for name in ['../secret.py', '/secret.py', 'C:/secret.py', 'vllm\\..\\secret.py']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                source_path(self.root, name)

    def test_fetch_reuses_verified_files_without_network(self):
        with patch('scripts.experiments.lesson0301.urlopen') as network:
            self.assertEqual(fetch_sources(self.root, self.manifest), [])
            network.assert_not_called()

    def test_fetch_preserves_modified_existing_file(self):
        self.path.write_bytes(b'my edit')
        with patch('scripts.experiments.lesson0301.urlopen') as network, self.assertRaises(ValueError):
            fetch_sources(self.root, self.manifest)
        network.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b'my edit')

    def test_fetch_rejects_wrong_download_before_saving(self):
        self.path.unlink()
        with patch('scripts.experiments.lesson0301.urlopen') as network:
            network.return_value.__enter__.return_value.read.return_value = b'wrong'
            with self.assertRaises(ValueError):
                fetch_sources(self.root, self.manifest)
        self.assertFalse(self.path.exists())

    def test_fetch_saves_verified_missing_file(self):
        self.path.unlink()
        with patch('scripts.experiments.lesson0301.urlopen') as network:
            network.return_value.__enter__.return_value.read.return_value = self.payload
            self.assertEqual(fetch_sources(self.root, self.manifest), ['vllm/core.py'])
            self.assertIn('/'+'a'*40+'/', network.call_args.args[0])
        self.assertEqual(self.path.read_bytes(), self.payload)

    def test_default_cli_is_offline_and_saves_a_static_report(self):
        with patch('scripts.experiments.lesson0301.read_manifest', return_value=self.manifest), \
                patch('scripts.experiments.lesson0301.urlopen') as network, redirect_stdout(io.StringIO()):
            code = main(['--source-dir', str(self.root), '--output-dir', str(self.root/'report')])
        network.assert_not_called()
        self.assertEqual(code, 0)
        self.assertTrue((self.root/'report/source-map.md').is_file())


if __name__ == '__main__':
    unittest.main()
