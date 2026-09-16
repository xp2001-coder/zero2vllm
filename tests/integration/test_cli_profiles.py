import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from zero2vllm.cli import main
from zero2vllm.profiles import inspect_profile

ROOT = Path(__file__).resolve().parents[2]


class ProfileTests(unittest.TestCase):
    def complete_profile(self, backend="cuda"):
        source = ROOT / "configs/platforms" / backend / ("a100.toml" if backend == "cuda" else "c500.toml")
        text = source.read_text(encoding="utf-8")
        text = text.replace('= ""', '= "test-value"')
        text = text.replace('image = "test-value"', 'image = "registry.example/course@sha256:' + "a" * 64 + '"')
        text = text.replace('path_in_container = "test-value"', 'path_in_container = "/models/test model"')
        text = text.replace('gpu_memory_gib = "test-value"', 'gpu_memory_gib = "40"')
        return text

    def inspect_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.toml"
            path.write_text(text, encoding="utf-8")
            return inspect_profile(path)

    def test_templates_do_not_claim_gpu_validation_or_emit_commands(self):
        for name in ("cuda-a100", "maca-c500"):
            backend, device = name.split('-')
            result = inspect_profile(ROOT / "configs/platforms" / backend / f"{device}.toml")
            self.assertFalse(result["configuration_complete"])
            self.assertFalse(result["gpu_execution_tested"])
            self.assertIsNone(result["command_argv"])
            self.assertIn("runtime.image", result["missing_or_unlocked"])

    def test_complete_profiles_emit_argv_without_claiming_validation(self):
        for backend in ("cuda", "maca"):
            result = self.inspect_text(self.complete_profile(backend))
            self.assertTrue(result["configuration_complete"])
            self.assertFalse(result["gpu_execution_tested"])
            self.assertEqual(result["command_argv"][:3], ["vllm", "serve", "/models/test model"])

    def test_maca_version_mismatch_is_rejected(self):
        text = self.complete_profile("maca").replace('vllm_metax = "test-value"', 'vllm_metax = "different"')
        with self.assertRaises(ValueError):
            self.inspect_text(text)

    def test_unpinned_image_is_reported(self):
        text = self.complete_profile().replace("registry.example/course@sha256:" + "a" * 64, "registry.example/course:latest")
        result = self.inspect_text(text)
        self.assertFalse(result["configuration_complete"])
        self.assertIsNone(result["command_argv"])

    def test_absent_runtime_and_model_fields_are_reported_as_gaps(self):
        text = self.complete_profile().replace('dtype = "float16"', '')
        text = text.replace('image = "registry.example/course@sha256:' + "a" * 64 + '"', '')
        result = self.inspect_text(text)
        self.assertIn("runtime.image", result["missing_or_unlocked"])
        self.assertIn("model.dtype", result["missing_or_unlocked"])
        self.assertIsNone(result["command_argv"])

    def test_bad_serving_settings_fail(self):
        substitutions = [("tensor_parallel_size = 1", "tensor_parallel_size = 2"),
                         ("max_model_len = 1024", "max_model_len = 0"),
                         ("max_num_seqs = 4", "max_num_seqs = true"),
                         ("gpu_memory_utilization = 0.70", "gpu_memory_utilization = 1.2"),
                         ("port = 8000", "port = 70000"),
                         ("enforce_eager = true", "enforce_eager = 1")]
        for before, after in substitutions:
            with self.subTest(after=after), self.assertRaises(ValueError):
                self.inspect_text(self.complete_profile().replace(before, after))

    def test_backend_must_be_explicit(self):
        with self.assertRaises(ValueError):
            self.inspect_text(self.complete_profile().replace('backend = "cuda"', 'backend = "auto"'))

    def test_unknown_fields_are_rejected_in_every_table(self):
        additions = [('', 'extra = "ignored"\n'),
                     ('[hardware]', 'gpu_memroy_gib = "40"'),
                     ('[runtime]', 'maca = "wrong backend"'),
                     ('[model]', 'quantization = "fp8"'),
                     ('[serving]', 'data_parallel_size = 4'),
                     ('[serving]', 'max_num_seq = 16')]
        for section, field in additions:
            text = self.complete_profile()
            text = (text.replace(section, section + '\n' + field)
                    if section else field + text)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Unknown"):
                self.inspect_text(text)

    def test_gpu_memory_must_be_positive_and_finite(self):
        for value in ('test-value', '0', '-1', 'NaN', 'Infinity', '-Infinity', '40 GiB'):
            text = self.complete_profile().replace('gpu_memory_gib = "40"',
                                                   f'gpu_memory_gib = "{value}"')
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "gpu_memory_gib"):
                self.inspect_text(text)

    def test_fractional_gpu_memory_is_valid_and_empty_is_a_gap(self):
        for value, complete in (('39.5', True), ('', False)):
            text = self.complete_profile().replace('gpu_memory_gib = "40"',
                                                   f'gpu_memory_gib = "{value}"')
            with self.subTest(value=value):
                result = self.inspect_text(text)
                self.assertEqual(result['configuration_complete'], complete)
                if not complete:
                    self.assertIn('hardware.gpu_memory_gib', result['missing_or_unlocked'])
                    self.assertIsNone(result['command_argv'])


class CLITests(unittest.TestCase):
    def call(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_demo_writes_parseable_report_in_unicode_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "实验 结果" / "demo.json"
            code, stdout, stderr = self.call(["demo", "--output", str(output)])
            self.assertEqual(code, 0, stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(report, json.loads(stdout))
            self.assertTrue(report["verification"]["passed"])

    def test_invalid_demo_returns_error(self):
        for args in (["--prompt", ""], ["--prompt", "1,99"], ["--max-new-tokens", "-1"]):
            code, stdout, stderr = self.call(["demo"] + args)
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("zero2vllm:", stderr)

    def test_doctor_is_inventory_only(self):
        code, stdout, stderr = self.call(["doctor"])
        self.assertEqual(code, 0, stderr)
        self.assertFalse(json.loads(stdout)["gpu_execution_tested"])

    def test_missing_profile_returns_error(self):
        code, _, stderr = self.call(["profile", str(ROOT / "configs" / "not-present.toml")])
        self.assertEqual(code, 2)
        self.assertIn("zero2vllm:", stderr)

    def test_require_complete_returns_nonzero_for_templates(self):
        code, stdout, _ = self.call(["profile", str(ROOT / "configs/platforms/maca/c500.toml"), "--require-complete"])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(stdout)["configuration_complete"])

    def test_unknown_profile_setting_returns_error_without_argv(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.toml"
            text = ProfileTests().complete_profile() + '\ndata_parallel_size = 4\n'
            path.write_text(text, encoding="utf-8")
            code, stdout, stderr = self.call(["profile", str(path)])
            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("serving.data_parallel_size", stderr)


if __name__ == "__main__":
    unittest.main()
