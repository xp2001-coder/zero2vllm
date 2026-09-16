import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from zero2vllm.reference.next_token import (encode_pieces, next_token_experiment,
                                          teaching_model, trace_forward)


class NextTokenLessonTests(unittest.TestCase):
    def test_hand_calculation_matches_the_report(self):
        report = next_token_experiment()
        trace = report["trace"]
        # Golden expression is calculated directly from the printed 2x2 example.
        alpha = 1.0 / (1.0 + math.exp(1.0 / math.sqrt(2.0)))
        expected = [alpha, 2.0 - alpha, 2.0, -alpha, alpha - 2.0]
        self.assertEqual(trace["attention_weights"][0], [1.0, 0.0])
        self.assertIsNone(trace["masked_scores"][0][1])
        for actual, wanted in zip(trace["next_token_logits"], expected):
            self.assertAlmostEqual(actual, wanted, delta=1e-12)
        self.assertAlmostEqual(trace["attention_weights"][1][0], alpha, delta=1e-12)
        self.assertEqual(trace["shapes"]["logits"], [2, 5])
        self.assertEqual(report["generation"]["new_token_ids"], [2, 2, 2])
        self.assertTrue(report["verification"]["passed"])

    def test_trace_agrees_with_reference_for_varied_shapes_and_positions(self):
        model = teaching_model()
        model.position[1] = [0.2, -0.3]
        model.wq = [[0.5, -1.0], [0.2, 0.3]]
        model.wk = [[0.1, 0.8], [-0.4, 1.0]]
        model.wv = [[0.3, -0.2], [0.7, 0.9]]
        model.wo = [[0.6, -0.1], [-0.8, 0.4]]
        for tokens in ([0], [3, 0, 2], [4, 1, 3, 0]):
            with self.subTest(tokens=tokens):
                trace = trace_forward(model, list(tokens))
                self.assertLessEqual(trace["max_abs_reference_error"], 1e-12)
                for i, weights in enumerate(trace["attention_weights"]):
                    self.assertAlmostEqual(sum(weights), 1.0)
                    self.assertTrue(all(w == 0.0 for w in weights[i + 1:]))

    def test_future_tokens_leave_existing_rows_unchanged(self):
        model = teaching_model()
        before = trace_forward(model, [0, 1])
        after = trace_forward(model, [0, 1, 4])
        self.assertEqual(before["all_position_logits"], after["all_position_logits"][:2])
        for i, row in enumerate(before["attention_weights"]):
            self.assertEqual(row, after["attention_weights"][i][:2])

    def test_generation_uses_last_row_and_grows_the_prefix(self):
        report = next_token_experiment()
        trace, steps = report["trace"], report["generation"]["steps"]
        self.assertNotEqual(trace["all_position_logits"][0], steps[0]["next_token_logits"])
        self.assertEqual(trace["all_position_logits"][-1], steps[0]["next_token_logits"])
        for index, step in enumerate(steps):
            self.assertEqual(step["selected_logit_row"], len(step["input_token_ids"]) - 1)
            if index:
                previous = steps[index - 1]
                self.assertEqual(step["input_token_ids"], previous["input_token_ids"] + [previous["next_token_id"]])
        # Tied maximum logits follow the existing generation loop's lowest-ID rule.
        self.assertEqual(next_token_experiment("我", 1)["generation"]["new_token_ids"], [0])

    def test_token_mapping_invalid_inputs_and_context_boundary(self):
        self.assertEqual(encode_pieces(" 我\t喜欢 \n推理 "), [0, 1, 2])
        for prompt in ("", "我喜欢", "我 未知"):
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                next_token_experiment(prompt)
        for count in (0, -1, True, 1.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                next_token_experiment(max_new_tokens=count)
        self.assertTrue(next_token_experiment("我", 16)["verification"]["passed"])
        with self.assertRaises(ValueError):
            next_token_experiment("我", 17)

    def test_cli_saves_the_trace_and_rejects_unknown_tokens(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "第一课" / "trace.json"
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            process = subprocess.run([sys.executable, "-m", "zero2vllm", "next-token-demo", "--output", str(output)],
                                     capture_output=True, text=True, encoding="utf-8", env=env)
            self.assertEqual(process.returncode, 0, process.stderr)
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(process.stdout), report)
            self.assertTrue(report["verification"]["passed"])
            bad = subprocess.run([sys.executable, "-m", "zero2vllm", "next-token-demo", "--prompt", "我喜欢"],
                                 capture_output=True, text=True, encoding="utf-8", env=env)
            self.assertEqual(bad.returncode, 2)
            self.assertIn("Unknown teaching token", bad.stderr)


if __name__ == "__main__":
    unittest.main()
