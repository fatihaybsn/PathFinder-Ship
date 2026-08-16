import unittest

from benchmarks.run import build_prompt
from benchmarks.metrics import (
    bootstrap_ci,
    classification_summary,
    exact_match,
    expected_calibration_error,
    lexical_context_support,
    no_answer_match,
    repetition_rate,
    rouge_l_f1,
    token_f1,
)


class MetricsTest(unittest.TestCase):
    def test_answer_normalization(self):
        self.assertEqual(exact_match("The Anchor!", "anchor"), 1.0)
        self.assertEqual(token_f1("red blue", "red green"), 0.5)

    def test_rouge_l(self):
        self.assertAlmostEqual(rouge_l_f1("a red ship", "red ship"), 1.0)

    def test_rag_diagnostics(self):
        self.assertEqual(no_answer_match("I don't know."), 1.0)
        self.assertEqual(lexical_context_support("arrival noon", "The arrival is at noon"), 1.0)

    def test_repetition(self):
        self.assertGreater(repetition_rate("red blue green red blue green", 3), 0.0)

    def test_classification(self):
        result = classification_summary(["a", "a", "b"], ["a", "b", "b"])
        self.assertAlmostEqual(result["accuracy"], 2 / 3)
        self.assertIn("a", result["per_class"])

    def test_calibration(self):
        self.assertAlmostEqual(expected_calibration_error([1.0, 1.0], [True, True]), 0.0)

    def test_bootstrap_is_deterministic(self):
        first = bootstrap_ci([0.0, 1.0, 1.0], lambda values: sum(values) / len(values), iterations=100)
        second = bootstrap_ci([0.0, 1.0, 1.0], lambda values: sum(values) / len(values), iterations=100)
        self.assertEqual(first, second)

    def test_project_rag_suite_builds_a_rag_prompt(self):
        prompt = build_prompt(
            "instruction_tail_v2",
            "rag_project_v1",
            {"input": "Context: Port is Izmir.\nQuestion: Which port?\nAnswer:"},
        )
        self.assertIn("Context: Port is Izmir.", prompt)
        self.assertTrue(prompt.endswith("Answer:"))


if __name__ == "__main__":
    unittest.main()
