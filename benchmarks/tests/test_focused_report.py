import csv
import tempfile
import unittest
from pathlib import Path

import yaml

from benchmarks.report import write_focused_comparison


class FocusedReportTest(unittest.TestCase):
    def test_rejected_runs_remain_and_best_is_marked(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            for name in ("tables", "figures", "logs"):
                (run_dir / name).mkdir()
            manifest = run_dir / "focused.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "experiments": [
                            {
                                "id": "flan_bad",
                                "display_name": "Rejected",
                                "status": "rejected",
                                "training_change": "q/v only",
                            },
                            {
                                "id": "flan_best",
                                "display_name": "Candidate",
                                "status": "final_candidate",
                                "training_change": "seven targets",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            metrics = [
                {
                    "file": "flan_bad.json",
                    "payload": {"experiment_id": "flan_bad", "status": "complete", "suites": {"chat_reference_v1": {"token_f1": {"estimate": 0.2}}}},
                },
                {
                    "file": "flan_best.json",
                    "payload": {"experiment_id": "flan_best", "status": "complete", "suites": {"chat_reference_v1": {"token_f1": {"estimate": 0.8}}}},
                },
            ]
            write_focused_comparison(run_dir, metrics, manifest)
            with (run_dir / "tables" / "focused_model_comparison.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["experiment_id"] for row in rows], ["flan_bad", "flan_best"])
            self.assertEqual(rows[0]["best_chat_token_f1"], "False")
            self.assertEqual(rows[1]["best_chat_token_f1"], "True")


if __name__ == "__main__":
    unittest.main()
