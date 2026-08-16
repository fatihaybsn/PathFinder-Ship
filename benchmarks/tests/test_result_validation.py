import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.tools.validate_result_bundle import validate


class ResultValidationTest(unittest.TestCase):
    def test_valid_minimal_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("metrics", "predictions", "figures", "logs", "status"):
                (root / name).mkdir()
            (root / "environment.json").write_text("{}", encoding="utf-8")
            (root / "RUN_COMPLETE.txt").write_text("complete", encoding="utf-8")
            artifact = root / "metrics" / "one.json"
            artifact.write_text("{}", encoding="utf-8")
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            (root / "artifact_manifest.json").write_text(
                json.dumps(
                    {
                        "files": [
                            {"relative_path": "metrics/one.json", "size_bytes": artifact.stat().st_size, "sha256": digest}
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(validate(root)["valid"])


if __name__ == "__main__":
    unittest.main()
