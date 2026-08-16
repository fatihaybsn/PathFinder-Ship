import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.decontam import FingerprintIndex, exact_hash, simhash64


class DecontaminationTest(unittest.TestCase):
    def test_exact_and_unrelated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fingerprints.jsonl"
            training = "open the camera now"
            path.write_text(
                json.dumps(
                    {
                        "source_family": "test",
                        "exact_sha256": exact_hash(training),
                        "simhash64": f"{simhash64(training):016x}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            index = FingerprintIndex(path)
            self.assertEqual(index.match("Open the camera now!")["reason"], "exact")
            self.assertFalse(index.match("Explain photosynthesis using a diagram")["contaminated"])


if __name__ == "__main__":
    unittest.main()
