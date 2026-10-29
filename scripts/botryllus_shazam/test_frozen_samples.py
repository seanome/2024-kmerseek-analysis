import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from frozen_samples import restore_frozen_samples


class FrozenSamplesTests(unittest.TestCase):
    def test_actual_frozen_design(self):
        repo = Path(__file__).resolve().parents[2]
        reports = repo / "reports/botryllus-shazam"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "samples.tsv"
            restore_frozen_samples(
                reports / "samples.tsv", reports / "abundance-design-20261024.json", out
            )
            design = json.loads(
                (reports / "abundance-design-20261024.json").read_text()
            )
            self.assertEqual(
                hashlib.sha256(out.read_bytes()).hexdigest(), design["samples_sha256"]
            )
            self.assertEqual(out.read_text(), (reports / "samples.tsv").read_text())
            with self.assertRaises(FileExistsError):
                restore_frozen_samples(
                    reports / "samples.tsv",
                    reports / "abundance-design-20261024.json",
                    out,
                )

    def test_changed_label_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "samples").write_bytes(b"run\tcondition\nSRR1\tTreatment\n")
            (root / "design").write_text(
                json.dumps(
                    {
                        "samples_sha256": hashlib.sha256(
                            b"run\tcondition\r\nSRR1\tControl\r\n"
                        ).hexdigest()
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "beyond line endings"):
                restore_frozen_samples(root / "samples", root / "design", root / "out")
            self.assertFalse((root / "out").exists())
