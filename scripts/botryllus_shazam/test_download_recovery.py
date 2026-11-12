"""Regression tests for complete-length corrupt ENA partials."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import download


class CorruptPartialRecovery(unittest.TestCase):
    def exercise(self, payload, collision=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "raw/TEST"
            folder.mkdir(parents=True)
            partial = folder / "TEST_1.fastq.gz.partial"
            partial.write_bytes(payload)
            digest = hashlib.md5(b"new").hexdigest()
            row = {
                "run_accession": "TEST",
                "fastq_md5": digest + ";" + digest,
                "fastq_bytes": "3;3",
                "fastq_ftp": "host/a;host/b",
                "read_count": "1",
                "base_count": "2",
            }
            quarantine = partial.with_name(
                partial.name + ".invalid-md5-" + hashlib.md5(payload).hexdigest()[:12]
            )
            if collision:
                quarantine.write_bytes(b"previous preserved bytes")
            outputs = []

            def transfer(args):
                p = Path(args[args.index("--output") + 1])
                outputs.append(p.name)
                p.write_bytes(b"new")

            with patch.object(download, "transfer", transfer), patch.object(
                download.subprocess,
                "check_output",
                return_value=json.dumps({"pairs": 1, "bases": 2}),
            ):
                if collision:
                    with self.assertRaises(ValueError):
                        download.recover(row, root, Path("/validator"))
                    self.assertEqual(partial.read_bytes(), payload)
                    self.assertEqual(
                        quarantine.read_bytes(), b"previous preserved bytes"
                    )
                    self.assertFalse((folder / "validated.json").exists())
                else:
                    download.recover(row, root, Path("/validator"))
                    self.assertEqual((folder / "TEST_1.fastq.gz").read_bytes(), b"new")
                    self.assertTrue((folder / "validated.json").exists())
                    if payload == b"bad":
                        self.assertEqual(quarantine.read_bytes(), b"bad")
                        self.assertEqual(len(outputs), 2)
                    else:
                        self.assertFalse(quarantine.exists())
                        self.assertEqual(outputs, ["TEST_2.fastq.gz.partial"])
            self.assertFalse((folder / ".active").exists())

    def test_corrupt_partial_preserved_before_fresh_download(self):
        self.exercise(b"bad")

    def test_good_complete_partial_reused(self):
        self.exercise(b"new")

    def test_quarantine_collision_preserves_both_files(self):
        self.exercise(b"bad", collision=True)


if __name__ == "__main__":
    unittest.main()
