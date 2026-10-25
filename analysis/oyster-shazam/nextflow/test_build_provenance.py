"""Lightweight provenance identity and no-clobber checks; no sketch computation."""

import tempfile
import unittest
from pathlib import Path
from artifacts import write_new
from build_provenance import code_record


class ProvenanceTest(unittest.TestCase):
    def test_code_dependencies_bound(self):
        root = Path(__file__).resolve().parents[1]
        result = code_record(root)
        self.assertIn("peptide_protocol.py", result)
        self.assertIn("sketch_corrected_peptides.py", result)
        self.assertEqual(len(result), 5)
        self.assertTrue(all(len(value) == 64 for value in result.values()))

    def test_completion_cannot_replace_existing(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "completed.json"
            write_new(path, {"originals": 2})
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                write_new(path, {"originals": 0})
            self.assertEqual(path.read_bytes(), before)
