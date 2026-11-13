"""Run against the installed native binary with MERGETAGS_BINARY set."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from dekupl_contigs import assemble


@unittest.skipUnless(os.environ.get("MERGETAGS_BINARY"), "Native binary required")
class NativeContigTests(unittest.TestCase):
    def test_overlap_direction_and_empty(self):
        seq = "ACGTTGCATGTCAGTACGATCGTAGCTAGCATGCAT"
        a, b = seq[:31], seq[1:32]
        for direction, expected in [(1, 1), (-1, 2)]:
            seeds = {
                a: [dict(contrast="toy", padj=0.01, log2foldchange="1")],
                b: [dict(contrast="toy", padj=0.02, log2foldchange=str(direction))],
            }
            with tempfile.TemporaryDirectory() as t:
                out = Path(t) / "out"
                dna, origins = assemble(
                    seeds, {a: "DK1", b: "DK2"}, os.environ["MERGETAGS_BINARY"], out
                )
                self.assertEqual(len(dna), expected)
                self.assertEqual(
                    sum(
                        x["assembled_kmers"]
                        for x in json.loads((out / "contig-metadata.json").read_text())[
                            "contigs"
                        ]
                    ),
                    2,
                )
                if direction == 1:
                    self.assertEqual(next(iter(dna.values())), seq[:32])
        with tempfile.TemporaryDirectory() as t:
            dna, origins = assemble(
                {}, {}, os.environ["MERGETAGS_BINARY"], Path(t) / "out"
            )
            self.assertEqual(dna, {})
