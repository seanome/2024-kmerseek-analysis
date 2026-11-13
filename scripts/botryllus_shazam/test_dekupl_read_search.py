"""Strict DE-kupl selection and canonical seed identity."""

import csv
import gzip
from pathlib import Path
import tempfile
import unittest
from dekupl_read_search import canonical, select


class SelectionTests(unittest.TestCase):
    def test_global_q_boundary_and_reverse_complement(self):
        seed = "ACGTTGCATGTCAGTACGATCGTAGCTAGCA"
        reverse = seed.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            for i in range(3):
                p = root / f"contrast{i}" / "kmers"
                p.mkdir(parents=True)
                with gzip.open(p / "all-features.tsv.gz", "wt") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["tag", "padj", "log2FoldChange"])
                    w.writerows(
                        [
                            (seed, 0.049, 1),
                            (reverse, 0.05, 2),
                            (reverse, "NA", 0),
                            (seed, "NaN", 0),
                        ]
                    )
            seeds, sources = select(root)
            self.assertEqual(len(seeds), 1)
            self.assertEqual(len(seeds[canonical(seed)]), 3)
            self.assertEqual(len(sources), 3)
            self.assertEqual(canonical(seed), canonical(reverse))

    def test_missing_contrast_and_bad_dna_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                select(Path(t))
        with self.assertRaises(ValueError):
            canonical("N" * 31)
