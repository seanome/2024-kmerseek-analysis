import tempfile, unittest
from pathlib import Path
from figure4_recovery import encode, windows, exact_gene_map


class ExactMapping(unittest.TestCase):
    def test_canonical_windows_and_ambiguity(self):
        seq = "ACGTTGCAACCTGATCGTAGCTAGCTAGCTA"
        self.assertEqual(len(seq), 31)
        rc = seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        self.assertEqual(encode(seq), encode(rc))
        self.assertEqual(list(windows(seq)), [encode(seq)])
        self.assertEqual(list(windows(seq + "N" + seq)), [encode(seq), encode(seq)])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ref.fa"
            p.write_text(">T1\n" + seq + "\n>T2\n" + rc + "\n")
            hits = exact_gene_map(p, {"T1": "g1", "T2": "g2"}, {encode(seq)})
            self.assertEqual(hits[encode(seq)], {"g1", "g2"})

    def test_bad_query_and_missing_reference_mapping_refused(self):
        with self.assertRaises(ValueError):
            encode("N" * 31)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ref.fa"
            p.write_text(">unknown\n" + "A" * 31 + "\n")
            with self.assertRaises(ValueError):
                exact_gene_map(p, {}, set())


if __name__ == "__main__":
    unittest.main()
