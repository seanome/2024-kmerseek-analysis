"""Compare independent word enumeration with exported stretches on adversarial fixtures."""

import random
import unittest
import sourmash
from verify_replacements import independent
from peptide_protocol import segments


class CorrectedWordsTest(unittest.TestCase):
    def compare(self, dna, positive=False):
        observed = independent(dna)
        for mol, k, flags in [
            ("protein", 7, {"is_protein": True}),
            ("dayhoff", 12, {"dayhoff": True}),
            ("hp", 30, {"hp": True}),
        ]:
            with self.subTest(mol=mol):
                expected = sourmash.MinHash(n=0, ksize=k, scaled=10, **flags)
                for segment in segments(dna):
                    expected.add_protein(segment.peptide)
                self.assertEqual(observed[mol], expected)
                if positive:
                    self.assertGreater(len(expected), 0)

    def test_boundaries_stops_and_x(self):
        for dna in [
            "GCT" * 29,
            "GCT" * 30,
            "GCT" * 31,
            "GCT" * 15 + "NNN" + "GCT" * 14,
            "GCT" * 30 + "TAA" + "GCT" * 31,
        ]:
            with self.subTest(dna=dna):
                self.compare(dna)
        self.assertTrue(all(len(mh) == 0 for mh in independent("GCT" * 29).values()))

    def test_nonempty_diverse_internal_x(self):
        rng = random.Random(35)
        codons = [
            "GCT",
            "TGT",
            "GAT",
            "GAA",
            "TTT",
            "GGT",
            "CAT",
            "ATT",
            "AAA",
            "CTT",
            "ATG",
            "AAT",
            "CCT",
            "CAA",
            "CGT",
            "TCT",
            "ACT",
            "GTT",
            "TGG",
            "TAT",
        ]
        dna = (
            "".join(rng.choices(codons, k=100))
            + "NNN"
            + "".join(rng.choices(codons, k=100))
        )
        self.compare(dna, positive=True)
