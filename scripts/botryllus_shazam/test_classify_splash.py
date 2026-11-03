"""Guard coordinate and ambiguity handling in the SPLASH evidence join."""

from pathlib import Path
import tempfile
import unittest
from classify_splash import (
    annotation,
    blocks_and_introns,
    genes_for_blocks,
    union_length,
)


class MappingTests(unittest.TestCase):
    def test_junction_and_unmapped(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "genes.gtf"
            p.write_text(
                'chr1\tx\texon\t21\t70\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\nchr1\tx\texon\t121\t170\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
            )
            index, junctions, names = annotation(p)
            blocks, introns = blocks_and_introns(20, "50M50N50M")
            self.assertEqual(introns, [(70, 120)])
            self.assertEqual(junctions[("chr1", 70, 120)], {"G1"})
            self.assertEqual(genes_for_blocks(index, "chr1", blocks), {"G1"})
            self.assertEqual(genes_for_blocks(index, "chr1", [(0, 10)]), set())
            self.assertEqual(genes_for_blocks(index, "missing", blocks), set())
            self.assertEqual(union_length([(0, 20), (10, 30)]), 30)

    def test_multiple_genes_not_silently_selected(self):
        index = {"chr1": ([(0, 100, "G1"), (0, 100, "G2")], [0, 0], [100, 100])}
        self.assertEqual(genes_for_blocks(index, "chr1", [(10, 90)]), {"G1", "G2"})


if __name__ == "__main__":
    unittest.main()
