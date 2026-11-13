import json
from pathlib import Path
import tempfile
import unittest

from Bio.Seq import Seq
from peptide_protocol import export, segments


class PeptideProtocolTest(unittest.TestCase):
    def test_boundary_lengths(self):
        for length in (29, 30, 31):
            forward = [s for s in segments("GCT" * length) if s.frame == 1]
            self.assertEqual(
                [s.peptide for s in forward], ["A" * length] if length >= 30 else []
            )

    def test_x_is_internal_and_stops_are_boundaries(self):
        dna = "GCT" * 15 + "NNN" + "GCT" * 15 + "TAA" + "GCT" * 30
        forward = [s for s in segments(dna) if s.frame == 1]
        self.assertEqual(
            [s.peptide for s in forward], ["A" * 15 + "X" + "A" * 15, "A" * 30]
        )
        self.assertEqual([(s.aa_start, s.aa_end) for s in forward], [(0, 31), (32, 62)])
        self.assertEqual(
            [(s.dna_start, s.dna_end) for s in forward], [(0, 93), (96, 186)]
        )

        # X counts toward the requested 30-aa stretch length.
        at_boundary = [
            s for s in segments("GCT" * 15 + "NNN" + "GCT" * 14) if s.frame == 1
        ]
        self.assertEqual([s.peptide for s in at_boundary], ["A" * 15 + "X" + "A" * 14])

    def test_all_frames_and_coordinates(self):
        dna = "GCT" * 35 + "NNN" + "GCT" * 35
        result = list(segments(dna))
        self.assertEqual({s.frame for s in result}, {-3, -2, -1, 1, 2, 3})
        for s in result:
            piece = Seq(dna[s.dna_start : s.dna_end])
            if s.frame < 0:
                piece = piece.reverse_complement()
            self.assertEqual(str(piece.translate()), s.peptide)
            self.assertEqual(s.dna_end - s.dna_start, 3 * len(s.peptide))
            self.assertNotIn("*", s.peptide)

    def test_membership_empty_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "dna.fa"
            source.write_text(
                ">short description\nACGT\n>withX\n"
                + "GCT" * 15
                + "NNN"
                + "GCT" * 15
                + "\n"
            )
            export(source, root / "out")
            rows = [
                json.loads(line)
                for line in (root / "out/origins.jsonl").read_text().splitlines()
            ]
            self.assertEqual([r["origin_id"] for r in rows], ["short", "withX"])
            self.assertEqual(rows[0]["segments"], 0)
            self.assertEqual(rows[1]["dna"], "GCT" * 15 + "NNN" + "GCT" * 15)
            with self.assertRaises(FileExistsError):
                export(source, root / "out")

    def test_duplicate_ids_and_empty_input_fail_without_completion(self):
        for text in ("", ">q\nACG\n>q\nACT\n"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "dna.fa").write_text(text)
                with self.assertRaises(ValueError):
                    export(root / "dna.fa", root / "out")
                self.assertFalse((root / "out/complete.json").exists())


if __name__ == "__main__":
    unittest.main()
