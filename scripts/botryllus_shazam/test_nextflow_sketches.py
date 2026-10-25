"""Pinned Branchwater executable parity, including X and empty populations."""

import json
import os
import csv
import subprocess
import sys
import sourmash
from pathlib import Path
import tempfile
import unittest
import nextflow_stages as stages


class SketchTests(unittest.TestCase):
    def test_x_retention_and_empty_queries(self):
        old = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                q = Path("queries")
                q.mkdir()
                dna = {
                    "seed": "ACG" * 10 + "A",
                    "context": "GCT" * 15 + "NNN" + "GCT" * 15,
                }
                stages.write_fasta(q / "dna.fasta", dna)
                stages.write_fasta(
                    q / "peptides.faa", {"X30": "A" * 14 + "X" + "A" * 15}
                )
                stages.save(q / "membership.json", {"X30": "context"})
                for mode, k in [("protein", 7), ("dayhoff", 12), ("hp", 30)]:
                    folder = Path(mode)
                    folder.mkdir()
                    os.chdir(folder)
                    stages.sketch("../queries", mode, str(k), "1")
                    proof = json.loads(Path("coverage.json").read_text())
                    self.assertEqual(proof["queries"], 2)
                    self.assertEqual(proof["eligible_fragments"], 1)
                    with open("x.csv", "w") as f:
                        writer = csv.writer(f)
                        writer.writerow(["name", "genome_filename", "protein_filename"])
                        writer.writerow(
                            [
                                "X30",
                                "",
                                str((Path("../queries") / "peptides.faa").resolve()),
                            ]
                        )
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "sourmash",
                            "scripts",
                            "manysketch",
                            "x.csv",
                            "--singleton",
                            "-p",
                            f"{mode},k={k},scaled=1",
                            "-c",
                            "1",
                            "-o",
                            "x.scaled1.sig.zip",
                        ],
                        check=True,
                    )
                    sigs = list(sourmash.load_file_as_signatures("x.scaled1.sig.zip"))
                    self.assertEqual(len(sigs), 1)
                    expected = sigs[0].minhash.copy_and_clear()
                    expected.add_protein("A" * 14 + "X" + "A" * 15)
                    self.assertGreater(len(expected), 0)
                    self.assertEqual(set(expected.hashes), set(sigs[0].minhash.hashes))
                    os.chdir("..")
                q = Path("empty")
                q.mkdir()
                stages.write_fasta(q / "dna.fasta", {})
                stages.write_fasta(q / "peptides.faa", {})
                stages.save(q / "membership.json", {})
                Path("empty-result").mkdir()
                os.chdir("empty-result")
                stages.sketch("../empty", "hp", "30", "1")
                self.assertEqual(
                    json.loads(Path("coverage.json").read_text())["queries"], 0
                )
                self.assertTrue(Path("query.sig.zip").exists())
            finally:
                os.chdir(old)


if __name__ == "__main__":
    unittest.main()
