"""Regression checks for family deduplication across batches and streamed gzip output."""

import csv
import gzip
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCORER = (
    Path(__file__).resolve().parents[2]
    / "nextflow-runs/de-kmer-landing/bin/landings.py"
)


class StreamingLandingsTest(unittest.TestCase):
    def test_batch_boundary_and_empty_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def table(name, fields, rows):
                with (root / name).open("w") as fh:
                    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                    w.writerow(fields.split())
                    w.writerows(rows)

            table(
                "key.tsv",
                "pfam_acc status",
                [(f"PF{i:05d}", "up" if i < 100 else "unchanged") for i in range(200)],
            )
            table(
                "domains.tsv",
                "target pfam_acc clan start0 end",
                [("t", "PF00001", "CL1", 0, 100), ("t", "PF00150", "CL2", 100, 200)],
            )
            table("map.tsv", "peptide_id labelling", [("q", "real"), ("s", "swap1")])
            table("labels.tsv", "labelling", [("real",), ("swap1",)])
            cuts = [1e-6, 1e-4, 0.01, 1, 10, 100]
            sets = {(cut, lab): set() for cut in cuts for lab in ("real", "swap1")}
            counts = {k: 0 for k in sets}
            with (root / "regions.csv").open("w") as fh:
                w = csv.writer(fh)
                w.writerow(
                    "query_name target_name target_start target_end region_start region_end region_evalue region_ka_bits region_poisson_score region_n_shared_kmers".split()
                )
                for i in range(50_003):
                    lab = "real" if i % 7 else "swap1"
                    start = 0 if i % 3 else 100
                    e = [1, 10, 100, 101][i % 4]
                    w.writerow(
                        [
                            "q" if lab == "real" else "s",
                            "t",
                            start,
                            start + 20,
                            0,
                            20,
                            e,
                            30,
                            4,
                            3,
                        ]
                    )
                    family = "PF00001" if start == 0 else "PF00150"
                    for c in cuts:
                        if e <= c:
                            sets[c, lab].add(family)
                            counts[c, lab] += 1
            args = [
                "--regions",
                str(root / "regions.csv"),
                "--domains",
                str(root / "domains.tsv"),
                "--peptide-map",
                str(root / "map.tsv"),
                "--key",
                str(root / "key.tsv"),
                "--labellings",
                str(root / "labels.tsv"),
                "--organism",
                "test",
                "--setting",
                "a.k7",
                "--status",
                "fitted",
                "--out-prefix",
                str(root / "result"),
            ]
            subprocess.run([sys.executable, str(SCORER), *args], check=True)
            with (root / "result.summary.tsv").open() as fh:
                for row in csv.DictReader(fh, delimiter="\t"):
                    key = float(row["evalue_cutoff"]), row["labelling"]
                    self.assertEqual(int(row["recovered"]), int("PF00001" in sets[key]))
                    self.assertEqual(int(row["outside"]), int("PF00150" in sets[key]))
                    self.assertEqual(int(row["n_landing_rows"]), counts[key])
            with gzip.open(root / "result.landings.tsv.gz", "rt") as fh:
                self.assertEqual(
                    sum(1 for _ in csv.DictReader(fh, delimiter="\t")), 50_003
                )
            args[1] = "-"
            args[-1] = str(root / "empty")
            subprocess.run(
                [sys.executable, str(SCORER), *args], input="", text=True, check=True
            )
            with (root / "empty.summary.tsv").open() as fh:
                rows = list(csv.DictReader(fh, delimiter="\t"))
                self.assertEqual(len(rows), 12)
                self.assertTrue(all(r["recovered"] == "0" for r in rows))


if __name__ == "__main__":
    unittest.main()
