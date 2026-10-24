"""Repeated collections do not increase the genotype replicate count."""

import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class PoolTests(unittest.TestCase):
    def test_balanced_collections(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            design = {
                "contrasts": [
                    {
                        "name": "test",
                        "genotypes": ["G1", "G2", "G3"],
                        "libraries": [
                            {"run": f"{g}{c}{i}", "genotype": g, "condition": c}
                            for g in ("G1", "G2", "G3")
                            for c in ("Control", "Treatment")
                            for i in (1, 2)
                        ],
                    }
                ]
            }
            (p / "design.json").write_text(json.dumps(design))
            names = [x["run"] for x in design["contrasts"][0]["libraries"]]
            for name in ("raw.gz", "genes.gz"):
                with gzip.open(p / name, "wt") as f:
                    w = csv.writer(f, delimiter="\t")
                    w.writerow(["tag", *names])
                    w.writerow(["KEEP", *[3] * len(names)])
                    w.writerow(["DROP", *[1] * len(names)])
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("pool_abundance.py")),
                    str(p / "raw.gz"),
                    str(p / "genes.gz"),
                    str(p / "design.json"),
                    str(p / "out"),
                ],
                check=True,
            )
            for assay in ("genes", "kmers"):
                folder = p / "out/test" / assay
                with gzip.open(folder / "pooled.tsv.gz", "rt") as f:
                    rows = list(csv.reader(f, delimiter="\t"))
                self.assertEqual(rows[1:], [["KEEP", *["6"] * 6]])
                self.assertEqual(
                    json.loads((folder / "eligible.json").read_text())["features"], 1
                )
                found = []
                for x in folder.glob("part-*.gz"):
                    with gzip.open(x, "rt") as f:
                        found.extend(csv.DictReader(f, delimiter="\t"))
                self.assertEqual(len(found), 1)


if __name__ == "__main__":
    unittest.main()
