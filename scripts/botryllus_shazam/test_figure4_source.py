"""Check the cached publication inputs and detect duplicated gene memberships."""

import csv, gzip, json, unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "reports/botryllus-shazam/figure4"


class PublishedInputs(unittest.TestCase):
    def test_gene_sets_and_count_matrix(self):
        rows = list(
            csv.DictReader((ROOT / "immune-gene-sets.tsv").open(), delimiter="\t")
        )
        self.assertEqual(
            Counter(r["role"] for r in rows), dict(proimmune=13, cell_repair=133)
        )
        self.assertEqual(len({r["gene_id"] for r in rows}), 146)
        with gzip.open(ROOT / "published-counts.tsv.gz", "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            self.assertEqual(len(set(header[1:])), 64)
            genes = {r[0] for r in reader}
        self.assertEqual(len(genes), 72113)
        self.assertTrue({r["gene_id"] for r in rows} <= genes)

    def test_hub_counts_and_explicit_ambiguity(self):
        rows = list(
            csv.DictReader((ROOT / "published-hub-counts.tsv").open(), delimiter="\t")
        )
        self.assertEqual(len(rows), 84)
        target = [
            r
            for r in rows
            if r["cohort"] == "young"
            and r["time"] == "2 Hours Young"
            and r["hub"] == "DNA / RNA / Epigenetic"
        ]
        self.assertEqual(
            {r["direction"]: int(r["count"]) for r in target},
            {"Up-reg": 9, "Down-reg": 31},
        )
        mappings = json.loads((ROOT / "hub-gene-memberships.json").read_text())
        for r in mappings:
            if r["status"].startswith("unique"):
                self.assertEqual(len(r["gene_ids"]), 1)
            elif r["status"].startswith("ambiguous"):
                self.assertGreater(len(r["gene_ids"]), 1)
            else:
                self.assertEqual(r["gene_ids"], [])


if __name__ == "__main__":
    unittest.main()
