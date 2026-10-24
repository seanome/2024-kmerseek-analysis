"""Exercise uncapped selection and rejection of capped or changed discoveries."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from full_query_provenance import validate, sha, prepare_dna_queries


class FullQueryTests(unittest.TestCase):
    def test_uncapped_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "discovery"
            (out / "all-significant").mkdir(parents=True)
            manifest = root / "runs.tsv"
            manifest.write_text(
                "run_accession\n" + "".join(f"R{i}\n" for i in range(64))
            )
            (root / "trimmed-inputs.validated.json").write_text(
                json.dumps([{"run": f"R{i}"} for i in range(64)])
            )
            anchors = [
                "A" + format(i, "09b").replace("0", "A").replace("1", "C")
                for i in range(301)
            ]
            source = out / "full.after_correction.all_anchors.tsv"
            source.write_text(
                "anchor\tpval_opt_corrected\n"
                + "".join(x + "\t0.01\n" for x in anchors)
                + "GGG\t0.05\n"
            )
            selected = out / "all-significant/selected_anchors.tsv"
            selected.write_text("anchor\n" + "\n".join(anchors) + "\n")
            selection = out / "all-significant/selection.json"
            record = dict(
                extension_cap=None,
                rule="pval_opt_corrected < 0.05",
                source_sha256=sha(source),
                tested_anchors=302,
                selected_anchors=301,
            )
            selection.write_text(json.dumps(record))
            for name in ["peptides.fasta", "extensions.fasta"]:
                (out / name).write_text(">q1\nACGT\n")
            (out / "discovery.validated.json").write_text(
                json.dumps(
                    dict(
                        selected_anchors=301,
                        unique_peptides=1,
                        extension_records=1,
                        outputs={
                            n: sha(out / n)
                            for n in ["peptides.fasta", "extensions.fasta"]
                        },
                    )
                )
            )
            self.assertEqual(validate(root, out, manifest)["selected_anchors"], 301)
            (out / "peptide_membership.tsv").write_text(
                "anchor\tpeptide_id\n" + anchors[0] + "\tq1\n"
            )
            query_dir = root / "queries"
            query_dir.mkdir()
            provenance = {}
            dna = prepare_dna_queries(out, query_dir, provenance)
            self.assertEqual(
                sum(x.startswith(">") for x in dna.read_text().splitlines()), 302
            )
            with (query_dir / "anchor-query-coverage.tsv").open() as stream:
                coverage = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(coverage), 301)
            self.assertEqual(sum(int(x["peptide_queries"]) > 0 for x in coverage), 1)
            self.assertIn("GGG", source.read_text())
            self.assertNotIn("anchor=GGG", dna.read_text())
            record["extension_cap"] = 300
            selection.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "Capped"):
                validate(root, out, manifest)
            record["extension_cap"] = None
            selection.write_text(json.dumps(record))
            selected.write_text("anchor\n" + "\n".join(anchors[:300]) + "\n")
            with self.assertRaisesRegex(ValueError, "complete strict"):
                validate(root, out, manifest)
            selected.write_text("anchor\n" + "\n".join(anchors) + "\n")
            (out / "peptides.fasta").write_text(">tampered\nTTTT\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                validate(root, out, manifest)


if __name__ == "__main__":
    unittest.main()
