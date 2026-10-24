"""Exercise uncapped selection and rejection of capped or changed discoveries."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from full_query_provenance import validate, sha


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
