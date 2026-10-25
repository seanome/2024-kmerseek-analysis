"""Check production scope, exact paper labels, and fail-closed reference gating."""

import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).with_name("prepare_production.py")


class ProductionTests(unittest.TestCase):
    def test_preparation_and_reference_gate(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            root = base / "botryllus-shazam-PEC-20261023"
            for name in [
                "figure4-inputs",
                "references",
                "checkpoints/pilot/results",
                "external-tunicate-pilot-20261023/RUN/trimmed",
            ]:
                (root / name).mkdir(parents=True)
            pre = base / "pre"
            pre.mkdir()
            (pre / "complete.json").write_text('{"genome_matches_all": false}')
            env = {**os.environ, "SCRATCH": str(base)}
            cmd = [
                sys.executable,
                str(SCRIPT),
                str(base / "code"),
                str(pre),
                str(base / "launch"),
            ]
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((base / "launch").exists())
            (pre / "complete.json").write_text('{"genome_matches_all": true}')
            (root / "trimmed-inputs.validated.json").write_text(
                json.dumps([dict(run=f"R{i}", pairs=i) for i in range(64)])
            )
            (root / "figure4-inputs/immune-gene-sets.tsv").write_text(
                "gene_id\tcategory\nG1\timmune\n"
            )
            h = dict(
                gene_ids=["G2"],
                hub="DNA repair",
                time="2 Hours Young",
                direction="Up-reg",
                paper_label="DDX31",
                paper_group=1,
                cohort="young",
                status="unique exact label within published comparison",
            )
            (root / "figure4-inputs/hub-gene-memberships.json").write_text(
                json.dumps([h, {**h, "status": "ambiguous"}])
            )
            (
                root / "checkpoints/pilot/results/botryllus_candidate_annotated.fasta"
            ).write_text(">B1 candidate\nM\n")
            with gzip.open(root / "references/uniprot_sprot.fasta.gz", "wt") as f:
                f.write(">sp|P1|NAME known protein\nM\n")
            ext = root / "external-tunicate-pilot-20261023"
            (ext / "external-runs.tsv").write_text(
                "run_accession\tsample_title\nRUN\tembryos\n"
            )
            for i in (1, 2):
                (ext / f"RUN/trimmed/RUN_{i}_val_{i}.fq.gz").touch()
            subprocess.run(cmd, env=env, check=True)
            with (base / "launch/inputs/targets.tsv").open() as f:
                targets = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(targets), 14)
            with (base / "launch/inputs/paper.tsv").open() as f:
                paper = list(csv.DictReader(f, delimiter="\t"))
            self.assertEqual(len(paper), 2)
            self.assertEqual(paper[1]["paper_label"], "DDX31")
            scope = json.loads((base / "launch/scope.json").read_text())
            self.assertEqual(scope["pairs"], sum(range(64)))
            self.assertEqual(len(scope["ambiguous_or_unresolved_hubs"]), 1)
            self.assertEqual(scope["external_runs"], ["RUN"])


if __name__ == "__main__":
    unittest.main()
