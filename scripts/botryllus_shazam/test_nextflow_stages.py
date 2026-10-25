"""Adversarial checks for migration helpers; no experimental reads."""

import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import nextflow_stages as stages


class StageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = Path.cwd()
        __import__("os").chdir(self.tmp.name)

    def tearDown(self):
        __import__("os").chdir(self.old)
        self.tmp.cleanup()

    def models(self):
        paths = []
        for i in range(3):
            p = Path(str(i))
            p.mkdir()
            paths.append(str(p))
            with gzip.open(p / "all-features.tsv.gz", "wt") as f:
                f.write("tag\tpadj\tlog2FoldChange\tcontrast\n")
                for seq, q in [("A" * 31, 0.049), ("C" * 31, 0.05), ("G" * 31, "NA")]:
                    f.write(f"{seq}\t{q}\t2\tcontrast{i}\n")
        return paths

    def test_strict_threshold_and_all_contrasts(self):
        stages.seeds(*self.models())
        result = json.loads(Path("selected/seeds.json").read_text())
        self.assertEqual(list(result), ["A" * 31])
        self.assertEqual(len(result["A" * 31]), 3)

    def test_missing_contrast_fails(self):
        with self.assertRaises(ValueError):
            stages.seeds(*self.models()[:2])

    def test_gene_mapping_and_count_truncation(self):
        Path("ref.fa").write_text(">T1\nACGT\n")
        Path("ref.gtf").write_text(
            'chr\tx\texon\t1\t4\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n'
        )
        Path("S.quant").mkdir()
        Path("S.quant/abundance.tsv").write_text("target_id\test_counts\nT1\t3.9\n")
        stages.genes("ref.fa", "ref.gtf", "S.quant")
        with gzip.open("genes.tsv.gz", "rt") as f:
            self.assertEqual(f.read(), "tag\tS\nG1\t3\n")
        Path("S.quant/abundance.tsv").write_text(
            "target_id\test_counts\nMISSING\t3.9\n"
        )
        with self.assertRaises(ValueError):
            stages.genes("ref.fa", "ref.gtf", "S.quant")

    def test_duplicate_fasta_fails(self):
        Path("x.fa").write_text(">x\nACGT\n>x\nAAAA\n")
        with self.assertRaises(ValueError):
            stages.fasta("x.fa")

    def test_empty_compactors_preserves_zero(self):
        Path("selected").mkdir()
        Path("reads").mkdir()
        Path("selected/ids.json").write_text("{}")
        with patch(
            "subprocess.run", side_effect=AssertionError("No tool for empty seeds")
        ):
            stages.compact("dekupl", "reads", "selected", "unused", "1")
        self.assertEqual(stages.fasta("context/dna.fasta"), {})

    def test_workflow_copies_do_not_mutate_staged_directories(self):
        workflow = (
            Path(__file__).resolve().parents[2] / "pipelines/botryllus-shazam/main.nf"
        )
        commands = [
            line.strip()
            for line in workflow.read_text().splitlines()
            if line.strip().startswith("cp -R")
        ]
        self.assertEqual(len(commands), 2)
        source = Path("source")
        source.mkdir()
        (source / "input").write_text("original")
        Path("staged").symlink_to(source.resolve(), target_is_directory=True)
        for command in commands:
            subprocess.run(
                ["bash", "-ue", "-c", command.replace("${folder}", "staged")],
                check=True,
            )
            dest = Path(command.split()[-1])
            self.assertFalse(dest.is_symlink())
            (dest / "input").write_text("edited")
            self.assertEqual((source / "input").read_text(), "original")

    def test_join_keeps_both_sources_and_unmatched_paper_gene(self):
        Path("reference.fa").write_text(">T1\n" + "A" * 40 + "\n")
        Path("mapping.tsv").write_text("T1\tG1\n")
        Path("paper.tsv").write_text("gene_id\tcategory\nG1\timmune\nG2\thub\n")
        Path("annotations.tsv").write_text("record_id\tname\nREF1\tKnown reference\n")
        folders = []
        for source, prefix in [("splash", "SA"), ("dekupl", "DK")]:
            q = Path(source)
            q.mkdir()
            folders.append(str(q))
            stages.write_fasta(q / "dna.fasta", {prefix: "A" * 31})
            stages.save(
                q / "origins.json",
                {prefix: dict(kind="selected seed", source=source, seed=prefix)},
            )
            stages.save(
                q / "protocol.json", dict(source=source, protocol=stages.PROTOCOL)
            )
            m = Path(source + "-matches")
            m.mkdir()
            folders.append(str(m))
            stages.save(
                m / "search.json",
                dict(source=source, database="swissprot", mode="hp", scaled=5),
            )
            (m / "matches.csv").write_text(
                "query_name,match_name,intersect_hashes,containment,scaled\n"
                + prefix
                + ",REF1,2,0.2,5\n"
            )
        for folder in folders:
            m = Path(folder)
            if (m / "search.json").exists():
                meta = json.loads((m / "search.json").read_text())
                meta["matches_sha256"] = stages.sha(m / "matches.csv")
                stages.save(m / "search.json", meta)
        stages.evidence(
            "reference.fa", "mapping.tsv", "paper.tsv", "annotations.tsv", *folders
        )
        data = json.loads(Path("gene-evidence.json").read_text())
        rows = {r["gene_id"]: r for r in data["genes"]}
        self.assertEqual(set(rows), {"G1", "G2"})
        self.assertEqual(set(rows["G1"]["sequences"]), {"SA", "DK"})
        self.assertEqual(len(rows["G1"]["matches"]), 2)
        self.assertEqual(rows["G2"]["matches"], [])
        with self.assertRaises(ValueError):
            stages.evidence(
                "reference.fa",
                "mapping.tsv",
                "paper.tsv",
                "annotations.tsv",
                *folders[:2],
            )


if __name__ == "__main__":
    unittest.main()
