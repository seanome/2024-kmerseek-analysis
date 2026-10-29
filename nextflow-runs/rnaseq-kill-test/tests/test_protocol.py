"""Attack rule boundaries, missing evidence, species leakage and label coverage."""

import copy
import hashlib
import importlib.util
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("protocol", ROOT / "protocol.py")
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def write_table(path, rows):
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return path


def sample_rows(species="mouse"):
    return [
        dict(
            sample=f"s{i}",
            condition="A" if i < 4 else "B",
            biological_replicate=f"animal{i}",
            taxid="10090" if species == "mouse" else "7955",
            study="test fixture",
            runs=f"run{i}",
            layout="PAIRED",
            strandedness="reverse",
            read_pairs="20000000",
            library_evidence="test fixture only",
            published_de="test fixture only",
            selection_date="2026-10-22",
        )
        for i in range(8)
    ]


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)

    def test_frozen_rules(self):
        p.check_rules()
        for name in ("decision_rules", "implementation_choices"):
            content = (ROOT / f"{name}.md").read_bytes()
            (self.path / f"{name}.md").write_bytes(content)
            (self.path / f"{name}.sha256").write_text(
                hashlib.sha256(content).hexdigest()
            )
        with (self.path / "decision_rules.md").open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            p.check_rules(self.path)

    def test_missing_calibration_blocks_before_mouse_manifest_is_opened(self):
        with patch.object(
            p, "check_samples", side_effect=AssertionError("mouse opened")
        ):
            with self.assertRaisesRegex(ValueError, "calibration is missing"):
                p.preflight(ROOT, self.path / "mouse.tsv", "mouse")

    def test_sample_eligibility_boundaries(self):
        rows = sample_rows()
        manifest = self.path / "samples.tsv"
        self.assertEqual(len(p.check_samples(write_table(manifest, rows), "mouse")), 8)
        mutations = [
            ("read_pairs", "19999999"),
            ("taxid", "7955"),
            ("strandedness", "auto"),
            ("layout", "SINGLE"),
            ("published_de", ""),
            ("read_pairs", "nan"),
            ("biological_replicate", "animal1"),
            ("runs", "run1"),
            ("selection_date", "not a date"),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(rows)
                changed[0][key] = value
                with self.assertRaises(ValueError):
                    p.check_samples(write_table(manifest, changed), "mouse")

    def test_duplicate_headers_are_not_silently_overwritten(self):
        path = self.path / "duplicate.tsv"
        path.write_text("sample\tsample\nfirst\tsecond\n")
        with self.assertRaisesRegex(ValueError, "duplicate.*column"):
            p.read_table(path, {"sample"})

    def test_sequencing_run_tokens_are_nonempty_and_unique(self):
        for bad_runs in ("run0;", ";run0", "run0; ", "run0; run1"):
            with self.subTest(runs=bad_runs):
                rows = sample_rows()
                rows[0]["runs"] = bad_runs
                with self.assertRaises(ValueError):
                    p.check_samples(
                        write_table(self.path / "samples.tsv", rows), "mouse"
                    )

    def test_distinct_sequencing_runs_allow_separator_spaces(self):
        rows = sample_rows()
        rows[0]["runs"] = "run0; run8"
        self.assertEqual(
            len(p.check_samples(write_table(self.path / "samples.tsv", rows), "mouse")),
            8,
        )

    def test_two_conditions_and_replicate_count(self):
        manifest = self.path / "samples.tsv"
        for rows in (
            sample_rows()[:4],
            sample_rows()[1:],
            sample_rows()[1:4] + sample_rows()[5:],
        ):
            with self.assertRaises(ValueError):
                p.check_samples(write_table(manifest, rows), "mouse")

    def test_enumeration_agrees_with_independent_binary_exhaustion(self):
        samples = [f"s{i}" for i in range(8)]
        rows = list(p.assignments(samples, [0] * 4 + [1] * 4))
        independent = {
            bits for bits in itertools.product((0, 1), repeat=8) if sum(bits) == 4
        }
        self.assertEqual({tuple(r["labels"].values()) for r in rows}, independent)
        self.assertEqual(len(rows), 70)
        self.assertEqual(sum(r["is_true"] for r in rows), 1)
        self.assertEqual(sum(r["is_complement"] for r in rows), 1)

    def test_equal_complement_count_fails_literal_swap_rule(self):
        expected = list(p.assignments(list("abcdefgh"), [0] * 4 + [1] * 4))
        rows = [
            {
                "assignment": r["assignment"],
                "significant_units": 100 if r["is_true"] or r["is_complement"] else 0,
            }
            for r in expected
        ]
        values = p.score_swaps(rows, expected)
        self.assertFalse(values["true_exceeds_every_swap"])
        self.assertEqual(values["swap_count"], 69)
        self.assertEqual(p.score_claims({"5": values})["5"]["verdict"], "FAIL")
        with self.assertRaisesRegex(ValueError, "every balanced"):
            p.score_swaps(rows[:-1], expected)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            p.score_swaps(rows + rows[:1], expected)

    def test_zero_true_count_is_not_divided_or_passed(self):
        expected = list(p.assignments(list("abcd"), [0, 0, 1, 1]))
        rows = [
            {"assignment": r["assignment"], "significant_units": 0} for r in expected
        ]
        values = p.score_swaps(rows, expected)
        self.assertIsNone(values["median_swap_to_true_ratio"])
        self.assertEqual(p.score_claims({"5": values})["5"]["verdict"], "FAIL")

    def test_unknown_lineage_and_vertebrates_rejected(self):
        row = dict(
            protein="P1",
            family="PF00001",
            taxid="7227",
            lineage_taxids="1;2759;7227",
            sequence_sha256="a" * 64,
        )
        path = self.path / "members.tsv"
        nodes = self.path / "nodes.dmp"
        nodes.write_text(
            "1 | 1 |\n2759 | 1 |\n7227 | 2759 |\n7742 | 2759 |\n"
            "7955 | 7742 |\n40674 | 7742 |\n9606 | 40674 |\n10090 | 40674 |\n"
        )
        self.assertEqual(
            p.check_index_members(write_table(path, [row]), 7742, nodes), 1
        )
        for lineage in ("", "7227", "1;7742;7227", "1;10090;7227"):
            changed = dict(row, lineage_taxids=lineage)
            with self.assertRaises(ValueError):
                p.check_index_members(write_table(path, [changed]), 7742, nodes)
        fish = dict(row, taxid="7955", lineage_taxids="1;7742;7955")
        fish["lineage_taxids"] = "1;2759;7742;7955"
        self.assertEqual(
            p.check_index_members(write_table(path, [fish]), 40674, nodes), 1
        )
        mammal = dict(row, taxid="9606", lineage_taxids="1;7742;40674;9606")
        with self.assertRaisesRegex(ValueError, "held-out"):
            p.check_index_members(write_table(path, [mammal]), 40674, nodes)
        truncated_human = dict(row, taxid="9606", lineage_taxids="1;9606")
        with self.assertRaisesRegex(ValueError, "held-out"):
            p.check_index_members(write_table(path, [truncated_human]), 7742, nodes)

    def test_taxonomy_cycles_and_unknown_taxa_rejected(self):
        with self.assertRaisesRegex(ValueError, "cycle"):
            p.ancestors("10", {"10": "11", "11": "10", "1": "1"})
        with self.assertRaisesRegex(ValueError, "missing"):
            p.ancestors("12", {"1": "1"})

    def test_no_measurements_cannot_pass(self):
        result = p.score_claims({})
        self.assertEqual(
            [result[str(i)]["verdict"] for i in range(1, 6)], ["not measured"] * 5
        )
        self.assertEqual(result["alphabet_action"], "not measured")

    def test_strict_and_inclusive_claim_limits(self):
        measurements = {
            "1": dict(
                wall_ratio_no_reference=0.1,
                cpu_ratio_no_reference=0.01,
                wall_ratio_with_reference=0.1,
                cpu_ratio_with_reference=0.1,
            ),
            "2": dict(genes_hit_top300=70, go_overlap_top20=10),
            "3": dict(
                in_domain_accuracy=0.8,
                outside_protein_false_family_rate=0.05,
                tag_to_hmmscan_cpu_ratio=0.099,
            ),
            "4": dict(recall_gain_at_equal_false_family_rate=0.05),
            "5": dict(true_exceeds_every_swap=True, median_swap_to_true_ratio=0.05),
        }
        result = p.score_claims(measurements)
        self.assertEqual(
            [result[str(i)]["verdict"] for i in range(1, 6)],
            ["FAIL", "PASS", "PASS", "PASS", "FAIL"],
        )
        measurements["4"]["recall_gain_at_equal_false_family_rate"] = -0.1
        self.assertEqual(
            p.score_claims(measurements)["alphabet_action"],
            "re-report claims 1 to 3 in protein20",
        )

    def test_invalid_values_do_not_become_passes(self):
        for value in (float("nan"), float("inf"), -1, True, "not measured"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                p.score_claims({"2": {"genes_hit_top300": value}})
        with self.assertRaises(ValueError):
            p.score_claims({"3": {"in_domain_accuracy": 80}})
        with self.assertRaises(ValueError):
            p.score_claims({"2": {"go_overlap_top20": 21}})
        with self.assertRaisesRegex(ValueError, "boolean"):
            p.score_claims({"4": {"recall_gain_at_equal_false_family_rate": True}})

    def test_calibration_requires_complete_settings_and_unchanged_development_files(
        self,
    ):
        manifest = write_table(self.path / "dev.tsv", sample_rows("zebrafish"))
        outputs = self.path / "dev.json"
        outputs.write_text('{"test_fixture": true}')
        record = dict(
            rules_sha256=p.digest(ROOT / "decision_rules.md"),
            development_taxid=7955,
            heldout_taxid=7742,
            mouse_accessed=False,
            source="development_outputs",
            frozen_date="2026-10-22",
            development_manifest="dev.tsv",
            development_outputs="dev.json",
            development_manifest_sha256=p.digest(manifest),
            development_outputs_sha256=p.digest(outputs),
            settings=[
                dict(alphabet=a, k=k, min_unique_kmers=2)
                for a, ks in (
                    ("protein20", (4, 5, 6, 7, 8, 9)),
                    ("hp_thomas_dill2", (19, 23, 27, 31, 35)),
                )
                for k in ks
            ],
        )
        path = self.path / "calibration.json"

        def seal():
            path.write_text(json.dumps(record))
            path.with_suffix(".sha256").write_text(p.digest(path))

        seal()
        self.assertEqual(p.check_calibration(path, record["rules_sha256"]), record)
        record["settings"].pop()
        seal()
        with self.assertRaisesRegex(ValueError, "every frozen"):
            p.check_calibration(path, record["rules_sha256"])
        outputs.write_text("changed")
        with self.assertRaisesRegex(ValueError, "source has changed"):
            p.check_calibration(path, record["rules_sha256"])

    def test_status_writer_rejects_modified_report(self):
        import shutil
        import subprocess
        import sys

        for name in (
            "write_status.py",
            "protocol.py",
            "decision_rules.md",
            "decision_rules.sha256",
            "implementation_choices.md",
            "implementation_choices.sha256",
        ):
            shutil.copyfile(ROOT / name, self.path / name)
        subprocess.run(
            [sys.executable, str(self.path / "write_status.py")],
            check=True,
            capture_output=True,
        )
        report = self.path / "report.md"
        report.write_text(
            report.read_text() + "\nBiological measurement: 123 seconds.\n"
        )
        before = report.read_bytes()
        process = subprocess.run(
            [sys.executable, str(self.path / "write_status.py")], capture_output=True
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual(report.read_bytes(), before)

    def test_overlap_counts_genes_once_and_excludes_19_bases(self):
        overlaps = [
            dict(unit="u1", gene="g1", overlap_bp="20"),
            dict(unit="u1", gene="g1", overlap_bp="21"),
            dict(unit="u2", gene="g1", overlap_bp="20"),
            dict(unit="u2", gene="g2", overlap_bp="19"),
        ]
        result = p.overlap_curve(["u1", "u2"], ["g1", "g2"], overlaps)
        self.assertEqual([r["genes_hit"] for r in result], [1] * 5)
        self.assertEqual([r["available_units"] for r in result], [2] * 5)
        with self.assertRaises(ValueError):
            p.overlap_curve(["u1", "u1"], ["g1"], overlaps)

    def test_taxonomy_skips_blank_lines_and_names_bad_line(self):
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            good = Path(folder) / "good.dmp"
            good.write_text("1\t|\t1\t|\n\n2\t|\t1\t|\n\n")
            self.assertEqual(p.read_taxonomy(good), {"1": "1", "2": "1"})
            bad = Path(folder) / "bad.dmp"
            bad.write_text("1\t|\t1\t|\nbroken\n")
            with self.assertRaisesRegex(ValueError, "line 2"):
                p.read_taxonomy(bad)


if __name__ == "__main__":
    unittest.main()
