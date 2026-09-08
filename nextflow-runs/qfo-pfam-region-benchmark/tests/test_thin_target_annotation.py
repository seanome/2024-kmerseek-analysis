"""A target species with almost no annotation must stop the report, not read as biology.

The failure this locks down was silent. On the midi run every task ran, exited 0, and
published a valid file, and the report drew Ciona intestinalis at 550 Mya as an
evolutionary cliff: every kmerseek arm at Fmax 0.001 with recall_reachable 0.000, phmmer
down 10x, hhblits down 2x.

It was not divergence. The primary truth set is Swiss-Prot, and the transfer table a
species is scored through is built from Swiss-Prot entries for that species. Ciona has 28
of them. Mouse has 17_228, arabidopsis 16_396, chicken 2_309 -- the smallest non-Ciona
target is 80x bigger. So the ciona transfer table was ~1/100th the size and EVERY tool
lost 30-130x of its calls, folddisco included (its recall_reachable fell 0.94 -> 0.15; only
its Fmax rose, because Fmax is precision-dominated and the handful of curated Ciona
features that survive are the easy ones).

The reachability bar was the thing meant to catch this and could not: on the Swiss-Prot
truth set `pfam_id` holds one of ~15 curated feature types, 28 proteins still carry almost
all of them, and Ciona scored 6_991 / 7_000 reachable. Hence two checks here -- one on
label vocabulary, one on annotated target proteins -- because the first is what blinded
the second.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parents[1] / "bin"
AGG = BIN / "aggregate_domain_metrics.py"

# Real UniProtKB/Swiss-Prot entry counts for the benchmark's nine target proteomes,
# counted from uniprot_sprot.dat.gz on 2026-09-01. Ciona is not a rounding difference.
SPROT_ENTRIES = {
    "mouse": 17_228, "arabidopsis": 16_396, "yeast": 6_733, "ecoli": 4_531,
    "worm": 4_489, "fly": 3_816, "zebrafish": 3_351, "chicken": 2_309, "ciona": 28,
}


def row(species, n_target_map_proteins, *, truth_set="swissprot", n_truth_families=12,
        tool="kmerseek", n_calls=5_000):
    return {"tool": tool, "variant": "v1", "species": species, "split": "heldout",
            "stratum_axis": "all", "stratum": "all", "truth_set": truth_set,
            "dedup_transfers": False, "n_calls": n_calls, "fmax": 0.2, "auprc": 0.2,
            "n_target_map_proteins": n_target_map_proteins,
            "n_truth_families": n_truth_families,
            "n_truth_instances": 7_000, "n_reachable_instances": 7_000}


def run(tmp_path, rows, *flags):
    metrics, curves = tmp_path / "metrics", tmp_path / "curves"
    metrics.mkdir(exist_ok=True)
    curves.mkdir(exist_ok=True)
    pl.DataFrame(rows).write_parquet(metrics / "a.metrics.parquet")
    pl.DataFrame({"split": ["heldout"], "recall": [0.5]}).write_parquet(
        curves / "a.curve.parquet")
    return subprocess.run(
        [sys.executable, str(AGG), *flags, str(metrics), str(curves),
         str(tmp_path / "m.parquet"), str(tmp_path / "m.csv"),
         str(tmp_path / "c.parquet")],
        capture_output=True, text=True)


REAL_SHAPE = [row(sp, n) for sp, n in SPROT_ENTRIES.items()]
HEALTHY = [row(sp, n) for sp, n in SPROT_ENTRIES.items() if sp != "ciona"]


def test_the_ciona_column_fails_the_run(tmp_path):
    proc = run(tmp_path, REAL_SHAPE)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "THIN TARGET ANNOTATION" in out
    assert "ciona" in out


def test_a_healthy_run_is_not_flagged(tmp_path):
    # chicken at 2_309 against a median of ~4_500 is a real spread between proteomes and
    # must not trip the check, or the guard gets turned off and stops guarding.
    proc = run(tmp_path, HEALTHY)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "THIN TARGET ANNOTATION" not in (proc.stdout + proc.stderr)


def test_the_tables_are_written_before_it_fails(tmp_path):
    run(tmp_path, REAL_SHAPE)
    assert (tmp_path / "m.parquet").exists() and (tmp_path / "m.csv").exists()


def test_allow_thin_targets_reports_anyway(tmp_path):
    proc = run(tmp_path, REAL_SHAPE, "--allow-thin-targets")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "THIN TARGET ANNOTATION" in (proc.stdout + proc.stderr)


def test_a_partial_report_waives_it_too(tmp_path):
    # A snapshot of a run still going can legitimately show a species as thin.
    proc = run(tmp_path, REAL_SHAPE, "--allow-dead-arms")
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_small_label_vocabulary_is_called_out(tmp_path):
    proc = run(tmp_path, HEALTHY)
    out = proc.stdout + proc.stderr
    assert "DEGENERATE REACHABILITY" in out
    assert "plain recall" in out


def test_a_family_vocabulary_is_not_called_out(tmp_path):
    rows = [row(sp, n, truth_set="pfam", n_truth_families=900)
            for sp, n in SPROT_ENTRIES.items() if sp != "ciona"]
    proc = run(tmp_path, rows)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "DEGENERATE REACHABILITY" not in (proc.stdout + proc.stderr)


def test_an_older_metrics_table_without_the_columns_still_aggregates(tmp_path):
    rows = [{k: v for k, v in r.items()
             if k not in ("n_target_map_proteins", "n_truth_families")}
            for r in HEALTHY]
    proc = run(tmp_path, rows)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_an_empty_domain_map_stops_scoring(tmp_path, pfam_truth):
    """The total case, caught on first occurrence rather than by comparing neighbours."""
    empty = tmp_path / "empty_domain_map.parquet"
    pl.read_parquet(pfam_truth["map"]).head(0).write_parquet(empty)
    regions = tmp_path / "regions.tsv.gz"
    subprocess.run(["bash", "-c", f"printf '' | gzip > {regions}"], check=True)
    proc = subprocess.run(
        [sys.executable, str(BIN / "evaluate_domain_calls.py"),
         "--truth", str(pfam_truth["truth"]), "--domain-map", str(empty),
         "--species", "ciona", "--truth-set", "pfam",
         "--tool", "kmerseek", "--variant", "v1", "--regions", str(regions),
         "--calls-out", str(tmp_path / "calls.parquet"),
         "--metrics-out", str(tmp_path / "metrics.parquet")],
        cwd=tmp_path, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "empty domain map" in (proc.stdout + proc.stderr)


def test_the_new_coverage_columns_reach_the_metrics_row(tmp_path, pfam_truth):
    """The counts have to be ON the row, or the cross-species check has nothing to read."""
    from conftest import score, write_perfect_regions

    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"],
                          tmp_path / "regions.tsv")
    m = score(pfam_truth["truth"], pfam_truth["map"], tmp_path / "regions.tsv",
              tmp_path / "run", "pfam")
    for col in ("n_target_map_proteins", "n_target_map_instances", "n_target_families",
                "n_truth_families"):
        assert col in m.columns, f"{col} missing from the metrics row"
    assert m["n_target_map_proteins"].max() > 0
    assert m["n_truth_families"].max() > 0
