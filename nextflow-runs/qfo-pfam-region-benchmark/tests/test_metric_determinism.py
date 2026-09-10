"""The metrics must not change when nothing about the input does.

Two columns were not reproducible. Four runs of evaluate_domain_calls.py with identical
arguments gave four distinct values of sens_first_fp_mean -- summed over the metric rows,
1.9249, 2.3062, 1.5438, 1.4750, a 56% spread -- and three of sens_first_fp_median. Every
other column was bit-identical, which is what made it read as a quirk of that one metric.

The cause is the one test_dedup_passes.py already guards for the suppression passes, at a
different stage: an ordered read of a table that has no order. sensitivity_to_first_fp
ranks a query's calls and counts what sits above the first false positive, and calls tied
on score came back in whatever order polars emitted them, so the boundary moved. Ties are
the common case rather than the corner one -- HP alphabets at low ksize produce large
blocks of identical region scores.

Because those two columns were excluded from old-vs-new metric comparison as known noise,
a real regression in them was invisible. These tests are what lets them be compared again.

The permutation tests are the load-bearing ones: they fail on the defect every time, by
feeding the same rows in different orders. Re-running the script only fails when polars
happens to reorder, which it does on a real sweep and may not on a small fixture.
"""

import sys
from pathlib import Path

import polars as pl

from conftest import score, write_decoy_regions, write_perfect_regions

BIN = Path(__file__).resolve().parents[1] / "bin"
sys.path.insert(0, str(BIN))

import cafa_metrics as cm  # noqa: E402
import evaluate_domain_calls as edc  # noqa: E402

N_ORDERINGS = 8


def tied_calls(n_proteins: int = 300, seed: int = 1):
    """A scored-calls table that is dense in score ties, plus its answer key.

    Unique on (query_acc, pfam_id, qstart, qend), which is what score_calls emits -- it
    groups on exactly those four. CALL_TIEBREAK is a total order only under that
    invariant, so a fixture that broke it would be testing a table the pipeline cannot
    produce. test_scored_calls_are_unique_on_the_call_key checks the real thing.
    """
    import random

    rng = random.Random(seed)
    truth, calls = [], []
    for p in range(n_proteins):
        acc = f"Q{p:05d}"
        for i in range(3):
            truth.append({"accession": acc, "pfam_id": f"PF{i:05d}",
                          "domain_start": 10 + 120 * i, "domain_end": 100 + 120 * i})
        for j in range(6):
            i = j % 4
            calls.append({
                "query_acc": acc, "pfam_id": f"PF{i:05d}",
                "qstart": 10 + 120 * i + j, "qend": 100 + 120 * i + j,
                "score": rng.choice([1.0, 1.0, 1.0, 2.0]),
                "iou": 1.0, "cover": 1.0,
                "true_start": 10 + 120 * i, "true_end": 100 + 120 * i,
                "is_tp": rng.random() < 0.5,
            })
    return pl.DataFrame(calls), pl.DataFrame(truth)


def test_fixture_is_unique_on_the_tiebreak():
    calls, _ = tied_calls()
    assert calls.select(cm.CALL_TIEBREAK).is_unique().all()


def test_sensitivity_to_first_fp_ignores_row_order():
    calls, truth = tied_calls()
    answers = {
        tuple(cm.sensitivity_to_first_fp(
            calls.sample(fraction=1.0, shuffle=True, seed=s), truth).items())
        for s in range(N_ORDERINGS)
    }
    assert len(answers) == 1, (
        f"{len(answers)} answers from {N_ORDERINGS} orderings of the same rows: "
        f"{sorted(answers)}"
    )


def test_sensitivity_to_first_fp_mean_is_bit_identical():
    """Equal to a tolerance is not enough. These columns are diffed between runs to find
    regressions, so a float that drifts in its last bits still shows up as a change."""
    calls, truth = tied_calls()
    means = {
        cm.sensitivity_to_first_fp(
            calls.sample(fraction=1.0, shuffle=True, seed=s), truth
        )["sens_first_fp_mean"].hex()
        for s in range(N_ORDERINGS)
    }
    assert len(means) == 1, f"the mean differs between orderings: {sorted(means)}"


def test_assign_instances_ignores_row_order():
    """Two calls tied on score and IoU, one annotation. Exactly one wins, and which one
    must not depend on which row polars emitted first."""
    calls = pl.DataFrame({
        "query_acc": ["A", "A"], "pfam_id": ["PF1", "PF1"],
        "qstart": [10, 12], "qend": [90, 92], "score": [1.0, 1.0],
        "iou": [1.0, 1.0], "cover": [1.0, 1.0],
        "true_start": [10, 10], "true_end": [90, 90],
    })
    patterns = {
        tuple(edc.assign_instances(
            calls, calls.sample(fraction=1.0, shuffle=True, seed=s), 0.5, "alignment"
        ).sort("qstart")["is_tp"].to_list())
        for s in range(N_ORDERINGS)
    }
    assert len(patterns) == 1, f"a different call wins depending on row order: {patterns}"
    assert sum(next(iter(patterns))) == 1, "one annotation, so exactly one true positive"


def test_cafa_scalars_report_the_lowest_threshold_reaching_the_optimum():
    """A plateau puts several thresholds at the same Fmax; the one reported is fixed."""
    curve = pl.DataFrame({
        "threshold": [0.1, 0.2, 0.3, 0.4],
        "f": [0.5, 0.9, 0.9, 0.4], "wf": [0.5, 0.8, 0.8, 0.4],
        "s": [3.0, 1.0, 1.0, 5.0], "pr": [0.5, 0.9, 0.9, 0.4],
        "rc": [0.5, 0.9, 0.9, 0.4], "ru": [1.0, 0.5, 0.5, 2.0],
        "mi": [1.0, 0.5, 0.5, 2.0],
    })
    for src in (curve, curve.reverse()):
        got = cm.cafa_scalars(src)
        assert got["fmax_threshold"] == 0.2
        assert got["smin_threshold"] == 0.2


# --- end to end, through the real script: the reproduction from the bug report ---

def _regions(pfam_truth, d: Path) -> Path:
    """One regions file, written once. Decoys are included so the calls are a mix of true,
    false and gray rather than the all-correct perfect set -- with no false positives there
    is no first false positive and the metric under test is trivially 1.0 everywhere."""
    d.mkdir(parents=True, exist_ok=True)
    out = d / "regions.tsv"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], out, pad=20)
    write_decoy_regions(pfam_truth["truth"], pfam_truth["map"], out)
    return out


def test_metrics_are_identical_across_runs(pfam_truth, tmp_path):
    """The reported reproduction: one input, scored twice from different directories."""
    regions = _regions(pfam_truth, tmp_path / "input")
    runs = [
        score(pfam_truth["truth"], pfam_truth["map"], regions, tmp_path / tag, "pfam")
        for tag in ("run_a", "run_b")
    ]
    first, second = runs
    assert first.columns == second.columns
    # Named rather than left to the frame comparison, so a failure says which of the two
    # regressed instead of only that something did.
    for col in ("sens_first_fp_mean", "sens_first_fp_median"):
        assert first[col].to_list() == second[col].to_list(), f"{col} differs between runs"
    assert first.equals(second), "a metric column differs between two identical runs"


def test_scored_calls_are_unique_on_the_call_key(pfam_truth, tmp_path):
    """The invariant CALL_TIEBREAK's totality rests on, checked against a real run rather
    than against the fixture that was built to satisfy it. Per arm, not pooled: the dedup
    arm scores the same calls again, so the two files legitimately share call keys."""
    regions = _regions(pfam_truth, tmp_path / "input")
    score(pfam_truth["truth"], pfam_truth["map"], regions, tmp_path / "run", "pfam")
    files = sorted((tmp_path / "run").glob("*.calls.parquet"))
    assert files, "no calls parquet written"
    for f in files:
        calls = pl.read_parquet(f)
        assert calls.height, f
        assert calls.select(
            "query_acc", "pfam_id", "qstart", "qend"
        ).is_unique().all(), f
