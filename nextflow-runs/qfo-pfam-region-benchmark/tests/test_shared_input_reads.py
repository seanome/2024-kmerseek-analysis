"""scoreDomainCalls must read the two shared frames once per task, not once per arm.

This is the regression behind the pipeline's largest I/O item. On the 2026-08-31 midi
trace scoreDomainCalls read 906.1 GB across 756 tasks -- more than folddiscoQuery's
409.5 GB and more than every search process put together -- while `rchar` over the same
tasks was 19.5 GB. The 46x gap is the signature: polars memory-maps a `scan_parquet`, a
scan is re-executed on every collect, and re-collecting a mapped file whose pages the
task's own working set keeps evicting re-faults them in from the block device, where they
count in read_bytes and never in rchar. Read volume fit 0.073 GB x arms + 0.114 GB, so
820 of the 906 GB scaled with the ARM COUNT rather than the task count.

The fix is read_shared_inputs: collect both frames once, hand back in-memory lazy views.
The test for "in memory" that cannot pass by accident is deleting the source files and
checking the frames still resolve -- a `scan_parquet` handle cannot.
"""

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bin"))

import evaluate_domain_calls as edc  # noqa: E402


class _Args:
    def __init__(self, truth, domain_map):
        self.truth = truth
        self.domain_map = domain_map


def test_shared_inputs_survive_deleting_the_files_they_came_from(pfam_truth, tmp_path):
    """A frame that still resolves after its file is gone was never a live scan."""
    truth_copy = tmp_path / "truth.parquet"
    map_copy = tmp_path / "map.parquet"
    pl.read_parquet(pfam_truth["truth"]).write_parquet(truth_copy)
    pl.read_parquet(pfam_truth["map"]).write_parquet(map_copy)

    truth_lf, map_lf = edc.read_shared_inputs(_Args(truth_copy, map_copy))
    n_truth = truth_lf.collect().height
    n_map = map_lf.collect().height
    assert n_truth > 0 and n_map > 0, "fixture produced empty frames"

    truth_copy.unlink()
    map_copy.unlink()

    # Both must still collect, and to the same thing. A scan_parquet handle raises here.
    assert truth_lf.collect().height == n_truth
    assert map_lf.collect().height == n_map
    # Repeated collects are what a batched task does once per (arm x dedup mode).
    for _ in range(3):
        assert truth_lf.collect().height == n_truth
        assert map_lf.collect().height == n_map


def test_domain_map_is_optional(pfam_truth):
    """--direct-annotation passes no map, and the reader must not invent one."""
    truth_lf, map_lf = edc.read_shared_inputs(_Args(pfam_truth["truth"], None))
    assert map_lf is None
    assert truth_lf.collect().height > 0


def test_score_one_still_falls_back_to_scanning_when_given_no_map(pfam_truth):
    """The keyword defaults to None so any caller predating the change still works."""
    import inspect
    sig = inspect.signature(edc.score_one)
    assert sig.parameters["domain_map_lf"].default is None


def _run_manifest(truth, domain_map, regions, workdir, n_arms):
    """Score n_arms identical arms in ONE task, the shape scoreDomainCalls batches into."""
    workdir.mkdir(parents=True, exist_ok=True)
    manifest = workdir / "manifest.tsv"
    manifest.write_text(
        "".join(f"perfect\tarm{i}\t{regions}\n" for i in range(n_arms)))
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parents[1] / "bin"
                             / "evaluate_domain_calls.py"),
         "--manifest", str(manifest), "--species", "yeast", "--species-mya", "1000",
         "--truth", str(truth), "--domain-map", str(domain_map),
         "--truth-set", "pfam"],
        cwd=workdir, capture_output=True, text=True, check=True)
    files = sorted(workdir.glob("*.metrics.parquet"))
    assert files, f"no metrics written in {workdir}"
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


@pytest.mark.parametrize("n_arms", [1, 8])
def test_every_arm_in_a_batched_task_scores_the_same_as_a_lone_arm(
        pfam_truth, tmp_path, n_arms):
    """Sharing one collected map across arms must not move a single scored number.

    The point of the fix is that it is invisible in the output: the same bytes were being
    read many times over, and now they are read once. Identical arms in one task must
    therefore produce identical metric rows, and they must match the single-arm run.
    """
    from conftest import write_perfect_regions

    regions = tmp_path / "regions.tsv.gz"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], regions)

    lone = _run_manifest(pfam_truth["truth"], pfam_truth["map"], regions,
                         tmp_path / "lone", 1)
    batched = _run_manifest(pfam_truth["truth"], pfam_truth["map"], regions,
                            tmp_path / f"batched{n_arms}", n_arms)

    # Two dedup modes per arm, so the row count is arms x modes x whatever splits exist.
    assert batched.height == lone.height * n_arms

    cols = [c for c in ("recall_reachable", "precision", "best_f1", "fmax")
            if c in lone.columns]
    assert cols, "fixture produced no rate columns to compare"
    for c in cols:
        assert sorted(batched[c].drop_nulls().to_list()) == sorted(
            lone[c].drop_nulls().to_list() * n_arms), f"{c} moved under batching"
