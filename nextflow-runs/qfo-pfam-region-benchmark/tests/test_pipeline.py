"""End-to-end test of everything downstream of search, on a 60-protein committed fixture.

Search itself is not exercised and cannot be: the baselines need multi-GB databases and the
structure arms need AlphaFold models. Every arm's output is a region table though, so the
scoring path can be driven with a synthetic one whose right answer is known in advance --
which is a stronger test than replaying a real tool, because a real tool's numbers only
tell you they changed, not which direction is correct.

The specific regressions guarded here all actually happened:
  * recall_reachable above 1.0 on instance-level strata (observed 2.11)
  * best_f1 coming back NaN when a threshold block was entirely gray
  * point features unscoreable by construction, making a short-feature deficit that was an
    artifact of IoU against a 1-residue interval rather than a result
"""

import json
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

from conftest import (BIN, FIXTURES, run, score, write_decoy_regions,
                      write_perfect_regions)

RATE_COLUMNS = ["recall_reachable", "recall", "precision", "coverage", "best_f1",
                "best_f1_precision", "best_f1_recall_reachable", "fmax", "auprc"]


# --------------------------------------------------------------------------
# truth builders
# --------------------------------------------------------------------------

def test_swissprot_parser_keeps_ranges_points_and_rejects_fuzzy(tmp_path):
    """Fuzzy endpoints and unwanted FT types must not reach the answer key."""
    run("build_swissprot_truth.py",
        "--sprot-dat", FIXTURES / "swissprot_edge_cases.dat",
        "--annotations", FIXTURES / "annotations_edge",
        "--truth-out", tmp_path / "t.parquet",
        "--map-outdir", tmp_path,
        "--summary-out", tmp_path / "s.json")
    # annotations_edge holds exactly these two accessions, so everything the parser kept
    # lands in the human truth. Read every parquet it wrote either way.
    rows = pl.concat(
        [pl.read_parquet(p) for p in tmp_path.glob("*.parquet")], how="diagonal_relaxed"
    ) if list(tmp_path.glob("*.parquet")) else pl.DataFrame()
    got = set(zip(rows["pfam_id"].to_list(), rows["is_point"].to_list())) if rows.height else set()

    assert ("TRANSMEM", False) in got, "range feature dropped"
    assert ("ACT_SITE", True) in got, "point feature dropped"
    assert ("BINDING", False) in got and ("BINDING", True) in got, \
        "BINDING occurs as both a range and a point and both must survive"
    # Fuzzy endpoints: an uncertain boundary cannot serve as boundary truth.
    assert "DOMAIN" not in rows["pfam_id"].to_list(), "DOMAIN <1..80 has a fuzzy start"
    assert "REGION" not in rows["pfam_id"].to_list(), "REGION 90..>150 has a fuzzy end"
    assert "MOTIF" not in rows["pfam_id"].to_list(), "MOTIF ?200..210 has a fuzzy start"
    # Types outside the requested vocabulary.
    for unwanted in ("STRAND", "HELIX", "VARIANT"):
        assert unwanted not in rows["pfam_id"].to_list()


def test_pfam_truth_has_split_and_no_point_column(pfam_truth):
    t = pl.read_parquet(pfam_truth["truth"])
    assert t.height > 0
    assert "split" in t.columns, "Pfam truth carries the selection/heldout split"
    assert "is_point" not in t.columns, "Pfam domains are intervals; there are no points"


# --------------------------------------------------------------------------
# a tool that is exactly right must score exactly 1.0
# --------------------------------------------------------------------------

@pytest.fixture(scope="session")
def perfect_pfam(pfam_truth, tmp_path_factory):
    d = tmp_path_factory.mktemp("perfect_pfam")
    n = write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], d / "regions.tsv")
    assert n > 0, "fixture produced no transferable regions"
    return score(pfam_truth["truth"], pfam_truth["map"], d / "regions.tsv",
                 d / "run", "pfam")


def _ungrouped(m: pl.DataFrame) -> dict:
    """The all-splits, all-strata row for the un-deduplicated arm.

    dedup_transfers is swept off/on, so this has to name which arm it wants; picking
    `.to_dicts()[0]` off an unfiltered frame silently reads whichever came first.
    """
    return m.filter((pl.col("split") == "all") & (pl.col("stratum_axis") == "all")
                    & (~pl.col("dedup_transfers"))).to_dicts()[0]


def test_perfect_tool_recovers_every_reachable_instance(perfect_pfam):
    row = _ungrouped(perfect_pfam)
    assert row["n_instances_found"] == row["n_reachable_instances"], \
        "a region placed exactly on every reachable instance must find all of them"
    assert row["recall_reachable"] == pytest.approx(1.0)
    assert row["median_iou_tp"] == pytest.approx(1.0)


def test_perfect_tools_only_errors_are_nested_transfers(perfect_pfam, perfect_pfam_calls):
    """Precision cannot reach 1.0 here, and that is correct behaviour rather than slack.

    Pfam domains nest. A region covering the target's outer domain also covers the inner
    ones, so the transfer step legitimately hands all of those families to one query
    interval and only the outer one is correctly placed. Human RNA Pol II RPB1 (P24928)
    is the case in this fixture: PF04998 spans 830-1427 and PF04990 / PF04992 sit inside
    it, so one region yields one true positive and two wrongly-placed calls.

    Asserting "precision > 0.99" would pass for the wrong reason the day something else
    broke. This asserts the REASON: every non-TP call shares its query interval with a
    call that was a true positive.
    """
    tp = perfect_pfam_calls.filter("is_tp").select("query_acc", "qstart", "qend").unique()
    fp = perfect_pfam_calls.filter(~pl.col("is_tp")).select(
        "query_acc", "qstart", "qend").unique()
    orphan = fp.join(tp, on=["query_acc", "qstart", "qend"], how="anti")
    assert orphan.height == 0, (
        f"{orphan.height} non-TP call intervals are not explained by a nested transfer:\n"
        f"{orphan}"
    )


def test_perfect_tool_places_every_true_positive_exactly(perfect_pfam):
    """Boundary error, not coverage.

    residue_recall is deliberately NOT asserted to be 1.0: it divides by every truth
    residue, and the fixture keeps instances whose family is absent from the target, which
    no transfer-based method can reach. Those belong in the denominator -- that is what
    makes recall honest -- so residue_recall is bounded by the reachable fraction, not by
    whether placement was right. (This used to name the duplicate ndo column, which was
    set to exactly this expression and has since been dropped.)
    """
    row = _ungrouped(perfect_pfam)
    assert row["dbd_median"] == pytest.approx(0.0)
    assert row["nterm_offset_median"] == pytest.approx(0.0)
    assert row["cterm_offset_median"] == pytest.approx(0.0)
    assert row["median_iou_tp"] == pytest.approx(1.0)
    assert row["n_reachable_instances"] < row["n_truth_instances"], \
        "the fixture must keep some unreachable instances, or recall is untested"


# --------------------------------------------------------------------------
# the regressions
# --------------------------------------------------------------------------

def test_no_rate_metric_exceeds_one_on_any_stratum(perfect_pfam):
    """Instance-level strata cut calls by protein but score by instance.

    Before subset() restricted the TP numerator to the cut, a call that correctly hit an
    instance OUTSIDE the stratum was counted inside it, against a denominator that never
    contained it. Observed: recall_reachable 2.11.
    """
    for col in RATE_COLUMNS:
        if col not in perfect_pfam.columns:
            continue
        s = perfect_pfam[col].drop_nulls()
        if s.len() == 0:
            continue
        assert s.max() <= 1.0 + 1e-9, (
            f"{col} reached {s.max()} -- a rate above 1.0 means the numerator and "
            "denominator describe different sets"
        )
        assert s.min() >= -1e-9, f"{col} went negative: {s.min()}"


def test_no_metric_is_nan(perfect_pfam):
    for col in RATE_COLUMNS:
        if col in perfect_pfam.columns:
            s = perfect_pfam[col].drop_nulls()
            assert int(s.is_nan().sum()) == 0, f"{col} contains NaN"


def test_no_metric_is_nan_when_the_top_block_is_all_gray(pfam_truth, tmp_path):
    """The condition that actually produced NaN, constructed on purpose.

    A leading threshold block of gray-only calls gives precision 0/0. polars sorts NaN as
    the largest float, so `sort("f1", descending=True).head(1)` handed that row back as
    best_f1. The perfect-tool fixture alone cannot catch this -- it emits no gray calls, so
    the branch is never reached and the guard would pass whether or not the fix is present.
    """
    regions = tmp_path / "regions.tsv"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], regions)
    n_decoy = write_decoy_regions(pfam_truth["truth"], pfam_truth["map"], regions)
    assert n_decoy > 0, "fixture produced no unannotated stretch to place a decoy in"

    m = score(pfam_truth["truth"], pfam_truth["map"], regions, tmp_path / "run", "pfam")
    assert m["n_gray_calls"].max() > 0, \
        "the decoys must land in unannotated territory, or this tests nothing"
    for col in RATE_COLUMNS:
        if col in m.columns:
            s = m[col].drop_nulls()
            assert int(s.is_nan().sum()) == 0, f"{col} contains NaN"
            assert s.max() <= 1.0 + 1e-9, f"{col} reached {s.max()}"


def test_scoring_is_deterministic_under_score_ties(swissprot_truth, tmp_path):
    """Identical inputs must give identical numbers, run to run.

    assign_instances is a greedy one-to-one walk, so which call claims an instance depends
    on row order -- and (score, is_point, elig) is not a unique sort key. Ties are common:
    HP alphabets at low ksize produce large blocks of identical region scores. polars does
    not promise a stable sort, so tied rows came back in different orders and the same arm
    scored differently between runs: five identical runs gave n_instances_found
    169, 169, 169, 169, 168.

    The synthetic arm makes every score identical, which is the worst case on purpose --
    a fixture with distinct scores would never revisit the tie path.
    """
    write_perfect_regions(swissprot_truth["truth"], swissprot_truth["map"],
                          tmp_path / "regions.tsv", pad=80)
    seen = set()
    for i in range(4):
        m = score(swissprot_truth["truth"], swissprot_truth["map"],
                  tmp_path / "regions.tsv", tmp_path / f"rep{i}", "swissprot")
        row = m.filter((pl.col("stratum_axis") == "all")
                       & (~pl.col("dedup_transfers"))).to_dicts()[0]
        seen.add((row["n_tp_calls"], row["n_instances_found"],
                  round(row["recall_reachable"], 12), round(row["best_f1"], 12)))
    assert len(seen) == 1, f"four identical runs produced {len(seen)} different results: {seen}"


def test_feature_length_bin_axis_is_populated(perfect_pfam):
    axis = perfect_pfam.filter(pl.col("stratum_axis") == "feature_length_bin")
    assert axis.height > 0, "the feature_length_bin axis produced no rows"
    assert axis["stratum"].n_unique() >= 3, \
        "the fixture must span several length bins or the axis proves nothing"
    assert axis["median_feature_length"].drop_nulls().len() == axis.height, \
        "every row needs the measured length the ratio is computed from"
    assert axis["n_stratum_proteins"].min() >= 1


def test_feature_type_axis_is_null_on_pfam_truth(perfect_pfam):
    """Pfam's pfam_id is a family accession with no type variation to cut on."""
    assert perfect_pfam.filter(pl.col("stratum_axis") == "feature_type").height == 0


def test_feature_type_axis_is_populated_on_swissprot(swissprot_truth, tmp_path):
    n = write_perfect_regions(swissprot_truth["truth"], swissprot_truth["map"],
                              tmp_path / "regions.tsv")
    assert n > 0
    m = score(swissprot_truth["truth"], swissprot_truth["map"],
              tmp_path / "regions.tsv", tmp_path / "run", "swissprot")
    axis = m.filter(pl.col("stratum_axis") == "feature_type")
    assert axis.height > 0, "feature_type produced no rows on the Swiss-Prot truth"
    types = set(axis["stratum"].to_list())
    assert {"DOMAIN", "REPEAT", "ZN_FING"} <= types, f"expected range types, got {types}"
    assert axis["point_fraction"].drop_nulls().len() == axis.height


def test_point_features_are_scoreable_by_containment_not_iou(swissprot_truth, tmp_path):
    """IoU against a 1-residue interval is 1/call_length, so at min-overlap 0.5 a point
    feature could never be a true positive. That is unsatisfiable, not strict, and it
    manufactured a short-feature deficit. Containment asks the answerable question."""
    # Padded to a realistic alignment width. An unpadded region on a 1-residue truth
    # interval is itself 1 residue and scores IoU 1.0 -- which no real tool could reach,
    # and which would make this test pass under both settings and prove nothing.
    n = write_perfect_regions(swissprot_truth["truth"], swissprot_truth["map"],
                              tmp_path / "regions.tsv", pad=80)
    assert n > 0

    def found_in_point_strata(*extra):
        # A directory per setting. Keying on len(extra) gave "--point-semantics iou" and
        # "--point-semantics cover" the same workdir, so the second run globbed a directory
        # the first had already written into -- results that depend on execution order.
        tag = "-".join(extra).replace("--", "") or "default"
        m = score(swissprot_truth["truth"], swissprot_truth["map"],
                  tmp_path / "regions.tsv", tmp_path / f"run_{tag}",
                  "swissprot", *extra)
        pts = m.filter((pl.col("stratum_axis") == "feature_type")
                       & (pl.col("point_fraction") > 0.5))
        return pts["n_instances_found"].sum() if pts.height else 0

    assert found_in_point_strata("--point-semantics", "iou") == 0, \
        "under IoU no point feature can be recovered -- that is the bug being guarded"
    cover = found_in_point_strata("--point-semantics", "cover")
    assert cover > 0, \
        "under containment a call covering the annotated residue must count"
    # The DEFAULT, with no flag. Asserted separately because the two explicit runs above
    # would both keep passing if the default flipped back to iou, while every production
    # number changed silently.
    assert found_in_point_strata() == cover, \
        "cover must be the default point-semantics, not merely an available option"


def test_boundary_metrics_exclude_point_features(swissprot_truth, tmp_path):
    write_perfect_regions(swissprot_truth["truth"], swissprot_truth["map"],
                          tmp_path / "regions.tsv")
    m = score(swissprot_truth["truth"], swissprot_truth["map"],
              tmp_path / "regions.tsv", tmp_path / "run", "swissprot")
    assert "n_point_instances_excluded" in m.columns
    pts = m.filter((pl.col("stratum_axis") == "feature_type")
                   & (pl.col("point_fraction") > 0.5))
    if pts.height:
        assert pts["n_point_instances_excluded"].max() > 0, \
            "a point-dominated stratum must exclude its points from boundary metrics"


# --------------------------------------------------------------------------
# aggregation and the report
# --------------------------------------------------------------------------

def test_aggregate_and_multiqc_sections_build(pfam_truth, swissprot_truth, tmp_path):
    work = tmp_path / "agg"
    (work / "metrics").mkdir(parents=True)
    (work / "curves").mkdir(parents=True)
    # Presented as a kmerseek variant: the alphabet and ceiling panels draw kmerseek rows
    # only, and parse the variant as <alphabet>_k<k>_lc<True|False>. An arm called
    # "perfect/default" produces a valid metrics table that those sections correctly skip,
    # which would make this test assert the panels are missing rather than that they work.
    for name, tr in (("pfam", pfam_truth), ("swissprot", swissprot_truth)):
        d = tmp_path / f"score_{name}"
        write_perfect_regions(tr["truth"], tr["map"], tmp_path / f"{name}.tsv")
        score(tr["truth"], tr["map"], tmp_path / f"{name}.tsv", d, name,
              tool="kmerseek", variant="hp_pbotc_1st_ed2_k19_lcFalse")
        for f in d.glob("*.metrics.parquet"):
            f.rename(work / "metrics" / f.name)
        for f in d.glob("*.curve.parquet"):
            f.rename(work / "curves" / f.name)

    run("aggregate_domain_metrics.py", work / "metrics", work / "curves",
        work / "all.parquet", work / "all.csv", work / "curves.parquet")
    agg = pl.read_parquet(work / "all.parquet")
    assert agg.height > 0
    assert set(agg["truth_set"].unique().to_list()) == {"pfam", "swissprot"}

    out = work / "mqc"
    run("build_multiqc_inputs.py",
        "--metrics", work / "all.parquet", "--curves", work / "curves.parquet",
        "--n-queries", 60, "--outdir", out)
    written = {p.name for p in out.glob("*_mqc.json")}
    assert written, "no MultiQC sections written"
    assert any("ceiling_length" in n for n in written), \
        "the feature-length panel is missing from the report"
    assert any("ceiling_feature_type" in n for n in written), \
        "the feature-type panel is missing from the report"
    for p in out.glob("*_mqc.json"):
        json.loads(p.read_text())   # every section must be valid JSON MultiQC can read


def test_bpe_diagnostic_self_test():
    """The hand-written BPE must agree with a reference implementation.

    It exists because pulling in HuggingFace tokenizers for a 4739-line merge table would
    have meant a new container for a script that otherwise needs only stdlib.
    """
    # Where `make bpe-tokenizer` puts it. resolve_tokenizer() accepts the tarball as well
    # as an unpacked directory, so the test does not care which form is on disk.
    tarball = BIN.parent / "data/protberta/ProtBERTa_tokenizers.tar.gz"
    if not tarball.exists():
        pytest.skip(f"no tokenizer at {tarball}; run `make bpe-tokenizer`")
    proc = subprocess.run(
        [sys.executable, str(BIN / "hp_bpe_boundary_diagnostic.py"),
         "--self-test", "--tokenizer", str(tarball)],
        capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "0 disagree" in proc.stdout, proc.stdout[-2000:]
