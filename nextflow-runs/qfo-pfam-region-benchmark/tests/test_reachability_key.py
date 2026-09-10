"""The reachability denominator has to be a property of the TARGET, not a constant.

`recall_reachable` divides by the number of human annotations the target proteome could
actually have supplied, so that a tool is not charged for missing something the target does
not contain. On the midi run it returned exactly 7_000 for eight of nine species and 6_991
for the ninth, which is not a denominator, it is the truth-set size with a label on it.

The cause was the join key. Reachability joined truth to target on `pfam_id`, which is
right on the Pfam truth sets, where `pfam_id` is a family and "the target has family F"
varies by species. On the Swiss-Prot truth set `pfam_id` is one of twelve curated FEATURE
TYPES, every proteome carries nearly all twelve, so the join matched everything. The 6_991
was ciona lacking INTRAMEM and nothing else: 7_000 minus the 9 human INTRAMEM features.

These tests fix the three things that would let it regress:

  * the Swiss-Prot denominator must respond to which target it is asked about;
  * the Pfam denominator must not move at all, because notebooks 220-226 and every
    published family-level number depend on that join staying exactly as it was;
  * the numerator must be counted over the same instances as the denominator, or
    recall_reachable exceeds 1.0 and auprc integrates a curve that runs off the top.
"""
import sys
from pathlib import Path

import polars as pl
import pytest

from conftest import FIXTURES, run, score, write_perfect_regions

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import build_swissprot_truth as sprot  # noqa: E402
import evaluate_domain_calls as edc  # noqa: E402


def thin_target(full: pl.DataFrame, n: int) -> pl.DataFrame:
    """A target proteome carrying only its n most common Pfam families.

    Stands in for what curation depth does in the real data: ciona has 23 annotated
    proteins against mouse's 15_634, so almost no human query has an annotated relative
    there to transfer from.
    """
    keep = (full.explode(sprot.ANCHOR_COL)
                .group_by(sprot.ANCHOR_COL).len()
                .sort("len", descending=True)
                .head(n)[sprot.ANCHOR_COL].to_list())
    return full.filter(
        pl.col(sprot.ANCHOR_COL).list.eval(pl.element().is_in(keep)).list.any()
    )


def test_the_swissprot_truth_carries_the_anchor_column(swissprot_truth):
    """Both sides need it: the human query's families and the target protein's."""
    truth = pl.read_parquet(swissprot_truth["truth"])
    tmap = pl.read_parquet(swissprot_truth["map"])
    assert sprot.ANCHOR_COL in truth.columns
    assert sprot.ANCHOR_COL in tmap.columns
    assert truth[sprot.ANCHOR_COL].list.len().min() >= 1, "an empty anchor reaches nothing"


def test_a_feature_type_join_matches_everything(swissprot_truth):
    """The defect itself, kept as a test so the old key cannot quietly come back.

    Twelve labels in the whole vocabulary means the old join was very nearly the identity,
    which is what made the denominator constant across species.
    """
    truth = pl.read_parquet(swissprot_truth["truth"])
    tmap = pl.read_parquet(swissprot_truth["map"])
    old = truth.join(tmap.select("pfam_id").unique(), on="pfam_id", how="inner")
    assert truth["pfam_id"].n_unique() < 50, "fixture no longer has a category vocabulary"
    assert old.height >= 0.99 * truth.height


def test_the_swissprot_denominator_responds_to_the_target(swissprot_truth):
    """The fix. A thinner target proteome must reach strictly fewer human instances."""
    truth = pl.read_parquet(swissprot_truth["truth"])
    full = pl.read_parquet(swissprot_truth["map"])
    fams = full.select("pfam_id").unique()

    rich = edc.reachable_instances(truth, fams, sprot.anchor_pairs(full))
    thin = thin_target(full, 1)
    poor = edc.reachable_instances(truth, fams, sprot.anchor_pairs(thin))

    assert poor.height < rich.height, (
        "a proteome with one annotated family reaches as much as the full one; "
        "the denominator is not reading the target"
    )
    assert rich.height <= truth.height


def test_the_pfam_family_join_is_untouched(pfam_truth):
    """Notebooks 220-226 and every family-level number depend on this join not moving.

    The Pfam truth set has no anchor column, so reachable_instances must fall through to
    exactly the join it always did -- asserted against that join written out here, not
    against a recorded number.
    """
    truth = pl.read_parquet(pfam_truth["truth"])
    tmap = pl.read_parquet(pfam_truth["map"])
    fams = tmap.select("pfam_id").unique()

    assert sprot.ANCHOR_COL not in truth.columns
    was = truth.join(fams, on="pfam_id", how="inner")
    for anchor in (None, sprot.anchor_pairs(
            tmap.with_columns(pl.col("pfam_id").reshape((-1, 1)).alias(sprot.ANCHOR_COL)))):
        # Even handed an anchor table, a truth set without the column keeps the family
        # join. The dispatch reads the data on BOTH sides, not just the target.
        now = edc.reachable_instances(truth, fams, anchor)
        assert now.height == was.height
        assert now.sort(truth.columns).equals(was.sort(truth.columns))


def test_a_family_vocabulary_still_excludes_absent_families(pfam_truth):
    """The Pfam join is meant to bite, and does. Guards against 'fixing' it into a no-op."""
    truth = pl.read_parquet(pfam_truth["truth"])
    tmap = pl.read_parquet(pfam_truth["map"])
    one = tmap.select("pfam_id").unique().head(1)
    assert edc.reachable_instances(truth, one, None).height < truth.height


@pytest.mark.parametrize("truth_set", ["pfam", "swissprot"])
def test_recall_reachable_never_exceeds_one(request, tmp_path, truth_set):
    """The numerator has to be restricted to the instances the denominator counts.

    A perfect tool finds every instance, including any the reachability rule rules out. If
    the numerator is not cut to the same set, recall_reachable goes above 1.0 -- it was
    observed at 2.77 -- and auprc integrates that curve, so best_f1 goes with it.
    """
    fx = request.getfixturevalue(f"{truth_set}_truth" if truth_set == "pfam"
                                 else "swissprot_truth")
    regions = tmp_path / "regions.tsv"
    write_perfect_regions(fx["truth"], fx["map"], regions)
    m = score(fx["truth"], fx["map"], regions, tmp_path / "run", truth_set)

    for col in ("recall_reachable", "best_f1_recall_reachable"):
        worst = m[col].max()
        assert worst is None or worst <= 1.0 + 1e-9, f"{col} reached {worst}"
    over = m.filter(pl.col("n_instances_found_reachable")
                    > pl.col("n_reachable_instances"))
    assert over.height == 0, "found more reachable instances than exist"


def test_the_summary_reports_coverage_and_the_new_denominator(tmp_path):
    """Curation depth is a property of the annotation database, not of the organism.

    Ciona has 28 reviewed Swiss-Prot entries in total against mouse's tens of thousands. It
    is not excluded for that -- the denominator is corrected so its number is honest -- but
    the coverage has to be on the record, or a thin species reads as an evolutionary
    result.
    """
    import json
    run("build_swissprot_truth.py",
        "--sprot-dat", FIXTURES / "uniprot_sprot_fixture.dat",
        "--annotations", FIXTURES / "annotations",
        "--truth-out", tmp_path / "truth.parquet",
        "--map-outdir", tmp_path,
        "--summary-out", tmp_path / "summary.json")
    s = json.loads((tmp_path / "summary.json").read_text())

    assert 0.0 <= s["human"]["coverage_fraction"] <= 1.0
    yeast = s["yeast"]
    assert 0.0 <= yeast["coverage_fraction"] <= 1.0
    assert yeast["coverage_fraction"] == pytest.approx(
        yeast["n_proteins"] / yeast["n_annotated_proteins"])
    assert 0 <= yeast["n_reachable_human_features"] <= s["human"]["n_features"]
