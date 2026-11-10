"""transfer_domains: where a target Pfam domain lands on the query.

The tandem-repeat case that motivated `projected` (2026-11-10). A C2H2 zinc-finger protein
has fingers of about 23 residues in a row. An alignment that covers six of them on the
target used to give six calls of PF00096, each spanning the whole 140-residue alignment on
the query, so none passed the IoU >= 0.5 test against a single 23-residue finger.

Coordinates are 0-based and end-exclusive, as in the truth and the domain maps.

  target T1, six fingers of 23 residues, starting at 200, 223, 246, 269, 292, 315
  alignment  query Q1 [50, 190)  <->  target T1 [195, 335)   offset 50 - 195 = -145

  span       six calls, each Q1 [50, 190)
  projected  Q1 [55, 78), [78, 101), [101, 124), [124, 147), [147, 170), [170, 190)
             the last finger, T1 [315, 338), sticks out 3 residues past the alignment's
             target end (335), so it is cut to the aligned stretch: 20 residues
"""

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import evaluate_domain_calls as edc  # noqa: E402
from conftest import score, write_perfect_regions  # noqa: E402

FINGER = 23
FINGER_STARTS = [200 + i * FINGER for i in range(6)]
QSTART, QEND, TSTART, TEND = 50, 190, 195, 335


def tandem_inputs():
    regions = pl.LazyFrame({
        "query_acc": ["Q1"], "target_acc": ["T1"],
        "qstart": [QSTART], "qend": [QEND], "tstart": [TSTART], "tend": [TEND],
        "score": [42.0],
    })
    domain_map = pl.LazyFrame({
        "accession": ["T1"] * 6 + ["T1"],
        "pfam_id": ["PF00096"] * 6 + ["PF00000"],
        "domain_start": FINGER_STARTS + [400],
        "domain_end": [s + FINGER for s in FINGER_STARTS] + [450],
    })
    return regions, domain_map


def test_span_gives_every_finger_the_whole_alignment():
    calls = edc.transfer_domains(*tandem_inputs(), min_overlap=0.5, mode="span").collect()
    assert calls.height == 6
    assert calls["qstart"].unique().to_list() == [QSTART]
    assert calls["qend"].unique().to_list() == [QEND]


def test_projected_gives_six_separate_finger_calls():
    calls = (edc.transfer_domains(*tandem_inputs(), min_overlap=0.5, mode="projected")
             .collect().sort("qstart"))
    offset = QSTART - TSTART
    expected = [(s + offset, min(s + FINGER + offset, QEND)) for s in FINGER_STARTS]
    assert list(zip(calls["qstart"], calls["qend"])) == expected
    lengths = (calls["qend"] - calls["qstart"]).to_list()
    assert lengths == [23, 23, 23, 23, 23, 20]
    # One call per finger, none overlapping the next.
    assert (calls["qstart"].shift(-1).drop_nulls() >= calls["qend"].head(5)).all()
    # PF00000 sits outside the alignment on the target, so neither mode keeps it.
    assert calls["pfam_id"].unique().to_list() == ["PF00096"]


def test_projected_keeps_the_same_domains_as_span():
    # Same keep rule in both modes: target overlap >= min_overlap of the domain length.
    # A finger half outside the alignment is kept or dropped the same way in both.
    regions, domain_map = tandem_inputs()
    regions = regions.with_columns(pl.lit(FINGER_STARTS[2] + 12).alias("tend"),
                                   pl.lit(QSTART + FINGER_STARTS[2] + 12 - TSTART)
                                   .alias("qend"))
    for min_overlap in (0.3, 0.5, 0.6):
        n = {m: edc.transfer_domains(regions, domain_map, min_overlap, mode=m)
             .collect().height for m in edc.TRANSFER_MODES}
        assert n["span"] == n["projected"], (min_overlap, n)
    # 12 of 23 residues of the third finger are aligned: kept at 0.5, dropped at 0.6.
    assert edc.transfer_domains(regions, domain_map, 0.5, "projected").collect().height == 3
    assert edc.transfer_domains(regions, domain_map, 0.6, "projected").collect().height == 2


def test_projected_drops_a_domain_cut_to_nothing():
    # Query side 10 residues long, target side 100: a domain at the far end of the target
    # shifts past qend and has no residues left on the query.
    regions = pl.LazyFrame({"query_acc": ["Q1"], "target_acc": ["T1"],
                            "qstart": [0], "qend": [10], "tstart": [0], "tend": [100],
                            "score": [1.0]})
    domain_map = pl.LazyFrame({"accession": ["T1", "T1"], "pfam_id": ["PF1", "PF2"],
                               "domain_start": [0, 60], "domain_end": [8, 90]})
    calls = edc.transfer_domains(regions, domain_map, 0.5, "projected").collect()
    assert calls["pfam_id"].to_list() == ["PF1"]
    assert (calls["qstart"][0], calls["qend"][0]) == (0, 8)


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="transfer mode"):
        edc.transfer_domains(*tandem_inputs(), min_overlap=0.5, mode="gapped")


def test_cli_stamps_transfer_mode_on_every_metrics_row(pfam_truth, tmp_path):
    regions = tmp_path / "regions.tsv"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], regions)
    rows = {}
    for mode in edc.TRANSFER_MODES:
        m = score(pfam_truth["truth"], pfam_truth["map"], regions, tmp_path / mode,
                  "pfam", "--transfer-mode", mode)
        assert m["transfer_mode"].unique().to_list() == [mode]
        rows[mode] = m
    # Default is span, so a run that does not pass the flag reproduces older scores.
    default = score(pfam_truth["truth"], pfam_truth["map"], regions, tmp_path / "default",
                    "pfam")
    assert default["transfer_mode"].unique().to_list() == ["span"]
    key = ["split", "stratum_axis", "stratum", "dedup_transfers"]
    cols = [c for c in ("fmax", "recall_reachable", "precision") if c in default.columns]
    a = default.sort(key).select(key + cols)
    b = rows["span"].sort(key).select(key + cols)
    assert a.equals(b)
