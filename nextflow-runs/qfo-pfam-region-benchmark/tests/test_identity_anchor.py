"""A Swiss-Prot feature has no identity of its own, so it takes the domain's it sits in.

The identity table is measured over Pfam domain instances: extract_domain_sequences emits
one record per Pfam domain and parse_domain_identity keys the result on (accession,
pfam_id, domain_start, domain_end). attach_identity joined the truth to it on that key,
which is right for the pfam and pfamn truth sets and matches nothing at all on Swiss-Prot,
where pfam_id holds a curated feature type from a six-value vocabulary. Every instance
landed in no_homolog, so the truth set the leaderboard selects on had no identity axis and
the twilight-zone panel had to be drawn on Pfam instead.

The pfam arm must not move. These tests hold both halves: the exact join is still the exact
join wherever it can match, and the anchored join only fires where it cannot.
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
import evaluate_domain_calls as ev  # noqa: E402


def ident(rows) -> pl.DataFrame:
    """Identity table rows: (accession, pfam_id, start, end, pident, target)."""
    return pl.DataFrame(
        [dict(zip(["accession", "pfam_id", "domain_start", "domain_end",
                   "best_pident", "best_target"], r)) for r in rows],
        schema={"accession": pl.String, "pfam_id": pl.String,
                "domain_start": pl.Int64, "domain_end": pl.Int64,
                "best_pident": pl.Float64, "best_target": pl.String})


def truth(rows) -> pl.DataFrame:
    return pl.DataFrame(
        [dict(zip(["accession", "pfam_id", "domain_start", "domain_end"], r))
         for r in rows],
        schema={"accession": pl.String, "pfam_id": pl.String,
                "domain_start": pl.Int64, "domain_end": pl.Int64})


# One protein, two Pfam domains, measured against some target proteome.
IDENT = ident([
    ("P1", "PF00001", 10, 110, 82.0, "T1"),
    ("P1", "PF00002", 200, 300, 24.0, "T2"),
    ("P2", "PF00003", 10, 110, 55.0, "T3"),
])


def bins(t, i=IDENT) -> dict:
    out = ev.attach_identity(t, i)
    return {r["pfam_id"] + str(r["domain_start"]): r["stratum_identity"]
            for r in out.to_dicts()}


# --- the pfam arm does not move ------------------------------------------------------

def test_a_pfam_instance_reads_its_own_measured_identity():
    assert bins(truth([("P1", "PF00001", 10, 110)])) == {"PF0000110": "60-100%"}


def test_a_pfam_instance_with_no_target_match_stays_no_homolog():
    # PF00009 is a Pfam accession, so the frame keys exactly and there is no fallback. It
    # must NOT inherit PF00001's 82% just because the two intervals overlap -- that would
    # move numbers on the truth set every published Pfam figure is drawn from.
    t = truth([("P1", "PF00001", 10, 110), ("P1", "PF00009", 20, 100)])
    assert bins(t) == {"PF0000110": "60-100%", "PF0000920": "no_homolog"}


# --- the Swiss-Prot arm gets an axis at all ------------------------------------------

def test_a_feature_inside_a_domain_takes_that_domains_identity():
    assert bins(truth([("P1", "BINDING", 50, 51)])) == {"BINDING50": "60-100%"}


def test_two_features_in_different_domains_of_one_protein_differ():
    t = truth([("P1", "BINDING", 50, 51), ("P1", "SITE", 250, 251)])
    assert bins(t) == {"BINDING50": "60-100%", "SITE250": "20-30%"}


def test_a_feature_in_no_measured_domain_is_no_homolog():
    assert bins(truth([("P1", "REGION", 400, 450)])) == {"REGION400": "no_homolog"}


def test_a_protein_absent_from_the_identity_table_is_no_homolog():
    assert bins(truth([("P9", "REGION", 10, 60)])) == {"REGION10": "no_homolog"}


def test_a_feature_clipping_a_domains_edge_does_not_inherit_it():
    # 100 residues, 10 of them inside PF00001. A transmembrane helix that ends where a
    # kinase domain begins is not in that kinase domain, and reporting 82% for it would
    # put a conserved core on the identity axis that this feature is not part of.
    assert bins(truth([("P1", "TRANSMEM", 100, 200)])) == {"TRANSMEM100": "no_homolog"}


def test_the_half_covered_feature_is_the_boundary_and_it_is_inclusive():
    # PF00001 ends at 110. A 100-residue feature starting at 60 has exactly half of itself
    # inside it and anchors; one residue further out and it does not.
    assert bins(truth([("P1", "TRANSMEM", 60, 160)])) == {"TRANSMEM60": "60-100%"}
    assert bins(truth([("P1", "TRANSMEM", 61, 161)])) == {"TRANSMEM61": "no_homolog"}


def test_a_feature_straddling_two_domains_takes_the_one_it_is_mostly_in():
    close = ident([("P3", "PF00001", 0, 100, 82.0, "T1"),
                   ("P3", "PF00002", 100, 200, 24.0, "T2")])
    # 60 residues in PF00002 against 40 in PF00001.
    assert bins(truth([("P3", "REGION", 60, 160)]), close) == {"REGION60": "20-30%"}


def test_the_anchor_is_the_same_one_whatever_order_the_rows_arrive_in():
    # Two domains covering the feature equally. Overlap alone does not break the tie, so
    # without a total sort key the pick falls to polars' row order and the bin moves
    # between identical runs.
    tie = ident([("P4", "PF00001", 0, 100, 82.0, "T1"),
                 ("P4", "PF00002", 0, 100, 24.0, "T2")])
    t = truth([("P4", "REGION", 20, 80)])
    assert bins(t, tie) == bins(t, tie.reverse()) == {"REGION20": "60-100%"}


# --- the degenerate inputs the pipeline actually passes ------------------------------

def test_no_identity_table_leaves_the_axis_null():
    out = ev.attach_identity(truth([("P1", "BINDING", 50, 51)]), None)
    assert out["stratum_identity"].to_list() == [None]


def test_an_empty_identity_table_leaves_the_axis_null():
    out = ev.attach_identity(truth([("P1", "BINDING", 50, 51)]), IDENT.head(0))
    assert out["stratum_identity"].to_list() == [None]


def test_an_identity_table_with_no_overlapping_row_keeps_every_column():
    # The no-candidate path builds the missing columns itself rather than joining. It has
    # to produce the same frame shape as the joining path or score_one loses best_target.
    out = ev.attach_identity(truth([("P9", "REGION", 10, 60)]), IDENT)
    assert {"best_pident", "best_target", "stratum_identity"} <= set(out.columns)
    assert out["best_target"].to_list() == [None]
