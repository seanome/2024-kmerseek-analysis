"""Tests for notebooks/isoform_blank_exons_359.py, on hand-built transcripts whose answer
is worked out in the comment beside each case."""

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import isoform_blank_exons_359 as ib  # noqa: E402

def test_codon_split_goes_to_exon_with_two_bases_plus_strand():
    # transcript: 2 UTR bases, then ATG GCT TGG AAA GAA TAA
    # exon 1 = 6 bases "xxATGG" -> holds A,T,G of codon 0 and G of codon 1 (1 of 3)
    # exon 2 = 5 bases "CTTGG" -> holds C,T of codon 1 (2 of 3), and codon 2 TGG
    # exon 3 = rest     "AAAGAATAA"
    exons = [ib.Exon(101, 106), ib.Exon(201, 205), ib.Exon(301, 309)]
    segs = ib.exon_protein_intervals(exons, "+", cds_start=3, n_codons=5)
    assert segs == [(0, 1), (1, 3), (3, 5)]


def test_codon_split_minus_strand_uses_transcript_order():
    # Same transcript on the minus strand: exon genomic order is reversed, so the exon
    # with the highest coordinates is the first in the transcript.
    exons = [ib.Exon(1000 - e.end, 1000 - e.start) for e in [ib.Exon(101, 106), ib.Exon(201, 205), ib.Exon(301, 309)]]
    segs = ib.exon_protein_intervals(exons, "-", cds_start=3, n_codons=5)
    assert segs == [(0, 1), (1, 3), (3, 5)]


def test_codon_with_two_bases_in_first_exon():
    # exon 1 = "ATGGC" holds codon 0 and G,C of codon 1 -> codon 1 goes to exon 1
    exons = [ib.Exon(1, 5), ib.Exon(11, 20)]
    segs = ib.exon_protein_intervals(exons, "+", cds_start=1, n_codons=5)
    assert segs == [(0, 2), (2, 5)]


def test_exon_with_no_majority_codon_is_none():
    # exon 2 is a single base between two exons: it holds 1 base of one codon only
    exons = [ib.Exon(1, 4), ib.Exon(11, 11), ib.Exon(21, 30)]
    segs = ib.exon_protein_intervals(exons, "+", cds_start=1, n_codons=5)
    assert segs[1] is None


def test_protein_to_genomic_round_trip_minus():
    exons = [ib.Exon(101, 106), ib.Exon(201, 205), ib.Exon(301, 309)]
    # minus strand: transcript order is 301-309, 201-205, 101-106
    pieces = ib.protein_to_genomic((0, 2), exons, "-", cds_start=1)
    # codons 0-1 = transcript bases 0..5 -> 309 down to 304
    assert pieces == [(304, 309)]
    pieces = ib.protein_to_genomic((2, 4), exons, "-", cds_start=1)
    # bases 6..11 -> 303..301 (3 bases), then 205..203 (3 bases)
    assert pieces == [(203, 205), (301, 303)]


def test_landing_is_fraction_of_exon_not_call():
    seg = (100, 120)  # 20 residues
    assert ib.lands((0, 500), seg)  # a whole-protein call covers all of it
    assert ib.lands((110, 130), seg)  # covers 10 of 20 = 50%
    assert not ib.lands((111, 130), seg)  # 9 of 20
    assert not ib.lands((0, 0), (5, 5))  # empty segment never lands


def test_shuffle_keeps_outside_and_composition():
    p = "MAAAAKKKKEEEEWWWW"
    s = ib.shuffle_segment(p, (5, 13), random.Random(1))
    assert len(s) == len(p)
    assert s[:5] == p[:5] and s[13:] == p[13:]
    assert sorted(s[5:13]) == sorted(p[5:13])


def test_threshold_at_decoy_rate_cap_and_ties():
    n = 300  # cap = 3 decoy exons at 1%
    decoy = {"a": 1e-9, "b": 1e-6, "c": 1e-3, "d": 1e-3, "e": 5.0}
    t = ib.threshold_at_decoy_rate(decoy, n)
    # the 4th smallest is 1e-3; it and its tie "c" must go, leaving 2 decoy exons
    assert len(ib.landed_at(decoy, t)) == 2
    assert t < 1e-3 and t >= 1e-6
    assert ib.threshold_at_decoy_rate({"a": 1.0}, n) == math.inf
    assert ib.landed_at({"a": 1.0}, ib.threshold_at_decoy_rate({"a": 1.0}, 50)) == set()


def test_union_thresholds_hold_the_union_cap():
    n = 200  # cap = 2
    d = {"s1": {"x": 1e-5, "y": 1e-2}, "s2": {"z": 1e-5, "x": 1e-4}}
    ts, q = ib.union_thresholds(d, n)
    union = ib.landed_at(d["s1"], ts["s1"]) | ib.landed_at(d["s2"], ts["s2"])
    assert len(union) <= 2


def test_scan_calls_match_notebook_231_function():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
    import swissprot_control_utils as su

    rng = random.Random(3)
    seq = "".join(rng.choice("AILVFKEDRSTGP") for _ in range(400))
    ref = su.kd_transmem_calls({"x": seq}, 19, 1.6)
    mine = ib.scan_calls(ib.window_means(seq), 1.6)
    # notebook 231's table is 1-based inclusive; mine is 0-based end-exclusive
    assert [(r["qstart"] - 1, r["qend"]) for r in ref.to_dicts()] == mine


def test_scan_best_landing_is_highest_landing_threshold():
    rng = random.Random(5)
    seq = "".join(rng.choice("KEDRSTGPN") for _ in range(100)) + "LIVLIVLIVLIVLIVLIVLIVL" + "KEDRSTGPN" * 10
    means = ib.window_means(seq)
    seg = (100, 122)
    score = ib.scan_best_landing(means, seg)
    t = -score
    assert any(ib.lands(c, seg) for c in ib.scan_calls(means, t))
    assert not any(ib.lands(c, seg) for c in ib.scan_calls(means, math.nextafter(t, math.inf) + 1e-9))


def test_region_name_and_named_rule():
    r = ib.parse_region_name("A0A017SE85|REGION|Disordered|50-153")
    assert r["feature_type"] == "REGION" and r["start"] == 50 and r["end"] == 153
    assert not ib.is_named(r["feature_type"], r["description"])
    r = ib.parse_region_name("A0A009IHW8|COILED|none|31-99")
    assert r["description"] == "" and not ib.is_named(r["feature_type"], r["description"])
    r = ib.parse_region_name("A0A017SE85|COMPBIAS|Low_complexity|80-102")
    assert r["description"] == "Low complexity" and ib.is_named(r["feature_type"], r["description"])
