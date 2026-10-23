"""Tests for scripts/ice_binding_sw_benchmark.py (notebook 349)."""

import importlib.util
import math
import random
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "ice_binding_sw_benchmark", REPO / "scripts" / "ice_binding_sw_benchmark.py"
)
bm = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bm
spec.loader.exec_module(bm)

ALPHABETS = bm.read_alphabets(REPO / "data" / "ice_binding_sw" / "alphabet_classes.tsv")
# Equal residue frequencies are enough to test the algebra of the collapse.
UNIFORM = np.full(20, 0.05)


def test_recall_at_budget_counts_ties_as_missed():
    pos = np.array([5.0, 3.0, 1.0])
    neg = np.array([4.0, 3.0, 2.0, 0.0])
    # Budget 0: above the best negative (4) -> only 5.
    assert bm.recall_at_budget(pos, neg, 0.0) == pytest.approx(1 / 3)
    # Budget 0.5 of 4 negatives = 2 false hits allowed -> above the 3rd best negative (2).
    # 3.0 ties nothing at that threshold, so 5 and 3 count.
    assert bm.recall_at_budget(pos, neg, 0.5) == pytest.approx(2 / 3)
    # Budget 0.25 -> 1 false hit, threshold is the 2nd best negative (3); the positive 3 ties.
    assert bm.recall_at_budget(pos, neg, 0.25) == pytest.approx(1 / 3)


def test_collapse_matrix_protein20_is_the_source_matrix():
    m = bm.collapse_matrix("BLOSUM62", ALPHABETS["protein20"], UNIFORM)
    assert m["W"]["W"] == 11 and m["A"]["R"] == -1


@pytest.mark.parametrize("alphabet", ["hp_lehninger2", "dayhoff6"])
@pytest.mark.parametrize("source", ["BLOSUM62", "BLOSUM45"])
def test_collapsed_matrix_has_negative_expected_score(alphabet, source):
    """Local alignment needs E[score] < 0 under the background, or alignments run on."""
    mapping = ALPHABETS[alphabet]
    m = bm.collapse_matrix(source, mapping, UNIFORM)
    letters = list(m.alphabet)
    cq = {c: sum(0.05 for a in bm.AMINO_ACIDS if mapping[a] == c) for c in letters}
    expected = sum(cq[a] * cq[b] * m[a][b] for a in letters for b in letters)
    assert expected < 0
    # Same class scores above different class, for every pair of classes.
    for a in letters:
        for b in letters:
            if a != b:
                assert m[a][a] > m[a][b]


def test_hp_translation_matches_kmerseek_classes():
    # hp_lehninger2 class 0 = AFGILMPVWY (hydrophobic), class 1 = CDEHKNQRST (polar).
    assert bm.translate("AATAATCD", ALPHABETS["hp_lehninger2"]) == "HHPHHPPP"
    assert bm.translate("AXB", ALPHABETS["hp_lehninger2"]) == "H"


def test_mature_span_prefers_longest_chain():
    rec = bm.FlatRecord(
        "X",
        "M" * 100,
        [("SIGNAL", "1..20"), ("CHAIN", "21..100"), ("CHAIN", "30..40")],
        "",
    )
    assert bm.mature_span(rec) == (21, 100, "longest CHAIN")
    rec = bm.FlatRecord(
        "Y", "M" * 50, [("PEPTIDE", "6..19"), ("PEPTIDE", "23..36")], ""
    )
    assert bm.mature_span(rec) == (6, 36, "span of PEPTIDE features")
    assert bm.parse_span("<1..5") == (1, 5)
    assert bm.parse_span("?..5") is None


def test_assign_origin_needs_every_rule_field():
    origins = pl.DataFrame(
        {
            "origin_id": ["A", "B"],
            "uniprot_family": ["Type-I AFP family", "Type-I AFP family"],
            "taxon_family": ["Pleuronectidae", "Cottidae"],
            "name_regex": ["", ""],
            "positive_status": ["positive", "positive"],
            "origin_evidence": ["x", "x"],
            "citation_doi": ["d", "d"],
            "note": ["", ""],
        }
    )
    row = {
        "accession": "P1",
        "uniprot_family": "Type-I AFP family",
        "taxon_family": "Cottidae",
        "protein_name": "AFP",
    }
    assert bm.assign_origin(row, origins)["origin_id"] == "B"
    row["taxon_family"] = "Gadidae"
    assert bm.assign_origin(row, origins)["status"] == "positive_unresolved_origin"


def test_gumbel_fit_recovers_parameters():
    """Draw scores from the Karlin-Altschul form and check the fit gets lambda and K back."""
    rng = np.random.default_rng(0)
    lam, K = 0.3, 0.05
    mn = rng.integers(50, 400, size=4000) * rng.integers(50, 400, size=4000)
    # Inverse CDF of P(S <= x) = exp(-K mn e^(-lam x)).
    u = rng.uniform(size=4000)
    x = (np.log(K * mn) - np.log(-np.log(u))) / lam
    lam_hat, k_hat = bm.fit_gumbel(x, mn.astype(float))
    assert lam_hat == pytest.approx(lam, rel=0.05)
    assert k_hat == pytest.approx(K, rel=0.25)


def test_gapped_rows_shows_real_residues_and_classes():
    mapping = ALPHABETS["hp_lehninger2"]
    aligner = bm.make_aligner(bm.collapse_matrix("BLOSUM62", mapping, UNIFORM), 11, 1)
    qa, ta = "AATAATAAT", "GGSGGSGGS"
    q_tr, t_tr = bm.translate(qa, mapping), bm.translate(ta, mapping)
    aln = bm.best_alignment(aligner, q_tr, t_tr)
    qrow, mrow, trow = bm.gapped_rows(aln, qa, ta, q_tr, t_tr)
    assert qrow == qa and trow == ta
    # No residue is identical, every class is.
    assert mrow == ":" * 9


def test_shuffle_preserves_composition_and_identity_coverage():
    rng = random.Random(1)
    s = "AATAATAATCDEFGHIKLMNPQ"
    letters = list(s)
    rng.shuffle(letters)
    assert sorted(letters) == sorted(s)
    ident, cov = bm.identity_coverage(bm.blosum62_aligner(), s, s)
    assert ident == 1.0 and cov == 1.0


def test_task_metrics_on_a_toy_ranking():
    df = pl.DataFrame(
        {
            "setting": ["s"] * 5,
            "query": ["q"] * 5,
            "q_origin": ["o1"] * 5,
            "role": [
                "cross_origin_positive",
                "real_negative",
                "cross_origin_positive",
                "real_negative",
                "shuffle",
            ],
            "sim": [9.0, 8.0, 1.0, 0.5, 100.0],
        }
    )
    m = bm.task_metrics(df, "cross_origin_positive", [0.0, 0.5]).row(0, named=True)
    # Shuffles never count as negatives here.
    assert m["n_negatives"] == 2
    assert m["average_precision"] == pytest.approx(
        (1 / 1 + 2 / 3) / 2
    )  # positives at ranks 1 and 3
    assert m["recall_fpr_0.0"] == pytest.approx(0.5)
    assert m["recall_fpr_0.5"] == pytest.approx(1.0)
    assert m["best_positive_rank"] == 1
    assert math.isnan(bm.recall_at_budget(np.array([]), np.array([1.0]), 0.0))


def test_curve_thresholds_include_complete_tie_groups():
    # Positives precede negatives in the stored array; neither can precede
    # the other when an actual threshold admits the entire tied group.
    scores = np.array([5.0, 5.0, 4.0, 3.0, 3.0])
    assert bm.curve_threshold_indices(scores).tolist() == [1, 2, 4]
    assert bm.curve_threshold_indices(np.ones(10)).tolist() == [9]
    assert bm.curve_threshold_indices(np.array([])).tolist() == []
