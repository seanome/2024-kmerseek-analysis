"""Tests for experiment 358. Run: python -m pytest -q test_peptide_source.py"""

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import make_fixtures
import peptide_source as ps

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ alignment


def reference_sw(profile, target_codes):
    """Plain O(L*n) Smith-Waterman with the same affine gaps."""
    L, n = len(profile), len(target_codes)
    o, e = ps.GAP_OPEN + ps.GAP_EXTEND, ps.GAP_EXTEND
    neg = -(10**9)
    H = [[0] * (n + 1) for _ in range(L + 1)]
    E = [[neg] * (n + 1) for _ in range(L + 1)]
    F = [[neg] * (n + 1) for _ in range(L + 1)]
    best = 0
    for i in range(1, L + 1):
        for j in range(1, n + 1):
            E[i][j] = max(H[i - 1][j] - o, E[i - 1][j] - e)
            F[i][j] = max(H[i][j - 1] - o, F[i][j - 1] - e)
            H[i][j] = max(0, H[i - 1][j - 1] + profile[i - 1][target_codes[j - 1]], E[i][j], F[i][j])
            best = max(best, H[i][j])
    return best


@pytest.mark.parametrize("seed", range(30))
def test_vectorised_sw_matches_reference(seed):
    rng = random.Random(seed)
    prots = [("".join(rng.choices(ps.CANONICAL, k=rng.randint(5, 40)))) for _ in range(4)]
    # plant a gapped copy of the query in one protein so gaps get used
    q = "".join(rng.choices(ps.CANONICAL, k=rng.randint(6, 14)))
    gapped = q[:3] + "".join(rng.choices(ps.CANONICAL, k=rng.randint(0, 3))) + q[3:]
    prots[1] = prots[1][:2] + gapped + prots[1][2:]
    db = ps.build_database([(str(i), s) for i, s in enumerate(prots)])
    query = ps.Query("q", q, {}, frozenset())
    got = ps.sw_best_per_protein(query.profile(), db.codes, db.starts)
    want = [reference_sw(query.profile(), ps.encode_protein(s)) for s in prots]
    assert list(got) == want


def test_i_and_l_score_the_same():
    q = ps.Query("q", "PEPTLDEK", {}, frozenset())
    db = ps.build_database([("a", "PEPTIDEK"), ("b", "PEPTLDEK")])
    got = ps.sw_best_per_protein(q.profile(), db.codes, db.starts)
    assert got[0] == got[1] == q.self_score()


def test_self_score_is_reached_by_identical_target():
    q = ps.Query("q", "PEPTLDEK", {2: "A"}, frozenset())
    db = ps.build_database([("a", "MMPEPTIDEKMM")])  # I in the protein, L in the call
    assert ps.sw_best_per_protein(q.profile(), db.codes, db.starts)[0] == q.self_score()


# ------------------------------------------------------------------ inputs


def write_inputs(tmp_path, calls, accepted, proteins="ACDEFGHIKLMNPQRSTVWYK"):
    (tmp_path / "p.fasta").write_text(f">sp|P1|X\n{proteins}\n")
    pd.DataFrame(calls).to_csv(tmp_path / "calls.tsv", sep="\t", index=False)
    pd.DataFrame(accepted).to_csv(tmp_path / "acc.tsv", sep="\t", index=False)
    (tmp_path / "in.json").write_text(json.dumps({
        "data_origin": "experimental", "proteins": [{"path": "p.fasta", "role": "human"}],
        "calls": "calls.tsv", "accepted": "acc.tsv"}))
    return tmp_path / "in.json"


GOOD = dict(call_id="c1", run_id="r", scan_id="1", called_sequence="PEPTIDEK", call_score=0.9)
ACC = dict(call_id="c1", accepted_peptide="PEPTIDEK", evidence_tool="t", evidence_qvalue=0.001)


@pytest.mark.parametrize("bad, message", [
    (dict(called_sequence="PEPT1DEK"), "letters"),
    (dict(residue_scores="0.9,0.9"), "residue scores"),
    (dict(alternatives="12:K"), "outside"),
    (dict(alternatives="2:Z"), "canonical"),
])
def test_validation_rejects_bad_calls(tmp_path, bad, message):
    path = write_inputs(tmp_path, [GOOD | bad], [ACC])
    with pytest.raises(ValueError, match=message):
        ps.load_inputs(path)


def test_validation_rejects_duplicate_call_ids(tmp_path):
    path = write_inputs(tmp_path, [GOOD, GOOD], [ACC])
    with pytest.raises(ValueError, match="duplicate"):
        ps.load_inputs(path)


def test_uniprot_accession_taken_from_header(tmp_path):
    inp = ps.load_inputs(write_inputs(tmp_path, [GOOD], [ACC]))
    assert inp.proteins[0][0] == "P1"


# ------------------------------------------------------------------ sources, split


def test_valid_sources_use_i_equals_l_and_list_every_protein():
    prots = [("a", "MMPEPTIDEKMM"), ("b", "PEPTLDEK"), ("c", "PEPTADEK"), ("d", "XXPEPTIDEK")]
    assert ps.valid_sources(["PEPTIDEK"], prots) == [["a", "b", "d"]]


def test_groups_are_transitive():
    g = ps.group_proteins([["a", "b"], ["b", "c"], ["d"]])
    assert g["a"] == g["b"] == g["c"] == "a" and g["d"] == "d"


@pytest.fixture(scope="module")
def fixture_run(tmp_path_factory):
    """The full pipeline on the fixtures, once."""
    fx = tmp_path_factory.mktemp("fixtures")
    make_fixtures.main(fx)
    work = tmp_path_factory.mktemp("work")
    verdict = ps.run_all(fx / "inputs.json", work)
    return fx, work, verdict


def test_no_protein_is_a_source_on_both_sides(fixture_run):
    _, work, _ = fixture_run
    cases = ps.read_cases(work)
    real = cases[cases["kind"].str.startswith("source")]
    side_of = {}
    for r in real.itertuples():
        for a in r.valid_sources.split(";"):
            assert side_of.setdefault(a, r.side) == r.side


def test_source_absent_share_and_full_removal(fixture_run):
    _, work, _ = fixture_run
    cases = ps.read_cases(work)
    for side, g in cases[cases["kind"].str.startswith("source")].groupby("side"):
        assert (g["kind"] == "source_absent").sum() == int(0.2 * len(g))
    absent = cases[cases["kind"] == "source_absent"]
    assert (absent["excluded"] == absent["valid_sources"]).all()


def test_exclusions_never_returned(fixture_run):
    _, work, _ = fixture_run
    cases = ps.read_cases(work).set_index("case_id")
    for m in ps.SEARCH_METHODS:
        cand = pd.read_csv(work / "candidates" / f"{m}.tsv.gz", sep="\t")
        cand = cand[cand["case_id"].map(cases["kind"]) == "source_absent"]
        for r in cand.itertuples():
            assert r.accession not in cases.loc[r.case_id, "excluded"].split(";")


def test_exclusion_equals_rebuilt_collection(fixture_run):
    """Masking a call's sources gives the same candidates as a collection without them."""
    fx, work, _ = fixture_run
    cases = ps.read_cases(work)
    prots = ps.read_fasta(work / "proteins.fasta")
    for r in cases[cases["kind"] == "source_absent"].head(8).itertuples():
        q = ps.to_query(r)
        rebuilt = [(a, s) for a, s in prots if a not in q.excluded]
        db_full, db_cut = ps.build_database(prots), ps.build_database(rebuilt)
        q_open = ps.Query(q.case_id, q.seq, q.alternatives, frozenset())
        for alphabet, k in [(None, ps.EXACT_K), ("massspec18", 5), ("dayhoff6", 8)]:
            i_full = ps.build_kmer_index(db_full, alphabet, k)
            i_cut = ps.build_kmer_index(db_cut, alphabet, k)
            if alphabet is None:
                a = ps.search_exact(q, db_full, i_full, ps.Timer())
                b = ps.search_exact(q_open, db_cut, i_cut, ps.Timer())
            else:
                a = ps.search_reduced(q, db_full, i_full, alphabet, ps.Timer())
                b = ps.search_reduced(q_open, db_cut, i_cut, alphabet, ps.Timer())
            assert a == b
        assert ps.search_align(q, db_full, ps.Timer()) == ps.search_align(q_open, db_cut, ps.Timer())


# ------------------------------------------------------------------ methods


def test_exact_uses_alternatives_and_x():
    db = ps.build_database([("a", "MMPEPTIDEKMM"), ("b", "MMPEPTQDEKMM")])
    idx = ps.build_kmer_index(db, None, ps.EXACT_K)
    q = ps.Query("q", "PEPTKDEK", {4: "I"}, frozenset())
    assert set(ps.search_exact(q, db, idx, ps.Timer())) == {"a"}
    q = ps.Query("q", "PEPTXDEK", {}, frozenset())
    assert set(ps.search_exact(q, db, idx, ps.Timer())) == {"a", "b"}
    q = ps.Query("q", "PEPT", {}, frozenset())  # shorter than k: scanned
    assert set(ps.search_exact(q, db, idx, ps.Timer())) == {"a", "b"}


def test_reduced_finds_kq_swap_that_exact_misses():
    db = ps.build_database([("a", "MMAGHWEKDYRMM"), ("z", "WWWWWWWWWWWW")])
    q = ps.Query("q", "AGHWEQDYR", {}, frozenset())
    assert ps.search_exact(q, db, ps.build_kmer_index(db, None, 5), ps.Timer()) == {}
    res = ps.search_reduced(q, db, ps.build_kmer_index(db, "massspec18", 5), "massspec18", ps.Timer())
    assert res["a"][1] == 1.0 and res["a"][0] < 1.0


# ------------------------------------------------------------------ evaluation rules


def test_rank_counts_only_wrong_proteins():
    cs = ps.CaseScores(0.9, np.array([0.95, 0.9, 0.5]), np.array([]))
    assert ps.rank_of_valid(cs) == 3
    assert ps.rank_of_valid(ps.CaseScores(float("nan"), np.zeros(0), np.zeros(0))) == float("inf")


def test_threshold_allows_one_percent_of_development_absent_calls():
    tops = [ps.CaseScores(float("nan"), np.zeros(0), np.array([s])) for s in np.linspace(0.3, 0.99, 200)]
    t = ps.fit_threshold(tops)
    assert sum(ps.confident(c, t) for c in tops) == 2


def test_threshold_ignores_evaluation_calls(fixture_run):
    _, work, _ = fixture_run
    summary = pd.read_csv(work / "results" / "summary.tsv", sep="\t").set_index("method")
    cases = ps.read_cases(work)
    cand = pd.read_csv(work / "candidates" / "align.tsv.gz", sep="\t")
    ev = set(cases.query("side == 'evaluation'")["case_id"])
    cand.loc[cand["case_id"].isin(ev), "match_fraction"] = 1.0
    cs = ps.case_scores(cases, cand, "match_fraction")
    dev_abs = cases.query("side == 'development' and kind == 'source_absent'")["case_id"]
    assert ps.fit_threshold([cs[c] for c in dev_abs]) == summary.loc["align", "threshold"]


def test_surviving_count_is_not_truncated():
    cs = ps.CaseScores(0.9, np.zeros(0), np.full(40, 0.95))
    assert ps.n_surviving(cs, 0.9) == 40


def test_wilson_zero_of_fifty():
    assert ps.wilson(0, 50)[1] == pytest.approx(0.0713, abs=1e-4)


def test_pair_shuffle_keeps_pairs():
    rng = random.Random(1)
    for _ in range(50):
        s = "".join(rng.choices("ACDEKLR", k=rng.randint(3, 25)))
        t = ps.shuffle_pairs(s, rng)
        assert Counter(zip(s, s[1:])) == Counter(zip(t, t[1:])) and t[0] == s[0] and t[-1] == s[-1]


def test_synthetic_data_end_in_insufficient_evidence(fixture_run):
    _, work, verdict = fixture_run
    assert verdict["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert any("synthetic" in r for r in verdict["reasons"])
    assert verdict["verdict_if_data_were_real"] in ("PASS", "FAIL")
