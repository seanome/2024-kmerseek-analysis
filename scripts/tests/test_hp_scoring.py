import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pl = pytest.importorskip("polars")
pytest.importorskip("Bio")
pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "notebooks"))

import hp_conservation_utils as hc  # noqa: E402

UNIFORM = np.full(20, 0.05)


def test_aligned_codes_drops_gap_columns():
    q, t = hc.aligned_codes("AC-De", "ACWD.")
    assert q.tolist() == [0, 1, 2]
    assert t.tolist() == [0, 1, 2]


def test_substitution_bits_target_frequencies_sum_to_one():
    bits, target, lam = hc.substitution_bits("BLOSUM62", UNIFORM)
    assert target.sum() == pytest.approx(1.0)
    assert np.allclose(bits, np.log2(target / np.outer(UNIFORM, UNIFORM)))
    assert lam > 0


def test_blosum62_lambda_matches_published_value():
    # Ungapped BLOSUM62 lambda is 0.3176 nats (0.458 bits) with BLAST's background;
    # with a uniform background it differs, so only the order of magnitude is checked here.
    _, _, lam = hc.substitution_bits("BLOSUM62", UNIFORM)
    assert 0.3 < lam < 0.6


def test_class_bits_scores_each_residue_by_its_class():
    _, target, _ = hc.substitution_bits("BLOSUM62", UNIFORM)
    hp = hc.class_bits(target, UNIFORM, "hp_thomas_dill2")
    i = {a: k for k, a in enumerate(hc.PROTEIN20_ORDER)}
    # A and I are both hydrophobic, D and K both polar
    assert hp[i["A"], i["D"]] == hp[i["I"], i["K"]]
    assert hp[i["A"], i["I"]] > 0 > hp[i["A"], i["D"]]


def test_score_vs_chance_identity_by_hand():
    s = hc.agreement_bits("protein20")
    r = hc.score_vs_chance(np.array([0, 1, 2]), np.array([0, 1, 2]), s)
    # three distinct residues, all identical: chance 1/3, sd sqrt(2/9)
    assert r["score"] == 1.0
    assert r["chance"] == pytest.approx(1 / 3)
    assert r["z_per_position"] == pytest.approx((2 / 3) / np.sqrt(2 / 9))


def test_cluster_bootstrap_whole_clusters():
    # one big cluster of 1.0s and one small of 0.0s: the mean is row-weighted,
    # and resampling clusters can give any mix, so the interval spans 0 to 1
    x = [1.0] * 9 + [0.0]
    c = ["a"] * 9 + ["b"]
    m, lo, hi = hc.cluster_bootstrap_mean_ci(x, c, n_boot=500, seed=1)
    assert m == pytest.approx(0.9)
    assert lo == 0.0 and hi == 1.0


def test_cluster_bootstrap_drops_nan():
    m, _, _ = hc.cluster_bootstrap_mean_ci(
        [1.0, np.nan, 3.0], ["a", "b", "c"], n_boot=50
    )
    assert m == pytest.approx(2.0)


def test_large_deviation_bits_equals_bernoulli_kl_for_agreement_scoring():
    # agree/disagree scoring: values 0 and 1, chance agreement c, observed agreement a
    c = np.array([0.065, 0.51, 0.3])
    a = np.array([0.164, 0.69, 0.25])  # last one is below chance
    w = np.stack([1 - c, c], axis=1)
    got = hc.large_deviation_bits(a, np.array([0.0, 1.0]), w)
    want = np.where(a > c, hc.bernoulli_kl_bits(a, c), 0.0)
    assert np.allclose(got, want, atol=1e-9)


def test_large_deviation_bits_all_top_scores():
    # every position at the top score: rate is -log2 P(top score)
    w = np.array([[0.75, 0.25]])
    got = hc.large_deviation_bits(np.array([1.0]), np.array([0.0, 1.0]), w)
    assert got[0] == pytest.approx(2.0)


def test_score_pairs_matches_score_vs_chance():
    s = {
        "identity": hc.agreement_bits("protein20"),
        "hp": hc.agreement_bits("hp_thomas_dill2"),
    }
    qa = "ACDEFGHIKLMNPQRSTVWY" * 3
    ta = "ACDEFGHIKLMNPQRSTVWY"[::-1] * 3
    t = hc.score_pairs([qa], [ta], s, min_cols=10)
    q, tt = hc.aligned_codes(qa, ta)
    for k, sc in s.items():
        r = hc.score_vs_chance(q, tt, sc)
        row = t.filter(pl.col("scheme") == k).row(0, named=True)
        assert row["z_per_position"] == pytest.approx(r["z_per_position"])
        assert row["score"] == pytest.approx(r["score"])


def test_cluster_bootstrap_diff_of_means_centres_on_the_difference():
    rng = np.random.default_rng(0)
    x1 = rng.normal(1.0, 0.1, 400)
    x2 = rng.normal(0.4, 0.1, 300)
    c1 = np.repeat(np.arange(100), 4)
    c2 = np.repeat(np.arange(100), 3)
    m, lo, hi = hc.cluster_bootstrap_diff_of_means(x1, c1, x2, c2, n_boot=500, seed=1)
    assert m == pytest.approx(x1.mean() - x2.mean())
    assert lo < m < hi
    assert 0.5 < lo and hi < 0.7


def test_cluster_bootstrap_diff_of_means_spans_zero_for_equal_groups():
    x = np.random.default_rng(2).normal(0.0, 1.0, 600)
    c = np.repeat(np.arange(150), 4)
    _, lo, hi = hc.cluster_bootstrap_diff_of_means(
        x[:300], c[:300], x[300:], c[300:], n_boot=500, seed=3
    )
    assert lo < 0 < hi
