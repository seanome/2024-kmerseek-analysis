"""sets_vs_weighted_score.py on a made-up example small enough to score by hand.

Run: python -m pytest tests/test_sets_vs_weighted.py
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import sets_vs_weighted_score as svw  # noqa: E402


def test_b_score_by_hand():
    # Q up {1,2,3}, Q down {4,5}; D up {1,2,5}, D down {3,9}
    # same = |{1,2}| + |{}| = 2; reversed = |{3}| + |{5}| = 2; denominator 5
    q_up, q_down = np.array([1, 2, 3]), np.array([4, 5])
    assert svw.b_score(q_up, q_down, np.array([1, 2, 5]), np.array([3, 9])) == (2, 2, 5, 0.0)
    assert svw.b_score(q_up, q_down, np.array([1, 2]), np.array([4])) == (3, 0, 5, 0.6)
    # exchanged Botryllus labels flip the sign
    assert svw.b_score(q_down, q_up, np.array([1, 2]), np.array([4])) == (0, 3, 5, -0.6)
    assert svw.b_score(np.array([], int), np.array([], int), np.array([1]), np.array([2]))[3] is None


def test_exclusive_drops_codes_in_both():
    up, down, both = svw.exclusive(np.array([1, 2, 3]), np.array([3, 4]))
    assert up.tolist() == [1, 2] and down.tolist() == [4] and both == 1


def test_member_matches_isin():
    rng = np.random.default_rng(0)
    d = np.unique(rng.integers(0, 1000, 300))
    x = rng.integers(0, 1200, 500)
    assert (svw.member(d, x) == np.isin(x, d)).all()
    assert not svw.member(np.empty(0, np.int64), x).any()
    assert svw.sorted_unique(np.array([3, 1, 3])).tolist() == [1, 3]


def test_run_weights():
    d = pl.DataFrame({"sample": ["a", "b", "c", "d"], "condition": ["PEC", "control", "PEC", "control"],
                      "subject": ["s1", "s1", "s2", "s2"]})
    assert svw.run_weights(d, True) == {"a": 0.5, "b": -0.5, "c": 0.5, "d": -0.5}
    u = pl.DataFrame({"sample": ["a", "b", "c"], "condition": ["PEC", "control", "control"], "subject": ["x", "y", "z"]})
    assert svw.run_weights(u, False) == {"a": 1.0, "b": -0.5, "c": -0.5}


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def codes_tsv(path, codes):
    write(path, "alphabet\tk\tcode\n" + "".join(f"protein\t2\t{c}\n" for c in codes))


def test_main_end_to_end(tmp_path):
    """One setting, two Botryllus sets (one real, one swap), one paired human comparison of two
    subjects, and its label swap for version B."""
    work, an, out = tmp_path / "work", tmp_path / "analysis", tmp_path / "out"
    d = an / "design" / "sets_vs_weighted"
    write(d / "settings.tsv", "alphabet\tk\nprotein\t2\n")
    write(d / "botryllus_sets.tsv", "name\tcomparison\tlabels\tdir\nb1\tb1\treal\tx\nb1__pairflip1\tb1\tpairflip1\tx\n")
    write(d / "human_sets.tsv", "name\tcomparison\tlabels\tkind\tseries\tpaired\tdir\tdesign\n"
          "h1\th1\treal\texercise_48h_or_more\tGSE1\ttrue\tx\tdesign/human/contrast_h1.design.tsv\n"
          "h1__swap_even\th1\tswap_even\texercise_48h_or_more\tGSE1\ttrue\tx\tdesign/human/contrast_h1.design.tsv\n")
    design = "sample\tcondition\tsubject\nr1\tPEC\ts1\nr2\tcontrol\ts1\nr3\tPEC\ts2\nr4\tcontrol\ts2\n"
    write(an / "design" / "human" / "contrast_h1.design.tsv", design)
    for c in svw.FIG4:  # the Fig. 4 groupings are scored as a check; here on the same four runs
        write(an / "design" / "human" / f"contrast_{c}.design.tsv", design)
    write(d / "human_runs.tsv", "series\trun\nGSE1\tr1\nGSE1\tr2\nGSE1\tr3\nGSE1\tr4\n")
    # b1: up {1,2,3}, down {3,4,5}; code 3 is in both and is dropped. swap: up {4}, down {1}
    codes_tsv(work / "botryllus" / "b1" / "up.codes.tsv", [1, 2, 3])
    codes_tsv(work / "botryllus" / "b1" / "down.codes.tsv", [3, 4, 5])
    codes_tsv(work / "botryllus" / "b1__pairflip1" / "up.codes.tsv", [4])
    codes_tsv(work / "botryllus" / "b1__pairflip1" / "down.codes.tsv", [1])
    codes_tsv(work / "query_codes.tsv", [1, 2, 3, 4, 5])
    # counts per run; every run has 1,000,000 k-mers, so counts per million = count
    counts = {"r1": {1: 15, 2: 7, 4: 3, 5: 1}, "r2": {1: 7, 2: 7, 4: 7, 5: 1},
              "r3": {1: 31, 2: 7, 4: 1, 5: 1}, "r4": {1: 7, 2: 7, 4: 3, 5: 1}}
    for run, c in counts.items():
        p = work / "human" / f"{run}.counts.tsv.gz"
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt") as f:
            f.write("alphabet\tk\tcode\tcount\nprotein\t2\t-1\t1000000\n" + "".join(f"protein\t2\t{k}\t{v}\n" for k, v in c.items()))
    # human sets: h1 up {1,5,9}, down {4}; its swap: up {2}, down {1}
    for name, up, down in (("h1", [1, 5, 9], [4]), ("h1__swap_even", [2], [1])):
        f = work / "human_sets" / name
        f.mkdir(parents=True)
        (f / "codes.done").touch()
        for direction, codes in (("up", up), ("down", down)):
            with gzip.open(f / f"{direction}.protein.2.codes.gz", "wt") as g:
                g.write("".join(f"{c}\n" for c in codes))

    svw.main(work, an, out)
    res = pl.read_csv(out / "scores.tsv", separator="\t")
    assert len(res) == 3 * 2  # b1, its exchanged-labels copy, the swap; two human sets

    def row(b, h):
        return res.filter((pl.col("botryllus_set") == b) & (pl.col("human_set") == h)).row(0, named=True)

    # Version B by hand. b1: Q up {1,2}, Q down {4,5}. h1: D up {1,5,9}, D down {4}
    r = row("b1", "h1")
    assert (r["q_up"], r["q_down"], r["q_in_both"], r["d_up"], r["d_down"]) == (2, 2, 1, 3, 1)
    assert (r["b_same"], r["b_reversed"], r["b_denominator"], r["b_score"]) == (2, 1, 4, 0.25)
    assert row("b1__fullswap", "h1")["b_score"] == -0.25
    # swap set: Q up {4}, Q down {1}: both are reversed against h1
    assert (row("b1__pairflip1", "h1")["b_same"], row("b1__pairflip1", "h1")["b_reversed"]) == (0, 2)
    # against the human label swap (D up {2}, D down {1}): Q up 2 is same, Q up 1 is reversed
    s = row("b1", "h1__swap_even")
    assert (s["b_same"], s["b_reversed"], s["b_score"]) == (1, 1, 0.0)
    # Version A by hand. Totals over the 4 runs: code 1: 60, 2: 28, 4: 14, 5: 4. Code 5 is under 10: not kept.
    # Kept up {1,2}, kept down {4}. log2(count + 0.5): 15 -> log2(15.5), 7 -> log2(7.5), 31 -> log2(31.5),
    # 3 -> log2(3.5), 1 -> log2(1.5).
    L = lambda c: np.log2(c + 0.5)  # noqa: E731
    v = {"r1": (L(15) + L(7)) / 2 - L(3), "r2": (L(7) + L(7)) / 2 - L(7),
         "r3": (L(31) + L(7)) / 2 - L(1), "r4": (L(7) + L(7)) / 2 - L(3)}
    expected = ((v["r1"] - v["r2"]) + (v["r3"] - v["r4"])) / 2
    assert (r["a_up_kept"], r["a_down_kept"]) == (2, 1)
    assert r["a_D"] == pytest.approx(expected, abs=1e-12)
    assert row("b1__fullswap", "h1")["a_D"] == pytest.approx(-expected, abs=1e-12)
    assert row("b1", "h1__swap_even")["a_D"] is None  # version A is not scored against human label swaps
    chk = pl.read_csv(out / "full_swap_check.tsv", separator="\t")
    assert len(chk) == 2 and chk["b_sum"].abs().max() == 0
    sizes = pl.read_csv(out / "sizes_b.tsv", separator="\t")
    h = sizes.filter(pl.col("side") == "human").row(0, named=True)
    assert h["kmers"] == 4 and h["bytes"] == (out / "tables" / "human.h1.protein.2.tsv.gz").stat().st_size
    assert pl.read_csv(out / "tables" / "human.h1.protein.2.tsv.gz", separator="\t")["direction"].to_list() == ["up"] * 3 + ["down"]


def test_unfinished_human_set_stops(tmp_path):
    with pytest.raises(SystemExit):
        svw.read_human_codes(tmp_path, "up", "protein", 7)
