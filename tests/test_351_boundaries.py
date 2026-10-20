"""Tests for notebook 351's scoring, on small made-up outputs from each tool."""

import gzip
import importlib.util
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

A = Path(__file__).resolve().parents[1] / "analysis/351_short_feature_boundaries"


def load(name):
    spec = importlib.util.spec_from_file_location(name, A / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


score = load("score_boundaries")
excl = load("exclude_close_sequences")
shuf = load("make_shuffles")

KMERSEEK_HEADER = ",".join(score.KMERSEEK_COLS)


def domtbl_line(prot, hmm, acc, env_from, env_to, ievalue):
    # 23 whitespace columns as hmmscan writes them; only some are read.
    c = [
        hmm,
        acc,
        "100",
        prot,
        "-",
        "100",
        "1e-5",
        "50",
        "0",
        "1",
        "1",
        "1e-5",
        str(ievalue),
        "40",
        "0",
        "1",
        "20",
        str(env_from),
        str(env_to),
        str(env_from),
        str(env_to),
        "0.9",
        "desc",
    ]
    return " ".join(c) + "\n"


@pytest.fixture
def setup(tmp_path):
    a = tmp_path / "a"
    (a / "data/filtered").mkdir(parents=True)
    shutil.copy(A / "preregistered.yaml", a / "preregistered.yaml")
    (a / "data/filtered/instances_kept.tsv").write_text(
        "instance_id\telm_class\telm_type\tprotein\tstart\tend\tlength\torganism\tfeature_residues\n"
        "I1\tX\tLIG\tP1\t40\t45\t6\tH\tAAAAAA\n"
        "I2\tX\tLIG\tP2\t10\t15\t6\tH\tAAAAAA\n"
    )
    (a / "data/filtered/query_proteins.fasta").write_text(
        ">P1\n" + "A" * 100 + "\n>P2\n" + "C" * 50 + "\n"
    )
    run = tmp_path / "run"
    for s in ("protein20_k5", "dayhoff6_k7", "hp_lehninger2_k12"):
        d = run / "kmerseek" / s
        d.mkdir(parents=True)
        rows = [KMERSEEK_HEADER]
        if s == "protein20_k5":
            # 0-based end-exclusive 39..45 is residues 40..45: exact.
            rows.append("P1,T1,39,45,AAAAAA,9,15,AAAAAA,0.001,2.0")
            # a far, better-E region; the truth-blind choice takes it
            rows.append("P1,T2,79,90,AAAAAAAAAAA,0,11,AAAAAAAAAAA,1e-9,2.0")
            # fails the cutoff: ignored
            rows.append("P2,T3,9,15,CCCCCC,0,6,CCCCCC,0.5,1.0")
            # one shuffle with a placement
            rows.append("P1__shuf03,T1,0,10,AAAAAAAAAA,0,10,AAAAAAAAAA,inf,1.0")
            rows.append("P2__shuf01,T1,9,15,CCCCCC,0,6,CCCCCC,0.001,1.0")
        with gzip.open(d / "chunk000.csv.gz", "wt") as fh:
            fh.write("\n".join(rows) + "\n")
        (d / "DONE").write_text("")
    (run / "hmmscan").mkdir()
    (run / "hmmscan/chunk000.domtbl").write_text(
        "# header\n" + domtbl_line("P1", "Dom", "PF00001.1", 20, 70, 1e-6)
    )
    (run / "blastp").mkdir()
    (run / "blastp/chunk000.tsv").write_text(
        "P1\tT9\t50\t30\t35\t50\t1\t16\t1e-4\t40\tAAAA\tAAAA\n"
        "P1\tT9\t50\t30\t1\t100\t1\t100\t1e-2\t20\tAAAA\tAAAA\n"
    )
    return a, run


def test_scores_each_method(setup):
    a, run = setup
    score.main(a, run)
    e = pd.read_csv(a / "results/boundary_errors.csv")
    k = e[(e.setting == "protein20_k5") & (e.instance_id == "I1")].iloc[0]
    assert (k.placed_start, k.placed_end, k.start_error, k.end_error) == (40, 45, 0, 0)
    assert (k.blind_start, k.blind_end) == (80, 90)
    # P2's only kmerseek region fails the cutoff, so I2 is unplaced
    assert e[(e.setting == "protein20_k5") & (e.instance_id == "I2")].start_error.iloc[
        0
    ] == float("inf")
    p = e[(e.setting == "pfam_hmmscan") & (e.instance_id == "I1")].iloc[0]
    assert (p.start_error, p.end_error) == (20, 25)
    b = e[(e.setting == "blastp") & (e.instance_id == "I1")].iloc[0]
    # best HSP of T9 is the E = 1e-4 one, 35..50
    assert (b.placed_start, b.placed_end) == (35, 50)

    m = pd.read_csv(a / "results/metrics.csv").set_index("setting")
    # pooled start and end errors of I1 (0, 0) and I2 (inf, inf): median is (0 + inf) / 2
    assert m.loc["protein20_k5", "within_10"] == 0.5
    # 1 of 20 shuffled proteins has a placement passing the cutoff (P1__shuf03 is E = inf)
    assert m.loc["protein20_k5", "shuffle_any_placement_rate"] == pytest.approx(1 / 20)
    v = json.load(open(a / "results/verdict.json"))
    assert v["answer"] in {"PASS", "FAIL", "DROP THE IDEA"}
    # dayhoff6_k7 and hp_lehninger2_k12 ran but placed nothing: median inf > k
    assert v["settings"]["dayhoff6_k7"]["median_above_k"]


def test_verdict_rules():
    import yaml

    prereg = yaml.safe_load(open(A / "preregistered.yaml"))
    rows = [
        {
            "method": "kmerseek",
            "setting": "protein20_k5",
            "median_error": 2.0,
            "within_10": 0.9,
            "shuffle_any_placement_rate": 0.01,
        },
        {
            "method": "kmerseek",
            "setting": "dayhoff6_k7",
            "median_error": 20.0,
            "within_10": 0.3,
            "shuffle_any_placement_rate": 0.01,
        },
        {
            "method": "profile",
            "setting": "pfam_hmmscan",
            "median_error": 6.0,
            "within_10": 0.5,
            "shuffle_any_placement_rate": 0.0,
        },
    ]
    ran = {"protein20_k5": True, "dayhoff6_k7": True, "hp_lehninger2_k12": False}
    v = score.verdict(pd.DataFrame(rows), prereg, ran)
    assert v["answer"] == "PASS"  # 6 >= 3 x 2
    rows[2]["median_error"] = 5.9
    assert score.verdict(pd.DataFrame(rows), prereg, ran)["answer"] == "FAIL"
    rows[0]["median_error"] = 5.5  # above k = 5, and dayhoff 20 > 7
    assert score.verdict(pd.DataFrame(rows), prereg, ran)["answer"] == "DROP THE IDEA"
    assert not score.verdict(pd.DataFrame(rows), prereg, ran)["settings"][
        "hp_lehninger2_k12"
    ]["ran"]


def test_uniform_random_stays_inside_protein():
    inst = pd.DataFrame(
        {
            "instance_id": ["I"] * 50,
            "protein": ["P"] * 50,
            "start": [5] * 50,
            "end": [9] * 50,
            "length": list(range(2, 52)),
        }
    )
    ur = score.uniform_random(inst, {"P": 30})
    assert (ur.placed_start >= 1).all() and (ur.placed_end <= 30).all()
    assert (ur.placed_end >= ur.placed_start).all()


def test_shuffles_keep_composition(tmp_path):
    (tmp_path / "q.fa").write_text(">P1\nACDEFGHIKLMNPQRSTVWY\n")
    shuf.main(tmp_path / "q.fa", tmp_path / "all.fa")
    seqs = excl.read_fasta(tmp_path / "all.fa")
    assert len(seqs) == 11 and sorted(seqs["P1__shuf07"]) == sorted(seqs["P1"])
    assert len({s for s in seqs.values()}) == 11


def test_exclusion(tmp_path):
    (tmp_path / "sp.fa").write_text(
        ">sp|P1|A_HUMAN x\nAAAA\n>sp|Q9|B_HUMAN y\nCCCC\n>sp|Q8|C_HUMAN\nDDDD\n>sp|Q7|D_HUMAN\nEEEE\n"
    )
    excl.prep(tmp_path / "sp.fa", tmp_path / "sp.acc.fa")
    (tmp_path / "q.fa").write_text(">P1-2\nGGGG\n>P5\nEEEE\n")
    # Q9: 95% over the whole length -> removed; Q8: 95% over 1 of 4 residues -> kept
    (tmp_path / "h.m8").write_text(
        "P5\tQ9\t0.95\t4\t1\t4\t1\t4\t4\t4\t1e-5\nP5\tQ8\t0.95\t1\t1\t1\t1\t1\t4\t4\t1\n"
    )
    excl.remove(
        tmp_path / "sp.acc.fa",
        tmp_path / "q.fa",
        tmp_path / "h.m8",
        tmp_path / "out.fa",
        tmp_path / "rm.tsv",
    )
    kept = set(excl.read_fasta(tmp_path / "out.fa"))
    assert kept == {"Q8"}  # P1 by accession, Q7 identical to P5, Q9 close
    (tmp_path / "after.m8").write_text("P5\tQ8\t0.95\t4\t1\t4\t1\t4\t4\t4\t1e-5\n")
    with pytest.raises(SystemExit):
        excl.check(tmp_path / "after.m8", tmp_path / "c.txt")
    (tmp_path / "after.m8").write_text("P5\tQ8\t0.89\t4\t1\t4\t1\t4\t4\t4\t1e-5\n")
    excl.check(tmp_path / "after.m8", tmp_path / "c.txt")
    assert "PASSED" in (tmp_path / "c.txt").read_text()
