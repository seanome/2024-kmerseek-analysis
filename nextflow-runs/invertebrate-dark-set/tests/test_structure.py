"""Tests for the Foldseek and ProstT5 steps of -entry darkStructure.

    python -m pytest nextflow-runs/invertebrate-dark-set/tests/test_structure.py

Real inputs where they fit in a test: human BCL2 (P10415, residues 1-60) with a made-up 3Di
string of the same length, and AlphaFold DB file names as the server writes them.
"""
import gzip
import importlib.machinery
import importlib.util
import io
import json
import math
import os
import random
import subprocess
import sys
import tarfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PIPE, "..", "shared"))
sys.path.insert(0, os.path.join(PIPE, "bin"))

import structure_decoys  # noqa: E402


def _load_tool(name):
    path = os.path.join(PIPE, "tools", name)
    loader = importlib.machinery.SourceFileLoader(name.replace("-", "_"), path)
    spec = importlib.util.spec_from_loader(loader.name, loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


score = _load_tool("dark-structure-score")

BCL2 = "MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPAA"[:60]
SS = "DDPPPVVVVVCVVVPDDDPVSVVVVVVCVVVVHDPPDDPPPPPPPDDDDDPDPPPDPPPDD"[:60]


def test_columns_shuffle_keeps_pairs_and_adjacent_pairs():
    assert len(BCL2) == len(SS) == 60
    da, ds = structure_decoys.shuffle_columns(BCL2, SS, random.Random(0))
    real = list(zip(BCL2, SS))
    dec = list(zip(da, ds))
    assert dec != real
    assert Counter(dec) == Counter(real)
    assert Counter(zip(dec, dec[1:])) == Counter(zip(real, real[1:]))
    assert dec[0] == real[0] and dec[-1] == real[-1]


def test_columns_shuffle_depends_only_on_seed_and_name():
    a = structure_decoys.shuffle_columns(BCL2, SS, structure_decoys.rng_for(0, "AF-P10415-F1-model_v6"))
    b = structure_decoys.shuffle_columns(BCL2, SS, structure_decoys.rng_for(0, "AF-P10415-F1-model_v6 extra"))
    c = structure_decoys.shuffle_columns(BCL2, SS, structure_decoys.rng_for(1, "AF-P10415-F1-model_v6"))
    assert a == b
    assert a != c


def test_sequence_mode_drops_long_and_shuffles(tmp_path):
    fa = tmp_path / "p.fasta"
    fa.write_text(">P10415\n%s\n>LONG1\n%s\n" % (BCL2, "A" * 70))
    env = dict(os.environ, PYTHONPATH=os.path.join(PIPE, "..", "shared"))
    subprocess.run([sys.executable, os.path.join(PIPE, "bin", "structure_decoys.py"), "sequence",
                    "--in", str(fa), "--max-len", "65", "--out-real", str(tmp_path / "r.fa"),
                    "--out-decoy", str(tmp_path / "d.fa"), "--skipped", str(tmp_path / "s.tsv")],
                   check=True, env=env)
    assert (tmp_path / "s.tsv").read_text() == "accession\tlength\nLONG1\t70\n"
    real = (tmp_path / "r.fa").read_text().split("\n")
    dec = (tmp_path / "d.fa").read_text().split("\n")
    assert real[:2] == [">P10415", BCL2]
    assert dec[0] == ">P10415" and sorted(dec[1]) == sorted(BCL2) and dec[1] != BCL2


def test_decoy_cutoff():
    # 100 decoys run, 5% allowed = 5 may pass.
    assert score.decoy_cutoff([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 5.0], 100) == (0.1, 5)
    # ties at the boundary: 4 below, then three at 0.1 -> only the 4 can pass
    assert score.decoy_cutoff([1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.1, 0.1], 100) == (1e-2, 4)
    # fewer decoy hits than allowed: every reported hit passes
    assert score.decoy_cutoff([0.5, 2.0], 100) == (math.inf, 2)
    # 10 decoys run, 0 allowed: nothing can be placed
    assert score.decoy_cutoff([3.0], 10) == (-math.inf, 0)


def test_judge_hit():
    to_clan = {"PF00452": "CL0581", "PF02180": "CL0581"}.get
    tc = lambda p: to_clan(p) or p
    assert score.judge_hit({"PF00452"}, {"PF00452", "PF02180"}, tc) == "family"
    assert score.judge_hit({"PF00452"}, {"PF02180"}, tc) == "clan"
    assert score.judge_hit({"PF00452"}, {"PF00069"}, tc) == "wrong"
    assert score.judge_hit({"PF00452"}, set(), tc) == "unknown"
    assert score.judge_hit(set(), {"PF00452"}, tc) == "no_query_family"


def _add(tf, name, data=b"data_x\n"):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tf.addfile(info, io.BytesIO(data))


def test_afdb_query_models_keeps_highest_version_and_fragments(tmp_path):
    fa = tmp_path / "q.fasta"
    fa.write_text(">P10415\nMAHAG\n>Q8WZ42\nMTTQA\n>FUN000001_FUN000001\nMKK\n")
    arch = tmp_path / "UP000005640_9606_HUMAN_v6.tar"
    with tarfile.open(arch, "w") as tf:
        _add(tf, "AF-P10415-F1-model_v6.cif.gz")
        _add(tf, "AF-P10415-F1-model_v6.pdb.gz")
        _add(tf, "AF-P10415-F1-confidence_v6.json.gz")
        _add(tf, "AF-Q8WZ42-F1-model_v6.cif.gz")
        _add(tf, "AF-Q8WZ42-F2-model_v6.cif.gz")
        _add(tf, "AF-O00000-F1-model_v6.cif.gz")   # not in the proteome
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "AF-P10415-F1-model_v4.cif").write_text("old")  # archive already has P10415
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(PIPE, "bin", "afdb_query_models.py"),
                    "--proteome", str(fa), "--archive", str(arch), "--extra-dir", str(extra),
                    "--out-dir", str(out / "models"), "--coverage", str(out / "cov.tsv"),
                    "--summary", str(out / "sum.json"), "--species", "human"], check=True)
    assert sorted(os.listdir(out / "models")) == [
        "AF-P10415-F1-model_v6.cif.gz", "AF-Q8WZ42-F1-model_v6.cif.gz", "AF-Q8WZ42-F2-model_v6.cif.gz"]
    rows = (out / "cov.tsv").read_text().splitlines()
    assert rows[1:] == [
        "FUN000001_FUN000001\t3\t0\t0\t\tnone",
        "P10415\t5\t1\t1\tv6\tUP000005640_9606_HUMAN_v6.tar",
        "Q8WZ42\t5\t1\t2\tv6\tUP000005640_9606_HUMAN_v6.tar",
    ]
    s = json.load(open(out / "sum.json"))
    assert (s["n_queries"], s["n_with_model"], s["model_versions"]) == (3, 2, {"v6": 2})


def _gz(path, lines):
    with gzip.open(path, "wt") as fh:
        fh.write("".join(l + "\n" for l in lines))


def test_placement_judge_report_end_to_end(tmp_path, capsys):
    """Three human proteins: P10415 dark with a model, Q07817 placed by the sequence tools,
    Q8WZ42 dark with no model. 40 decoys would be needed for one to pass at 5%, so with
    3 run proteins no decoy may hit and the cutoff is set by the one decoy hit."""
    import polars as pl
    sd = tmp_path / "struct" / "human" / "structure"
    sd.mkdir(parents=True)
    (sd / "human_afdb_query_coverage.tsv").write_text(
        "accession\tlength\thas_model\tn_fragments\tmodel_version\tsource\n"
        "P10415\t239\t1\t1\tv6\ta\nQ07817\t233\t1\t1\tv6\ta\nQ8WZ42\t34350\t0\t0\t\tnone\n")
    (sd / "human_prostt5_skipped.tsv").write_text("accession\tlength\nQ8WZ42\t34350\n")
    (sd / "human_afdb_target_summary.json").write_text('{"n_reference": 10, "n_with_model": 9, "share_with_model": 0.9}')
    (sd / "human_afdb_query_summary.json").write_text('{"model_versions": {"v6": 2}}')
    _gz(sd / "human.chunk_0000.foldseek.tsv.gz",
        ["P10415\tP49950\t1\t200\t1\t200\t300\t1e-30", "Q07817\tP10415\t1\t200\t1\t200\t300\t1e-50"])
    _gz(sd / "human.chunk_0000.foldseek_decoy.tsv.gz", ["P10415\tP00001\t1\t20\t1\t20\t20\t0.5"])
    _gz(sd / "human.chunk_0000.prostt5.tsv.gz", ["P10415\tP00001\t1\t50\t1\t50\t30\t1.0"])
    _gz(sd / "human.chunk_0000.prostt5_decoy.tsv.gz", [])
    ev = tmp_path / "ev"
    ev.mkdir()
    _gz(ev / "human_protein_tool_min_evalue.csv.gz",
        ["accession,phmmer,jackhmmer,mmseqs2", "P10415,5.0,,", "Q07817,1e-80,1e-90,1e-70", "Q8WZ42,,,"])
    out = tmp_path / "out"
    score.placement(str(tmp_path / "struct"), str(ev), str(out), "human")
    pp = pl.read_parquet(out / "placement_proteins.parquet")
    fs = {r["accession"]: r for r in pp.filter(pl.col("tool") == "foldseek").iter_rows(named=True)}
    # 3 run decoys allow 0 to pass; the one decoy hit (E 0.5) sets the cutoff below 0.5
    assert fs["P10415"]["placed"] is False and fs["Q07817"]["placed"] is False
    assert fs["Q8WZ42"]["run"] is False
    s = pl.read_csv(out / "placement_summary.csv")
    row = s.filter((pl.col("tool") == "foldseek") & (pl.col("dark_set") == "dark_E1")).row(0, named=True)
    assert (row["n_dark"], row["n_dark_run"], row["n_dark_placed"]) == (2, 1, 0)
    ps = {r["accession"]: r for r in pp.filter(pl.col("tool") == "prostt5").iter_rows(named=True)}
    # no ProstT5 decoy hit: every reported hit passes
    assert ps["P10415"]["placed"] is True and ps["Q8WZ42"]["run"] is False
    clans = tmp_path / "clans.tsv.gz"
    _gz(clans, ["PF00452\tCL0581\tBcl-2"])
    qp = tmp_path / "q.tsv"
    qp.write_text("accession\tpfam_id\nP10415\tPF00452\nQ07817\tPF00452\n")
    rp = tmp_path / "r.parquet"
    pl.DataFrame({"accession": ["P00001", "P49950"], "pfam_id": ["PF00069", "PF00452"]}).write_parquet(rp)
    score.judge(str(out), str(clans), "human:%s:%s" % (qp, rp))
    m = pl.read_csv(out / "marginal_gain.csv")
    pr = m.filter(pl.col("tool") == "prostt5").row(0, named=True)
    assert (pr["n_dark_placed"], pr["wrong"], pr["family"]) == (1, 1, 0)
    d = pl.read_csv(out / "decoy_hits_judged.csv")
    assert d.filter(pl.col("tool") == "foldseek")["wrong"].to_list() == [1]
    # Loose and strict calls with Prompt C's score-bands (PR 163, pinned commit 13749b87).
    # P10415's only domain is PF00452 at 10-80. ProstT5's best hit (P00001) has another
    # family: loose and strict wrong.
    score.bands_inputs(str(out))
    placed = (out / "bands" / "human.prostt5_placed.tsv").read_text().splitlines()
    assert placed == ["query\ttarget\tevalue\tquery_from\tquery_to", "P10415\tP00001\t1.0\t1\t50"]
    dsa = _dark_set_accuracy(tmp_path)
    qd = tmp_path / "qd.tsv"
    qd.write_text("accession\tpfam_id\tdomain_start\tdomain_end\nP10415\tPF00452\t10\t80\n"
                  "Q07817\tPF00452\t90\t160\n")
    # A Foldseek source with two placed dark hits to a PF00452 target: one overlapping the
    # domain (strict right), one outside it (family right, strict wrong).
    fsrc = tmp_path / "fs.tsv"
    fsrc.write_text("query\ttarget\tevalue\tquery_from\tquery_to\nP10415\tP49950\t1e-30\t5\t20\n")
    fsrc2 = tmp_path / "fs2.tsv"
    fsrc2.write_text("query\ttarget\tevalue\tquery_from\tquery_to\nP10415\tP49950\t1e-30\t100\t200\n")
    (out / "bands").mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(dsa), "score-bands", "--species", "human",
                    "--query-pfam", str(qp), "--query-domains", str(qd), "--ref-pfam", str(rp),
                    "--clans", str(clans), "--ev-dir", str(ev),
                    "--source", "prostt5_placed=%s" % (out / "bands" / "human.prostt5_placed.tsv"),
                    "--source", "foldseek_placed=%s" % fsrc,
                    "--source", "foldseek2_placed=%s" % fsrc2,
                    "--out-tsv", str(out / "bands" / "human_bands.tsv"),
                    "--out-json", str(out / "bands" / "human_bands.json")], check=True)
    score.bands_collect(str(out))
    sj = {r["tool"]: r for r in pl.read_csv(out / "strict_judgement.csv").iter_rows(named=True)}
    assert (sj["prostt5"]["n_judged"], sj["prostt5"]["loose_right"], sj["prostt5"]["strict_right"]) == (1, 0, 0)
    assert (sj["foldseek"]["family"], sj["foldseek"]["strict_right"]) == (1, 1)
    assert (sj["foldseek2"]["family"], sj["foldseek2"]["strict_right"]) == (1, 0)
    score.report(str(out))
    text = capsys.readouterr().out
    assert "P1 UNDECIDED" in text and "strict right 1 of 1 judged" in text


def _dark_set_accuracy(tmp_path):
    """Prompt C's tools/dark-set-accuracy at the pinned commit, read out of git."""
    import pytest
    out = tmp_path / "dark-set-accuracy"
    r = subprocess.run(["git", "show", "13749b87:nextflow-runs/invertebrate-dark-set/tools/dark-set-accuracy"],
                       cwd=PIPE, capture_output=True)
    if r.returncode != 0:
        pytest.skip("commit 13749b87 not fetched: git fetch origin olgabot/dark-set-weak-hit-accuracy")
    out.write_bytes(r.stdout)
    return out


def test_afdb_query_models_with_placeholder_archive(tmp_path):
    """The run's Borrelia and Botryllus tasks got archives/NO_ARCHIVE (job 48521315)."""
    (tmp_path / "archives").mkdir()
    (tmp_path / "archives" / "NO_ARCHIVE").write_text(open(os.path.join(PIPE, "assets", "NO_ARCHIVE")).read())
    fa = tmp_path / "q.fasta"
    fa.write_text(">O51076\nMGERG\n>FUN000001_FUN000001\nMKK\n")
    extra = tmp_path / "extra"
    extra.mkdir()
    (extra / "AF-O51076-F1-model_v6.cif").write_text("data_x\n")
    out = tmp_path / "out"
    subprocess.run([sys.executable, os.path.join(PIPE, "bin", "afdb_query_models.py"),
                    "--proteome", str(fa), "--archive", str(tmp_path / "archives" / "NO_ARCHIVE"),
                    "--extra-dir", str(extra), "--out-dir", str(out / "models"),
                    "--coverage", str(out / "cov.tsv"), "--summary", str(out / "sum.json"),
                    "--species", "borrelia"], check=True)
    assert os.listdir(out / "models") == ["AF-O51076-F1-model_v6.cif"]
    s = json.load(open(out / "sum.json"))
    assert (s["n_with_model"], s["archives"]) == (1, [])
    # Botryllus: placeholder archive, placeholder (non-directory) extra
    out2 = tmp_path / "out2"
    subprocess.run([sys.executable, os.path.join(PIPE, "bin", "afdb_query_models.py"),
                    "--proteome", str(fa), "--archive", str(tmp_path / "archives" / "NO_ARCHIVE"),
                    "--extra-dir", os.path.join(PIPE, "assets", "NO_EXTRA"), "--out-dir", str(out2 / "models"),
                    "--coverage", str(out2 / "cov.tsv"), "--summary", str(out2 / "sum.json"),
                    "--species", "botryllus"], check=True)
    assert json.load(open(out2 / "sum.json"))["n_with_model"] == 0


def test_read_hits_keeps_smallest_evalue(tmp_path):
    p = tmp_path / "h.tsv.gz"
    with gzip.open(p, "wt") as fh:
        fh.write("P10415\tQ07817\t1\t200\t1\t200\t300\t1e-40\n")
        fh.write("P10415\tP0DUM1\t1\t50\t1\t50\t40\t1e-3\n")
        fh.write("Q8WZ42\tP11532\t5\t90\t5\t90\t60\t2.0\n")
    best = score.read_hits([str(p)])
    assert best == {"P10415": ("Q07817", 1e-40, 300.0, 1, 200), "Q8WZ42": ("P11532", 2.0, 60.0, 5, 90)}
