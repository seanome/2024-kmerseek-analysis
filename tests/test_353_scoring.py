"""Notebook 353: query maker and scorer on a hand-made case (no kmerseek needed).

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 -m pytest tests/test_353_scoring.py
"""

import csv
import gzip
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "353"))
sys.path.insert(0, str(REPO / "nextflow-runs" / "shared"))

from alphabets import class_of, encode  # noqa: E402
from make_queries import substitute  # noqa: E402

COLS = "query_name target_name region_start region_end target_start target_end region_subseq target_subseq region_length region_n_mismatches region_n_shared_kmers region_evalue region_evalue_source region_run_evalue region_ka_evalue".split()


def test_substitutions_keep_or_change_the_class():
    classes = class_of("dayhoff6")
    window = "MSEYQPSLFALNPMGFSPLDGSKSTNENVSA"
    for seed in range(50):
        seq, pos = substitute(window, 2, 1, classes, random.Random(seed))
        changed = [i for i in range(len(window)) if seq[i] != window[i]]
        assert sorted(changed) == sorted(pos)
        within, cross = pos[:2], pos[2:]
        assert all(encode(seq[i], "dayhoff6") == encode(window[i], "dayhoff6") for i in within)
        assert all(encode(seq[i], "dayhoff6") != encode(window[i], "dayhoff6") for i in cross)


def test_score_found_placed_other(tmp_path):
    rec = tmp_path / "records.tsv"
    seq_a = "MSEYQPSLFALNPMGFSPLDGSKSTNENVSASTSTAKPMVGQLIFDKFIKTEE"
    seq_b = "DNSKEWTSLFDNDIPVTTDDVSLADKAIESTEEVSLVPSNLEVSTTSFLPTPV"
    with open(rec, "w") as f:
        f.write("record_id\tsource\tbase_accession\tsequence\tsplit\n")
        f.write(f"T:A:0\tyeast_tile\tA\t{seq_a}\theld_out\n")
        f.write(f"T:B:0\tyeast_tile\tB\t{seq_b}\theld_out\n")
    q = tmp_path / "queries.tsv"
    with open(q, "w") as f:
        f.write("query_id\trecord_id\tkind\tlength\toffset\tn_within\tn_cross\tsub_positions\tshuffle_index\tfrac_hydrophobic\twindow\tsequence\n")
        f.write(f"q1\tT:A:0\treal\t15\t10\t0\t0\t\t\t0.3\t{seq_a[10:25]}\t{seq_a[10:25]}\n")
        f.write(f"q2\tT:A:0\treal\t15\t10\t1\t0\t3\t\t0.3\t{seq_a[10:25]}\t{seq_a[10:25]}\n")
        f.write(f"q3\tT:A:0\tdipeptide\t15\t10\t0\t0\t\t0\t0.3\t{seq_a[10:25]}\tXXXXXXXXXXXXXXX\n")
    reg = tmp_path / "regions.tsv.gz"
    with gzip.open(reg, "wt") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(COLS)
        w.writerow(["q1", "T:A:0", 0, 15, 10, 25, "x", "x", 15, 0, 5, 0.001, "ka", 0.01, 0.001])  # found, placed
        w.writerow(["q2", "T:A:0", 0, 5, 11, 16, "x", "x", 5, 0, 2, 0.5, "run", 0.5, ""])  # found at 1, off diagonal
        w.writerow(["q2", "T:B:0", 0, 5, 0, 5, "x", "x", 5, 0, 2, 0.05, "run", 0.05, ""])  # other protein
        w.writerow(["q3", "T:B:0", 0, 5, 0, 5, "x", "x", 5, 0, 2, 2.0, "run", 2.0, ""])
    subprocess.run([sys.executable, str(REPO / "scripts/353/score.py"), "--split", "held_out", "--records", rec, "--queries", q, "--regions", reg, "--out-dir", tmp_path], check=True)
    real = pl.read_csv(tmp_path / "real_scores.tsv", separator="\t").sort("query_id")
    assert real["found_0.01"].to_list() == [True, False]
    assert real["placed_0.01"].to_list() == [True, False]
    assert real["found_1"].to_list() == [True, True]
    assert real["placed_1"].to_list() == [True, False]
    assert real["n_other_0.1"].to_list() == [0, 1]
    shuf = pl.read_csv(tmp_path / "shuffle_scores.tsv", separator="\t")
    assert shuf["n_regions_1"].to_list() == [0] and shuf["n_regions_10"].to_list() == [1]
    assert shuf["n_other_10"].to_list() == [1]
