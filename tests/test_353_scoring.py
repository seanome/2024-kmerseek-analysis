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
        assert all(
            encode(seq[i], "dayhoff6") == encode(window[i], "dayhoff6") for i in within
        )
        assert all(
            encode(seq[i], "dayhoff6") != encode(window[i], "dayhoff6") for i in cross
        )


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
        f.write(
            "query_id\trecord_id\tkind\tlength\toffset\tn_within\tn_cross\tsub_positions\tshuffle_index\tfrac_hydrophobic\twindow\tsequence\n"
        )
        f.write(
            f"q1\tT:A:0\treal\t15\t10\t0\t0\t\t\t0.3\t{seq_a[10:25]}\t{seq_a[10:25]}\n"
        )
        f.write(
            f"q2\tT:A:0\treal\t15\t10\t1\t0\t3\t\t0.3\t{seq_a[10:25]}\t{seq_a[10:25]}\n"
        )
        f.write(
            f"q3\tT:A:0\tdipeptide\t15\t10\t0\t0\t\t0\t0.3\t{seq_a[10:25]}\tXXXXXXXXXXXXXXX\n"
        )
    reg = tmp_path / "regions.tsv.gz"
    with gzip.open(reg, "wt") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(COLS)
        w.writerow(
            ["q1", "T:A:0", 0, 15, 10, 25, "x", "x", 15, 0, 5, 0.001, "ka", 0.01, 0.001]
        )  # found, placed
        w.writerow(
            ["q2", "T:A:0", 0, 5, 11, 16, "x", "x", 5, 0, 2, 0.5, "run", 0.5, ""]
        )  # found at 1, off diagonal
        w.writerow(
            ["q2", "T:B:0", 0, 5, 0, 5, "x", "x", 5, 0, 2, 0.05, "run", 0.05, ""]
        )  # other protein
        w.writerow(["q3", "T:B:0", 0, 5, 0, 5, "x", "x", 5, 0, 2, 2.0, "run", 2.0, ""])
    subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/353/score.py"),
            "--split",
            "held_out",
            "--records",
            rec,
            "--queries",
            q,
            "--regions",
            reg,
            "--out-dir",
            tmp_path,
        ],
        check=True,
    )
    real = pl.read_csv(tmp_path / "real_scores.tsv", separator="\t").sort("query_id")
    assert real["found_0.01"].to_list() == [True, False]
    assert real["placed_0.01"].to_list() == [True, False]
    assert real["found_1"].to_list() == [True, True]
    assert real["placed_1"].to_list() == [True, False]
    assert real["n_other_0.1"].to_list() == [0, 1]
    shuf = pl.read_csv(tmp_path / "shuffle_scores.tsv", separator="\t")
    assert shuf["n_regions_1"].to_list() == [0] and shuf["n_regions_10"].to_list() == [
        1
    ]
    assert shuf["n_other_10"].to_list() == [1]


def test_stage1_sheet_blinds_dedups_and_scores(tmp_path):
    stage1 = REPO / "scripts/353/stage1.py"
    qlog = tmp_path / "query_log.tsv"
    with open(qlog, "w") as f:
        f.write(
            "query_number\treceived_at\twork_item\tprotein_accession\tstart\tend\tsequence\tchosen_before_results\tnotes\n"
        )
        f.write("1\t2026-10-21\tx\tP1\t10\t24\tMSEYQPSLFALNPMG\tyes\t\n")
    cols = [
        "query_number",
        "method",
        "record_id",
        "query_start",
        "query_end",
        "query_segment",
        "record_segment",
        "match_line",
        "identical",
        "aligned",
        "evalue",
        "record_coordinates",
        "measurement",
        "reference",
        "record_accession",
    ]
    rows = [
        [
            "1",
            "kmerseek",
            "T:P2:0",
            "1",
            "15",
            "MSEYQPSLFALNPMG",
            "MSEYQPSLFALNPMG",
            "|" * 15,
            "15",
            "15",
            "0.01",
            "P2 1-15",
            "act 5",
            "doi:x",
            "P2",
        ],
        [
            "1",
            "comparison:blastp",
            "T:P2:0",
            "1",
            "15",
            "MSEYQPSLFALNPMG",
            "MSEYQPSLFALNPMG",
            "|" * 15,
            "15",
            "15",
            "0.02",
            "P2 1-15",
            "act 5",
            "doi:x",
            "P2",
        ],
        [
            "1",
            "kmerseek",
            "E:X",
            "1",
            "15",
            "MSEYQPSLFALNPMG",
            "MSEYQPSLFALNPMG",
            "|" * 15,
            "15",
            "15",
            "0.01",
            "P1 12-26",
            "self",
            "pmid",
            "P1",
        ],
        [
            "1",
            "kmerseek",
            "E:Y",
            "1",
            "15",
            "MSEYQPSLFALNPMG",
            "MTEYQPSLFALNPMG",
            "|" * 15,
            "14",
            "15",
            "0.05",
            "P3 5-19",
            "motif",
            "pmid",
            "P3",
        ],
    ]
    cards = tmp_path / "cards.tsv"
    with open(cards, "w") as f:
        f.write("\t".join(cols) + "\n" + "\n".join("\t".join(r) for r in rows) + "\n")
    subprocess.run(
        [
            sys.executable,
            str(stage1),
            "sheet",
            "--cards",
            cards,
            "--query-log",
            qlog,
            "--out-dir",
            tmp_path,
        ],
        check=True,
    )
    sheet = list(csv.DictReader(open(tmp_path / "review_sheet.tsv"), delimiter="\t"))
    key = {
        r["blind_id"]: r["found_by"]
        for r in csv.DictReader(open(tmp_path / "review_key.tsv"), delimiter="\t")
    }
    assert len(sheet) == 2  # the self-record is dropped, the duplicate merged
    assert "method" not in sheet[0] and "found_by" not in sheet[0]
    assert sorted(key.values()) == ["comparison+kmerseek", "kmerseek"]
    # The judge marks the kmerseek-only card useful.
    for r in sheet:
        only_k = key[r["blind_id"]] == "kmerseek"
        r["analyst_had_seen_it (yes/no)"] = "no"
        r["kept_for_investigation (yes/no)"] = "yes" if only_k else "no"
        r["one_line_why"] = "new motif evidence" if only_k else ""
    with open(tmp_path / "filled.tsv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]), delimiter="\t")
        w.writeheader()
        w.writerows(sheet)
    with open(tmp_path / "stopwatch.tsv", "w") as f:
        f.write(
            "query_number\tworkflow\tused_first\tstart_time\tstop_time\tseconds\tanalyst\tnotes\n1\tkmerseek\tyes\t\t\t40\tO\t\n1\tcomparison\tno\t\t\t100\tO\t\n"
        )
    out = subprocess.run(
        [
            sys.executable,
            str(stage1),
            "score",
            "--sheet",
            tmp_path / "filled.tsv",
            "--key",
            tmp_path / "review_key.tsv",
            "--stopwatch",
            tmp_path / "stopwatch.tsv",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "useful measurement from kmerseek: 1 (threshold >= 20): fails" in out
    assert "kmerseek 40 s vs comparison 100 s" in out and out.strip().endswith("FAIL")
