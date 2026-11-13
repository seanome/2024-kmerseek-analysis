"""HP k-mers from probe-set proteins, on two made-up probe sets that share one 30-residue stretch."""

import csv
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "fig4" / "hp_kmer_table.py"
SHARED = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"  # 30 residues
LEHNINGER_H = set("AFGILMPVWY")


def letters(protein):
    return "".join("h" if r in LEHNINGER_H else "p" for r in protein)


def test_shared_stretch_gets_the_mean_of_both_probe_sets(tmp_path):
    first = "MKT" + SHARED  # 33 residues: 4 windows
    second = SHARED + "XWW"  # the X stops the windows that reach it: 1 window
    (tmp_path / "p.fasta").write_text(f">P1 NM_1\n{first}\n>P2 NM_2\n{second}\n>P3 NM_3\n{'A' * 20}\n")
    with gzip.open(tmp_path / "pairs.tsv.gz", "wt") as f:
        f.write("array\tprobe_set\tsource\tsequence\tlength\n")
        f.write(
            f"A 1\tTC1\trefseq_mrna\tP1\t33\nA 1\tTC2\trefseq_mrna\tP2\t33\nA 1\tTC3\trefseq_mrna\tP3\t20\nA 1\tTC4\test\tAI1\t500\n"
        )
    (tmp_path / "gpl.txt").write_text("ID\tprobeset_id\nTC1\t11\nTC2\t22\nTC3\t33\n")
    with gzip.open(tmp_path / "GSE1_series_matrix.txt.gz", "wt") as f:
        f.write(
            '!series_matrix_table_begin\n"ID_REF"\t"GSM1"\t"GSM2"\n"11"\t2\t4\n"22"\t6\t10\n"33"\t1\t1\n!series_matrix_table_end\n'
        )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(tmp_path / "pairs.tsv.gz"),
            "A 1",
            str(tmp_path / "gpl.txt"),
            str(tmp_path),
            str(tmp_path / "p.fasta"),
            str(tmp_path / "GSE1_series_matrix.txt.gz"),
        ],
        check=True,
    )

    with open(tmp_path / "A_1.hp_lehninger2.summary.tsv") as f:
        summary = next(csv.DictReader(f, delimiter="\t"))
    expected = {letters(first[i : i + 30]) for i in range(4)} | {letters(SHARED)}
    assert int(summary["distinct_kmers"]) == len(expected)
    assert summary["probe_sets_with_a_kmer"] == "2"  # TC3's protein is too short, TC4 is an EST
    with gzip.open(tmp_path / "GSE1.hp_lehninger2.shared_kmers.tsv.gz", "rt") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert [(r["kmer"], r["probe_sets"], r["GSM1"], r["GSM2"]) for r in rows] == [
        (letters(SHARED), "2", "4.0000", "7.0000")
    ]
    saved = np.load(tmp_path / "A_1.hp_lehninger2.kmers.npz")
    assert sorted(saved["names"]) == ["TC1", "TC2", "TC3"]
