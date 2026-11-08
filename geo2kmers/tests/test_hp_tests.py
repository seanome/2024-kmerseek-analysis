"""HP rows that stand for several k-mers, on small made-up tables: the scores must equal those from the same k-mers
written out one per row."""

import csv
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np

from geo2kmers import held_out as ho

FIG4 = Path(__file__).parents[1] / "fig4"


def grouped_and_expanded(rng, people=6, rows=60):
    """Rows of values, each standing for 1 to 400 k-mers, and the same k-mers one per row."""
    values = rng.normal(8, 1, (rows, people * 2))
    post = np.array([False, True] * people)
    values[:5][:, post] += 1.5
    weights = rng.integers(1, 400, rows)
    return values, post, np.repeat(np.arange(people), 2), weights, np.repeat(values, weights, axis=0)


def test_first_kmers_cuts_the_row_that_reaches_the_size():
    assert ho.first_kmers([400, 400, 400], 1000).tolist() == [400, 400, 200]
    assert ho.first_kmers([5, 5], 1000).tolist() == [5, 5]
    assert ho.first_kmers([1000, 3], 1000).tolist() == [1000, 0]


def test_grouped_rows_score_as_the_expanded_kmers():
    values, post, people, weights, expanded = grouped_and_expanded(np.random.default_rng(4))
    sizes = (100, 1000, 10000)
    grouped, _ = ho.held_out_scores(values, values * values, post, people, sizes, weights)
    one_per_row, _ = ho.held_out_scores(expanded, expanded * expanded, post, people, sizes)
    for size in sizes:
        assert np.allclose(grouped[size], one_per_row[size])


def test_read_table_keeps_only_the_designs_samples_and_does_not_log_hp_values(tmp_path):
    with gzip.open(tmp_path / "t.tsv.gz", "wt") as f:
        f.write("feature\tkmers\tprobe_sets\tGSM1\tGSM9\tGSM2\nr0\t3\tA,B\t1.5\t99\t2.5\n")
    (tmp_path / "d.tsv").write_text("GSM1\tpre\nGSM2\tpost\n")
    names, samples, values, post, weights, probe_sets = ho.read_table(tmp_path / "t.tsv.gz", tmp_path / "d.tsv")
    assert (names, samples, values.tolist(), post.tolist()) == (["r0"], ["GSM1", "GSM2"], [[1.5, 2.5]], [False, True])
    assert (weights.tolist(), probe_sets) == ([3], ["A,B"])


def test_value_table_gives_one_row_per_set_of_probe_sets(tmp_path):
    # k-mers 0-2 in probe set 0, k-mer 3 in probe set 1, k-mers 4-5 in probe sets 0 and 1, k-mer 6 in probe set 2
    sharing = np.array([1, 1, 1, 1, 2, 2, 1])
    owner = np.array([0, 0, 0, 1, 0, 1, 0, 1, 2], dtype=np.int32)
    np.savez(
        tmp_path / "A.hp.kmers.npz",
        kmer=np.arange(7, dtype=np.uint64),
        start=np.concatenate([[0], np.cumsum(sharing)[:-1]]),
        probe_sets=sharing,
        owner=owner,
        names=np.array(["PS0", "PS1", "PS2"]),
    )
    with gzip.open(tmp_path / "values.tsv.gz", "wt") as f:
        f.write("probe_set\tGSM1\tGSM2\nPS0\t2\t4\nPS1\t6\t10\nPS2\t3\t3\n")
    subprocess.run(
        [
            sys.executable,
            str(FIG4 / "hp_value_table.py"),
            *(str(tmp_path / n) for n in ("A.hp.kmers.npz", "values.tsv.gz", "out.tsv.gz")),
        ],
        check=True,
    )
    with gzip.open(tmp_path / "out.tsv.gz", "rt") as f:
        rows = [(r["kmers"], r["probe_sets"], r["GSM1"], r["GSM2"]) for r in csv.DictReader(f, delimiter="\t")]
    # PS2's k-mer is the same in every sample and is left out
    assert rows == [
        ("3", "PS0", "2.0000", "4.0000"),
        ("1", "PS1", "6.0000", "10.0000"),
        ("2", "PS0,PS1", "4.0000", "7.0000"),
    ]


def write_hp_series(folder, name, people, rng, rise):
    """An HP value table whose first 5 of 300 rows rise after exercise, and its design."""
    samples = [f"GSM{name}{i:02d}" for i in range(people * 2)]
    post = np.array([False, True] * people)
    values = rng.normal(8, 1, (300, len(samples)))
    values[:5][:, post] += rise
    weights = np.random.default_rng(9).integers(1, 400, 300)  # the same rows on one array
    with gzip.open(folder / f"{name}.tsv.gz", "wt") as f:
        f.write("feature\tkmers\tprobe_sets\t" + "\t".join(samples) + "\n")
        for r, row in enumerate(values):
            f.write(f"r{r}\t{weights[r]}\tPS{r}\t" + "\t".join(f"{v:.4f}" for v in row) + "\n")
    with open(folder / f"{name}.design.tsv", "w") as f:
        f.writelines(f"{s}\t{'post' if p else 'pre'}\n" for s, p in zip(samples, post, strict=True))
    return samples


def run_cross(tmp_path, scored_rise):
    rng = np.random.default_rng(3)
    write_hp_series(tmp_path, "train", 6, rng, rise=3)
    samples = write_hp_series(tmp_path, "scored", 5, rng, rise=scored_rise)
    (tmp_path / "series" / "GSE28998").mkdir(parents=True)
    with gzip.open(tmp_path / "series/GSE28998/GSE28998_series_matrix.txt.gz", "wt") as f:
        f.write("!Sample_title\t" + "\t".join(f'"{s}"' for s in samples) + "\n")
        f.write("!Sample_geo_accession\t" + "\t".join(f'"{s}"' for s in samples) + "\n")
        f.write(
            "!Sample_characteristics_ch1\t"
            + "\t".join(f'"subject identifier: P{i // 2}"' for i in range(len(samples)))
            + "\n"
        )
    t = str(tmp_path)
    subprocess.run(
        [
            sys.executable,
            str(FIG4 / "cross_series_auc.py"),
            f"{t}/train.tsv.gz",
            f"{t}/train.design.tsv",
            f"{t}/scored.tsv.gz",
            f"{t}/scored.design.tsv",
            f"{t}/series",
            "TRAIN",
            "GSE28998",
            "main",
            "50",
            f"{t}/out",
        ],
        check=True,
        env={"PYTHONPATH": str(FIG4.parent)},
    )
    with open(tmp_path / "out.cross.tsv") as f:
        return {r["kmers"]: r for r in csv.DictReader(f, delimiter="\t")}


def test_hp_rows_that_rise_in_both_series_carry_over(tmp_path):
    rows = run_cross(tmp_path, scored_rise=3)
    assert float(rows["100"]["auc"]) > 0.9
    assert float(rows["100"]["p_random_kmers"]) < 0.05
    assert int(rows["100"]["probe_sets_behind"]) <= 5  # 100 k-mers come from at most the 5 rising rows
    weights = np.random.default_rng(9).integers(1, 400, 300)
    assert rows["1000"]["kmers_shared"] == str(weights.sum())


def test_hp_rows_with_no_rise_in_the_scored_series_do_not_carry_over(tmp_path):
    rows = run_cross(tmp_path, scored_rise=0)
    assert float(rows["1000"]["p_random_kmers"]) > 0.05
