"""Scoring one series with k-mers chosen on another, on small made-up series whose answers are known."""

import csv
import gzip
import subprocess
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parents[1] / "fig4" / "cross_series_auc.py"


def write_series(folder, name, people, rng, rise=0.0, shift=0.0):
    """A KaMRaT table and design for pairs of samples, and a series matrix naming each sample's person.

    The first 40 of 2000 k-mers rise by `rise` after exercise; every k-mer rises by `shift`, as a scan day would do."""
    samples = [f"GSM{name}{i:02d}" for i in range(people * 2)]
    post = np.array([False, True] * people)
    values = rng.normal(8, 1, (2000, len(samples)))
    values[:40][:, post] += rise
    values[:, post] += shift
    with open(folder / f"{name}.table.tsv", "w") as f:
        f.write("kmer\t" + "\t".join(samples) + "\n")
        for k, row in enumerate(values):
            f.write(f"K{k:05d}\t" + "\t".join(f"{2**v - 1:.3f}" for v in row) + "\n")
    with open(folder / f"{name}.design.tsv", "w") as f:
        f.writelines(f"{s}\t{'post' if p else 'pre'}\n" for s, p in zip(samples, post, strict=True))
    (folder / "series" / "GSE28998").mkdir(parents=True, exist_ok=True)
    return samples


def run(tmp_path, scored_rise, scored_shift):
    rng = np.random.default_rng(3)
    write_series(tmp_path, "train", 6, rng, rise=2)
    samples = write_series(tmp_path, "scored", 5, rng, rise=scored_rise, shift=scored_shift)
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
            str(SCRIPT),
            f"{t}/train.table.tsv",
            f"{t}/train.design.tsv",
            f"{t}/scored.table.tsv",
            f"{t}/scored.design.tsv",
            f"{t}/series",
            "TRAIN",
            "GSE28998",
            "main",
            "50",
            f"{t}/out",
        ],
        check=True,
        env={"PYTHONPATH": str(SCRIPT.parents[1])},
    )
    with open(tmp_path / "out.cross.tsv") as f:
        return {r["kmers"]: r for r in csv.DictReader(f, delimiter="\t")}


def test_kmers_that_rise_in_both_series_carry_over(tmp_path):
    row = run(tmp_path, scored_rise=2, scored_shift=0)["100"]
    assert float(row["auc"]) > 0.9
    assert float(row["p_labels_swapped"]) < 0.05
    assert float(row["p_random_kmers"]) < 0.05
    assert row["people_post_above_pre"] == "5"


def test_a_shift_of_the_whole_scored_series_is_matched_by_random_kmers(tmp_path):
    row = run(tmp_path, scored_rise=0, scored_shift=1)["100"]
    assert float(row["p_labels_swapped"]) < 0.05
    assert float(row["p_random_kmers"]) > 0.05
