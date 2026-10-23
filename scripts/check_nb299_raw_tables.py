"""Independently check notebook 299's result using CSV rows and Python sets.

Run from any directory. Does not import the notebook's scoring helper.
"""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "notebooks/299_data"


def check_raw_tables():
    """Raise on disagreement; return the independently counted summary."""
    with (DATA / "queries.tsv").open() as handle:
        queries = list(csv.DictReader(handle, delimiter="\t"))
    real = {r["query_id"] for r in queries if r["shuffled"] == "false"}
    shuffled = {r["query_id"] for r in queries if r["shuffled"] == "true"}
    truth, decoy = set(), set()
    with (DATA / "sherlock/diamond_hits.tsv").open() as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if float(row[8]) < 1e-5:
                assert row[0] in real | shuffled, row[0]
                (truth if row[0] in real else decoy).add((row[0], row[1]))
    with (
        ROOT / "tables/299_kill_test_recall_vs_shuffled_rate_omrgc_v2.csv"
    ).open() as handle:
        table = list(csv.DictReader(handle))
    settings = [r for r in table if r["method"] == "sourmash manysearch"]
    expected = {
        f"{a}.k{k}.s{s}"
        for a in ("protein", "dayhoff", "hp")
        for k in (7, 10, 13)
        for s in (10, 100, 1_000)
    }
    assert (
        len(settings) == len(expected) and {r["setting"] for r in settings} == expected
    )
    passing = []
    for row in settings:
        folder = DATA / "sherlock/search" / row["setting"]
        with (folder / "recall.pairs.tsv").open() as handle:
            found = {
                (r[0], r[1])
                for r in csv.reader(handle, delimiter="\t")
                if float(r[2]) < 1e-5
            } & truth
        count_file = (
            "full.counts.tsv" if row["count_source"] == "full" else "sample.counts.tsv"
        )
        with (folder / count_file).open() as handle:
            counts = list(csv.reader(handle, delimiter="\t"))
        assert all(q in real | shuffled for q, _ in counts)
        n_real = sum(int(n) for q, n in counts if q in real)
        n_shuffled = sum(int(n) for q, n in counts if q in shuffled)
        ratio = n_shuffled / (20 * n_real) if n_real else None
        recall = len(found) / len(truth)
        assert len(found) == int(row["truth_hits_found"]), row["setting"]
        assert math.isclose(recall, float(row["recall"])), row["setting"]
        if ratio is None:
            assert not row["shuffled_rate"], row["setting"]
        else:
            assert math.isclose(ratio, float(row["shuffled_rate"])), row["setting"]
        passes = recall >= 0.8 and ratio is not None and ratio < 0.01
        assert passes == (row["passes"] == "true"), row["setting"]
        if passes:
            passing.append(row["setting"])
    return dict(
        real_queries=len(real),
        shuffled_queries=len(shuffled),
        truth_pairs=len(truth),
        queries_with_truth=len({q for q, _ in truth}),
        shuffled_diamond_pairs=len(decoy),
        settings_checked=len(settings),
        passing_settings=passing,
    )


if __name__ == "__main__":
    import json

    print(json.dumps(check_raw_tables(), indent=2))
