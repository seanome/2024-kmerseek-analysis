"""--scored-calls must give back exactly the metrics the full scoring path wrote.

The mode exists to rescore finished runs against a corrected answer key (Pfam-N cut to the
run's query proteins, PR 165) from their published call files, without rerunning searches.
It skips region loading, transfer and call scoring and reuses everything after. If it
drifted from the full path, a rescore would silently change numbers that have nothing to do
with the answer key. So: score a fixture arm the normal way, feed its call files back in,
and require every metrics row and every column to match.
"""

import subprocess
import sys

import polars as pl

from conftest import BIN, score, write_perfect_regions


def test_scored_calls_reproduces_every_metric(pfam_truth, tmp_path):
    regions = tmp_path / "perfect.regions.tsv"
    write_perfect_regions(pfam_truth["truth"], pfam_truth["map"], regions, pad=7)
    full = tmp_path / "full"
    first = score(pfam_truth["truth"], pfam_truth["map"], regions, full, "pfamn")
    assert first["stratum_axis"].n_unique() > 1  # strata, not only the whole set

    calls = sorted(full.glob("pfamn.*.calls.parquet"))
    assert len(calls) == 2  # dedup off and on
    manifest = tmp_path / "scored.tsv"
    manifest.write_text(
        "".join(
            f"perfect\tdefault\t{c}\t{'on' if c.name.endswith('.dedup.calls.parquet') else 'off'}\n"
            for c in calls
        )
    )
    again = tmp_path / "again"
    again.mkdir()
    subprocess.run(
        [
            sys.executable,
            str(BIN / "evaluate_domain_calls.py"),
            "--manifest",
            str(manifest),
            "--scored-calls",
            "--species",
            "yeast",
            "--species-mya",
            "1000",
            "--truth",
            str(pfam_truth["truth"]),
            "--domain-map",
            str(pfam_truth["map"]),
            "--truth-set",
            "pfamn",
        ],
        cwd=again,
        capture_output=True,
        text=True,
        check=True,
    )
    second = pl.concat(
        [pl.read_parquet(f) for f in sorted(again.glob("*.metrics.parquet"))],
        how="diagonal_relaxed",
    )
    assert not list(again.glob("*.calls.parquet"))  # the calls are read, not rewritten

    keys = ["dedup_transfers", "split", "stratum_axis", "stratum"]
    a = first.sort(keys)
    b = second.select(a.columns).sort(keys)
    assert a.equals(b), "scored-calls metrics differ from the full scoring path"

    for name in (
        "pfamn.perfect.default.yeast.curve.parquet",
        "pfamn.perfect.default.yeast.dedup.curve.parquet",
    ):
        assert pl.read_parquet(full / name).equals(pl.read_parquet(again / name))
