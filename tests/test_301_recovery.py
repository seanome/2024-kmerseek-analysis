"""Regression checks for tied decoy E-values and interrupted index recovery."""

import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "scoring", ROOT / "scripts/score_301_completed.py"
)
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


def test_threshold_strict_and_tied():
    assert scoring.threshold([0] * 55, 1356) == 0
    assert scoring.threshold([0.1] * 14, 1356) == 0.1
    assert scoring.threshold([0.1] * 13, 1356) == 10
    assert scoring.threshold([0] * 2, 200) == 0


def test_interrupted_index_is_rebuilt_and_finished_chunk_reused(tmp_path):
    import os
    import gzip
    import json

    repo = tmp_path / "repo"
    (repo / "notebooks").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "notebooks/241_plan.json").write_text(
        json.dumps([dict(alphabet="protein20", ksize=5, penalty="0.14", xdrop="0.57")])
    )
    (repo / "scripts/filter_301_regions.py").write_text(
        (ROOT / "scripts/filter_301_regions.py").read_text()
    )
    base = tmp_path / "base"
    (base / "step2").mkdir(parents=True)
    (base / "step3").mkdir()
    d = base / "step5/protein20/protein20.k5.s1.lctrue.c0.14.x0.57"
    (d / "idx").mkdir(parents=True)
    (d / "idx/CURRENT").touch()
    (d / "index.log").write_text("killed during calibration")
    (base / "step2/swissprot_minus_query_clusters.fasta").write_text(">T\nAAAAA\n")
    (base / "step2/truth.tsv").write_text("query_id\tstart\tend\nQ\t1\t5\n")
    (base / "step3/decoys.tsv").write_text("decoy_id\tstart\tend\tunchanged\n")
    (base / "step5/subset_queries.fasta").write_text(">Q\nAAAAA\n")
    # Finished chunk was promoted only after the writer and filter succeeded.
    with gzip.open(d / "regions_filtered.chunk000.tsv.gz", "wt") as f:
        f.write("kept\n")
    (d / "filter.chunk000.log").write_text("finished filter\n")
    (d / "search.chunk000.time").write_text(
        "Elapsed: 0:01.00\nMaximum resident: 1024\n"
    )
    binary = tmp_path / "fake_kmerseek"
    binary.write_text("""#!/bin/bash
[ "$1" = index ] || exit 99
while [ $# -gt 0 ]; do if [ "$1" = -o ]; then shift; mkdir -p "$1"; touch "$1/CURRENT"; fi; shift; done
echo 'fitted now'
echo 'Indexing completed successfully!'
""")
    binary.chmod(0o755)
    # GNU time is present on Linux; use a portable stand-in on macOS.
    timer = tmp_path / "time"
    timer.write_text("""#!/bin/bash
shift; shift; out=$1; shift
"$@"; rc=$?
printf 'Elapsed: 0:01.00\nMaximum resident: 1024\n' > "$out"
exit $rc
""")
    timer.chmod(0o755)
    env = dict(
        os.environ,
        SCRATCH=str(tmp_path),
        SLURM_SUBMIT_DIR=str(repo),
        ALPHABET="protein20",
        KMERSEEK=str(binary),
        NB301_DIR=str(base),
        TIME_BIN=str(timer),
        CHUNK="100",
    )
    subprocess.run(
        ["bash", str(ROOT / "scripts/301_kmerseek_alphabet.sbatch")],
        env=env,
        check=True,
        capture_output=True,
    )
    assert not list(d.glob("idx.failed.*"))
    assert (d / "index.log.failed.test").read_text() == "killed during calibration"
    assert "fitted" in (d.parent / "status.tsv").read_text()
    assert gzip.open(d / "regions_filtered.chunk000.tsv.gz", "rt").read() == "kept\n"


def test_real_snapshot_zero_evalue_groups_exceed_one_percent():
    import csv
    import gzip

    raw = ROOT / "data/301_short_feature_boundaries/partial/raw/step5"
    for alphabet, expected in [("protein20", 55), ("uniprot18", 45), ("wass14", 35)]:
        queries = set()
        for path in (raw / alphabet).glob("*/regions_filtered.chunk*.tsv.gz"):
            with gzip.open(path, "rt") as f:
                for row in csv.DictReader(f, delimiter="\t"):
                    if (
                        row["query_id"].startswith("decoy_")
                        and float(row["region_evalue"]) == 0
                    ):
                        queries.add(row["query_id"])
        assert len(queries) == expected
        assert len(queries) / 1356 > 0.01
