"""kmerseek_dark_gain.py: reach in the dark set, in the placed set, at stricter cutoffs,
and on the shuffled null, from the per-chunk lists kmerseekSearch writes."""
import json
import subprocess
import sys
from pathlib import Path

import polars as pl

BIN = Path(__file__).resolve().parent.parent / "bin"
PY = sys.executable


def fasta(path: Path, seqs: dict[str, str]) -> None:
    path.write_text("".join(f">{a}\n{s}\n" for a, s in seqs.items()))


def run_gain(tmp: Path, *, scores: bool, shuffled: bool) -> dict:
    fasta(tmp / "q.fasta", {"A": "MKLLV", "B": "MSTEQ", "C": "MAAAA", "D": "MPPPP"})
    pl.DataFrame({"accession": ["A", "B"]}).write_parquet(tmp / "dark.parquet")
    q = tmp / "queries"
    q.mkdir()
    # hp k23 mask on: reaches A (dark) and C (placed); mask off: A, B, C.
    (q / "chunk_0000.hp_thomas_dill2.k23.lctrue.queries.txt").write_text("A\nC\n")
    (q / "chunk_0000.hp_thomas_dill2.k23.lcfalse.queries.txt").write_text("A\nB\nC\n")
    if scores:
        (q / "chunk_0000.hp_thomas_dill2.k23.lctrue.query_scores.tsv").write_text("A\t1.5\nC\t12.0\n")
        (q / "chunk_0000.hp_thomas_dill2.k23.lcfalse.query_scores.tsv").write_text("A\t1.5\nB\t4.0\nC\t12.0\n")
    sh = tmp / "shuffled"
    sh.mkdir()
    if shuffled:
        (sh / "chunk_0000.hp_thomas_dill2.k23.lctrue.queries.txt").write_text("A\n")
        (sh / "chunk_0000.hp_thomas_dill2.k23.lcfalse.queries.txt").write_text("A\nB\n")
    cmd = [PY, str(BIN / "kmerseek_dark_gain.py"), "--dark", str(tmp / "dark.parquet"),
           "--query", str(tmp / "q.fasta"), "--species", "toy",
           "--queries", *map(str, sorted(q.glob("*.queries.txt"))),
           "--scores", *map(str, sorted(q.glob("*.query_scores.tsv"))),
           "--thresholds", "1.3", "3", "10",
           "--shuffled", *map(str, sorted(sh.glob("*.queries.txt"))),
           "--out", str(tmp / "gain.parquet"), "--summary-out", str(tmp / "gain.json")]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads((tmp / "gain.json").read_text())


def combo(summary: dict, mask: bool) -> dict:
    return next(r for r in summary["by_combo"] if r["low_complexity_mask"] is mask)


def test_dark_and_placed_reach_are_counted_side_by_side(tmp_path):
    s = run_gain(tmp_path, scores=False, shuffled=False)
    on, off = combo(s, True), combo(s, False)
    assert (on["dark_reached"], on["placed_reached"]) == (1, 1)
    assert (off["dark_reached"], off["placed_reached"]) == (2, 1)
    assert s["mask_pairs"][0]["lost_to_masking"] == 1
    assert s["thresholds"] == [] and s["has_shuffled_control"] is False
    assert "dark_reached_at" not in on


def test_reach_at_stricter_cutoffs_comes_from_the_best_score_per_query(tmp_path):
    s = run_gain(tmp_path, scores=True, shuffled=False)
    off = combo(s, False)
    # A scores 1.5, B 4.0 (dark); C 12.0 (placed).
    assert off["dark_reached_at"] == {"1.3": 2, "3.0": 1, "10.0": 0}
    assert off["placed_reached_at"] == {"1.3": 1, "3.0": 1, "10.0": 1}
    assert s["thresholds"] == [1.3, 3.0, 10.0]
    # The parquet keeps the flat columns only.
    df = pl.read_parquet(tmp_path / "gain.parquet")
    assert "dark_reached_at" not in df.columns and df.height == 2


def test_the_shuffled_null_is_counted_against_the_same_dark_set(tmp_path):
    s = run_gain(tmp_path, scores=True, shuffled=True)
    assert s["has_shuffled_control"] is True
    assert combo(s, True)["shuffled_dark_reached"] == 1
    assert combo(s, False)["shuffled_dark_reached"] == 2
    assert combo(s, False)["fraction_shuffled_dark_reached"] == 1.0


def test_shuffle_keeps_length_composition_and_accessions(tmp_path):
    fasta(tmp_path / "q.fasta", {"A": "MKLLVAAAGG", "B": "MSTEQ", "C": "MAAAA"})
    pl.DataFrame({"accession": ["A", "B"]}).write_parquet(tmp_path / "dark.parquet")
    out = tmp_path / "sh"
    subprocess.run([PY, str(BIN / "shuffle_dark_queries.py"), "--query", str(tmp_path / "q.fasta"),
                    "--dark", str(tmp_path / "dark.parquet"), "--outdir", str(out),
                    "--chunk-size", "1"], check=True, capture_output=True, text=True)
    files = sorted(out.glob("chunk_*.fasta"))
    assert [f.name for f in files] == ["chunk_0000.fasta", "chunk_0001.fasta"]
    got = {}
    for f in files:
        lines = f.read_text().splitlines()
        got[lines[0][1:]] = lines[1]
    assert set(got) == {"A", "B"}, "only the dark proteins, under their own accessions"
    assert sorted(got["A"]) == sorted("MKLLVAAAGG") and len(got["A"]) == 10
    assert sorted(got["B"]) == sorted("MSTEQ")
