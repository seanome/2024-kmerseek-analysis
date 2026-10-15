"""Tests for scripts/build_leaderboard_tables.py: trace parsing and the no-invented-rows rule."""

import importlib.util
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_leaderboard_tables.py"
spec = importlib.util.spec_from_file_location("blt", SCRIPT)
blt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(blt)


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("4.4s", 4.4),
        ("1m 6s", 66.0),
        ("1h 2m 3s", 3723.0),
        ("34ms", 0.034),
        ("1d 1h", 90000.0),
        ("-", None),
        (None, None),
    ],
)
def test_duration_seconds(text, seconds):
    got = blt.duration_seconds(text)
    if seconds is None:
        assert got is None
    else:
        assert got == pytest.approx(seconds)


def test_size_gb():
    assert blt.size_gb("1 GB") == pytest.approx(1.0)
    assert blt.size_gb("512 MB") == pytest.approx(0.5)
    assert blt.size_gb("-") is None


@pytest.mark.parametrize(
    "process, tag, expected",
    [
        (
            "kmerseekSearch",
            "mouse_hp_thomas_dill2_k24_lcfalse",
            ("kmerseek", "hp_thomas_dill2_k24_lcFalse", "mouse"),
        ),
        (
            "kmerseekSearch",
            "ecoli_protein20_k7_lctrue",
            ("kmerseek", "protein20_k7_lcTrue", "ecoli"),
        ),
        (
            "mmseqs2Search",
            "human_vs_fly [mmseqs2_iterative]",
            ("mmseqs2_iterative", "s7", "fly"),
        ),
        (
            "phmmerSearch",
            "human_vs_zebrafish",
            ("hmmer3_phmmer", "default", "zebrafish"),
        ),
        ("hmmscanAnnotate", "hmmscan_human", ("hmmscan", "direct", "all")),
        ("foldseekDb", "fly", None),
    ],
)
def test_tool_setting_species(process, tag, expected):
    assert blt.tool_setting_species(process, tag) == expected


def trace_row(complete, realtime, cpu, status="COMPLETED"):
    return {
        "process": "QFO:phmmerSearch",
        "tag": "human_vs_fly",
        "status": status,
        "hash": complete,
        "complete": complete,
        "cpus": "8",
        "realtime": realtime,
        "%cpu": cpu,
        "peak_rss": "2 GB",
        "trace_file": "t",
    }


def test_build_compute_keeps_latest_completion_and_skips_cached():
    traces = pl.DataFrame(
        [
            trace_row("2026-09-01 10:00:00.000", "1h", "400%"),
            trace_row("2026-09-05 10:00:00.000", "30m", "200%"),
            trace_row("2026-09-09 10:00:00.000", "9h", "800%", status="CACHED"),
        ]
    )
    out = blt.build_compute(traces, n_queries=500)
    assert out.height == 1
    row = out.row(0, named=True)
    assert row["cpu_h"] == pytest.approx(1.0)  # 0.5 h x 200 %
    assert row["cpu_h_per_1000_queries"] == pytest.approx(2.0)
    assert row["cpu_h_reserved"] == pytest.approx(4.0)  # 0.5 h x 8 cpus
    assert row["provenance"] == "measured" and row["cpu_h_reported"] is None


def test_coverage_marks_missing_bins_not_run():
    results = pl.DataFrame(
        {
            "truth_set": ["pfam"],
            "dedup": [False],
            "tool": ["phmmer"],
            "setting": ["d"],
            "species": ["fly"],
            "species_mya": [600.0],
            "stratum_axis": ["identity"],
            "stratum": ["20-30%"],
        }
    )
    md = blt.coverage_markdown(results, ["fly"])
    assert "| identity | 20-30% | ■ |" in md
    assert "| identity | 0-20% | □ |" in md
    assert "| feature_length_bin | whole_protein | □ |" in md
