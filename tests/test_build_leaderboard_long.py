"""Tests for scripts/build_leaderboard_long.py: trace parsing, long-format reshaping, and
the rules that keep a number from being counted twice or filled in."""

import importlib.util
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_leaderboard_long.py"
spec = importlib.util.spec_from_file_location("bll", SCRIPT)
bll = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bll)


@pytest.mark.parametrize(
    "text, seconds",
    [
        ("4.4s", 4.4),
        ("1m 6s", 66.0),
        ("1h 2m 3s", 3723.0),
        ("34ms", 0.034),
        ("-", None),
    ],
)
def test_duration_seconds(text, seconds):
    got = bll.duration_seconds(text)
    assert got == (pytest.approx(seconds) if seconds is not None else None)


def test_image_name():
    path = "/scratch/users/olgabot/apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img"
    assert bll.image_name(path) == "docker.io-olgabot-kmerseek-0.4.0-rc5"
    assert bll.image_name("-") is None


@pytest.mark.parametrize(
    "step, tag, style, expected",
    [
        (
            "kmerseekSearch",
            "mouse_hp_thomas_dill2_k24_lcfalse",
            "metrics",
            ("kmerseek", "hp_thomas_dill2_k24_lcFalse", "mouse"),
        ),
        (
            "kmerseekSearch",
            "chicken_sdm12_k7.s5_lctrue",
            "elm",
            ("kmerseek", "sdm12.k7.s5.lctrue", "chicken"),
        ),
        (
            "kmerseekIndex",
            "yeast_protein20_k5_lcfalse",
            "elm",
            ("kmerseek", "protein20.k5.lcfalse", "yeast"),
        ),
        (
            "mmseqs2Search",
            "human_vs_fly [mmseqs2_iterative]",
            "metrics",
            ("mmseqs2_iterative", "s7", "fly"),
        ),
        (
            "jackhmmerSearch",
            "human_vs_worm",
            "metrics",
            ("hmmer3_jackhmmer", "n3", "worm"),
        ),
        (
            "jackhmmerSearch",
            "human_vs_worm",
            "elm",
            ("hmmer3_jackhmmer", "default", "worm"),
        ),
        (
            "prostt5Db",
            "human-c9f92874",
            "metrics",
            ("prostt5", "database build", "human-c9f92874"),
        ),
        ("buildDomainTruth", "-", "metrics", None),
    ],
)
def test_qfo_task(step, tag, style, expected):
    assert bll.qfo_task(step, tag, style) == expected


def test_dark_and_scope_tags():
    assert bll.dark_task(
        "kmerseekSearch", "worm.chunk_0001.protein20.k5.s1.lcfalse.extend-c0.14"
    ) == ("kmerseek", "protein20.k5.s1.lcfalse.extend-c0.14", "worm")
    assert bll.scope_task("searchAndScore", "wass14.k8.c_min") == (
        "kmerseek",
        "wass14.k8.c_min",
        "scope40",
    )
    assert bll.twilight_task("kmerseekArm", "funcgroups8 k12") == (
        "kmerseek",
        "funcgroups8.k12",
        "db",
    )


def task(target, complete, wall, cores=8, step="phmmerSearch"):
    return {
        "tool": "hmmer3_phmmer",
        "setting": "default",
        "step": step,
        "target": target,
        "complete": complete,
        "wall_s": wall,
        "cores": cores,
        "peak_ram_gb": 1.0,
        "gpu_h": None,
        "source": "trace.txt",
    }


def test_summarise_keeps_newest_completion_per_target():
    rows = [
        task("fly", "2026-09-01", 3600.0),
        task("fly", "2026-09-02", 1800.0),
        task("worm", "2026-09-01", 3600.0),
    ]
    (out,) = bll.summarise_tasks(rows, "exp", 998)
    assert out["n_tasks"] == 2
    assert out["wall_s"] == 5400.0
    assert out["cpu_h"] == pytest.approx(8 * 5400 / 3600)
    assert out["cores"] == 8


def test_summarise_mixed_cores_is_null_not_a_guess():
    rows = [task("fly", "a", 60.0, cores=8), task("worm", "a", 60.0, cores=16)]
    (out,) = bll.summarise_tasks(rows, "exp", None)
    assert out["cores"] is None
    assert out["cpu_h"] == pytest.approx((8 * 60 + 16 * 60) / 3600)


def test_summarise_missing_cores_gives_null_cpu_h():
    (out,) = bll.summarise_tasks([task("fly", "a", 60.0, cores=None)], "exp", None)
    assert out["cpu_h"] is None
    assert out["wall_s"] == 60.0


def test_trace_tasks_skips_cached():
    t = pl.DataFrame(
        {
            "step": ["phmmerSearch", "phmmerSearch"],
            "tag": ["human_vs_fly", "human_vs_worm"],
            "status": ["COMPLETED", "CACHED"],
            "realtime": ["1m", "1m"],
            "cpus": ["8", "8"],
            "peak_rss": ["1 GB", "1 GB"],
            "complete": ["x", "y"],
            "trace_file": ["t", "t"],
            "hash": ["a", "b"],
            "container": [None, None],
        }
    )
    rows = bll.trace_tasks(t, lambda s, g: bll.qfo_task(s, g, "metrics"))
    assert [r["target"] for r in rows] == ["fly"]


def metrics_fixture() -> pl.DataFrame:
    base = {
        "truth_set": "pfam",
        "tool": "kmerseek",
        "variant": "protein20_k5_lcFalse",
        "species": "fly",
        "species_mya": 600,
        "dedup_transfers": False,
        "n_truth_instances": 10,
        "fmax": 0.5,
        "auprc": None,
    }
    return pl.DataFrame(
        [
            base | {"split": "all", "stratum_axis": "all", "stratum": "all"},
            base | {"split": "all", "stratum_axis": "identity", "stratum": "0-20%"},
            base | {"split": "heldout", "stratum_axis": "all", "stratum": "all"},
            base | {"split": "all", "stratum_axis": "disorder", "stratum": "high"},
        ]
    )


def test_qfo_metrics_keeps_only_all_split_and_carried_axes(tmp_path, monkeypatch):
    p = tmp_path / "m.parquet"
    metrics_fixture().write_parquet(p)
    monkeypatch.setitem(bll.COMMIT, "test_exp", "abc1234")
    out = bll.qfo_metrics(p, "x/m.parquet", "test_exp", "q", {"kmerseek": "img"})
    # Two kept rows x one non-null metric (auprc is null, so it has no row).
    assert out.height == 2
    assert set(out["stratum_axis"]) == {"all", "identity"}
    assert set(out["metric"]) == {"fmax"}
    assert out["target_db"].unique().to_list() == ["QfO fly"]
    assert out["dedup_or_raw"].unique().to_list() == ["raw"]
    assert out["truth_version"].unique().to_list() == ["Pfam 38.2"]
    assert out["source_path"][0] == "/scratch/users/olgabot/x/m.parquet"
    assert out.columns == list(bll.SCHEMA)


def test_coverage_counts_an_instance_once_across_settings():
    row = {c: None for c in bll.SCHEMA}
    rows = [
        row
        | {
            "experiment": "e",
            "tool": "t",
            "setting": s,
            "query_set": "q",
            "target_db": "d",
            "truth_source": "x",
            "stratum_axis": "identity",
            "stratum_bin": "0-20%",
            "metric": m,
            "n_instances": 7,
        }
        for s in ("a", "b")
        for m in ("fmax", "auprc")
    ]
    md = bll.coverage_markdown(pl.DataFrame(rows, schema=bll.SCHEMA))
    line = next(ln for ln in md.splitlines() if ln.startswith("| t |"))
    assert line == "| t |  | 7 |  |  |  |"


def test_local_copy_maps_sherlock_and_laptop_paths(tmp_path):
    assert (
        bll.local_copy(tmp_path, "/scratch/users/olgabot/a/b.tsv")
        == tmp_path / "a/b.tsv"
    )
    assert (
        bll.local_copy(tmp_path, "/Users/olga/data/x.parquet")
        == tmp_path / "laptop/data/x.parquet"
    )
    with pytest.raises(ValueError):
        bll.local_copy(tmp_path, "/tmp/elsewhere.tsv")


def write_pair_metrics(path, tool, species, auc, n_found):
    pl.DataFrame(
        {
            "tool": [tool],
            "species": [species],
            "auc_roc": [auc],
            "auc_pr": [auc],
            "sens_at_99prec": [0.5],
            "prec_at_50rec": [0.9],
            "n_positives": [20000],
            "n_negatives": [40000],
            "n_found": [n_found],
        }
    ).write_parquet(path)


def test_pfam_pair_roc_drops_empty_output_and_prefixes_metrics(tmp_path):
    d = bll.local_copy(tmp_path, bll.PFAM_TOOLS)
    d.mkdir(parents=True)
    write_pair_metrics(
        d / "mmseqs2_seqseq.mouse.metrics.parquet",
        "mmseqs2_seqseq",
        "mouse",
        0.75,
        10025,
    )
    write_pair_metrics(
        d / "diamond.zebrafish.metrics.parquet", "diamond", "zebrafish", float("nan"), 0
    )
    out = bll.pfam_pair_roc(tmp_path)
    assert set(out["tool"]) == {"mmseqs2_seqseq"}
    assert all(m.startswith("pair_") for m in out["metric"])
    assert out.filter(pl.col("metric") == "pair_roc_auc")["value"].to_list() == [0.75]
    assert out.filter(pl.col("metric") == "pair_n_with_hit")["value"].to_list() == [
        10025.0
    ]
    assert out["n_instances"].unique().to_list() == [20000]
    assert out["target_db"].unique().to_list() == ["QfO mouse"]
    assert out["source_commit"].null_count() == out.height
    assert out.columns == list(bll.SCHEMA)
