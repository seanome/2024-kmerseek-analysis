"""scripts/delimitation_pre_transfer.py, full-run mode, on a hand-built results directory.

Two queries, three Pfam instances (0-based, end-exclusive, as in the truth):

  query  length  instance  interval
  Q1     400     PF00001   [100, 140)   40 aa, 10% of the protein
  Q1     400     PF00002   [200, 300)
  Q2     300     PF00003   [50, 150)    Q2 has no significant region in any file

kmerseek, human_vs_mouse.hp_pbotc_1st_ed2.k19.lctrue (cutoff region_poisson_score >= 3):

  Q1 [0, 400)    score 5   a call covering the whole protein: covers PF00001 (cover 1.0) but
                           its IoU with it is 40 / 400 = 0.1, so it does not delimit it
  Q1 [200, 300)  score 5   equal to PF00002: IoU 1
  Q1 [100, 140)  score 2   below the cutoff; if it were kept PF00001 would be delimited
  Q2 [50, 150)   score 1   below the cutoff, so Q2 has no significant region at all

phmmer, human_vs_mouse.hmmer3_phmmer.tsv.gz (cutoff evalue <= 1e-3), headerless and with
UniProt headers, as the pipeline writes it:

  Q1 100..140  1e-10   equal to PF00001: IoU 1
  Q2 50..150   1.0     above the cutoff
"""

import gzip
import re
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "delimitation_pre_transfer.py"


def kmerseek_file(path: Path, rows) -> None:
    pl.DataFrame(
        {
            "query_name": [f"sp|{q}|{q}_HUMAN" for q, *_ in rows],
            "region_start": [r[1] for r in rows],
            "region_end": [r[2] for r in rows],
            "region_poisson_score": [float(r[3]) for r in rows],
        },
        schema_overrides={"region_start": pl.Int64, "region_end": pl.Int64},
    ).write_parquet(path)


def tool_file(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for q, s, e, ev in rows:
            fh.write(f"sp|{q}|{q}_HUMAN\tsp|T1|T1_MOUSE\t{s}\t{e}\t1\t40\t50.0\t{ev}\n")


@pytest.fixture
def results(tmp_path):
    res = tmp_path / "results"
    (res / "kmerseek").mkdir(parents=True)
    kmerseek_file(
        res / "kmerseek" / "human_vs_mouse.hp_pbotc_1st_ed2.k19.lctrue.regions.parquet",
        [
            ("Q1", 0, 400, 5),
            ("Q1", 200, 300, 5),
            ("Q1", 100, 140, 2),
            ("Q2", 50, 150, 1),
        ],
    )
    tool_file(
        res / "regions" / "hmmer3_phmmer" / "human_vs_mouse.hmmer3_phmmer.tsv.gz",
        [("Q1", 100, 140, "1e-10"), ("Q2", 50, 150, "1.0")],
    )
    # More jobs for the task split, and files that must not become jobs: a species outside
    # the midi-plus ten, hmmscan's human-only table, and ProstT5's uncompressed skip list.
    for sp in ["chicken", "fly", "worm"]:
        kmerseek_file(
            res / "kmerseek" / f"human_vs_{sp}.protein20.k10.lcfalse.regions.parquet",
            [("Q1", 0, 10, 5)],
        )
        tool_file(
            res / "regions" / "hmmer3_phmmer" / f"human_vs_{sp}.hmmer3_phmmer.tsv.gz",
            [("Q1", 1, 10, "1e-5")],
        )
    kmerseek_file(
        res / "kmerseek" / "human_vs_tmaritima.protein20.k10.lcfalse.regions.parquet",
        [("Q1", 0, 10, 5)],
    )
    tool_file(
        res / "regions" / "hmmscan" / "human.hmmscan.tsv.gz", [("Q1", 1, 10, "1e-5")]
    )
    # A search that wrote nothing: not a job, named on --list.
    (
        res
        / "kmerseek"
        / "human_vs_mouse.hp_thomas_dill2_ext2.k12.lcfalse.regions.parquet"
    ).touch()
    (res / "regions" / "prostt5").mkdir()
    (res / "regions" / "prostt5" / "human_vs_mouse.prostt5_skipped.tsv").write_text(
        "Q1\n"
    )
    pl.DataFrame(
        {
            "accession": ["Q1", "Q1", "Q2"],
            "pfam_id": ["PF00001", "PF00002", "PF00003"],
            "domain_start": [100, 200, 50],
            "domain_end": [140, 300, 150],
            "protein_length": [400, 400, 300],
            "split": ["test"] * 3,
        },
        schema_overrides={
            "domain_start": pl.Int32,
            "domain_end": pl.Int32,
            "protein_length": pl.Int32,
        },
    ).write_parquet(tmp_path / "truth.parquet")
    return tmp_path


def run(tmp: Path, *args) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--results",
        str(tmp / "results"),
        "--truth",
        str(tmp / "truth.parquet"),
        *map(str, args),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-4000:]
    return proc


def scored(tmp: Path, name: str) -> pl.DataFrame:
    out = tmp / "out"
    run(tmp, "--out-dir", out, "--only", f"^{re.escape(name)}$")
    return pl.read_parquet(out / f"{name}.domains.parquet").sort("pfam_id")


def test_list_is_the_midi_plus_species_only(results):
    proc = run(results, "--list")
    names = [line.split("\t")[0] for line in proc.stdout.splitlines()]
    assert names == sorted(names)
    assert len(names) == 8
    assert "kmerseek.hp_pbotc_1st_ed2_k19_lcTrue.mouse" in names
    assert "hmmer3_phmmer.mouse" in names
    assert not [
        n for n in names if "tmaritima" in n or "hmmscan" in n or "prostt5" in n
    ]
    assert "8 jobs (1 zero-byte files left out)" in proc.stderr
    assert (
        "human_vs_mouse.hp_thomas_dill2_ext2.k12.lcfalse.regions.parquet" in proc.stderr
    )


def test_whole_protein_call_covers_but_does_not_delimit(results):
    d = scored(results, "kmerseek.hp_pbotc_1st_ed2_k19_lcTrue.mouse")
    short = d.row(by_predicate=pl.col("pfam_id") == "PF00001", named=True)
    assert short["covered"] and not short["delimited"]
    assert short["best_cover"] == 1.0
    # [0, 400) against [100, 140): 40 / 400. The [100, 140) region at score 2 is below the
    # cutoff; had it been kept the IoU would be 1.
    assert short["best_iou"] == pytest.approx(0.1)


def test_region_equal_to_domain_has_iou_one(results):
    d = scored(results, "kmerseek.hp_pbotc_1st_ed2_k19_lcTrue.mouse")
    assert (
        d.row(by_predicate=pl.col("pfam_id") == "PF00002", named=True)["best_iou"]
        == 1.0
    )
    p = scored(results, "hmmer3_phmmer.mouse")
    assert (
        p.row(by_predicate=pl.col("pfam_id") == "PF00001", named=True)["best_iou"]
        == 1.0
    )
    assert p["arm"].unique().to_list() == ["hmmer3_phmmer"]
    hits = pl.read_parquet(results / "out" / "hmmer3_phmmer.mouse.hits.parquet")
    assert hits.columns == [
        "accession",
        "pfam_id",
        "domain_start",
        "domain_end",
        "protein_length",
        "qstart",
        "qend",
        "iou",
        "cover",
        "arm",
    ]


@pytest.mark.parametrize(
    "name", ["kmerseek.hp_pbotc_1st_ed2_k19_lcTrue.mouse", "hmmer3_phmmer.mouse"]
)
def test_query_without_regions_scores_zero_not_null(results, name):
    d = scored(results, name)
    q2 = d.filter(pl.col("accession") == "Q2")
    assert q2.height == 1
    row = q2.row(0, named=True)
    assert row["best_iou"] == 0.0 and row["best_cover"] == 0.0
    assert row["covered"] is False and row["delimited"] is False
    assert row["query_has_regions"] is False
    assert d.null_count().sum_horizontal().item() == 0


@pytest.mark.parametrize("n_tasks", [1, 2, 3, 5])
def test_only_and_task_split_cover_every_job_once(results, n_tasks):
    only = "hmmer3_phmmer|protein20"
    expected = [
        line.split("\t")[0]
        for line in run(results, "--list", "--only", only).stdout.splitlines()
    ]
    assert len(expected) == 7
    out = results / f"split{n_tasks}"
    seen = []
    for t in range(n_tasks):
        proc = run(
            results, "--out-dir", out, "--only", only, "--task", t, "--n-tasks", n_tasks
        )
        seen += re.findall(
            r"^(\S+): [\d_]+ significant regions", proc.stdout, flags=re.M
        )
    assert sorted(seen) == sorted(expected)
    assert sorted(
        f.name.removesuffix(".domains.parquet") for f in out.glob("*.domains.parquet")
    ) == sorted(expected)


def test_rerun_skips_finished_jobs(results):
    name = "hmmer3_phmmer.mouse"
    scored(results, name)
    proc = run(results, "--out-dir", results / "out", "--only", f"^{re.escape(name)}$")
    assert f"{name}: already scored, skipped" in proc.stdout


def test_concat_adds_labels(results):
    out = results / "out"
    run(results, "--out-dir", out)
    full = results / "full.parquet"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "concat_delimitation_pre_transfer.py"),
            "--in-dir",
            str(out),
            "--out",
            str(full),
        ],
        check=True,
        capture_output=True,
    )
    d = pl.read_parquet(full)
    assert d.height == 8 * 3
    k = d.filter(pl.col("arm") == "kmerseek.hp_pbotc_1st_ed2_k19_lcTrue").row(
        0, named=True
    )
    assert (k["tool"], k["alphabet"], k["ksize"], k["lc"], k["species"]) == (
        "kmerseek",
        "hp_pbotc_1st_ed2",
        19,
        True,
        "mouse",
    )
    t = d.filter(pl.col("arm") == "hmmer3_phmmer", pl.col("species") == "fly").row(
        0, named=True
    )
    assert (t["tool"], t["alphabet"], t["ksize"], t["lc"]) == (
        "hmmer3_phmmer",
        None,
        None,
        None,
    )
    h = pl.read_parquet(results / "full.hits.parquet")
    assert h.height == sum(
        pl.read_parquet(f).height for f in out.glob("*.hits.parquet")
    )
