"""Best-hit domain transfer (bin/annotator_best_hit.py) and its leaderboard rows
(bin/annotator_tool_rows.py).

The fixture under tests/fixtures/annotator is real tool output: DIAMOND and MMseqs2
searches of the 60 human fixture queries against the 120 yeast fixture proteins, and
hmmscan --cut_ga on the best-hit proteins (tests/make_annotator_fixture.sh).
"""

import ast
import gzip
import json
import shutil
import sys
from pathlib import Path

import polars as pl
import pytest

from conftest import BIN, FIXTURES, run

sys.path.insert(0, str(BIN))
import annotator_best_hit as abh  # noqa: E402
import annotator_tool_rows as atr  # noqa: E402

ANNOT = FIXTURES / "annotator"
SETTINGS = [
    ("diamond_besthit", "sensitive"),
    ("diamond_besthit", "ultra-sensitive"),
    ("mmseqs2_besthit", "s7.5"),
]
SPECIES = "swissprot_minus_mammalia"


def test_alignment_map_skips_gap_columns():
    # query  10 A C - D E
    # target 20 A - B D E
    m = abh.target_to_query_map(10, 20, "AC-DE", "A-BDE")
    assert m == {20: 10, 22: 12, 23: 13}


def test_alignment_strings_must_match_in_length():
    with pytest.raises(ValueError):
        abh.target_to_query_map(1, 1, "AC", "A")


def hit(**kw) -> pl.DataFrame:
    base = dict(
        query="q",
        target="t",
        qstart=1,
        qend=10,
        tstart=1,
        tend=10,
        bits=50.0,
        evalue=1e-5,
        pident=40.0,
        qaln="A" * 10,
        taln="A" * 10,
    )
    base.update(kw)
    return pl.DataFrame([base], schema=abh.HIT_SCHEMA)


def domains(*rows) -> pl.DataFrame:
    return pl.DataFrame(
        [dict(target=t, pfam_id=p, dstart=s, dend=e) for t, p, s, e in rows],
        schema={
            "target": pl.String,
            "pfam_id": pl.String,
            "dstart": pl.Int64,
            "dend": pl.Int64,
        },
    )


def test_best_hit_breaks_ties_by_evalue_then_target():
    h = pl.concat(
        [
            hit(target="B", bits=60.0, evalue=1e-6),
            hit(target="A", bits=60.0, evalue=1e-6),
            hit(target="C", bits=60.0, evalue=1e-9),
            hit(target="D", bits=10.0, evalue=1e-30),
        ]
    )
    assert abh.best_hits(h)["target"].to_list() == ["C"]
    assert abh.best_hits(h.filter(pl.col("target") != "C"))["target"].to_list() == ["A"]


def test_gap_moves_the_straight_line_but_not_the_exact_placement():
    # Query has 5 extra residues inside the hit: query 1-15 aligns to target 1-10.
    qaln = "AAAAA" + "CCCCC" + "AAAAA"
    taln = "AAAAA" + "-----" + "AAAAA"
    h = hit(qstart=1, qend=15, tstart=1, tend=10, qaln=qaln, taln=taln)
    # Target domain 6-10 sits after the gap: exactly query 11-15.
    calls, qc = abh.place_domains(h, domains(("t", "PF1", 6, 10)), 0.5)
    r = calls.row(0, named=True)
    assert (r["qstart"], r["qend"]) == (11, 15)
    # The straight line (PR 121) puts target 6 at query 1 + 5 * 14/9 = 8.8 -> 9.
    assert (r["linear_qstart"], r["linear_qend"]) == (9, 15)
    assert qc["n_calls_linear_differs"] == 1 and qc["max_linear_shift"] == 2


def test_domain_must_be_half_covered_on_the_target():
    h = hit(tstart=1, tend=10)
    # Domain 6-25 (20 residues): the hit covers 5 of them, a quarter.
    calls, qc = abh.place_domains(h, domains(("t", "PF1", 6, 25)), 0.5)
    assert calls.height == 0 and qc["n_domains_below_min_overlap"] == 1
    # Domain 6-15: half covered, transfers as the covered part only.
    calls, _ = abh.place_domains(h, domains(("t", "PF1", 6, 15)), 0.5)
    assert calls.select("qstart", "qend").row(0) == (6, 10)


def test_hit_on_a_protein_with_no_domain_makes_no_call():
    calls, _ = abh.place_domains(hit(target="other"), domains(("t", "PF1", 1, 10)), 0.5)
    assert calls.height == 0


@pytest.mark.parametrize("tool,setting", SETTINGS)
def test_place_reproduces_the_committed_fixture(tool, setting, tmp_path):
    stem = f"{tool}.{setting}"
    run(
        "annotator_best_hit.py",
        "place",
        "--hits",
        ANNOT / f"{stem}.hits.tsv.gz",
        "--target-domains",
        ANNOT / "hit_target_domains.tsv.gz",
        "--queries",
        FIXTURES / "qfo" / "human.fasta",
        "--tool",
        tool,
        "--setting",
        setting,
        "--out",
        tmp_path / "r.tsv.gz",
        "--qc-out",
        tmp_path / "qc.json",
    )
    with gzip.open(tmp_path / "r.tsv.gz", "rt") as a, gzip.open(
        ANNOT / f"{stem}.regions.tsv.gz", "rt"
    ) as b:
        assert a.read() == b.read()
    assert json.loads((tmp_path / "qc.json").read_text()) == json.loads(
        (ANNOT / f"{stem}.place_qc.json").read_text()
    )


def test_every_placed_call_lies_inside_its_hit(tmp_path):
    for tool, setting in SETTINGS:
        best = abh.best_hits(abh.read_hits(ANNOT / f"{tool}.{setting}.hits.tsv.gz"))
        calls, _ = abh.place_domains(
            best, abh.read_domains(ANNOT / "hit_target_domains.tsv.gz"), 0.5
        )
        j = calls.join(
            best.select(
                "query",
                "target",
                pl.col("qstart").alias("hs"),
                pl.col("qend").alias("he"),
            ),
            on=["query", "target"],
        )
        assert j.height == calls.height > 0
        assert (
            j.filter(
                (pl.col("qstart") < pl.col("hs"))
                | (pl.col("qend") > pl.col("he"))
                | (pl.col("qstart") > pl.col("qend"))
            ).height
            == 0
        )


def test_targets_refuses_a_hit_missing_from_the_reference(tmp_path):
    ref = tmp_path / "ref.fasta"
    ref.write_text(">P00000\nMA\n")
    proc = __import__("subprocess").run(
        [
            sys.executable,
            str(BIN / "annotator_best_hit.py"),
            "targets",
            "--hits",
            str(ANNOT / "mmseqs2_besthit.s7.5.hits.tsv.gz"),
            "--reference",
            str(ref),
            "--out",
            str(tmp_path / "t.fasta"),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0 and "not in ref.fasta" in proc.stderr


def test_direct_annotation_scoring_reads_placed_calls(pfam_truth, tmp_path):
    stem = "pfam.diamond_besthit.sensitive." + SPECIES
    run(
        "evaluate_domain_calls.py",
        "--regions",
        ANNOT / "diamond_besthit.sensitive.regions.tsv.gz",
        "--tool",
        "diamond_besthit",
        "--variant",
        "sensitive",
        "--species",
        SPECIES,
        "--truth",
        pfam_truth["truth"],
        "--direct-annotation",
        "--calls-out",
        tmp_path / f"{stem}.calls.parquet",
        "--metrics-out",
        tmp_path / f"{stem}.metrics.parquet",
        "--curve-out",
        tmp_path / f"{stem}.curve.parquet",
    )
    calls = pl.read_parquet(tmp_path / f"{stem}.calls.parquet")
    n_regions = sum(
        1 for _ in gzip.open(ANNOT / "diamond_besthit.sensitive.regions.tsv.gz", "rt")
    )
    assert calls.height == n_regions
    assert calls["is_tp"].sum() > 0


# ---------------------------------------------------------------------------------------
# Rows, cost rows, pile-up calls and cards


def results_layout(tmp_path: Path) -> Path:
    """The annotator_tools.nf outdir, built from the committed fixture."""
    res = tmp_path / "results"
    for sub in ("metrics", "calls", "regions", "hits"):
        (res / sub).mkdir(parents=True)
    for tool, setting in SETTINGS:
        stem = f"pfam.{tool}.{setting}.{SPECIES}"
        shutil.copy(ANNOT / f"{stem}.metrics.parquet", res / "metrics")
        shutil.copy(ANNOT / f"{stem}.calls.parquet", res / "calls")
        shutil.copy(ANNOT / f"{tool}.{setting}.place_qc.json", res / "regions")
        (res / "hits" / f"{tool}.{setting}.run.txt").write_text(
            f"command: {tool} --{setting}\nversion: 1.0\ncpu_model: test cpu\n"
        )
    return res


TRACE_HEAD = (
    "task_id\tprocess\ttag\tstatus\tcpus\trealtime\tpeak_rss\tcontainer\tcomplete\n"
)


def write_trace(path: Path) -> None:
    rows = [
        (
            "1",
            "cladeExcludedReference",
            "swissprot minus Mammalia",
            "COMPLETED",
            "1",
            "2m",
            "1.5 GB",
            "img:py",
            "1",
        ),
        (
            "2",
            "DIAMOND_BEST_HIT:diamondMakedb",
            "diamond makedb",
            "COMPLETED",
            "4",
            "1m",
            "2 GB",
            "quay.io/biocontainers/diamond@sha256:x",
            "2",
        ),
        # A killed first attempt and its retry: only the retry is counted.
        (
            "3",
            "DIAMOND_BEST_HIT:diamondBestHitSearch",
            "diamond sensitive",
            "FAILED",
            "8",
            "1h",
            "8 GB",
            "quay.io/biocontainers/diamond@sha256:x",
            "3",
        ),
        (
            "4",
            "DIAMOND_BEST_HIT:diamondBestHitSearch",
            "diamond sensitive",
            "COMPLETED",
            "8",
            "30m",
            "3 GB",
            "quay.io/biocontainers/diamond@sha256:x",
            "4",
        ),
        (
            "5",
            "placeBestHitDomains",
            "diamond_besthit sensitive",
            "COMPLETED",
            "1",
            "10s",
            "500 MB",
            "img:py",
            "5",
        ),
    ]
    path.write_text(TRACE_HEAD + "".join("\t".join(r) + "\n" for r in rows))


def test_rows_cost_pileup_and_cards(tmp_path):
    res = results_layout(tmp_path)
    write_trace(tmp_path / "trace.txt")
    out = tmp_path / "lb"
    run(
        "annotator_tool_rows.py",
        "--results",
        res,
        "--trace",
        tmp_path / "trace.txt",
        "--out",
        out,
        "--source-commit",
        "abc1234",
    )

    rows = pl.read_parquet(out / "rows.parquet")
    assert rows.schema == pl.Schema(atr.ROW_SCHEMA)
    assert set(rows["tool_version"].drop_nulls()) == {"1.0"}
    assert rows["container_digest"].null_count() < rows.height
    assert set(rows["tool"]) == {"diamond_besthit", "mmseqs2_besthit"}
    assert rows["n_instances"].null_count() == 0
    # Row values are the metrics file's own numbers.
    m = pl.read_parquet(
        res / "metrics" / f"pfam.mmseqs2_besthit.s7.5.{SPECIES}.metrics.parquet"
    )
    want = m.filter((pl.col("split") == "all") & (pl.col("stratum_axis") == "all"))[
        "fmax"
    ][0]
    got = rows.filter(
        (pl.col("setting") == "s7.5")
        & (pl.col("stratum_axis") == "all")
        & (pl.col("metric") == "fmax")
    )["value"]
    assert got.to_list() == [want]

    comp = pl.read_parquet(out / "compute.parquet")
    assert comp.schema == pl.Schema(atr.COMPUTE_SCHEMA)
    search = comp.filter(pl.col("step") == "diamondBestHitSearch").row(0, named=True)
    assert search["n_tasks"] == 1 and search["cpu_h"] == pytest.approx(8 * 0.5)
    assert search["peak_ram_gb"] == pytest.approx(3.0)
    shared = comp.filter(pl.col("tool") == "annotator_shared")
    assert shared["step"].to_list() == ["cladeExcludedReference"]

    pile = pl.read_parquet(out / "pileup_calls.parquet")
    assert pile.columns == [
        "tool",
        "setting",
        "accession",
        "pfam_id",
        "call_start",
        "call_end",
        "score",
        "is_tp",
        "target",
    ]
    assert set(pile["target"]) == {"Swiss-Prot minus Mammalia"}

    c = json.loads((out / "cards" / "diamond_besthit.sensitive.json").read_text())
    assert c["container"] == "quay.io/biocontainers/diamond@sha256:x"
    assert c["public"] is False and c["needs_gpu"] is False
    assert c["cost_this_setting"]["steps"] == [
        "diamondBestHitSearch",
        "placeBestHitDomains",
    ]
    assert c["cost_database_build_shared_by_tool_settings"]["steps"] == [
        "diamondMakedb"
    ]
    assert c["scores_split_all"]["grey"] is False
    assert c["scores_split_all"]["n_truth_instances"] >= 30


def test_card_marks_small_strata_grey_and_missing_coordinates_not_applicable(
    tmp_path, monkeypatch
):
    m = pl.read_parquet(
        ANNOT / f"pfam.diamond_besthit.sensitive.{SPECIES}.metrics.parquet"
    )
    m = m.with_columns(pl.lit(12, dtype=pl.Int64).alias("n_truth_instances"))
    info = dict(atr.TOOL_INFO["diamond_besthit"], has_query_coordinates=False)
    monkeypatch.setitem(atr.TOOL_INFO, "diamond_besthit", info)

    class A:
        query_set, target_db, target_db_version = "q", "t", "v"
        truth_version, source_commit = "Pfam", None

    c = atr.card(
        "diamond_besthit",
        "sensitive",
        m,
        {},
        {},
        None,
        pl.DataFrame(schema=atr.COMPUTE_SCHEMA),
        A,
    )
    assert c["scores_split_all"]["grey"] is True
    assert c["scores_split_all"]["median_iou_tp"] == "not applicable"
    assert c["scores_split_all"]["fmax"] == "not applicable"
    assert isinstance(c["scores_split_all"]["family_fmax"], float)


def test_schema_matches_pr148_when_present():
    """PR 148's long-table schema, read from its source without importing it."""
    src = Path(__file__).resolve().parents[3] / "scripts" / "build_leaderboard_long.py"
    if not src.exists():
        pytest.skip("PR 148 (scripts/build_leaderboard_long.py) is not on this branch")
    tree = ast.parse(src.read_text())
    names = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("SCHEMA", "COMPUTE_SCHEMA")
        ):
            names[node.targets[0].id] = [k.value for k in node.value.keys]
    assert names["SCHEMA"] == list(atr.SCHEMA)
    assert names["COMPUTE_SCHEMA"] == list(atr.COMPUTE_SCHEMA)


def test_container_digest_maps_the_apptainer_cache_file_back_to_the_image():
    refs = atr.module_containers(BIN.parent / "modules")
    ref = next(r for r in refs if "diamond" in r)
    cached = (
        "/scratch/x/apptainer-cache/" + ref.replace("/", "-").replace(":", "-") + ".img"
    )
    assert atr.container_ref(cached, refs) == ref
    assert atr.container_ref("docker.io/other:1", refs) == "docker.io/other:1"


def test_score_tasks_tagged_with_their_target_keep_their_setting():
    assert atr.task_setting(
        "scoreAnnotatorTool", "diamond_besthit sensitive swissprot_minus_mammalia"
    ) == ("diamond_besthit", "sensitive")
