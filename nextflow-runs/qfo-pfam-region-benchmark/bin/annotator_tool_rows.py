#!/usr/bin/env python3
"""Leaderboard rows, cost rows, pile-up calls and tool cards for the annotator tools.

Reads one annotator_tools.nf output folder and its Nextflow trace, and writes:

    rows.parquet          long rows, in the schema of PR 148's results.parquet
                          (scripts/build_leaderboard_long.py SCHEMA)
    compute.parquet       cost rows from the trace, in PR 148's compute.parquet schema
    pileup_calls.parquet  each tool's calls in the shape PR 146's load_calls() returns
                          (tool, setting, accession, pfam_id, call_start, call_end, score,
                          is_tp), for tools that place calls on query residues
    cards/<tool>.<setting>.json   one tool card per row: version, image, command as
                          run, hardware, scores with n, cost, and whether the parameters
                          have been checked by the tool's authors

Nothing is estimated. A value the outputs do not carry is null. Scores are read from
evaluate_domain_calls.py's metrics at split "all"; a stratum with fewer than
GREY_BELOW truth instances is marked grey on the card, never dropped.

A tool whose output has no per-residue query coordinates (TOOL_INFO has_query_coordinates
False) gets "not applicable" for IoU and no pile-up rows, not zero.
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl

GREY_BELOW = 30

# Copied from PR 148 (scripts/build_leaderboard_long.py, branch
# olgabot/leaderboard-all-experiments). tests/test_annotator_tool_rows.py compares the
# two whenever that file is present, so they cannot drift apart silently.
SCHEMA = {
    "experiment": pl.String,
    "tool": pl.String,
    "tool_version": pl.String,
    "setting": pl.String,
    "query_set": pl.String,
    "target_db": pl.String,
    "target_db_version": pl.String,
    "target_mya_to_human": pl.Float64,
    "stratum_axis": pl.String,
    "stratum_bin": pl.String,
    "metric": pl.String,
    "value": pl.Float64,
    "n_instances": pl.Int64,
    "truth_source": pl.String,
    "truth_version": pl.String,
    "dedup_or_raw": pl.String,
    "source_path": pl.String,
    "source_commit": pl.String,
}
COMPUTE_SCHEMA = {
    "experiment": pl.String,
    "tool": pl.String,
    "setting": pl.String,
    "step": pl.String,
    "n_tasks": pl.Int64,
    "n_target_dbs": pl.Int64,
    "wall_s": pl.Float64,
    "cores": pl.Int64,
    "cpu_h": pl.Float64,
    "peak_ram_gb": pl.Float64,
    "gpu_type": pl.String,
    "gpu_h": pl.Float64,
    "n_queries": pl.Int64,
    "n_targets": pl.Int64,
    "source": pl.String,
}
METRICS = [
    "fmax",
    "fmax_precision",
    "fmax_recall",
    "fmax_threshold",
    "auprc",
    "recall_reachable",
    "median_iou_tp",
    "family_fmax",
]
# Metrics that need the call's position on the query. A tool without per-residue
# coordinates gets "not applicable" for these on its card.
# Every score except family_fmax counts a call as right only when it overlaps the true
# domain (IoU), so it needs the call's position on the query. A tool without per-residue
# coordinates gets "not applicable" for these on its card and no rows for them.
POSITION_METRICS = set(METRICS) - {"family_fmax"}
STRATUM_AXES = ["all", "identity", "plddt", "feature_type", "feature_length_bin"]

# One entry per tool. `group` is the site's column group. `graded_on` says what the
# calls are scored against, in the words the site prints.
TOOL_INFO = {
    "diamond_besthit": {
        "name": "DIAMOND best hit",
        "group": "best-hit transfer",
        "graded_on": "Pfam domains of the best Swiss-Prot hit (Mammalia removed), "
        "placed on the query through the alignment",
        "has_query_coordinates": True,
        "needs_gpu": False,
        "authors": "Benjamin Buchfink (DIAMOND)",
        "search_step": "diamondBestHitSearch",
        "db_steps": ["diamondMakedb"],
    },
    "mmseqs2_besthit": {
        "name": "MMseqs2 best hit",
        "group": "best-hit transfer",
        "graded_on": "Pfam domains of the best Swiss-Prot hit (Mammalia removed), "
        "placed on the query through the alignment",
        "has_query_coordinates": True,
        "needs_gpu": False,
        "authors": "Martin Steinegger, Johannes Söding (MMseqs2)",
        "search_step": "mmseqsBestHitSearch",
        "db_steps": ["mmseqsBestHitDb"],
    },
    "interproscan": {
        "name": "InterProScan (all member databases)",
        "group": "reference-database tool, graded on its own models",
        "graded_on": "its own Pfam member matches; InterPro's Pfam models are the source "
        "of the Pfam truth, and its databases contain human",
        "has_query_coordinates": True,
        "needs_gpu": False,
        "authors": "InterPro team, EMBL-EBI (InterProScan)",
        "search_step": "interproscanSearch",
        "db_steps": [],
        "target_db": "InterPro member databases (InterProScan 5.78-109.0 data)",
        "target_db_version": "InterProScan 5.78-109.0",
    },
    "eggnog_mapper": {
        "name": "eggNOG-mapper (default DIAMOND search)",
        "group": "reference-database tool, graded on its own models",
        "graded_on": "the PFAMs column (family names from the seed ortholog's group, no "
        "coordinates); eggNOG 5 contains human",
        "has_query_coordinates": False,
        "needs_gpu": False,
        "authors": "Carlos P. Cantalapiedra, Jaime Huerta-Cepas (eggNOG-mapper)",
        "search_step": "eggnogMapperSearch",
        "db_steps": ["pfamNameMap"],
        "target_db": "eggNOG 5.0 (emapperdb 5.0.2)",
        "target_db_version": "emapperdb 5.0.2",
    },
    # Embedding search runs once per QfO target proteome: target_db is "QfO <species>",
    # as in the main run's rows (PR 148), so the rows sit beside them.
    "tmvec": {
        "name": "TM-Vec (+ DeepBLAST alignment)",
        "group": "embedding search",
        "graded_on": "Pfam domains of the best hit in the target proteome, placed through "
        "DeepBLAST's structural alignment",
        "has_query_coordinates": True,
        "needs_gpu": True,
        "authors": "Tymor Hamamsy, Jamie Morton, Richard Bonneau (TM-Vec, DeepBLAST)",
        "search_step": "tmvecSearch",
        "db_steps": [],
        "per_target": True,
    },
    "plmsearch": {
        "name": "PLMSearch (embedding search)",
        "group": "embedding search",
        "graded_on": "Pfam families of the best hit in the target proteome; PLMSearch "
        "reports no alignment, so no position",
        "has_query_coordinates": False,
        "needs_gpu": True,
        "authors": "Wei Liu, Hong-Bin Shen and colleagues (PLMSearch)",
        "search_step": "plmsearchSearch",
        "db_steps": [],
        "per_target": True,
    },
}
QFO_TARGET_VERSION = "QfO 2020_04 (UP000008143 updated)"
# Steps that hold one GPU per task (-G 1): their wall time is GPU time.
GPU_STEPS = {"tmvecSearch", "plmsearchEmbed", "plmsearchSearch"}
# Embedding steps, tagged "<tool> <species>" or "plmsearch embed <name>".
EMBEDDING_STEPS = {
    "tmvecSearch": "tmvec",
    "plmsearchEmbed": "plmsearch",
    "plmsearchSearch": "plmsearch",
}
# Steps every tool of one run uses once. Their cost is reported once, as its own rows,
# and on each card as "shared", never split among the tools by a made-up rule.
SHARED_STEPS = {"cladeExcludedReference", "hitTargets", "hmmscanHitTargets"}
PER_SETTING_STEPS = {
    "placeBestHitDomains",
    "scoreAnnotatorTool",
    "interproscanSearch",
    "interproscanCalls",
    "eggnogMapperSearch",
    "eggnogCalls",
}
SHARED_EMBEDDING_STEPS = {"shortHeaders"}

_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)")
_SECONDS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_SIZE = re.compile(r"([\d.]+)\s*([KMGT]?B)")
_GB = {"B": 1 / 1024**3, "KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}


def duration_seconds(text: str | None) -> float | None:
    """Nextflow's '1h 2m 3.4s' / '34ms' / '-' as seconds."""
    if text is None or text.strip() in ("", "-"):
        return None
    return sum(float(v) * _SECONDS[u] for v, u in _DURATION.findall(text))


def size_gb(text: str | None) -> float | None:
    if text is None or text.strip() in ("", "-"):
        return None
    m = _SIZE.search(text)
    return float(m.group(1)) * _GB[m.group(2)] if m else None


def tidy(df: pl.DataFrame, schema: dict) -> pl.DataFrame:
    for col, dtype in schema.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return df.select([pl.col(c).cast(t) for c, t in schema.items()])


def read_run_info(path: Path) -> dict:
    """<tool>.<setting>.run.txt: 'command: ...', 'version: ...', 'cpu_model: ...'."""
    info = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(": ")
        info[key.strip()] = value.strip()
    return info


def task_setting(step: str, tag: str | None) -> tuple[str, str] | None:
    """(tool, setting) a trace row belongs to, or None for a shared step."""
    tag = tag or ""
    if step in SHARED_STEPS or step in SHARED_EMBEDDING_STEPS:
        return None
    if step == "diamondMakedb":
        return ("diamond_besthit", "database build")
    if step == "mmseqsBestHitDb":
        return ("mmseqs2_besthit", "database build")
    if step == "pfamNameMap":
        return ("eggnog_mapper", "database build")
    if step == "diamondBestHitSearch":
        return ("diamond_besthit", tag.removeprefix("diamond ").strip())
    if step == "mmseqsBestHitSearch":
        return ("mmseqs2_besthit", tag.removeprefix("mmseqs2 ").strip())
    if step in EMBEDDING_STEPS:
        return (EMBEDDING_STEPS[step], None)
    if step == "embeddingCalls":
        return (tag.partition(" ")[0], None)
    if step in PER_SETTING_STEPS:
        # "<tool> <setting>", or "<tool> <setting> <target>" for scoring.
        parts = tag.split()
        return (parts[0], parts[1] if len(parts) > 1 else "")
    return None


def task_target(step: str, tag: str | None) -> str | None:
    """The target proteome a task searched or scored, from its tag; None when the tag
    names none (one-target tools) or names the query set."""
    words = (tag or "").split()
    if step in EMBEDDING_STEPS or step == "embeddingCalls":
        t = words[-1] if words else None
    elif step == "scoreAnnotatorTool" and len(words) == 3:
        t = words[2]
    else:
        t = None
    return None if t == "query" else t


def read_trace(path: Path) -> pl.DataFrame:
    t = pl.read_csv(path, separator="\t", infer_schema=False)
    return t.with_columns(pl.col("process").str.split(":").list.last().alias("step"))


def compute_rows(
    trace: pl.DataFrame,
    experiment: str,
    n_queries: int | None,
    source: str,
    settings: dict[str, list[str]] | None = None,
) -> pl.DataFrame:
    """Per (tool, setting, step): the newest COMPLETED attempt of each task, summed.

    `settings` maps a tool to the settings its metrics hold; an embedding step's tag
    names the target, not the setting, so a tool with one setting gets it from there.
    """
    settings = settings or {}
    done = trace.filter(pl.col("status") == "COMPLETED")
    rows = []
    for r in done.iter_rows(named=True):
        key = task_setting(r["step"], r.get("tag"))
        tool, setting = key if key else ("annotator_shared", "shared")
        if setting is None:
            known = settings.get(tool, [])
            setting = known[0] if len(known) == 1 else "all settings"
        wall = duration_seconds(r.get("realtime"))
        cores = int(r["cpus"]) if r.get("cpus") not in (None, "", "-") else None
        rows.append(
            {
                "tool": tool,
                "setting": setting,
                "step": r["step"],
                "task": r.get("tag") or r["step"],
                "complete": r.get("complete"),
                "wall_s": wall,
                "cores": cores,
                "peak_ram_gb": size_gb(r.get("peak_rss")),
                "target": task_target(r["step"], r.get("tag")),
                "gpu_h": (
                    wall / 3600 if r["step"] in GPU_STEPS and wall is not None else None
                ),
            }
        )
    if not rows:
        return pl.DataFrame(schema=COMPUTE_SCHEMA)
    df = (
        pl.DataFrame(rows, infer_schema_length=None)
        .sort("complete", nulls_last=True)
        .unique(["tool", "setting", "step", "task"], keep="last", maintain_order=True)
    )
    out = []
    for (tool, setting, step), g in df.group_by(
        ["tool", "setting", "step"], maintain_order=True
    ):
        cores = g["cores"].drop_nulls().unique().to_list()
        complete = g["wall_s"].null_count() == 0 and g["cores"].null_count() == 0
        out.append(
            {
                "experiment": experiment,
                "tool": tool,
                "setting": setting,
                "step": step,
                "n_tasks": g.height,
                "n_target_dbs": max(1, g["target"].drop_nulls().n_unique()),
                "wall_s": (
                    float(g["wall_s"].sum()) if g["wall_s"].null_count() == 0 else None
                ),
                "cores": (
                    cores[0]
                    if len(cores) == 1 and g["cores"].null_count() == 0
                    else None
                ),
                "cpu_h": (
                    float((g["cores"] * g["wall_s"] / 3600).sum()) if complete else None
                ),
                "peak_ram_gb": g["peak_ram_gb"].max(),
                "gpu_type": (
                    "1 GPU per task (-G 1); model in run.txt"
                    if step in GPU_STEPS
                    else "none"
                ),
                "gpu_h": (
                    float(g["gpu_h"].sum())
                    if step in GPU_STEPS and g["gpu_h"].null_count() == 0
                    else None
                ),
                "n_queries": n_queries,
                "n_targets": None,
                "source": source,
            }
        )
    return tidy(pl.DataFrame(out, infer_schema_length=None), COMPUTE_SCHEMA)


def target_db(tool: str, a, species: str | None = None) -> str:
    """The best-hit tools search --target-db; a reference-database tool its own; an
    embedding search one QfO proteome per row."""
    if TOOL_INFO[tool].get("per_target"):
        return f"QfO {species}"
    return TOOL_INFO[tool].get("target_db", a.target_db)


def target_db_version(tool: str, a) -> str:
    if TOOL_INFO[tool].get("per_target"):
        return QFO_TARGET_VERSION
    return TOOL_INFO[tool].get("target_db_version", a.target_db_version)


def metric_rows(
    m: pl.DataFrame, a: argparse.Namespace, images: dict, rel: str
) -> pl.DataFrame:
    m = m.filter(
        (pl.col("split") == "all") & pl.col("stratum_axis").is_in(STRATUM_AXES)
    )
    metrics = [c for c in METRICS if c in m.columns]
    if not TOOL_INFO[m["tool"][0]]["has_query_coordinates"]:
        metrics = [c for c in metrics if c not in POSITION_METRICS]
    index = [
        "tool",
        "variant",
        "stratum_axis",
        "stratum",
        "dedup_transfers",
        "n_truth_instances",
    ]
    long = (
        m.select(*index, *metrics)
        .unpivot(index=index, on=metrics, variable_name="metric", value_name="value")
        .filter(pl.col("value").is_not_null())
    )
    return tidy(
        long.with_columns(
            pl.lit(a.experiment).alias("experiment"),
            pl.col("tool")
            .replace_strict(images, default=None, return_dtype=pl.String)
            .alias("tool_version"),
            pl.col("variant").alias("setting"),
            pl.lit(a.query_set).alias("query_set"),
            pl.lit(target_db(m["tool"][0], a, m["species"][0])).alias("target_db"),
            pl.lit(target_db_version(m["tool"][0], a)).alias("target_db_version"),
            pl.lit(None, dtype=pl.Float64).alias("target_mya_to_human"),
            pl.col("stratum").cast(pl.String).alias("stratum_bin"),
            pl.col("n_truth_instances").cast(pl.Int64).alias("n_instances"),
            pl.lit("Pfam-A domain instances").alias("truth_source"),
            pl.lit(a.truth_version).alias("truth_version"),
            pl.when(pl.col("dedup_transfers"))
            .then(pl.lit("dedup"))
            .otherwise(pl.lit("raw"))
            .alias("dedup_or_raw"),
            pl.lit(f"{a.source_root}/{rel}").alias("source_path"),
            pl.lit(a.source_commit).alias("source_commit"),
        ),
        SCHEMA,
    )


def pileup_calls(
    calls_path: Path, tool: str, setting: str, target: str
) -> pl.DataFrame:
    """PR 146's load_calls() for a non-kmerseek tool: coordinates kept as they are
    (1-based, end included), plus the target the tool searched, so a tool run against
    several proteomes keeps each set of calls apart."""
    c = pl.read_parquet(calls_path)
    return c.select(
        pl.lit(tool).alias("tool"),
        pl.lit(setting).alias("setting"),
        pl.col("query_acc").alias("accession"),
        "pfam_id",
        pl.col("qstart").cast(pl.Int64).alias("call_start"),
        pl.col("qend").cast(pl.Int64).alias("call_end"),
        pl.col("score").cast(pl.Float64),
        "is_tp",
        pl.lit(target).alias("target"),
    )


def card(
    tool: str,
    setting: str,
    m: pl.DataFrame,
    qc: dict,
    run: dict,
    image: str | None,
    compute: pl.DataFrame,
    a: argparse.Namespace,
) -> dict:
    info = TOOL_INFO[tool]
    by_target = {}
    for species, mt in m.group_by("species", maintain_order=True):
        row = mt.filter(
            (pl.col("split") == "all")
            & (pl.col("stratum_axis") == "all")
            & (~pl.col("dedup_transfers").fill_null(False))
        )
        scores = {}
        if row.height:
            r = row.row(0, named=True)
            n = r["n_truth_instances"]
            for k in METRICS:
                if k in POSITION_METRICS and not info["has_query_coordinates"]:
                    scores[k] = "not applicable"
                elif k in r:
                    scores[k] = r[k]
            scores["n_truth_instances"] = n
            scores["grey"] = n is None or n < GREY_BELOW
        by_target[species[0]] = scores
    # One target (best-hit and reference-database tools): its scores directly.
    scores = (
        next(iter(by_target.values()))
        if len(by_target) == 1
        else "one set per target: see scores_by_target"
    )
    # Three cost parts, never summed into one: the setting's own search and scoring, the
    # tool's database build (shared by the tool's settings), and the steps every tool of
    # the run shares. Summing the last two into each card would count them once per card.
    own = compute.filter((pl.col("tool") == tool) & (pl.col("setting") == setting))
    db = compute.filter(
        (pl.col("tool") == tool) & (pl.col("setting") == "database build")
    )
    shared = compute.filter(pl.col("tool") == "annotator_shared")

    def cost(df: pl.DataFrame) -> dict:
        return {
            "steps": df["step"].to_list(),
            "cpu_h": (
                df["cpu_h"].sum()
                if df.height and df["cpu_h"].null_count() == 0
                else None
            ),
            "wall_s": (
                df["wall_s"].sum()
                if df.height and df["wall_s"].null_count() == 0
                else None
            ),
            "peak_ram_gb": df["peak_ram_gb"].max() if df.height else None,
        }

    return {
        "tool": tool,
        "setting": setting,
        "name": info["name"],
        "group": info["group"],
        "graded_on": info["graded_on"],
        "version": run.get("version"),
        "container": image,
        "command": run.get("command"),
        "cpu_model": run.get("cpu_model"),
        "needs_gpu": info["needs_gpu"],
        "has_query_coordinates": info["has_query_coordinates"],
        "query_set": a.query_set,
        "target_db": (
            sorted(f"QfO {t}" for t in m["species"].unique().to_list())
            if info.get("per_target")
            else target_db(tool, a)
        ),
        "target_db_version": target_db_version(tool, a),
        "truth": f"Pfam-A domain instances, {a.truth_version}",
        # The step's own QC counts, as written (place_qc.json or calls_qc.json).
        "queries": {k: v for k, v in qc.items() if k not in ("tool", "setting")},
        "gpu_model": run.get("gpu_model"),
        "database_size": run.get("database_size"),
        "scores_split_all": scores,
        "scores_by_target": by_target,
        "cost_this_setting": cost(own),
        "cost_database_build_shared_by_tool_settings": cost(db),
        "cost_shared_by_all_tools_in_run": cost(shared),
        "author_check": {
            "authors": info["authors"],
            "status": "not sent: draft waiting for Olga",
            "draft": f"docs/tool_author_messages/{tool}.md",
        },
        "public": False,
        "source_commit": a.source_commit,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--results", type=Path, required=True, help="annotator_tools.nf --outdir"
    )
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--experiment", default="annotator_tools_midi_plus")
    ap.add_argument("--query-set", default="midi-plus")
    ap.add_argument("--target-db", default="Swiss-Prot minus Mammalia")
    ap.add_argument("--target-db-version", default="UniProtKB/Swiss-Prot 2026_03")
    ap.add_argument("--truth-version", default="Pfam 38.2")
    ap.add_argument(
        "--source-root",
        default="",
        help="where --results lives on Sherlock, for source_path",
    )
    ap.add_argument("--source-commit", default=None)
    a = ap.parse_args()

    metric_files = sorted((a.results / "metrics").glob("pfam.*.metrics.parquet"))
    if not metric_files:
        raise SystemExit(f"no metrics under {a.results / 'metrics'}")
    trace = read_trace(a.trace) if a.trace else None
    images = {}
    if trace is not None and "container" in trace.columns:
        for tool, info in TOOL_INFO.items():
            imgs = (
                trace.filter(
                    (pl.col("step") == info["search_step"])
                    & (pl.col("status") == "COMPLETED")
                )["container"]
                .drop_nulls()
                .unique()
                .sort()
                .to_list()
            )
            if imgs:
                images[tool] = "; ".join(imgs)

    frames, pile, cards = [], [], {}
    n_queries = None
    for f in metric_files:
        m = pl.read_parquet(f)
        tool, setting, species = m["tool"][0], m["variant"][0], m["species"][0]
        if tool not in TOOL_INFO:
            raise SystemExit(f"{f.name}: tool {tool!r} has no TOOL_INFO entry")
        rel = f.relative_to(a.results).as_posix()
        frames.append(metric_rows(m, a, images, rel))
        calls = a.results / "calls" / f.name.replace(".metrics.", ".calls.")
        if TOOL_INFO[tool]["has_query_coordinates"] and calls.exists():
            pile.append(pileup_calls(calls, tool, setting, target_db(tool, a, species)))
        # QC: <tool>.<setting>.<species>.calls_qc.json for a per-target tool, else the
        # step's one json (place_qc or calls_qc).
        own = a.results / "regions" / f"{tool}.{setting}.{species}.calls_qc.json"
        qc_paths = (
            [own]
            if own.exists()
            else sorted((a.results / "regions").glob(f"{tool}.{setting}.*qc.json"))
        )
        qc = json.loads(qc_paths[0].read_text()) if qc_paths else {}
        n_queries = n_queries or qc.get("n_queries")
        runs = [
            a.results / "hits" / f"{tool}.{setting}.run.txt",
            a.results / "hits" / tool / f"{tool}.{species}.run.txt",
        ]
        run = next((read_run_info(r) for r in runs if r.exists()), {})
        entry = cards.setdefault((tool, setting), {"m": [], "qc": {}, "run": {}})
        entry["m"].append(m)
        entry["qc"][species] = qc
        entry["run"] = entry["run"] or run

    settings = {}
    for tool, setting in cards:
        settings.setdefault(tool, []).append(setting)
    compute = (
        compute_rows(trace, a.experiment, n_queries, str(a.trace), settings)
        if trace is not None
        else pl.DataFrame(schema=COMPUTE_SCHEMA)
    )
    a.out.mkdir(parents=True, exist_ok=True)
    rows = pl.concat(frames)
    rows.write_parquet(a.out / "rows.parquet")
    compute.write_parquet(a.out / "compute.parquet")
    if pile:
        pl.concat(pile).write_parquet(a.out / "pileup_calls.parquet")
    (a.out / "cards").mkdir(exist_ok=True)
    for (tool, setting), e in cards.items():
        qc = next(iter(e["qc"].values())) if len(e["qc"]) == 1 else e["qc"]
        c = card(
            tool,
            setting,
            pl.concat(e["m"], how="diagonal_relaxed"),
            qc,
            e["run"],
            images.get(tool),
            compute,
            a,
        )
        (a.out / "cards" / f"{tool}.{setting}.json").write_text(
            json.dumps(c, indent=1, default=str) + "\n"
        )
    print(
        json.dumps(
            {
                "rows": rows.height,
                "compute_rows": compute.height,
                "pileup_calls": sum(p.height for p in pile),
                "cards": len(cards),
            }
        )
    )


if __name__ == "__main__":
    main()
