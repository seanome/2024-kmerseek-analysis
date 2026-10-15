#!/usr/bin/env python3
"""Build the leaderboard's long-format tables from every finished benchmark output.

    results/leaderboard/results.parquet  one row per (experiment, tool, setting, query set,
                                         target database, stratum, metric)
    results/leaderboard/compute.parquet  one row per (experiment, tool, setting, step), from
                                         Nextflow traces and run logs
    results/leaderboard/coverage.md      tool x stratum axis: instances measured
    results/leaderboard/inventory.md     every output found, with date, commit, and whether
                                         it was used and why not
    results/leaderboard/provenance.json  sha256 of every input and output

--data is a copy of the needed files from Sherlock, laid out exactly as under
/scratch/users/olgabot (`make leaderboard-fetch` makes it). source_path in both tables is
the Sherlock path, so a row can be checked against the original file.

Nothing is estimated. A metric the output does not carry has no row; a compute number the
trace does not carry is null. Strata are only the ones the outputs already hold: target
species (and its divergence from human), identity bin, Swiss-Prot feature type, feature
length bin, pLDDT bin.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import polars as pl

SCRATCH = "/scratch/users/olgabot"
QFO = "2024-kmerseek-analysis/nextflow-runs/qfo-pfam-region-benchmark"
DARK = "invertebrate-dark-set/results-0.4"
DARK_CLONE = (
    "2024-kmerseek-invertebrate-dark-set-0.4/nextflow-runs/invertebrate-dark-set"
)
TWILIGHT = (
    "2024-kmerseek-analysis-deep-twilight/nextflow-runs/deep-twilight-controls/run"
)
SCOPE_RUNS = "2024-kmerseek-scope40-v040/nextflow-runs/scope40-alphabet-ksize-v040/runs"

# Commit of the Sherlock clone that wrote each output: HEAD at the run's launch, read from
# that clone's `git log -g` (HEAD reflog) on 2026-10-15. dark-fmax records its own commit in
# code/COMMIT. Every commit below exists on GitHub.
COMMIT = {
    "qfo_region_mini": "dd17259",
    "qfo_region_midi": "4cd14cc",
    "qfo_region_midi_plus": "a47ea25",
    "elm_cover_chicken": "b26328b",
    "elm_cover_seven_species": "ca051a6",
    "dark_set_gain": "0dad375",
    "dark_set_accuracy": "645e771",
    "dark_set_pfam_fmax": "0cb74bb",
    "alphabet_ranking_three_cases": "bd9cc3b",
    "deep_twilight": "c8cd2dd",
    "scope40_v040": "702de2c",
    "scope40_v040_below_kmin": "46dfecb",
    "scope40_v040_shuffled_k2_seed1": "80ec410",
}

QFO_TARGET_VERSION = "QfO 2020_04 (UP000008143 updated)"
# Truth sets of evaluate_domain_calls.py: Pfam-A regions (Pfam 38.2), Pfam-N (Pfam 35.0
# Pfam-N.gz, nextflow-runs/qfo-pfam-region-benchmark/Makefile PFAMN_RSYNC), Swiss-Prot
# feature table (uniprot_sprot.dat.gz; the release was not recorded, so null).
QFO_TRUTH = {
    "pfam": ("Pfam-A domain instances", "Pfam 38.2"),
    "pfamn": ("Pfam-N domain instances", "Pfam 35.0"),
    "swissprot": ("Swiss-Prot features", None),
}
QFO_METRICS = [
    "fmax",
    "fmax_precision",
    "fmax_recall",
    # The score cut at which fmax is reached (tau in CAFA); its unit is the tool's own
    # score, so it compares only within one tool and setting.
    "fmax_threshold",
    "auprc",
    "recall_reachable",
    "median_iou_tp",
    "family_fmax",
]
STRATUM_AXES = ["all", "identity", "plddt", "feature_type", "feature_length_bin"]

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

# One name per tool across experiments (the dark set and deep-twilight runs say "phmmer").
TOOL_NAME = {"phmmer": "hmmer3_phmmer", "jackhmmer": "hmmer3_jackhmmer"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_DURATION = re.compile(r"(\d+(?:\.\d+)?)(ms|d|h|m|s)")
_SECONDS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def duration_seconds(text: str | None) -> float | None:
    """Nextflow's '1h 2m 3.4s' / '34ms' / '-' as seconds."""
    if text is None or text.strip() in ("", "-"):
        return None
    parts = _DURATION.findall(text)
    if not parts:
        raise ValueError(f"unreadable duration {text!r}")
    return sum(float(n) * _SECONDS[u] for n, u in parts)


_SIZE = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}


def size_gb(text: str | None) -> float | None:
    """Nextflow's '3.3 MB' / '1.1 GB' as GiB."""
    if text is None or text.strip() in ("", "-"):
        return None
    num, unit = text.split()
    return float(num) * _SIZE[unit] / 1024**3


def image_name(container: str | None) -> str | None:
    """'/scratch/.../apptainer-cache/docker.io-olgabot-kmerseek-0.4.0-rc5.img' -> the
    image file's name without the cache folder and '.img'."""
    if container is None or container in ("", "-"):
        return None
    return Path(container).name.removesuffix(".img")


def frame(rows: list[dict], schema: dict) -> pl.DataFrame:
    return (
        pl.DataFrame(rows, schema=schema, orient="row")
        if rows
        else pl.DataFrame(schema=schema)
    )


def tidy(df: pl.DataFrame) -> pl.DataFrame:
    """Add any missing schema column as null and order the columns."""
    for col, dtype in SCHEMA.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    return df.select([pl.col(c).cast(t) for c, t in SCHEMA.items()])


# ---------------------------------------------------------------------------------------
# Traces


def read_trace(path: Path) -> pl.DataFrame:
    """A Nextflow trace as strings, with `step` (the process name) and the file it came from.

    The dark-set runs write a short trace with `name` ("darkSet:kmerseekSearch (tag)")
    and no `process`, `tag`, `cpus` or `container` columns.
    """
    t = pl.read_csv(path, separator="\t", infer_schema=False)
    if "process" not in t.columns:
        t = t.with_columns(
            pl.col("name").str.extract(r"^(\S+)", 1).alias("process"),
            pl.col("name").str.extract(r"\((.*)\)$", 1).alias("tag"),
        )
    for col in ("cpus", "container", "complete"):
        if col not in t.columns:
            t = t.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    return t.with_columns(pl.col("process").str.split(":").list.last().alias("step"))


def read_traces(data: Path, rels: list[str]) -> pl.DataFrame:
    frames = []
    for rel in rels:
        for p in sorted((data / rel).glob("*trace*.txt")):
            t = read_trace(p).with_columns(
                pl.lit(f"{SCRATCH}/{p.relative_to(data)}").alias("trace_file")
            )
            frames.append(t)
    return pl.concat(frames, how="diagonal") if frames else pl.DataFrame()


QFO_SEARCH = {
    "phmmerSearch": "hmmer3_phmmer",
    "jackhmmerSearch": "hmmer3_jackhmmer",
    "hhblitsSearch": "hhblits",
    "foldseekSearch": "foldseek",
    "prostt5Search": "prostt5",
    "reseekSearch": "reseek",
    "hmmscanAnnotate": "hmmscan",
    "folddiscoQuery": "folddisco",
}
QFO_DB = {
    "foldseekDb": "foldseek",
    "prostt5Db": "prostt5",
    "hhblitsBuildDB": "hhblits",
    "mmseqsDb": "mmseqs2",
    "reseekConvert": "reseek",
}
_KMERSEEK_TAG = re.compile(r"([a-z0-9]+)_(.+?)_k(\d+)(?:\.s(\d+))?_lc(false|true)")


def qfo_task(step: str, tag: str, style: str) -> tuple[str, str, str] | None:
    """(tool, setting, target species) of one region-benchmark trace row.

    style "metrics": settings as all_domain_metrics.parquet names them
    ('hp_thomas_dill2_k24_lcFalse'); style "elm": as the ELM cover summaries name them
    ('hp_thomas_dill2.k24.s5.lcfalse').
    """
    tag = tag or ""
    if step in ("kmerseekSearch", "kmerseekIndex", "kmerseekIndexAndSearch"):
        m = _KMERSEEK_TAG.fullmatch(tag)
        if not m:
            return None
        species, alphabet, k, scaled, lc = m.groups()
        if style == "elm":
            s = f".s{scaled}" if scaled else ""
            return "kmerseek", f"{alphabet}.k{k}{s}.lc{lc}", species
        s = f".s{scaled}" if scaled else ""
        return "kmerseek", f"{alphabet}_k{k}{s}_lc{lc.capitalize()}", species
    if step == "mmseqs2Search":
        m = re.fullmatch(r"human_vs_(\w+) \[(mmseqs2_seqseq|mmseqs2_iterative)\]", tag)
        return (m[2], "default" if style == "elm" else "s7", m[1]) if m else None
    if step == "hmmscanAnnotate":
        return "hmmscan", "direct", "all"
    if step in QFO_SEARCH:
        m = re.fullmatch(r"human_vs_(\w+)", tag)
        if not m:
            return None
        setting = {
            "hmmer3_phmmer": "default",
            "hmmer3_jackhmmer": "n3",
            "hhblits": "single_seq",
            "foldseek": "3di_aa",
            "prostt5": "3di_from_seq",
        }.get(QFO_SEARCH[step], "default")
        return QFO_SEARCH[step], "default" if style == "elm" else setting, m[1]
    if step in QFO_DB:
        return QFO_DB[step], "database build", tag
    return None


def summarise_tasks(
    rows: list[dict], experiment: str, n_queries: int | None
) -> list[dict]:
    """Per (tool, setting, step): newest COMPLETED attempt of each task, summed.

    A task is (tool, setting, step, target). CACHED rows are reuse, not a second run, and
    are skipped. cores is the requested CPU count when every task asked the same number,
    else null; cpu_h sums cores x wall time per task.
    """
    if not rows:
        return []
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.sort("complete", nulls_last=True).unique(
        ["tool", "setting", "step", "target"], keep="last", maintain_order=True
    )
    out = []
    for (tool, setting, step), g in df.group_by(
        ["tool", "setting", "step"], maintain_order=True
    ):
        cores = g["cores"].drop_nulls().unique().to_list()
        wall = g["wall_s"]
        cpu = (
            (g["cores"] * g["wall_s"] / 3600) if g["cores"].null_count() == 0 else None
        )
        gpu_rows = g.filter(pl.col("gpu_h").is_not_null())
        out.append(
            {
                "experiment": experiment,
                "tool": tool,
                "setting": setting,
                "step": step,
                "n_tasks": g.height,
                "n_target_dbs": g["target"].n_unique(),
                "wall_s": float(wall.sum()) if wall.null_count() == 0 else None,
                "cores": (
                    cores[0]
                    if len(cores) == 1 and g["cores"].null_count() == 0
                    else None
                ),
                "cpu_h": (
                    float(cpu.sum())
                    if cpu is not None and cpu.null_count() == 0
                    else None
                ),
                "peak_ram_gb": g["peak_ram_gb"].max(),
                "gpu_type": (
                    "1 GPU via --gres=gpu:1 (model not recorded)"
                    if gpu_rows.height
                    else "none"
                ),
                "gpu_h": float(gpu_rows["gpu_h"].sum()) if gpu_rows.height else None,
                "n_queries": n_queries,
                "n_targets": None,
                "source": "; ".join(sorted(g["source"].unique().to_list())),
            }
        )
    return out


def trace_tasks(traces: pl.DataFrame, mapper) -> list[dict]:
    rows = []
    if traces.is_empty():
        return rows
    for r in traces.filter(pl.col("status") == "COMPLETED").iter_rows(named=True):
        key = mapper(r["step"], r["tag"])
        if key is None:
            continue
        tool, setting, target = key
        wall = duration_seconds(r["realtime"])
        cores = int(r["cpus"]) if r["cpus"] not in (None, "", "-") else None
        is_gpu = r["step"] == "prostt5Db"
        rows.append(
            {
                "tool": tool,
                "setting": setting,
                "step": r["step"],
                "target": target,
                "complete": r["complete"],
                "wall_s": wall,
                "cores": cores,
                "peak_ram_gb": size_gb(r["peak_rss"]),
                "gpu_h": wall / 3600 if is_gpu and wall is not None else None,
                "source": r["trace_file"],
            }
        )
    return rows


def tool_images(traces: pl.DataFrame, mapper) -> dict[str, str]:
    """tool -> the image(s) its COMPLETED search tasks ran in, from the trace."""
    if traces.is_empty() or traces["container"].null_count() == traces.height:
        return {}
    seen: dict[str, set] = {}
    for r in traces.filter(pl.col("status") == "COMPLETED").iter_rows(named=True):
        key = mapper(r["step"], r["tag"])
        img = image_name(r["container"])
        if key is None or img is None or key[1] == "database build":
            continue
        seen.setdefault(key[0], set()).add(img)
    return {t: "; ".join(sorted(v)) for t, v in seen.items()}


# ---------------------------------------------------------------------------------------
# Results readers


def qfo_metrics(
    path: Path,
    rel: str,
    experiment: str,
    query_set: str,
    images: dict[str, str],
    *,
    truth: dict | None = None,
    target_db: pl.Expr | None = None,
    target_db_version: str | None = QFO_TARGET_VERSION,
    query_from_truth_set: bool = False,
) -> pl.DataFrame:
    """all_domain_metrics-shaped table (evaluate_domain_calls.py) to long rows."""
    if path.suffix == ".csv":
        m = pl.read_csv(path, infer_schema_length=200_000)
    else:
        m = pl.read_parquet(path)
    m = m.filter(
        (pl.col("split") == "all") & pl.col("stratum_axis").is_in(STRATUM_AXES)
    )
    # The region runs' identity step never matched Swiss-Prot features to a target: their
    # only identity bin is "no_homolog", and its numbers equal the whole set's (57,148 of
    # 57,148 rows, checked 2026-10-15). A label that says nothing true is left out.
    m = m.filter(
        ~((pl.col("truth_set") == "swissprot") & (pl.col("stratum_axis") == "identity"))
    )
    metrics = [c for c in QFO_METRICS if c in m.columns]
    truth = truth or QFO_TRUTH
    long = (
        m.select(
            "truth_set",
            "tool",
            "variant",
            "species",
            "species_mya",
            "stratum_axis",
            "stratum",
            "dedup_transfers",
            "n_truth_instances",
            *metrics,
        )
        .unpivot(
            index=[
                "truth_set",
                "tool",
                "variant",
                "species",
                "species_mya",
                "stratum_axis",
                "stratum",
                "dedup_transfers",
                "n_truth_instances",
            ],
            on=metrics,
            variable_name="metric",
            value_name="value",
        )
        .filter(pl.col("value").is_not_null())
    )
    truth_source = pl.col("truth_set").replace_strict(
        {k: v[0] for k, v in truth.items()}, default=pl.col("truth_set")
    )
    truth_version = pl.col("truth_set").replace_strict(
        {k: v[1] for k, v in truth.items()}, default=None, return_dtype=pl.String
    )
    qs = (
        (pl.col("species") + " " + pl.col("truth_set"))
        if query_from_truth_set
        else pl.lit(query_set)
    )
    tool = pl.col("tool").replace(TOOL_NAME)
    return tidy(
        long.with_columns(
            pl.lit(experiment).alias("experiment"),
            tool.alias("tool"),
            tool.replace_strict(images, default=None, return_dtype=pl.String).alias(
                "tool_version"
            ),
            pl.col("variant").alias("setting"),
            qs.alias("query_set"),
            (
                target_db
                if target_db is not None
                else pl.when(pl.col("tool") == "hmmscan")
                .then(pl.lit("Pfam-A HMM library"))
                .otherwise("QfO " + pl.col("species"))
            ).alias("target_db"),
            pl.lit(target_db_version).alias("target_db_version"),
            pl.col("species_mya").cast(pl.Float64).alias("target_mya_to_human"),
            pl.col("stratum").cast(pl.String).alias("stratum_bin"),
            pl.col("value").cast(pl.Float64),
            pl.col("n_truth_instances").cast(pl.Int64).alias("n_instances"),
            truth_source.alias("truth_source"),
            truth_version.alias("truth_version"),
            pl.when(pl.col("dedup_transfers"))
            .then(pl.lit("dedup"))
            .otherwise(pl.lit("raw"))
            .alias("dedup_or_raw"),
            pl.lit(f"{SCRATCH}/{rel}").alias("source_path"),
            pl.lit(COMMIT[experiment]).alias("source_commit"),
        )
    )


def elm_cover(data: Path, images: dict[str, str]) -> pl.DataFrame:
    """scripts/reduce_elm_landing.py summaries: one JSON per (species, setting)."""
    rows = []
    for p in sorted((data / "250-elm-cover/cover").glob("*/*.summary.json")):
        s = json.loads(p.read_text())
        if s.get("status") != "ok":
            continue
        species = p.parent.name
        arm = s["arm"]
        tool = "kmerseek" if re.match(r".+\.k\d+", arm) else TOOL_NAME.get(arm, arm)
        experiment = (
            "elm_cover_chicken" if species == "chicken" else "elm_cover_seven_species"
        )
        base = {
            "experiment": "elm_cover",
            "tool": tool,
            "tool_version": images.get(tool),
            "setting": arm if tool == "kmerseek" else "default",
            "query_set": "human ELM true-positive instances (ELM, downloaded 2026-09-24)",
            "target_db": f"QfO {species}",
            "target_db_version": QFO_TARGET_VERSION,
            "stratum_axis": "all",
            "stratum_bin": "all",
            "truth_version": "ELM 2026-09-24",
            "dedup_or_raw": "raw",
            "source_path": f"{SCRATCH}/{p.relative_to(data)}",
            "source_commit": COMMIT[experiment],
        }
        # (metric, numerator, denominator, truth). Shares are the file's own two counts.
        for metric, num, den, truth in [
            (
                "motif_on_ortholog_position_share",
                "n_ortholog_on_position",
                "n_with_projection",
                "ELM motif position projected onto the 1:1 OMA ortholog",
            ),
            (
                "ortholog_covered_placed_share",
                "n_ortholog_covered_placed",
                "n_with_ortholog",
                "ELM motif with a 1:1 OMA ortholog",
            ),
            (
                "instance_covered_placed_share",
                "n_covered_placed",
                "n_instances",
                "ELM motif instance",
            ),
        ]:
            if s.get(den):
                rows.append(
                    base
                    | {
                        "metric": metric,
                        "value": s[num] / s[den],
                        "n_instances": s[den],
                        "truth_source": truth,
                    }
                )
            rows.append(
                base
                | {
                    "metric": num,
                    "value": float(s[num]),
                    "n_instances": s[den],
                    "truth_source": truth,
                }
            )
    return tidy(frame(rows, SCHEMA)) if rows else tidy(pl.DataFrame())


def dark_gain(data: Path) -> pl.DataFrame:
    """kmerseekDarkGain: share of dark (and of placed) proteins with any kmerseek region."""
    frames = []
    for p in sorted((data / DARK).glob("*/*_kmerseek_dark_gain.parquet")):
        g = pl.read_parquet(p)
        setting = (
            pl.col("alphabet")
            + ".k"
            + pl.col("ksize").cast(pl.String)
            + ".s"
            + pl.col("scaled").cast(pl.String)
            + ".lc"
            + pl.col("low_complexity_mask").cast(pl.String)
            + "."
            + pl.col("extension")
            + pl.when(pl.col("evalue_max").is_not_null())
            .then(pl.lit(".E<=") + pl.col("evalue_max").cast(pl.String))
            .otherwise(pl.lit(""))
        )
        for group, n_col, hit_col, frac_col in [
            ("dark", "dark_proteins", "dark_reached", "fraction_dark_reached"),
            ("placed", "placed_proteins", "placed_reached", "fraction_placed_reached"),
        ]:
            base = g.with_columns(setting.alias("setting"))
            for metric, col in [
                (f"{group}_reached", hit_col),
                (f"fraction_{group}_reached", frac_col),
            ]:
                frames.append(
                    base.select(
                        pl.lit("dark_set_gain").alias("experiment"),
                        pl.lit("kmerseek").alias("tool"),
                        pl.lit("olgabot/kmerseek:0.4.0-rc5").alias("tool_version"),
                        "setting",
                        (
                            pl.col("species")
                            + f" {group} proteins (no sequence-tool hit at E<1)"
                            if group == "dark"
                            else pl.col("species") + " placed proteins"
                        ).alias("query_set"),
                        pl.lit("Swiss-Prot reviewed, query clade removed").alias(
                            "target_db"
                        ),
                        pl.lit("all").alias("stratum_axis"),
                        pl.lit("all").alias("stratum_bin"),
                        pl.lit(metric).alias("metric"),
                        pl.col(col).cast(pl.Float64).alias("value"),
                        pl.col(n_col).cast(pl.Int64).alias("n_instances"),
                        pl.lit("none: any kmerseek region counts as reached").alias(
                            "truth_source"
                        ),
                        pl.lit("raw").alias("dedup_or_raw"),
                        pl.lit(f"{SCRATCH}/{p.relative_to(data)}").alias("source_path"),
                        pl.lit(COMMIT["dark_set_gain"]).alias("source_commit"),
                    )
                )
    return tidy(pl.concat(frames, how="diagonal")) if frames else tidy(pl.DataFrame())


def dark_accuracy(data: Path) -> pl.DataFrame:
    """tools/dark-set-accuracy: best hit per protein judged by Pfam family or clan."""
    rows = []
    for p in sorted((data / DARK / "accuracy").glob("*_accuracy.json")):
        j = json.loads(p.read_text())
        for tool_key, by_set in j["counts"].items():
            tool = TOOL_NAME.get(tool_key, tool_key)
            is_kmerseek = tool_key not in ("phmmer", "jackhmmer", "mmseqs2")
            for set_name, per_cut in by_set.items():
                n = j["sets"][set_name]["n"]
                for cut, counts in zip(j["hit_cutoffs"], per_cut):
                    for col, v in zip(j["columns"], counts):
                        rows.append(
                            {
                                "experiment": "dark_set_accuracy",
                                "tool": "kmerseek" if is_kmerseek else tool,
                                "setting": f"{tool_key if is_kmerseek else 'default'}|best hit E<={cut}",
                                "query_set": f"{j['species']} {set_name}",
                                "target_db": "Swiss-Prot reviewed, query clade removed",
                                "stratum_axis": "all",
                                "stratum_bin": "all",
                                "metric": f"n_best_hit_{col}",
                                "value": float(v),
                                "n_instances": n,
                                "truth_source": "Pfam family or clan of the UniProt entry",
                                "dedup_or_raw": "raw",
                                "source_path": f"{SCRATCH}/{p.relative_to(data)}",
                                "source_commit": COMMIT["dark_set_accuracy"],
                            }
                        )
    return tidy(frame(rows, {**SCHEMA})) if rows else tidy(pl.DataFrame())


def alphabet_ranking(data: Path) -> pl.DataFrame:
    """nb 241 ranks.csv: where the known partner ranks among 19,732 human proteins."""
    rel = "241-alphabet-ranking/ranks.csv"
    r = pl.read_csv(data / rel).filter(pl.col("partner").is_not_null())
    base = r.with_columns(
        (
            pl.col("alphabet")
            + ".k"
            + pl.col("ksize").cast(pl.String)
            + "|"
            + pl.col("metric")
        ).alias("setting"),
        (pl.col("query") + " (known partner " + pl.col("partner") + ")").alias(
            "query_set"
        ),
    )
    common = [
        pl.lit("alphabet_ranking_three_cases").alias("experiment"),
        pl.lit("kmerseek").alias("tool"),
        "setting",
        "query_set",
        pl.lit("GENCODE human canonical proteins (19,732)").alias("target_db"),
        pl.lit("GENCODE v49").alias("target_db_version"),
        pl.lit("all").alias("stratum_axis"),
        pl.lit("all").alias("stratum_bin"),
        pl.lit(1, dtype=pl.Int64).alias("n_instances"),
        pl.lit("known partner protein").alias("truth_source"),
        pl.lit("raw").alias("dedup_or_raw"),
        pl.lit(f"{SCRATCH}/{rel}").alias("source_path"),
        pl.lit(COMMIT["alphabet_ranking_three_cases"]).alias("source_commit"),
    ]
    found = base.filter(pl.col("partner_found").is_not_null()).select(
        *common,
        pl.lit("partner_found").alias("metric"),
        pl.col("partner_found").cast(pl.Float64).alias("value"),
    )
    rank = base.filter(pl.col("rank").is_not_null()).select(
        *common,
        pl.lit("partner_rank").alias("metric"),
        pl.col("rank").cast(pl.Float64).alias("value"),
    )
    return tidy(pl.concat([found, rank]))


def deep_twilight(data: Path, images: dict[str, str]) -> pl.DataFrame:
    """nb 252 outcomes.parquet: per (tool, setting, protein family)."""
    rel = f"{TWILIGHT}/results/outcomes.parquet"
    o = pl.read_parquet(data / rel).with_columns(
        pl.col("tool").replace(TOOL_NAME),
        pl.when(pl.col("tool") == "kmerseek")
        .then(pl.col("alphabet") + ".k" + pl.col("ksize").cast(pl.String))
        .otherwise(pl.lit("default"))
        .alias("setting"),
    )
    g = o.group_by("tool", "setting", "family").agg(
        pl.len().alias("n_pairs"),
        (pl.col("outcome") == "correct").sum().alias("n_correct"),
        (pl.col("outcome") == "carried wrongly").sum().alias("n_carried_wrongly"),
        pl.col("needle_identity_pct")
        .filter(pl.col("outcome") == "correct")
        .min()
        .alias("lowest_identity_pct_correct"),
    )
    long = g.unpivot(
        index=["tool", "setting", "family", "n_pairs"],
        on=["n_correct", "n_carried_wrongly", "lowest_identity_pct_correct"],
        variable_name="metric",
        value_name="value",
    ).filter(pl.col("value").is_not_null())
    return tidy(
        long.with_columns(
            pl.lit("deep_twilight").alias("experiment"),
            pl.col("tool")
            .replace_strict(images, default=None, return_dtype=pl.String)
            .alias("tool_version"),
            (pl.col("family") + " label-transfer pairs").alias("query_set"),
            pl.lit("QfO human proteome + 8 non-human family members").alias(
                "target_db"
            ),
            pl.lit("all").alias("stratum_axis"),
            pl.lit("all").alias("stratum_bin"),
            pl.col("value").cast(pl.Float64),
            pl.col("n_pairs").cast(pl.Int64).alias("n_instances"),
            pl.lit("UniProt functional site label on the target").alias("truth_source"),
            pl.lit("raw").alias("dedup_or_raw"),
            pl.lit(f"{SCRATCH}/{rel}").alias("source_path"),
            pl.lit(COMMIT["deep_twilight"]).alias("source_commit"),
        )
    )


SCOPE_EXPERIMENTS = {
    "scope40_v040": "scope/results-scope40-alphabet-ksize-v040",
    "scope40_v040_below_kmin": "scope/results-scope40-alphabet-ksize-v040-below-kmin",
    "scope40_v040_shuffled_k2_seed1": "scope/results-scope40-alphabet-ksize-v040-shuffled-k2-seed1",
}


def scope_setting() -> pl.Expr:
    return (
        pl.col("alphabet")
        + ".k"
        + pl.col("ksize").cast(pl.String)
        + "."
        + pl.col("setting")
        + "|"
        + pl.col("score")
    )


def scope40(
    data: Path, experiment: str, rel_dir: str, image: str | None
) -> pl.DataFrame:
    """scope40_v040_auc.tsv (AUC to first false positive per level) and, where written,
    scope40_v040_identity_bands.tsv (true pairs above the first false positive, per
    identity bin)."""
    frames = []
    query = (
        "SCOPe40 2.08 all queries"
        if "shuffled" not in experiment
        else ("SCOPe40 2.08 all queries, alphabet letters shuffled (k=2, seed 1)")
    )
    common = [
        pl.lit(experiment).alias("experiment"),
        pl.lit("kmerseek").alias("tool"),
        pl.lit(image).alias("tool_version"),
        scope_setting().alias("setting"),
        pl.lit("SCOPe40 2.08 (ASTRAL 40%)").alias("target_db"),
        pl.lit("2.08").alias("target_db_version"),
        pl.lit("SCOPe 2.08").alias("truth_version"),
        pl.lit("raw").alias("dedup_or_raw"),
        pl.lit(COMMIT[experiment]).alias("source_commit"),
    ]
    auc_rel = f"{rel_dir}/scope40_v040_auc.tsv"
    a = pl.read_csv(data / auc_rel, separator="\t", infer_schema_length=200_000).filter(
        (pl.col("status") == "ok") & pl.col("level").is_not_null()
    )
    for metric, n_col, qs in [
        ("auc_all", "n_queries_all", query),
        ("auc_ref", "n_queries_ref", query.replace("all queries", "reference queries")),
    ]:
        frames.append(
            a.select(
                *common,
                pl.lit(qs).alias("query_set"),
                pl.lit("all").alias("stratum_axis"),
                pl.lit("all").alias("stratum_bin"),
                pl.lit(metric).alias("metric"),
                pl.col(metric).cast(pl.Float64).alias("value"),
                pl.col(n_col).cast(pl.Int64).alias("n_instances"),
                ("SCOPe " + pl.col("level")).alias("truth_source"),
                pl.lit(f"{SCRATCH}/{auc_rel}").alias("source_path"),
            )
        )
    bands_rel = f"{rel_dir}/scope40_v040_identity_bands.tsv"
    if (data / bands_rel).exists():
        b = pl.read_csv(
            data / bands_rel, separator="\t", infer_schema_length=200_000
        ).filter(
            (pl.col("status") == "ok")
            & pl.col("category").is_not_null()
            & pl.col("identity_band").is_not_null()
        )
        for metric, col in [
            ("pairs_above_first_fp", "n_above_first_fp"),
            ("pairs_reported", "n_reported"),
        ]:
            frames.append(
                b.select(
                    *common,
                    pl.lit(query).alias("query_set"),
                    pl.lit("identity").alias("stratum_axis"),
                    pl.col("identity_band").alias("stratum_bin"),
                    pl.lit(metric).alias("metric"),
                    pl.col(col).cast(pl.Float64).alias("value"),
                    pl.col("n_pairs").cast(pl.Int64).alias("n_instances"),
                    ("SCOPe pairs: " + pl.col("category")).alias("truth_source"),
                    pl.lit(f"{SCRATCH}/{bands_rel}").alias("source_path"),
                )
            )
    return tidy(pl.concat(frames, how="diagonal"))


# ---------------------------------------------------------------------------------------
# Compute readers


def runs_jsonl_compute(data: Path) -> list[dict]:
    """nb 241: runs.jsonl records wall seconds of each index and search; no core count."""
    rel = "241-alphabet-ranking/runs.jsonl"
    out = []
    for i, line in enumerate((data / rel).read_text().splitlines(), start=1):
        r = json.loads(line)
        for step, key, rc in [
            ("index", "index_s", "index_rc"),
            ("search", "search_s", "search_rc"),
        ]:
            if r.get(rc) != 0 or r.get(key) is None:
                continue
            out.append(
                {
                    "experiment": "alphabet_ranking_three_cases",
                    "tool": "kmerseek",
                    "setting": f"{r['alphabet']}.k{r['ksize']}",
                    "step": step,
                    "n_tasks": 1,
                    "n_target_dbs": 1,
                    "wall_s": float(r[key]),
                    "cores": None,
                    "cpu_h": None,
                    "peak_ram_gb": None,
                    "gpu_type": "none",
                    "gpu_h": None,
                    "n_queries": 3 if step == "search" else None,
                    "n_targets": 19_732,
                    "source": f"{SCRATCH}/{rel} line {i}",
                }
            )
    return out


def twilight_task(step: str, tag: str):
    if step == "kmerseekArm":
        m = re.fullmatch(r"(\S+) k(\d+)", tag or "")
        return ("kmerseek", f"{m[1]}.k{m[2]}", "db") if m else None
    tool = {
        "phmmerSearch": "hmmer3_phmmer",
        "mmseqsSearch": "mmseqs2",
        "foldseekSearch": "foldseek",
    }
    return (tool[step], "default", "db") if step in tool else None


def scope_task(step: str, tag: str):
    if step == "searchAndScore":
        m = re.fullmatch(r"(\S+)\.k(\d+)\.(\S+)", tag or "")
        return ("kmerseek", f"{m[1]}.k{m[2]}.{m[3]}", "scope40") if m else None
    if step == "kmerseekIndex":
        m = re.fullmatch(r"(\S+)\.k(\d+)", tag or "")
        return ("kmerseek", f"{m[1]}.k{m[2]}", "scope40") if m else None
    return None


def dark_task(step: str, tag: str):
    if step == "kmerseekSearch":
        m = re.fullmatch(r"(\w+)\.chunk_\d+\.(.+)", tag or "")
        return ("kmerseek", m[2], m[1]) if m else None
    return None


# ---------------------------------------------------------------------------------------
# Coverage and inventory


COVERAGE_AXES = [
    "target species",
    "identity",
    "plddt",
    "feature_type",
    "feature_length_bin",
]


def coverage_markdown(results: pl.DataFrame) -> str:
    """Tool x stratum axis, each square the number of truth instances measured.

    An instance is counted once per (experiment, query set, target database, truth source,
    stratum bin), whatever the number of settings or metrics scored on it: the largest
    n_instances over those rows. "target species" counts the whole-set rows (stratum
    'all') of every experiment, each target database once.
    """
    keys = ["experiment", "query_set", "target_db", "truth_source", "stratum_bin"]
    r = results.filter(pl.col("n_instances").is_not_null()).with_columns(
        pl.when(pl.col("stratum_axis") == "all")
        .then(pl.lit("target species"))
        .otherwise(pl.col("stratum_axis"))
        .alias("axis")
    )
    per = (
        r.group_by("tool", "axis", *keys)
        .agg(pl.col("n_instances").max())
        .group_by("tool", "axis")
        .agg(pl.col("n_instances").sum())
    )
    tools = sorted(results["tool"].unique().to_list())
    cell = {(t, a): n for t, a, n in per.iter_rows()}

    def grid(rows, cols, lookup, corner):
        lines = [
            f"| {corner} | " + " | ".join(cols) + " |",
            "|---|" + "---:|" * len(cols),
        ]
        for row in rows:
            vals = [lookup.get((row, c)) for c in cols]
            lines.append(
                f"| {row} | " + " | ".join(f"{v:,}" if v else "" for v in vals) + " |"
            )
        return lines

    exp = (
        r.group_by("tool", "experiment", *keys[1:])
        .agg(pl.col("n_instances").max())
        .group_by("tool", "experiment")
        .agg(pl.col("n_instances").sum())
    )
    experiments = sorted(results["experiment"].unique().to_list())
    cell_exp = {(t, e): n for t, e, n in exp.iter_rows()}
    lines = [
        "# Coverage: what was measured, and what was not run",
        "",
        "Each square is the number of truth instances measured for that tool on that axis,",
        "summed over experiments, query sets, target databases and truth sources. A blank",
        "square means nothing was run. An instance scored under many settings counts once.",
        "Axes are only the strata the outputs carry: whole set per target species, sequence",
        "identity bin, AlphaFold pLDDT bin, Swiss-Prot feature type, feature length bin.",
        "",
        *grid(tools, COVERAGE_AXES, cell, "tool \\ axis"),
        "",
        "## By experiment",
        "",
        *grid(tools, experiments, cell_exp, "tool \\ experiment"),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("results/leaderboard"))
    args = ap.parse_args()
    data = args.data

    # Traces per experiment, and the tool images they record.
    qfo_runs = {
        "qfo_region_mini": [f"{QFO}/run-mini"],
        "qfo_region_midi": [f"{QFO}/run-midi"],
        "qfo_region_midi_plus": [
            f"{QFO}/run-midi-plus",
            f"{QFO}/run-midi-plus-all-qfo",
        ],
        "elm_cover": [f"{QFO}/run-elm-search", f"{QFO}/run-elm-search-all"],
    }
    traces = {k: read_traces(data, v) for k, v in qfo_runs.items()}
    traces["deep_twilight"] = read_traces(data, [TWILIGHT])
    traces["dark_set_gain"] = read_traces(data, [f"{DARK_CLONE}/runs/ladder-0.4-midi"])
    for exp in SCOPE_EXPERIMENTS:
        sub = {
            "scope40_v040": "main",
            "scope40_v040_below_kmin": "below-kmin",
            "scope40_v040_shuffled_k2_seed1": "shuffled-k2-seed1",
        }[exp]
        traces[exp] = read_traces(data, [f"{SCOPE_RUNS}/{sub}"])

    def mapper(style):
        return lambda step, tag: qfo_task(step, tag, style)

    n_queries = {}
    for exp, name in [
        ("qfo_region_mini", "mini/mini"),
        ("qfo_region_midi", "midi/midi"),
        ("qfo_region_midi_plus", "midi-plus/midi"),
    ]:
        n_queries[exp] = json.loads(
            (data / f"{QFO}/data/{name}_testset_summary.json").read_text()
        )["n_queries"]

    parts = []
    for exp, sub, label in [
        ("qfo_region_mini", "mini", "mini"),
        ("qfo_region_midi", "midi", "midi"),
        ("qfo_region_midi_plus", "midi-plus", "midi-plus"),
    ]:
        rel = f"{QFO}/data/{sub}/results/all_domain_metrics.parquet"
        parts.append(
            qfo_metrics(
                data / rel,
                rel,
                exp,
                # No query count: the proteins scored differ by truth set (midi: 964
                # Pfam-A, 932 Pfam-N, 899 Swiss-Prot); compute.parquet has n_queries.
                f"human Pfam-annotated proteins, {label} set",
                tool_images(traces[exp], mapper("metrics")),
            )
        )
    dark_rel = f"{DARK}/dark-fmax-2026-10-14/dark_fmax_metrics.csv"
    parts.append(
        qfo_metrics(
            data / dark_rel,
            dark_rel,
            "dark_set_pfam_fmax",
            "",
            {"kmerseek": "olgabot/kmerseek:0.4.0-rc5"},
            truth={
                t: (
                    "Pfam domains from InterPro protein2ipr",
                    "Pfam 38.2 / InterPro 110.0",
                )
                for t in ("placed", "dark_E1", "dark_E10")
            },
            target_db=pl.lit("Swiss-Prot reviewed, query clade removed"),
            target_db_version=None,
            query_from_truth_set=True,
        )
    )
    parts.append(elm_cover(data, tool_images(traces["elm_cover"], mapper("elm"))))
    parts.append(dark_gain(data))
    parts.append(dark_accuracy(data))
    parts.append(alphabet_ranking(data))
    parts.append(
        deep_twilight(data, tool_images(traces["deep_twilight"], twilight_task))
    )
    for exp, rel_dir in SCOPE_EXPERIMENTS.items():
        img = tool_images(traces[exp], scope_task).get("kmerseek")
        parts.append(scope40(data, exp, rel_dir, img))
    results = pl.concat(parts, how="vertical").sort(
        "experiment",
        "tool",
        "setting",
        "query_set",
        "target_db",
        "truth_source",
        "dedup_or_raw",
        "stratum_axis",
        "stratum_bin",
        "metric",
    )
    dup = (
        results.group_by(
            "experiment",
            "tool",
            "setting",
            "query_set",
            "target_db",
            "truth_source",
            "dedup_or_raw",
            "stratum_axis",
            "stratum_bin",
            "metric",
        )
        .len()
        .filter(pl.col("len") > 1)
    )
    if dup.height:
        raise SystemExit(f"duplicate result keys:\n{dup.head(10)}")

    # Compute.
    compute_rows = []
    for exp, style in [
        ("qfo_region_mini", "metrics"),
        ("qfo_region_midi", "metrics"),
        ("qfo_region_midi_plus", "metrics"),
        ("elm_cover", "elm"),
    ]:
        # The traces also hold searches of species these results never scored (the
        # all-QfO launch folder searched 77 targets); keep the scored species, the
        # pooled hmmscan step ("all") and the query-side database builds ("human-<hash>").
        scored = set(
            results.filter(pl.col("experiment") == exp)["target_db"]
            .str.strip_prefix("QfO ")
            .unique()
            .to_list()
        ) | {"all"}
        tasks = [
            t
            for t in trace_tasks(traces[exp], mapper(style))
            if t["target"] in scored or t["target"].startswith("human")
        ]
        compute_rows += summarise_tasks(tasks, exp, n_queries.get(exp))
    compute_rows += summarise_tasks(
        trace_tasks(traces["deep_twilight"], twilight_task), "deep_twilight", 16
    )
    for exp in SCOPE_EXPERIMENTS:
        compute_rows += summarise_tasks(trace_tasks(traces[exp], scope_task), exp, None)
    compute_rows += runs_jsonl_compute(data)
    dark_rows = summarise_tasks(
        trace_tasks(traces["dark_set_gain"], dark_task), "dark_set_gain", None
    )
    # Keep a tool's cost only where the same experiment scored that tool (Reseek and
    # Folddisco searched in the midi-plus run but have no rows in its metrics). A database
    # build named after a tool family ("mmseqs2") belongs to its modes ("mmseqs2_seqseq").
    scored_tools = {
        (e, t) for e, t in results.select("experiment", "tool").unique().iter_rows()
    }
    compute_rows = [
        r
        for r in compute_rows
        if any(
            e == r["experiment"] and (t == r["tool"] or t.startswith(r["tool"] + "_"))
            for e, t in scored_tools
        )
    ]
    compute = frame(compute_rows + dark_rows, COMPUTE_SCHEMA)
    # The dark-set run's last trace holds only CACHED rows (its start truncated the earlier
    # heads' traces), so its searches have a row with every number null.
    dark_trace = (
        f"{SCRATCH}/{DARK_CLONE}/runs/ladder-0.4-midi/trace.ladder-0.4-midi.txt"
    )
    measured = set(
        compute.filter(pl.col("experiment") == "dark_set_gain")["setting"].to_list()
    )
    missing_dark = (
        results.filter(pl.col("experiment") == "dark_set_gain")
        .select(pl.col("setting").str.replace(r"\.E<=.*$", ""))
        .unique()
        .filter(~pl.col("setting").is_in(list(measured)))
        .sort("setting")["setting"]
        .to_list()
    )
    compute = pl.concat(
        [
            compute,
            frame(
                [
                    {
                        "experiment": "dark_set_gain",
                        "tool": "kmerseek",
                        "setting": s,
                        "step": "kmerseekSearch",
                        "n_tasks": None,
                        "n_target_dbs": None,
                        "wall_s": None,
                        "cores": None,
                        "cpu_h": None,
                        "peak_ram_gb": None,
                        "gpu_type": None,
                        "gpu_h": None,
                        "n_queries": None,
                        "n_targets": None,
                        "source": f"{dark_trace}: only CACHED rows, no measured run",
                    }
                    for s in missing_dark
                ],
                COMPUTE_SCHEMA,
            ),
        ]
    ).sort("experiment", "tool", "setting", "step")

    args.outdir.mkdir(parents=True, exist_ok=True)
    results.write_parquet(args.outdir / "results.parquet", statistics=False)
    compute.write_parquet(args.outdir / "compute.parquet", statistics=False)
    (args.outdir / "coverage.md").write_text(coverage_markdown(results))

    summaries = {
        f"{SCRATCH}/{QFO}/data/{name}_testset_summary.json"
        for name in ("mini/mini", "midi/midi", "midi-plus/midi")
    }
    inputs = sorted(
        {r for r in results["source_path"].unique().to_list()}
        | summaries
        | {
            s.split(" line ")[0].split(": only")[0]
            for src in compute["source"].unique().to_list()
            for s in src.split("; ")
        }
    )
    provenance = {
        "inputs": {p: sha256(data / Path(p).relative_to(SCRATCH)) for p in inputs},
        "outputs": {
            n: sha256(args.outdir / n)
            for n in ("results.parquet", "compute.parquet", "coverage.md")
        },
    }
    (args.outdir / "provenance.json").write_text(
        json.dumps(provenance, indent=1) + "\n"
    )
    print(results.group_by("experiment").len().sort("experiment"))
    print(f"results.parquet  {results.height:>9,} rows")
    print(f"compute.parquet  {compute.height:>9,} rows")


if __name__ == "__main__":
    main()
