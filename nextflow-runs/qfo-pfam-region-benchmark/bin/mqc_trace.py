#!/usr/bin/env python3
"""Parse a Nextflow trace file into a tidy table of per-task resource use.

Nextflow writes the trace incrementally, one row per task as it completes, in
human-readable units ("2h 13m 7s", "65.2 MB"). Everything here converts those back to
seconds and bytes so they can be summed and plotted. The alternative -- setting
`raw = true` in the trace scope -- would give machine units for free but makes the file
useless to read at the terminal, which is what it is mostly used for.

Cached tasks keep the resource figures from the execution that filled the cache, so a
`-resume` run still reports honest numbers for work it did not repeat. Rows whose status
never reached a terminal state have empty duration fields and are dropped from the
resource plots, but still counted in the status tally.
"""

import re
from pathlib import Path

import polars as pl

# Nextflow's own units. Memory is binary (MemoryUnit), durations are what Duration.toString
# emits: a space-separated run of <number><unit> tokens, largest first.
_MEM_UNITS = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4,
              "PB": 1024**5, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}
_TIME_UNITS = {"ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
_TIME_TOKEN = re.compile(r"(\d+(?:\.\d+)?)\s*(ms|[smhd])")

# The arm each process belongs to, using the same tool labels the metrics table uses, so a
# resource number can be joined to the accuracy number for the same tool. Processes that
# serve every arm (truth building, scoring, aggregation) are grouped as overhead rather
# than charged to any one tool.
PROCESS_TO_TOOL = {
    "kmerseekIndexAndSearch": "kmerseek",
    "phmmerSearch": "hmmer3_phmmer",
    "jackhmmerSearch": "hmmer3_jackhmmer",
    "mmseqs2Search": "mmseqs2",
    "hhblitsSearch": "hhblits",
    "hhblitsBuildDB": "hhblits",
    "foldseekSearch": "foldseek",
    "reseekSearch": "reseek",
    "reseekConvert": "reseek",
    "prostt5Search": "prostt5",
    "prostt5Weights": "prostt5",
    "folddiscoIndex": "folddisco",
    "folddiscoQuery": "folddisco",
    "folddiscoMerge": "folddisco",
    "hmmscanAnnotate": "hmmscan",
}

# Processes that actually run a search. Throughput is only meaningful for these: a task
# that builds a database or scores a parquet is not answering queries.
SEARCH_PROCESSES = {
    "kmerseekIndexAndSearch", "phmmerSearch", "jackhmmerSearch", "mmseqs2Search",
    "hhblitsSearch", "foldseekSearch", "reseekSearch", "prostt5Search",
    "folddiscoQuery", "hmmscanAnnotate",
}


def parse_duration(text: str | None) -> float | None:
    """'2h 13m 7s' -> 7987.0 seconds. Returns None for '-' and empty fields."""
    if not text or text in ("-", "0"):
        return None
    tokens = _TIME_TOKEN.findall(text)
    if not tokens:
        return None
    return sum(float(v) * _TIME_UNITS[u] for v, u in tokens)


def parse_memory(text: str | None) -> float | None:
    """'65.2 MB' -> 68_366_336.0 bytes."""
    if not text or text == "-":
        return None
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*([KMGTP]?B?)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    unit = (m.group(2) or "B").upper()
    return float(m.group(1)) * _MEM_UNITS.get(unit, 1)


def parse_percent(text: str | None) -> float | None:
    if not text or text == "-":
        return None
    try:
        return float(text.rstrip("%"))
    except ValueError:
        return None


def _apply(df: pl.DataFrame, col: str, fn, name: str) -> pl.DataFrame:
    if col not in df.columns:
        return df.with_columns(pl.lit(None, dtype=pl.Float64).alias(name))
    return df.with_columns(
        pl.col(col).map_elements(fn, return_dtype=pl.Float64).alias(name)
    )


def load_trace(path: Path) -> pl.DataFrame:
    """Read one trace file. Returns an empty frame with the right schema if unusable."""
    empty = pl.DataFrame(
        schema={"process": pl.String, "tag": pl.String, "status": pl.String,
                "exit": pl.String, "attempt": pl.Int64, "cpus": pl.Int64,
                "realtime_s": pl.Float64, "duration_s": pl.Float64,
                "peak_rss_b": pl.Float64, "requested_mem_b": pl.Float64,
                "pct_cpu": pl.Float64, "read_b": pl.Float64, "write_b": pl.Float64,
                "tool": pl.String, "is_search": pl.Boolean, "cpu_hours": pl.Float64,
                "mem_used_frac": pl.Float64}
    )
    if path is None or not Path(path).exists():
        return empty
    df = pl.read_csv(path, separator="\t", infer_schema_length=0, truncate_ragged_lines=True)
    if df.height == 0 or "process" not in df.columns:
        return empty

    df = _apply(df, "realtime", parse_duration, "realtime_s")
    df = _apply(df, "duration", parse_duration, "duration_s")
    df = _apply(df, "peak_rss", parse_memory, "peak_rss_b")
    df = _apply(df, "memory", parse_memory, "requested_mem_b")
    df = _apply(df, "%cpu", parse_percent, "pct_cpu")
    df = _apply(df, "read_bytes", parse_memory, "read_b")
    df = _apply(df, "write_bytes", parse_memory, "write_b")

    for col, dtype in (("cpus", pl.Int64), ("attempt", pl.Int64)):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(dtype, strict=False))
        else:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    for col in ("tag", "status", "exit"):
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))

    return df.with_columns(
        pl.col("process").replace_strict(PROCESS_TO_TOOL, default="overhead").alias("tool"),
        pl.col("process").is_in(list(SEARCH_PROCESSES)).alias("is_search"),
        (pl.col("realtime_s") * pl.col("cpus") / 3600).alias("cpu_hours"),
        # Peak RSS against what the job asked SLURM for. Well under 1 means the request is
        # oversized and the job waits longer in the queue than it needs to; near or over 1
        # is the shape that gets OOM-killed on the next combo.
        (pl.col("peak_rss_b") / pl.col("requested_mem_b")).alias("mem_used_frac"),
    )


def variant_from_tag(process: str, tag: str | None) -> str:
    """Recover a variant label from a process tag, matching the metrics table's spelling.

    kmerseekIndexAndSearch tags read `<species>_<alphabet>_k<k>_lc<bool>`; the metrics
    table spells the same combo `<alphabet>_k<k>_lc(True|False)`. mmseqs2 carries its
    variant in brackets. Everything else has one variant, called default.
    """
    if not tag or tag == "-":
        return "default"
    if process == "kmerseekIndexAndSearch":
        m = re.match(r"^(?P<sp>[^_]+)_(?P<rest>.+_k\d+_lc(?:true|false))$", tag)
        if m:
            rest = m.group("rest")
            return rest.replace("_lctrue", "_lcTrue").replace("_lcfalse", "_lcFalse")
        return tag
    m = re.search(r"\[(.+?)\]", tag)
    return m.group(1) if m else "default"


def species_from_tag(tag: str | None) -> str | None:
    """Target species out of a tag: `human_vs_yeast ...` or `yeast_hp_...`."""
    if not tag or tag == "-":
        return None
    m = re.match(r"^human_vs_([A-Za-z0-9]+)", tag)
    if m:
        return m.group(1)
    m = re.match(r"^([a-z]+)_", tag)
    return m.group(1) if m else tag
