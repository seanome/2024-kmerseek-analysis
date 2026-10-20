"""Kind 4 labels for notebook 342 (Stop 3): what kmerseek finds for each exon in the regions index.

Reads the call files of scripts/342_stop3.sbatch (one folder per kmerseek setting, regions
with E <= 10, 1-based inclusive coordinates) and writes, under --out:

* cutoffs.tsv: per setting, the decoy-matched cutoff, set on the tune half: the E below
  which at most 5% of tune-half decoy queries have a call on a real target (own-gene targets
  removed). Calls at E below the cutoff count as hits.
* kind4_features.tsv: per exon, the kind 4 features of rule 4 (has a hit at the cutoff in
  any setting; -log10 of its best E; feature type of the best hit; number of settings with
  a hit), the best E per setting, and P2's "any call at E < 1".
* own_gene_hits.tsv: per setting and half, the calls the own-gene filter removed, and calls
  kept whose target gene name starts with the query's gene symbol (a paralog the filter
  does not catch).
* best_hits.tsv: per exon and setting, the best call on another gene, with both residue
  strings, for printing aligned in the notebook.

Only calls whose query stretch overlaps at least one exon residue count. The call files
themselves are the record of every call; no copy of them is written (2026-10-20: about 470
million calls for aro_hyd_arg_chg_pol5 k9 alone).

A target is "own gene" when any of its UniProt gene names (Name= or Synonyms= on its GN line
in the Swiss-Prot file) equals the query's gene symbol, ignoring case (Olga, 2026-10-16:
model 4 sees only hits to other genes). Decoy targets (DECOY_ entries of the index) never
give a feature; they are counted only.

    python scripts/342_kind4_features.py --calls DIR --queries tables/342/342_setA_exon_queries.tsv \
        --swissprot-dat uniprot_sprot.2026_03.dat.gz --out DIR
"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import re
from pathlib import Path

import polars as pl

DECOY = "DECOY_"
MAX_DECOY_RATE = 0.05
#: P2 counts a call at E below this.
P2_EVALUE = 1.0
#: Peak memory = 1 GB + MEMORY_PER_FILE_GB per GB of the largest gzipped call file, as
#: files are read one at a time. Set from a measured run (see tables/342/342_stop3_resources.tsv).
MEMORY_BASE_GB = 1.0
MEMORY_PER_FILE_GB = 32.0  # job 48687067: 40.5 GB on a 1.36 GB file
SETTING_RE = re.compile(r"^(?P<alphabet>.+)_k(?P<k>\d+)_s1$")


def gene_names(dat: Path, cache: Path) -> pl.DataFrame:
    """Primary accession -> upper-cased gene names (Name= and Synonyms=) from a Swiss-Prot .dat."""
    if cache.exists():
        return pl.read_csv(cache, separator="\t")
    rows, acc, names = [], None, []
    opener = gzip.open if str(dat).endswith(".gz") else open
    with opener(dat, "rt") as fh:
        for line in fh:
            tag = line[:2]
            if tag == "AC" and acc is None:
                acc = line[5:].split(";")[0].strip()
            elif tag == "GN":
                for key in ("Name", "Synonyms"):
                    for m in re.finditer(rf"\b{key}=([^;]+);", line):
                        names += [
                            re.sub(r"\{.*?\}", "", n).strip().upper()
                            for n in m.group(1).split(",")
                        ]
            elif tag == "//":
                for n in dict.fromkeys(x for x in names if x):
                    rows.append((acc, n))
                acc, names = None, []
    table = pl.DataFrame(rows, schema=["accession", "gene_name"], orient="row")
    table.write_csv(cache, separator="\t")
    return table


#: Columns read from a call file in the first pass; the residue strings are read only for
#: the best calls, in a second pass.
FIRST_PASS_COLUMNS = ["query", "qstart", "qend", "target", "tstart", "tend", "evalue"]
CALL_SCHEMA = {
    "query": pl.String,
    "qstart": pl.Int32,
    "qend": pl.Int32,
    "target": pl.String,
    "tstart": pl.Int32,
    "tend": pl.Int32,
    "evalue": pl.Float64,
    "qseq": pl.String,
    "tseq": pl.String,
}


def call_files(calls_dir: Path) -> list[tuple[str, Path]]:
    """(setting, file) for every call file, settings in name order."""
    out = []
    for folder in sorted(p for p in calls_dir.iterdir() if p.is_dir()):
        if not SETTING_RE.match(folder.name):
            continue
        files = sorted(folder.glob("chunk_*.calls.tsv.gz"))
        if not files:
            raise SystemExit(f"no call files in {folder}")
        out += [(folder.name, f) for f in files]
    return out


def query_table(queries: pl.DataFrame) -> pl.DataFrame:
    """One row per query name, the exon and its decoy: the exon's row number in the query table,
    whether the query is the decoy, the exon's stretch in the query, and the gene's number."""
    exons = queries.with_row_index("event_idx").with_columns(
        pl.col("gene_name").rank("dense").cast(pl.UInt32).alias("gene_idx")
    )
    cols = ["event_idx", "gene_idx", "exon_start_in_query", "exon_end_in_query"]
    return pl.concat(
        [
            exons.select(pl.col("event_id").alias("query"), pl.lit(False).alias("query_is_decoy"), *cols),
            exons.select(
                (pl.lit(DECOY) + pl.col("event_id")).alias("query"),
                pl.lit(True).alias("query_is_decoy"),
                *cols,
            ),
        ]
    ).with_columns(
        pl.col("exon_start_in_query").cast(pl.Int32), pl.col("exon_end_in_query").cast(pl.Int32)
    )


def reduce_file(
    f: Path, qtab: pl.DataFrame, gene_symbols: pl.DataFrame, genes: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame, int, int]:
    """Read one call file and keep only what the outputs need: per exon and query kind (exon or
    its decoy), the call with the smallest E on a real target of another gene (ties to the
    smaller target name, then the earlier call), and per exon the own-gene counts. Only calls
    whose query stretch overlaps at least one exon residue count. Returns those two tables, the
    number of calls read and the number on the exon."""
    calls = pl.read_csv(
        f, separator="\t", columns=FIRST_PASS_COLUMNS, schema_overrides=CALL_SCHEMA
    ).with_row_index("row")
    n_read = calls.height
    calls = calls.join(qtab, on="query", how="left")
    assert calls["event_idx"].null_count() == 0, f"a call's query in {f} is not in the query table"
    calls = calls.filter(
        (pl.col("qstart") <= pl.col("exon_end_in_query"))
        & (pl.col("qend") >= pl.col("exon_start_in_query"))
    ).drop("query", "exon_start_in_query", "exon_end_in_query")
    n_on_exon = calls.height
    # Targets, and own gene / similar name, per distinct value, not per call: a file holds tens
    # of millions of calls but far fewer distinct targets and (target, query gene) pairs.
    targets = (
        calls.select("target")
        .unique()
        .with_row_index("tid")
        .with_columns(
            pl.col("target").str.starts_with(DECOY).alias("target_is_decoy"),
            pl.col("target").str.strip_prefix(DECOY).str.split("|").list.get(0).alias("target_accession"),
        )
    )
    calls = calls.join(targets.select("target", "tid", "target_is_decoy"), on="target", how="left")
    flags = (
        calls.select("tid", "gene_idx")
        .unique()
        .join(targets.select("tid", "target_accession"), on="tid")
        .join(gene_symbols, on="gene_idx")
        .join(
            genes.rename({"gene_name": "_name"}),
            left_on="target_accession",
            right_on="accession",
            how="left",
        )
        .group_by("tid", "gene_idx")
        .agg(
            (pl.col("_name") == pl.col("_symbol")).any().alias("own_gene"),
            # A target gene name that starts with the query's symbol but is not equal to it.
            ((pl.col("_name") != pl.col("_symbol")) & pl.col("_name").str.starts_with(pl.col("_symbol")))
            .any()
            .alias("similar_name"),
        )
    )
    calls = calls.join(flags, on=["tid", "gene_idx"], how="left").drop("tid")
    keys = ["event_idx", "query_is_decoy"]
    kept = calls.filter(~pl.col("target_is_decoy") & ~pl.col("own_gene"))
    best = best_per_query(kept.filter(pl.col("evalue") == pl.col("evalue").min().over(keys)), keys)
    counts = (
        calls.filter(~pl.col("target_is_decoy") & ~pl.col("query_is_decoy"))
        .group_by("event_idx")
        .agg(
            pl.len().alias("n"),
            pl.col("own_gene").sum().alias("own"),
            (pl.col("similar_name") & ~pl.col("own_gene")).sum().alias("similar"),
        )
    )
    return best, counts, n_read, n_on_exon


def best_per_query(calls: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    """Per group of `keys`, the call with the smallest E, ties to the smaller target name, then
    to the earlier call."""
    return calls.sort("evalue", "target", maintain_order=True).group_by(keys, maintain_order=True).first()


def residue_strings(f: Path, rows: pl.Series) -> pl.DataFrame:
    """qseq and tseq of the given rows of one call file (row = 0-based line after the header)."""
    return (
        pl.read_csv(f, separator="\t", columns=["qseq", "tseq"], schema_overrides=CALL_SCHEMA)
        .with_row_index("row")
        .filter(pl.col("row").is_in(rows.implode()))
    )


def check_memory(files: list[tuple[str, Path]]) -> None:
    """Stop before reading if one call file is predicted to need more memory than this job has.
    Files are read one at a time; what is kept from each is one row per exon and setting."""
    largest_gb = max(f.stat().st_size for _, f in files) / 1e9
    total_gb = sum(f.stat().st_size for _, f in files) / 1e9
    predicted = MEMORY_BASE_GB + MEMORY_PER_FILE_GB * largest_gb
    asked = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) / 1024
    print(
        f"{len(files)} call files, {total_gb:.3f} GB gzipped, largest {largest_gb:.3f} GB; "
        f"predicted peak {predicted:.1f} GB; job has {asked:.0f} GB"
    )
    if asked and predicted > asked:
        raise SystemExit(f"resubmit the score step with --mem={math.ceil(predicted * 1.25)}G")


def decoy_cutoff(best: pl.Series, n_decoys: int) -> float:
    """The E below which at most MAX_DECOY_RATE of n_decoys decoy queries have a call."""
    allowed = int(MAX_DECOY_RATE * n_decoys)
    values = best.drop_nulls().sort()
    if values.len() <= allowed:
        return math.inf
    return float(
        values[allowed]
    )  # a hit is E strictly below this: at most `allowed` decoys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--calls", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--swissprot-dat", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    files = call_files(args.calls)
    check_memory(files)
    queries = pl.read_csv(args.queries, separator="\t")
    exon = queries.select(
        "event_id", "gene_name", "outcome", "half", "exon_start_in_query", "exon_end_in_query"
    )
    genes = gene_names(args.swissprot_dat, args.out / "swissprot_gene_names.tsv")
    target_genes = genes.group_by("accession").agg(pl.col("gene_name").alias("target_gene_names"))
    qtab = query_table(queries)
    gene_symbols = (
        queries.select(
            pl.col("gene_name").rank("dense").cast(pl.UInt32).alias("gene_idx"),
            pl.col("gene_name").str.to_uppercase().alias("_symbol"),
        )
        .unique()
    )

    # One file at a time (2026-10-20: aro_hyd_arg_chg_pol5 k9 alone wrote 470 million calls at
    # E <= 10, 88% of them on the exon, too many to hold). What stays in memory is the best call
    # per exon, setting and query kind, and the own-gene counts per exon: all the outputs need.
    bests, counts, n_read, n_on_exon = [], [], 0, 0
    for setting, f in files:
        b, c, n, m = reduce_file(f, qtab, gene_symbols, genes)
        bests.append(b.with_columns(pl.lit(setting).alias("setting"), pl.lit(str(f)).alias("file")))
        counts.append(c.with_columns(pl.lit(setting).alias("setting")))
        n_read += n
        n_on_exon += m
        print(f"{f}: {n} calls, {m} on the exon", flush=True)
    settings = sorted({s for s, _ in files})
    print(f"{n_read} calls at E <= 10 in {len(settings)} settings: {settings}; {n_on_exon} on the exon")
    events = queries.with_row_index("event_idx").select("event_idx", "event_id", "gene_name", "outcome", "half")
    best = (
        best_per_query(pl.concat(bests), ["event_idx", "setting", "query_is_decoy"])
        .join(events, on="event_idx")
        .with_columns(pl.col("target").str.strip_prefix(DECOY).str.split("|").alias("_t"))
        .with_columns(
            pl.col("_t").list.get(1, null_on_oob=True).alias("feature_type"),
            pl.col("_t").list.get(2, null_on_oob=True).alias("feature_note"),
            pl.col("_t").list.get(0).alias("target_accession"),
        )
        .drop("_t")
        .join(target_genes, left_on="target_accession", right_on="accession", how="left")
        .with_columns(pl.col("target_gene_names").fill_null(pl.lit([], dtype=pl.List(pl.String))))
    )

    # Decoy-matched cutoff per setting, on the tune half only.
    n_tune = queries.filter(pl.col("half") == "tune").height
    cutoffs = []
    for s in settings:
        decoy_best = best.filter(
            (pl.col("setting") == s) & pl.col("query_is_decoy") & (pl.col("half") == "tune")
        )["evalue"]
        cutoff = decoy_cutoff(decoy_best, n_tune)
        cutoffs.append(
            {
                "setting": s,
                "tune_decoy_queries": n_tune,
                "tune_decoys_with_a_call": decoy_best.len(),
                "cutoff_evalue": cutoff,
                "tune_decoys_below_cutoff": int((decoy_best < cutoff).sum()),
            }
        )
    cutoffs = pl.DataFrame(cutoffs)
    cutoffs.write_csv(args.out / "cutoffs.tsv", separator="\t")
    print(cutoffs)

    # One row per exon and setting: its best call on another gene's real target.
    real = best.filter(~pl.col("query_is_decoy")).join(
        cutoffs.select("setting", "cutoff_evalue"), on="setting"
    )
    hits = real.filter(pl.col("evalue") < pl.col("cutoff_evalue"))
    best_hit = (
        hits.sort("evalue", "setting", "target", maintain_order=True)
        .group_by("event_id", maintain_order=True)
        .first()
    )
    per_setting = real.pivot(on="setting", index="event_id", values="evalue")
    per_setting = per_setting.select(
        "event_id", *[s for s in settings if s in per_setting.columns]
    ).rename({s: f"k4_best_e_{s}" for s in settings if s in per_setting.columns})
    features = (
        queries.select("event_id", "gene_name", "outcome", "half")
        .join(
            best_hit.select(
                "event_id",
                pl.col("evalue").alias("k4_best_e_at_cutoff"),
                pl.col("feature_type").alias("k4_best_feature_type"),
                pl.col("setting").alias("k4_best_setting"),
            ),
            on="event_id",
            how="left",
        )
        .join(
            hits.group_by("event_id").agg(
                pl.col("setting").n_unique().alias("k4_n_settings_with_hit")
            ),
            on="event_id",
            how="left",
        )
        .join(
            real.group_by("event_id").agg(pl.col("evalue").min().alias("k4_best_e_any")),
            on="event_id",
            how="left",
        )
        .join(per_setting, on="event_id", how="left")
        .with_columns(
            pl.col("k4_best_e_at_cutoff").is_not_null().alias("k4_has_hit"),
            (-pl.col("k4_best_e_at_cutoff").log10()).alias("k4_neglog10_best_e"),
            (-pl.col("k4_best_e_any").log10()).alias("k4_neglog10_best_e_any"),
            pl.col("k4_n_settings_with_hit").fill_null(0),
            (pl.col("k4_best_e_any") < P2_EVALUE).fill_null(False).alias("k4_any_call_e_below_1"),
        )
    )
    features.write_csv(args.out / "kind4_features.tsv", separator="\t")

    own = (
        pl.concat(counts)
        .group_by("setting", "event_idx")
        .sum()
        .join(events.select("event_idx", "half"), on="event_idx")
        .group_by("setting", "half")
        .agg(
            pl.col("n").sum().alias("calls_on_real_targets"),
            pl.col("own").sum().alias("removed_own_gene"),
            (pl.col("own") > 0).sum().alias("exons_with_own_gene_call"),
            pl.col("similar").sum().alias("kept_similar_name"),
        )
        .sort("setting", "half")
    )
    own.write_csv(args.out / "own_gene_hits.tsv", separator="\t")
    print(own)

    seqs = pl.concat(
        [
            residue_strings(Path(f), g["row"]).with_columns(pl.lit(f).alias("file"))
            for (f,), g in real.group_by("file")
        ]
    )
    real = real.join(seqs, on=["file", "row"], how="left")
    assert real["qseq"].null_count() == 0
    best_rows = real.sort("evalue", "target", "event_id", "setting").select(
        "event_id",
        "gene_name",
        "outcome",
        "half",
        "setting",
        "evalue",
        "cutoff_evalue",
        "qstart",
        "qend",
        "qseq",
        "target",
        "feature_type",
        "feature_note",
        "tstart",
        "tend",
        "tseq",
        pl.col("target_gene_names").list.join(";"),
    )
    best_rows.write_csv(args.out / "best_hits.tsv", separator="\t")
    print(
        f"{features['k4_has_hit'].sum()} of {features.height} exons have a hit at the decoy-matched cutoff; "
        f"{features['k4_any_call_e_below_1'].sum()} have a call at E < 1"
    )


if __name__ == "__main__":
    main()
