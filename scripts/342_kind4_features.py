"""Kind 4 labels for notebook 342 (Stop 3): what kmerseek finds for each exon in the regions index.

Reads the call files of scripts/342_stop3.sbatch (one folder per kmerseek setting, regions
with E <= 10, 1-based inclusive coordinates) and writes, under --out:

* calls_on_exon.tsv.gz: every call whose query stretch overlaps at least one exon residue,
  with the target's Swiss-Prot accession, feature type and gene names, and flags for decoy
  queries, decoy targets, own-gene targets and similar-name targets.
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
#: PR 114's scoring rule for the regions index: peak memory = 1 GB + 18 GB per GB of gzipped calls.
MEMORY_BASE_GB = 1.0
MEMORY_PER_CALL_GB = 18.0
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


def read_calls(calls_dir: Path) -> pl.DataFrame:
    frames = []
    for folder in sorted(p for p in calls_dir.iterdir() if p.is_dir()):
        m = SETTING_RE.match(folder.name)
        if not m:
            continue
        files = sorted(folder.glob("chunk_*.calls.tsv.gz"))
        if not files:
            raise SystemExit(f"no call files in {folder}")
        for f in files:
            frames.append(
                pl.read_csv(
                    f, separator="\t", schema_overrides={"evalue": pl.Float64}
                ).with_columns(pl.lit(folder.name).alias("setting"))
            )
    return pl.concat(frames)


def check_memory(calls_dir: Path) -> None:
    """Stop before reading if PR 114's rule predicts more memory than this job was given."""
    gz_gb = sum(f.stat().st_size for f in calls_dir.rglob("*.calls.tsv.gz")) / 1e9
    predicted = MEMORY_BASE_GB + MEMORY_PER_CALL_GB * gz_gb
    asked = int(os.environ.get("SLURM_MEM_PER_NODE", "0")) / 1024
    print(
        f"{gz_gb:.3f} GB of gzipped calls; PR 114's rule predicts {predicted:.1f} GB; job has {asked:.0f} GB"
    )
    if asked and predicted > asked:
        raise SystemExit(
            f"resubmit the score step with --mem={math.ceil(predicted * 1.25)}G"
        )


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

    check_memory(args.calls)
    queries = pl.read_csv(args.queries, separator="\t")
    genes = gene_names(args.swissprot_dat, args.out / "swissprot_gene_names.tsv")
    calls = read_calls(args.calls)
    settings = sorted(calls["setting"].unique())
    print(f"{calls.height} calls at E <= 10 in {len(settings)} settings: {settings}")

    calls = (
        calls.with_columns(
            pl.col("query").str.starts_with(DECOY).alias("query_is_decoy"),
            pl.col("query").str.strip_prefix(DECOY).alias("event_id"),
            pl.col("target").str.starts_with(DECOY).alias("target_is_decoy"),
        )
        .with_columns(
            pl.col("target").str.strip_prefix(DECOY).str.split("|").alias("_t")
        )
        .with_columns(
            pl.col("_t").list.get(0).alias("target_accession"),
            pl.col("_t").list.get(1, null_on_oob=True).alias("feature_type"),
            pl.col("_t").list.get(2, null_on_oob=True).alias("feature_note"),
        )
        .drop("_t")
    )
    calls = calls.join(
        queries.select(
            "event_id",
            "gene_name",
            "outcome",
            "half",
            "exon_start_in_query",
            "exon_end_in_query",
        ),
        on="event_id",
        how="left",
    )
    assert (
        calls["gene_name"].null_count() == 0
    ), "a call's query is not in the query table"
    on_exon = calls.filter(
        (pl.col("qstart") <= pl.col("exon_end_in_query"))
        & (pl.col("qend") >= pl.col("exon_start_in_query"))
    )

    target_genes = genes.group_by("accession").agg(
        pl.col("gene_name").alias("target_gene_names")
    )
    on_exon = (
        on_exon.join(
            target_genes, left_on="target_accession", right_on="accession", how="left"
        )
        .with_columns(
            pl.col("target_gene_names").fill_null(pl.lit([], dtype=pl.List(pl.String)))
        )
        .with_columns(
            pl.col("target_gene_names")
            .list.contains(pl.col("gene_name").str.to_uppercase())
            .alias("own_gene"),
        )
    )
    # Similar name: a target gene name that starts with the query's symbol but is not equal to it.
    on_exon = on_exon.with_columns(
        pl.struct("target_gene_names", "gene_name")
        .map_elements(
            lambda r: any(
                n != r["gene_name"].upper() and n.startswith(r["gene_name"].upper())
                for n in r["target_gene_names"]
            ),
            return_dtype=pl.Boolean,
        )
        .alias("similar_name")
    )
    with gzip.open(args.out / "calls_on_exon.tsv.gz", "wb") as fh:
        on_exon.with_columns(pl.col("target_gene_names").list.join(";")).write_csv(
            fh, separator="\t"
        )

    kept = on_exon.filter(~pl.col("target_is_decoy") & ~pl.col("own_gene"))

    # Decoy-matched cutoff per setting, on the tune half only.
    n_tune = queries.filter(pl.col("half") == "tune").height
    cutoffs = []
    for s in settings:
        best = (
            kept.filter(
                (pl.col("setting") == s)
                & pl.col("query_is_decoy")
                & (pl.col("half") == "tune")
            )
            .group_by("event_id")
            .agg(pl.col("evalue").min())["evalue"]
        )
        cutoff = decoy_cutoff(best, n_tune)
        cutoffs.append(
            {
                "setting": s,
                "tune_decoy_queries": n_tune,
                "tune_decoys_with_a_call": best.len(),
                "cutoff_evalue": cutoff,
                "tune_decoys_below_cutoff": int((best < cutoff).sum()),
            }
        )
    cutoffs = pl.DataFrame(cutoffs)
    cutoffs.write_csv(args.out / "cutoffs.tsv", separator="\t")
    print(cutoffs)

    real = kept.filter(~pl.col("query_is_decoy")).join(
        cutoffs.select("setting", "cutoff_evalue"), on="setting"
    )
    hits = real.filter(pl.col("evalue") < pl.col("cutoff_evalue"))
    best_hit = (
        hits.sort("evalue", "setting", "target")
        .group_by("event_id", maintain_order=True)
        .first()
    )
    per_setting = (
        real.group_by("event_id", "setting")
        .agg(pl.col("evalue").min())
        .pivot(on="setting", index="event_id", values="evalue")
    )
    per_setting = per_setting.rename(
        {s: f"k4_best_e_{s}" for s in settings if s in per_setting.columns}
    )
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
            real.group_by("event_id").agg(
                pl.col("evalue").min().alias("k4_best_e_any")
            ),
            on="event_id",
            how="left",
        )
        .join(
            real.filter(pl.col("evalue") < P2_EVALUE)
            .group_by("event_id")
            .agg(pl.len().alias("_p2")),
            on="event_id",
            how="left",
        )
        .join(per_setting, on="event_id", how="left")
        .with_columns(
            pl.col("k4_best_e_at_cutoff").is_not_null().alias("k4_has_hit"),
            (-pl.col("k4_best_e_at_cutoff").log10()).alias("k4_neglog10_best_e"),
            (-pl.col("k4_best_e_any").log10()).alias("k4_neglog10_best_e_any"),
            pl.col("k4_n_settings_with_hit").fill_null(0),
            pl.col("_p2").is_not_null().alias("k4_any_call_e_below_1"),
        )
        .drop("_p2")
    )
    features.write_csv(args.out / "kind4_features.tsv", separator="\t")

    own = (
        on_exon.filter(~pl.col("target_is_decoy") & ~pl.col("query_is_decoy"))
        .group_by("setting", "half")
        .agg(
            pl.len().alias("calls_on_real_targets"),
            pl.col("own_gene").sum().alias("removed_own_gene"),
            pl.col("event_id")
            .filter(pl.col("own_gene"))
            .n_unique()
            .alias("exons_with_own_gene_call"),
            (pl.col("similar_name") & ~pl.col("own_gene"))
            .sum()
            .alias("kept_similar_name"),
        )
        .sort("setting", "half")
    )
    own.write_csv(args.out / "own_gene_hits.tsv", separator="\t")
    print(own)

    best_rows = (
        real.sort("evalue", "target")
        .group_by("event_id", "setting", maintain_order=True)
        .first()
        .select(
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
    )
    best_rows.write_csv(args.out / "best_hits.tsv", separator="\t")
    print(
        f"{features['k4_has_hit'].sum()} of {features.height} exons have a hit at the decoy-matched cutoff; "
        f"{features['k4_any_call_e_below_1'].sum()} have a call at E < 1"
    )


if __name__ == "__main__":
    main()
