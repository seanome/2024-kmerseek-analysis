"""Shared loaders for the independent nb356 check. Written from the task definitions only."""

import os
import polars as pl

REPO = "/Users/olga/code/2024-kmerseek-analysis-356"
OUT = "/Users/olga/data/356-exon-landing/out"
CALLS = f"{OUT}/calls"
HERE = os.path.dirname(os.path.abspath(__file__))


def pairs():
    return pl.read_csv(f"{REPO}/notebooks/356_data/orthologs_1to1.tsv", separator="\t")


def models():
    p = pl.read_parquet(f"{OUT}/plddt.parquet")
    return p.group_by("side", "species", "accession").agg(
        pl.col("has_model").any().alias("has_model"), pl.len().alias("n_res")
    )


def exons():
    return pl.read_parquet(f"{REPO}/notebooks/356_data/exons.parquet")


def verdict_exons(dedup=False):
    e = exons()
    pr = pairs()
    m = models()
    qm = m.filter(pl.col("side") == "query").select(
        pl.col("accession").alias("query_acc"), pl.col("has_model").alias("q_model")
    )
    tm = m.filter(pl.col("side") == "target").select(
        "species",
        pl.col("accession").alias("target_acc"),
        pl.col("has_model").alias("t_model"),
    )
    v = (
        e.filter(pl.col("internal") & (pl.col("exon_len") <= 60))
        .join(pr, on=["species", "target_acc"], how="inner")
        .join(qm, on="query_acc", how="left")
        .join(tm, on=["species", "target_acc"], how="left")
        .filter(pl.col("q_model").fill_null(False) & pl.col("t_model").fill_null(False))
    )
    if dedup:
        v = v.unique(["species", "target_acc", "prot_start", "prot_end"], keep="first")
    return v


def call_files(tool):
    out = []
    for f in sorted(os.listdir(CALLS)):
        parts = f.split(".")
        if parts[0] == tool:
            out.append((f, parts[1], parts[2]))
    return out


def load_calls(tool):
    """Calls in 0-based end-exclusive target coords, with setting and species columns."""
    dfs = []
    for f, setting, species in call_files(tool):
        d = pl.read_parquet(f"{CALLS}/{f}")
        if (
            d.height == 0
        ):  # 4 hp_thomas_dill2_ext2 kmerseek settings are empty, no columns
            continue
        if tool != "kmerseek":
            d = d.with_columns(pl.col("tstart") - 1, pl.col("qstart") - 1)
        d = d.with_columns(
            pl.lit(setting).alias("setting"), pl.lit(species).alias("species")
        )
        dfs.append(
            d.select(
                "query_acc",
                "target_acc",
                "qstart",
                "qend",
                "tstart",
                "tend",
                "score",
                "setting",
                "species",
            )
        )
    return pl.concat(dfs)


def landing_expr(a="prot_start", b="prot_end"):
    ov = pl.min_horizontal(pl.col("tend"), pl.col(b)) - pl.max_horizontal(
        pl.col("tstart"), pl.col(a)
    )
    L = pl.col("tend") - pl.col("tstart")
    E = pl.col(b) - pl.col(a)
    return (ov >= 0.8 * L) & (ov >= 0.3 * E) & (ov > 0)


def landing_rows(ex, calls):
    j = ex.select(
        "eid", "query_acc", "species", "target_acc", "prot_start", "prot_end"
    ).join(
        calls.select("query_acc", "species", "target_acc", "tstart", "tend", "setting"),
        on=["query_acc", "species", "target_acc"],
        how="inner",
    )
    return j.filter(landing_expr())


def lands(ex, calls):
    return set(landing_rows(ex, calls)["eid"].to_list())
