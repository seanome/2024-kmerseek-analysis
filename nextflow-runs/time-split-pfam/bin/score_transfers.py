#!/usr/bin/env python3
"""Score family transfers for the time-split test (notebook 280).

Every tool's hits become *calls*: (setting, query, query interval, family, score). A hit
on a Swiss-Prot 2019_08 protein transfers family F when its target interval overlaps a
Pfam 33.1 domain of F on that protein (kmerseek rows arrive already joined, by
compact_kmerseek.awk); an hmmscan hit names its Pfam 33.1 family directly. Scores are
oriented so higher is better:
    kmerseek exact    region_poisson_score
    kmerseek extend   -log10(region_evalue)
    MMseqs2           -log10(evalue)
    phmmer, hmmscan   -log10(domain i-Evalue)
All coordinates are 1-based with the end included.

Rules, fixed before any result (notebook 280, first cell):
- found: a call of the same family whose residues lie at least half inside the t1 domain
  (landing fraction >= 0.5). Coverage of the t1 domain and same-clan credit are reported
  as secondary readouts.
- errors are counted on decoys only: a transfer of any family to a decoy query.
- matched cutoff: on the tune half, the lowest score at which decoys receive at most
  1 family transfer (distinct decoy-family pair) per 100 decoy queries. Recall is read on
  the test half at that cutoff.

Writes calls.parquet, instance_calls.parquet (best call per instance and setting),
setting_summary.csv, controls.csv and parse_checks.csv.
"""

from __future__ import annotations

import argparse
import gzip
import math
from pathlib import Path

import polars as pl

LANDING_MIN = 0.5
DECOY_RATE_MAX = 0.01  # family transfers per decoy query
CALL_COLS = ["setting", "tool", "query", "qstart", "qend", "pfam_acc", "score", "target"]


def neg_log10(col: str) -> pl.Expr:
    """-log10 of an E-value column; 0 becomes 400 (beyond any double), inf or null -inf."""
    e = pl.col(col).cast(pl.Float64, strict=False)
    return (pl.when(e <= 0).then(400.0)
            .when(e.is_infinite() | e.is_null()).then(float("-inf"))
            .otherwise(-e.log10()))


def read_kmerseek(dirpath: Path) -> tuple[pl.DataFrame, list[dict]]:
    frames, checks = [], []
    for f in sorted(dirpath.glob("*.transfers.tsv.gz")):
        alphabet, k, arm = f.name.split(".")[0], f.name.split(".")[1], f.name.split(".")[2]
        setting = f"kmerseek {alphabet} {k} {arm}"
        df = pl.read_csv(f, separator="\t", infer_schema_length=0)
        checks.append({"file": f.name, "rows": df.height, "columns": ",".join(df.columns)})
        if df.height == 0 or "region_evalue" not in df.columns:
            continue
        score = (pl.col("region_poisson_score").cast(pl.Float64) if arm == "exact"
                 else neg_log10("region_evalue"))
        frames.append(df.select(
            pl.lit(setting).alias("setting"), pl.lit("kmerseek").alias("tool"), "query",
            pl.col("qstart").cast(pl.Int64), pl.col("qend").cast(pl.Int64), "pfam_acc",
            score.alias("score"), "target"))
    return (pl.concat(frames) if frames else pl.DataFrame(schema={c: pl.String for c in CALL_COLS})), checks


def transfer_by_overlap(hits: pl.DataFrame, domains: pl.DataFrame) -> pl.DataFrame:
    """Hits with tstart/tend on a Swiss-Prot protein -> one row per overlapped domain."""
    j = hits.join(domains, left_on="target", right_on="sp_acc", how="inner")
    return j.filter((pl.col("tstart") <= pl.col("sp_end")) & (pl.col("sp_start") <= pl.col("tend")))


def read_mmseqs(dirpath: Path, domains: pl.DataFrame) -> pl.DataFrame:
    cols = ["query", "target", "qstart", "qend", "tstart", "tend", "pident", "bits", "evalue"]
    fs = sorted(dirpath.glob("*.mmseqs2.tsv.gz"))
    hits = pl.concat([pl.read_csv(f, separator="\t", has_header=False, new_columns=cols,
                                  schema_overrides={"query": pl.String, "target": pl.String})
                      for f in fs if f.stat().st_size > 20]) if fs else None
    if hits is None or hits.height == 0:
        return pl.DataFrame(schema={c: pl.String for c in CALL_COLS})
    t = transfer_by_overlap(hits, domains)
    return t.select(pl.lit("MMseqs2").alias("setting"), pl.lit("MMseqs2").alias("tool"), "query",
                    pl.col("qstart").cast(pl.Int64), pl.col("qend").cast(pl.Int64), "pfam_acc",
                    neg_log10("evalue").alias("score"), "target")


def read_domtbl(path: Path) -> list[list[str]]:
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if not line.startswith("#") and line.strip():
                rows.append(line.split(None, 22))
    return rows


def read_phmmer(dirpath: Path, domains: pl.DataFrame) -> pl.DataFrame:
    # domtblout: 1 target, 4 query, 13 i-Evalue, 16-17 hmm from/to (= query positions,
    # the query is the profile), 18-19 ali from/to (= target positions).
    rows = [r for f in sorted(dirpath.glob("*.domtbl.gz")) for r in read_domtbl(f)]
    if not rows:
        return pl.DataFrame(schema={c: pl.String for c in CALL_COLS})
    hits = pl.DataFrame({"target": [r[0] for r in rows], "query": [r[3] for r in rows],
                         "ievalue": [float(r[12]) for r in rows],
                         "qstart": [int(r[15]) for r in rows], "qend": [int(r[16]) for r in rows],
                         "tstart": [int(r[17]) for r in rows], "tend": [int(r[18]) for r in rows]})
    t = transfer_by_overlap(hits, domains)
    return t.select(pl.lit("phmmer").alias("setting"), pl.lit("phmmer").alias("tool"), "query",
                    "qstart", "qend", "pfam_acc", neg_log10("ievalue").alias("score"), "target")


def read_hmmscan(dirpath: Path) -> pl.DataFrame:
    # domtblout: 2 Pfam accession (PF00001.21), 4 query, 13 i-Evalue, 18-19 ali from/to
    # (query positions; for hmmscan the query is the sequence).
    rows = [r for f in sorted(dirpath.glob("*.domtbl.gz")) for r in read_domtbl(f)]
    if not rows:
        return pl.DataFrame(schema={c: pl.String for c in CALL_COLS})
    return pl.DataFrame({
        "setting": "hmmscan Pfam 33.1", "tool": "hmmscan", "query": [r[3] for r in rows],
        "qstart": [int(r[17]) for r in rows], "qend": [int(r[18]) for r in rows],
        "pfam_acc": [r[1].split(".")[0] for r in rows],
        "score": [400.0 if float(r[12]) <= 0 else -math.log10(float(r[12])) for r in rows],
        "target": None}).with_columns(pl.col("target").cast(pl.String))


def matched_cutoff(dec: pl.DataFrame, n_decoys: int) -> float:
    """Lowest score at which distinct decoy-family transfers <= DECOY_RATE_MAX x n_decoys.
    `dec` holds one row per (decoy, family) with its best score."""
    allowed = math.floor(DECOY_RATE_MAX * n_decoys)
    s = sorted(dec["score"].to_list(), reverse=True)
    if len(s) <= allowed:
        return float("-inf")
    # Keep the top `allowed` pairs: the cutoff sits just above the next one.
    return math.nextafter(s[allowed], math.inf)


def main() -> None:
    ap = argparse.ArgumentParser()
    for a in ("kmerseek", "stats", "mmseqs2", "phmmer", "hmmscan", "inputs", "reference", "settings", "out"):
        ap.add_argument(f"--{a}", type=Path, required=True)
    ap.add_argument("--read-test-half", action="store_true",
                    help="compute test-half recall and decoy rate; off for the pilot, whose "
                         "queries are part of the approved sample")
    a = ap.parse_args()
    q = pl.read_csv(a.inputs / "queries.tsv", separator="\t")
    inst = pl.read_csv(a.inputs / "instances.tsv", separator="\t")
    dom = pl.read_csv(a.reference / "sprot_domains.tsv", separator="\t")

    km, checks = read_kmerseek(a.kmerseek)
    calls = pl.concat([km.with_columns(pl.col("qstart").cast(pl.Int64), pl.col("qend").cast(pl.Int64),
                                       pl.col("score").cast(pl.Float64)),
                       read_mmseqs(a.mmseqs2, dom), read_phmmer(a.phmmer, dom), read_hmmscan(a.hmmscan)],
                      how="vertical_relaxed")
    calls = calls.join(q.select("query", "is_decoy", "half", "role", pl.col("length").alias("query_length")),
                       on="query", how="left")
    unknown = calls.filter(pl.col("is_decoy").is_null())["query"].unique().to_list()
    calls.write_parquet(a.out / "calls.parquet")

    # Parse checks: every call names a query we searched and sits inside it.
    pc = (calls.group_by("setting").agg(
        n_calls=pl.len(),
        unknown_query=pl.col("is_decoy").is_null().sum(),
        start_below_1=(pl.col("qstart") < 1).sum(),
        end_past_query=(pl.col("qend") > pl.col("query_length")).sum(),
        start_after_end=(pl.col("qstart") > pl.col("qend")).sum()).sort("setting"))
    pc.write_csv(a.out / "parse_checks.csv")
    pl.DataFrame(checks).write_csv(a.out / "kmerseek_files.csv")
    if unknown:
        print(f"WARNING: {len(unknown)} query names in calls are not in queries.tsv, e.g. {unknown[:5]}")

    # Best call per (setting, instance) of the same family, with landing and coverage.
    real = calls.filter(~pl.col("is_decoy"))
    ic = (inst.join(real.select("setting", "query", "qstart", "qend", "pfam_acc", "score"),
                    on=["query", "pfam_acc"], how="inner")
          .with_columns(ov=(pl.min_horizontal("qend", "t1_end") - pl.max_horizontal("qstart", "t1_start") + 1).clip(0))
          .with_columns(landing=pl.col("ov") / (pl.col("qend") - pl.col("qstart") + 1),
                        coverage=pl.col("ov") / pl.col("length"))
          .filter(pl.col("landing") >= LANDING_MIN)
          .sort("score", descending=True)
          .unique(["setting", "query", "pfam_acc", "t1_start", "t1_end"], keep="first"))
    ic.write_parquet(a.out / "instance_calls.parquet")

    settings = calls["setting"].unique().sort().to_list()
    n_dec = {h: q.filter(pl.col("is_decoy") & (pl.col("half") == h) & (pl.col("role") == "sample")).height
             for h in ("tune", "test")}
    inst_s = inst.filter(pl.col("role") == "sample").join(q.select("query", "half"), on="query")
    rows = []
    for s in settings:
        c = calls.filter(pl.col("setting") == s)
        dec_pairs = (c.filter(pl.col("is_decoy") & (pl.col("role") == "sample"))
                     .group_by("query", "pfam_acc", "half").agg(pl.col("score").max()))
        cut = matched_cutoff(dec_pairs.filter(pl.col("half") == "tune"), n_dec["tune"])
        dec_test = dec_pairs.filter((pl.col("half") == "test") & (pl.col("score") >= cut)).height
        f = ic.filter((pl.col("setting") == s) & (pl.col("score") >= cut))
        found_test = inst_s.filter(pl.col("half") == "test").join(
            f.select("query", "pfam_acc", "t1_start", "t1_end"), on=["query", "pfam_acc", "t1_start", "t1_end"],
            how="semi").height
        ctrl = inst.filter(pl.col("role") == "control")
        ctrl_any = ctrl.join(ic.filter(pl.col("setting") == s).select("query", "pfam_acc", "t1_start", "t1_end"),
                             on=["query", "pfam_acc", "t1_start", "t1_end"], how="semi").height
        rows.append({"setting": s, "n_calls": c.height,
                     "real_queries_with_call": c.filter(~pl.col("is_decoy"))["query"].n_unique(),
                     "decoy_queries_with_call": c.filter(pl.col("is_decoy"))["query"].n_unique(),
                     "controls_found_any_score": ctrl_any, "controls": ctrl.height,
                     "cutoff_tune": cut, "decoy_transfers_test": dec_test, "decoys_test": n_dec["test"],
                     "instances_test": inst_s.filter(pl.col("half") == "test").height,
                     "found_test": found_test})
    summ = pl.DataFrame(rows)
    if not a.read_test_half:
        summ = summ.drop("decoy_transfers_test", "found_test")
    else:
        summ = summ.with_columns(
            recall_test=pl.col("found_test") / pl.col("instances_test"),
            decoy_rate_test=pl.col("decoy_transfers_test") / pl.col("decoys_test"))
    summ.write_csv(a.out / "setting_summary.csv")
    ctrl_tab = (inst.filter(pl.col("role") == "control")
                .join(ic.select("setting", "query", "pfam_acc", "t1_start", "t1_end", "landing", "coverage", "score"),
                      on=["query", "pfam_acc", "t1_start", "t1_end"], how="left"))
    ctrl_tab.write_csv(a.out / "controls.csv")
    with pl.Config(tbl_rows=400, tbl_cols=20, tbl_width_chars=250):
        print(pc)
        print(summ)


if __name__ == "__main__":
    main()
