#!/usr/bin/env python3
"""Subset the midi run down to the MHC region so it can leave the cluster.

The midi run scores every human chr6 gene against nine proteomes with ~3,700 arms per
truth set. Nothing at that scale can be pulled to a laptop, so this collapses it three
ways and writes only what the MHC notebooks read:

  A gene-level    every arm, MHC-window genes, one row per (arm, gene)
  B domain-level  every arm, core MHC genes only, one row per (arm, gene, pfam_id)
  C raw calls     a focused arm list, all 964 chr6 genes, rows untouched
  D/E region hits target-side accessions, which A-C do not carry, for the synteny work

Only `pfam` truth and non-dedup files are read. Both are deliberate: rows duplicate across
the three truth sets, and the dedup arm answers a different question than this one.

`--only` re-runs a subset of the products, so a fix to one does not cost a re-read of the
3,735 call files the others already consumed.
"""
import argparse, json, os, re, sys, glob
from concurrent.futures import ProcessPoolExecutor
import polars as pl


def parse_call_name(path):
    # <truth_set>.<tool>.<variant>.<species>.calls.parquet
    b = os.path.basename(path)[: -len(".calls.parquet")]
    parts = b.split(".")
    if len(parts) != 4:
        return None
    return dict(zip(["truth_set", "tool", "variant", "species"], parts))


def gene_level(path, keep):
    meta = parse_call_name(path)
    df = pl.read_parquet(path).filter(pl.col("query_acc").is_in(keep))
    if df.is_empty():
        return None
    g = df.group_by("query_acc").agg(
        pl.len().alias("n_calls"),
        pl.col("is_tp").sum().alias("n_tp"),
        pl.col("is_gray").sum().alias("n_gray"),
        pl.col("pfam_id").filter(pl.col("is_tp")).n_unique().alias("n_families_found"),
        pl.col("pfam_id").n_unique().alias("n_families_called"),
        pl.col("score").max().alias("best_score"),
        pl.col("score").filter(pl.col("is_tp")).max().alias("best_tp_score"),
        pl.col("iou").filter(pl.col("is_tp")).median().alias("median_iou_tp"),
        pl.col("cover").max().alias("best_cover"),
    )
    return g.with_columns([pl.lit(v).alias(k) for k, v in meta.items()])


def domain_level(path, keep):
    meta = parse_call_name(path)
    df = pl.read_parquet(path).filter(pl.col("query_acc").is_in(keep))
    if df.is_empty():
        return None
    g = df.group_by("query_acc", "pfam_id").agg(
        pl.len().alias("n_calls"),
        pl.col("is_tp").any().alias("found"),
        pl.col("score").max().alias("best_score"),
        pl.col("iou").max().alias("best_iou"),
        pl.col("cover").max().alias("best_cover"),
        pl.col("true_start").min().alias("true_start"),
        pl.col("true_end").max().alias("true_end"),
        pl.col("qstart").min().alias("qstart"),
        pl.col("qend").max().alias("qend"),
    )
    return g.with_columns([pl.lit(v).alias(k) for k, v in meta.items()])


def _gene_job(a):
    try:
        return gene_level(a[0], a[1])
    except Exception as e:
        print(f"  SKIP {os.path.basename(a[0])}: {type(e).__name__} {e}", file=sys.stderr)
        return None


def _domain_job(a):
    try:
        return domain_level(a[0], a[1])
    except Exception as e:
        print(f"  SKIP {os.path.basename(a[0])}: {type(e).__name__} {e}", file=sys.stderr)
        return None


def concat(frames):
    frames = [f for f in frames if f is not None and not f.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else None


def write(df, path, label):
    if df is None:
        print(f"  {label}: EMPTY, not written", flush=True)
        return
    df.write_parquet(path, compression="zstd")
    print(f"  {label}: {df.height:_} rows -> {path} ({os.path.getsize(path)/1e6:.1f} MB)",
          flush=True)


REGION_COLS = ["query_name", "target_name", "containment", "max_containment", "jaccard",
               "n_intersecting_hashes", "query_poisson_pvalue", "region_start", "region_end",
               "target_start", "target_end", "region_length", "region_n_shared_kmers",
               "region_poisson_score", "region_tail_probability", "region_enrichment",
               "ksize", "moltype", "remove_low_complexity"]

BASELINE_COLS = ["query_name", "target_name", "qstart", "qend", "tstart", "tend",
                 "score", "evalue"]

BASELINE_NUMERIC = {"qstart": pl.Int64, "qend": pl.Int64, "tstart": pl.Int64,
                    "tend": pl.Int64, "score": pl.Float64, "evalue": pl.Float64,
                    "extra": pl.Float64}


def strip_acc(col):
    """sp|ACC|NAME -> ACC, leaving a bare accession untouched.

    An extract-then-coalesce rather than when/then/otherwise, because polars evaluates
    BOTH arms of a when/then across every row and only then selects between them. A
    `.then(col.str.split("|").list.get(1))` arm therefore also ran on the rows with no pipe
    at all, and took out whole files with "get index is out of bounds".
    """
    return pl.coalesce(col.str.extract(r"^[^|]*\|([^|]+)\|", 1), col)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--mhc-window", required=True, help="one accession per line")
    p.add_argument("--mhc-core", required=True, help="one accession per line")
    p.add_argument("--focus-arms", required=True, help="JSON list of tool.variant strings")
    p.add_argument("--region-combos",
                   default="hp_pbotc_1st_ed2:19,hp_pbotc_1st_ed2:21,hp_pbotc_1st_ed2:24,"
                           "hp_thomas_dill2:26,protein20:10,dayhoff6:12,funcgroups8:12",
                   help="alphabet:ksize pairs, lctrue only. Every ksize of every alphabet "
                        "is ~11 GB of reads and tens of millions of rows; the notebooks "
                        "only need the arms the calls tables single out.")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--only", default="ABCDE", help="which of the five products to build")
    a = p.parse_args()

    R, OUT, only = a.results, a.outdir, a.only.upper()
    os.makedirs(OUT, exist_ok=True)
    window = set(open(a.mhc_window).read().split())
    core = set(open(a.mhc_core).read().split())
    focus = set(json.load(open(a.focus_arms)))
    print(f"MHC window genes: {len(window)}   core: {len(core)}   focus arms: {len(focus)}",
          flush=True)

    all_calls = sorted(glob.glob(f"{R}/calls/pfam.*.calls.parquet"))
    calls = [c for c in all_calls if not c.endswith(".dedup.calls.parquet")]
    print(f"pfam call files: {len(calls)} (of {len(all_calls)} incl. dedup)", flush=True)

    # ---- A + B: every arm, aggregated -------------------------------------------------
    if "A" in only or "B" in only:
        with ProcessPoolExecutor(a.threads) as ex:
            if "A" in only:
                print("A: gene-level over all arms...", flush=True)
                gl = list(ex.map(_gene_job, [(c, window) for c in calls], chunksize=8))
                write(concat(gl), f"{OUT}/mhc_gene_level_all_arms.parquet", "A gene-level")
            if "B" in only:
                print("B: domain-level over all arms (core genes)...", flush=True)
                dl = list(ex.map(_domain_job, [(c, core) for c in calls], chunksize=8))
                write(concat(dl), f"{OUT}/mhc_domain_level_all_arms.parquet", "B domain-level")

    # ---- C: raw calls for the focused arms, all chr6 -----------------------------------
    if "C" in only:
        print("C: raw calls for focus arms...", flush=True)
        frames = []
        for c in calls:
            m = parse_call_name(c)
            if m and f"{m['tool']}.{m['variant']}" in focus:
                frames.append(pl.read_parquet(c).with_columns(
                    [pl.lit(v).alias(k) for k, v in m.items()]))
        write(concat(frames), f"{OUT}/chr6_calls_focus_arms.parquet", "C raw calls")

    # ---- D: kmerseek region hits (target side) ----------------------------------------
    if "D" in only:
        print("D: kmerseek region hits...", flush=True)
        combos = {(c.split(":")[0], int(c.split(":")[1])) for c in a.region_combos.split(",")}
        frames = []
        for f in sorted(glob.glob(f"{R}/kmerseek/human_vs_*.regions.parquet")):
            b = os.path.basename(f)
            m = re.match(r"human_vs_(\w+?)\.([\w]+)\.k(\d+)\.lc(true|false)\.regions\.parquet", b)
            if not m or m.group(4) != "true" or (m.group(2), int(m.group(3))) not in combos:
                continue
            sp, alpha, k = m.group(1), m.group(2), int(m.group(3))
            try:
                df = (pl.scan_parquet(f)
                        .with_columns(strip_acc(pl.col("query_name")).alias("query_acc"))
                        .filter(pl.col("query_acc").is_in(window))
                        .select(REGION_COLS + ["query_acc"])
                        .collect())
            except Exception as e:
                print(f"  SKIP {b}: {type(e).__name__} {e}", file=sys.stderr)
                continue
            if df.is_empty():
                continue
            frames.append(df.with_columns(species=pl.lit(sp), alphabet=pl.lit(alpha),
                                          ksize=pl.lit(k, dtype=pl.Int64)))
        write(concat(frames), f"{OUT}/mhc_kmerseek_regions.parquet", "D kmerseek regions")

    # ---- E: baseline region hits ------------------------------------------------------
    if "E" in only:
        print("E: baseline region hits...", flush=True)
        frames = []
        for f in sorted(glob.glob(f"{R}/regions/*/human_vs_*.tsv.gz")):
            tool = os.path.basename(os.path.dirname(f))
            b = os.path.basename(f)
            m = re.match(r"human_vs_(\w+?)\.", b)
            if not m:
                continue
            names = (BASELINE_COLS[:7] + ["extra", "evalue"] if tool == "folddisco"
                     else BASELINE_COLS)
            # Read every column as text and cast afterwards. These TSVs are headerless and
            # differ between tools; hhblits put a float where inference had already
            # committed to i64 off the first 10,000 rows, 13 MB into the file.
            try:
                df = (pl.read_csv(f, separator="\t", has_header=False, new_columns=names,
                                  infer_schema_length=0)
                        .with_columns([pl.col(c).cast(t, strict=False)
                                       for c, t in BASELINE_NUMERIC.items() if c in names])
                        .with_columns(strip_acc(pl.col("query_name")).alias("query_acc"),
                                      strip_acc(pl.col("target_name")).alias("target_acc"))
                        .filter(pl.col("query_acc").is_in(window)))
            except Exception as e:
                print(f"  SKIP {tool}/{b}: {type(e).__name__} {e}", file=sys.stderr)
                continue
            if df.is_empty():
                continue
            frames.append(df.with_columns(tool=pl.lit(tool), species=pl.lit(m.group(1))))
        write(concat(frames), f"{OUT}/mhc_baseline_regions.parquet", "E baseline regions")
    print("done", flush=True)


if __name__ == "__main__":
    main()
