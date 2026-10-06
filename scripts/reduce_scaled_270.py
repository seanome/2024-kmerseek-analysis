#!/usr/bin/env python3
"""Reduce one --scaled run to the small per-instance tables notebook 270 reads.

The midi region tables are too large to copy off Sherlock (an estimated 500 M rows per
run for hp_thomas_dill_no_c2 k19 at scaled 1), so the matching that notebook 270 does on
the mini set is done here, with the same functions (notebooks/scaled_270_utils.py), one
region file at a time and QUERY_BATCH queries at a time so memory stays bounded.

For a run directory <set>/<run>/ holding kmerseek/ and truth_swissprot/, writes to
<set>/reduced/<run>/:

  <file stem>.instances.parquet  one row per human instance (range feature x species) that
                                 any region matched, with:
                                   seeded        matched by any region (rule a + rule b)
                                   n_kmers       k-mers in the distinct regions that matched
                                   landed        matched by a region with >= half its
                                                 length inside the instance
                                   iou, start_err, end_err, target_acc
                                                 of the best landed region (highest IoU,
                                                 ties to start error, end error, target,
                                                 start), null when none landed
                                   seeded_pfam, landed_pfam, iou_pfam, start_err_pfam,
                                   end_err_pfam  the same, counting only regions on a
                                                 query-target pair that shares a Pfam family
  <file stem>.calls.parquet      yeast only: every region's (query, target, coordinates),
                                 for the test that extended calls do not depend on scaled
  summary.parquet                per file: rows, rows with finite region_evalue, rows with
                                 a mismatch, rows with region_evalue < 1, queries seen

For the decoy run (--decoy) the DECOY_ prefix is stripped from query accessions before
matching, so each decoy is scored against the features of the protein it was shuffled from.

    python3 scripts/reduce_scaled_270.py --set-dir <data>/midi-scaled-270 --run extend \\
        --annotations <data>/midi/annotations
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import scaled_270_utils as u  # noqa: E402

QUERY_BATCH = 50
COLUMNS = [
    "query_name", "target_name", "region_start", "region_end", "target_start",
    "target_end", "region_n_shared_kmers", "region_n_mismatches", "region_evalue",
    "region_mean_idf", "scaled",
]


def pfam_pairs(annotations: Path, species: list[str]) -> pl.DataFrame:
    h = pl.read_parquet(annotations / "human_pfam_domains.parquet").select(
        query_acc="accession", pfam_id="pfam_id").unique()
    parts = []
    for sp in species:
        t = pl.read_parquet(annotations / f"{sp}_pfam_domains.parquet").select(
            target_acc="accession", pfam_id="pfam_id").unique()
        parts.append(h.join(t, on="pfam_id").select("query_acc", "target_acc").unique()
                     .with_columns(species=pl.lit(sp)))
    return pl.concat(parts)


def best_landed(m: pl.DataFrame, suffix: str) -> pl.DataFrame:
    key = ["species", "query_acc", "ftype", "ds", "de"]
    landed = m.filter(pl.col("inside") >= u.LANDED_MIN)
    return (
        landed.sort(["iou", "start_err", "end_err", "target_acc", "qs"],
                    descending=[True, False, False, False, False])
        .group_by(key, maintain_order=True).first()
        .select(key + [pl.col("iou").alias(f"iou{suffix}"),
                       pl.col("start_err").alias(f"start_err{suffix}"),
                       pl.col("end_err").alias(f"end_err{suffix}"),
                       pl.col("target_acc").alias(f"target_acc{suffix}")])
        .with_columns(pl.lit(True).alias(f"landed{suffix}"))
    )


def reduce_chunk(reg: pl.DataFrame, inst: pl.DataFrame, tf: pl.DataFrame,
                 pairs: pl.DataFrame) -> pl.DataFrame:
    key = ["species", "query_acc", "ftype", "ds", "de"]
    m = u.match_instances(u.typed_regions(reg, tf), inst)
    if m.height == 0:
        return pl.DataFrame()
    k = reg["k"][0]
    seeded = (m.select(key + ["target_acc", "qs", "qe", "ts", "te"]).unique()
              .group_by(key).agg(n_kmers=(pl.col("qe") - pl.col("qs") + 1 - k + 1).sum())
              .with_columns(seeded=pl.lit(True)))
    out = seeded.join(best_landed(m, ""), on=key, how="left")
    mp = m.join(pairs, on=["species", "query_acc", "target_acc"], how="semi")
    if mp.height:
        out = (out.join(mp.select(key).unique().with_columns(seeded_pfam=pl.lit(True)),
                        on=key, how="left")
               .join(best_landed(mp, "_pfam"), on=key, how="left"))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set-dir", type=Path, required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--species", default="mouse,chicken,zebrafish,ciona,fly,worm,yeast,arabidopsis,ecoli")
    p.add_argument("--decoy", action="store_true")
    args = p.parse_args()

    species = args.species.split(",")
    u.SPECIES = species
    run_dir = args.set_dir / args.run
    out_dir = args.set_dir / "reduced" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    # Truth from this run's own truth_swissprot/ (the same tables in every run).
    u.DATA = args.set_dir
    inst = u.load_instances(args.run)
    tf = u.load_target_features(args.run)
    pairs = pfam_pairs(args.annotations, species)
    inst_queries = set(inst["query_acc"])

    summary = []
    for f in sorted((run_dir / "kmerseek").glob("human_vs_*.regions.parquet")):
        m = u._REGION_RE.match(f.name)
        if not m or m["sp"] not in species or m["alpha"] not in u.ARM_K:
            continue
        t0 = time.time()
        stem = f.name.removesuffix(".regions.parquet")
        # Streamed, so the full UniProt headers (~100 characters each) are never all in
        # memory at once; only the accessions are kept.
        reg = pl.scan_parquet(f).select(COLUMNS).with_columns(
            species=pl.lit(m["sp"]), alphabet=pl.lit(m["alpha"]), k=pl.lit(int(m["k"])),
            query_acc=u.accession("query_name"), target_acc=u.accession("target_name"),
        ).drop("query_name", "target_name").collect(engine="streaming")
        if args.decoy:
            reg = reg.with_columns(pl.col("query_acc").str.strip_prefix("DECOY_"))
        reg = reg.with_columns(
            qs=pl.col("region_start") + 1, qe=pl.col("region_end"),
            ts=pl.col("target_start") + 1, te=pl.col("target_end"),
        ).with_columns(call_length=pl.col("qe") - pl.col("qs") + 1)
        assert reg.filter(pl.col("scaled") != int(m["s"] or 1)).height == 0, f.name

        summary.append(dict(
            file=stem, species=m["sp"], alphabet=m["alpha"], k=int(m["k"]),
            scaled=int(m["s"] or 1), n_rows=reg.height,
            n_finite_evalue=int(reg["region_evalue"].is_finite().sum()),
            n_with_mismatch=int((reg["region_n_mismatches"] > 0).sum()),
            n_evalue_lt1=int((reg["region_evalue"] < 1).sum()),
            n_queries=reg["query_acc"].n_unique(),
        ))
        if m["sp"] == "yeast":
            reg.select("query_acc", "target_acc", "qs", "qe", "ts", "te").unique().write_parquet(
                out_dir / f"{stem}.calls.parquet")

        reg = reg.filter(pl.col("query_acc").is_in(inst_queries))
        queries = sorted(reg["query_acc"].unique())
        parts = []
        for i in range(0, len(queries), QUERY_BATCH):
            batch = queries[i:i + QUERY_BATCH]
            chunk = reduce_chunk(reg.filter(pl.col("query_acc").is_in(batch)),
                                 inst.filter(pl.col("query_acc").is_in(batch)), tf, pairs)
            if chunk.height:
                parts.append(chunk)
        res = pl.concat(parts, how="diagonal_relaxed") if parts else pl.DataFrame()
        res = res.with_columns(alphabet=pl.lit(m["alpha"]), k=pl.lit(int(m["k"])),
                               scaled=pl.lit(int(m["s"] or 1)))
        res.write_parquet(out_dir / f"{stem}.instances.parquet")
        print(f"{stem}: {summary[-1]['n_rows']:_} regions -> {res.height:_} instances "
              f"in {time.time() - t0:.0f} s", flush=True)
        del reg

    pl.DataFrame(summary).write_parquet(out_dir / "summary.parquet")
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
