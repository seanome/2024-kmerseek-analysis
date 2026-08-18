#!/usr/bin/env python3
"""Turn aligned regions into Pfam domain calls and score them against the answer key.

The question is domain finding, not orthology: for a human query protein, which stretch of
it is a given Pfam family, and did the tool put the region in the right place?

Pipeline for a search-based tool:

  1. Read the tool's regions: (query, target, qstart, qend, tstart, tend, score).
  2. Transfer. Look up the target protein's Pfam domains. A region claims a family when it
     covers at least --min-overlap of that domain on the *target* side. The region's query
     interval, carrying that family label, is now a domain call.
  3. Score. A call is a true positive when the query protein really does have that family
     AND the call's interval reciprocally overlaps (IoU) a real instance of it by at least
     --min-overlap. Right family in the wrong place is a false positive, not a hit -- that
     distinction is the whole reason for scoring regions instead of protein pairs.

With --direct-annotation (hmmscan against Pfam-A) step 2 is skipped: the tool already names
the family, so its intervals are query-side calls as they stand.

Two overlap criteria, deliberately different:
  transfer (target side)  overlap / domain_length -- did the region land on that domain
  scoring  (query side)   IoU = overlap / union   -- is the call in the right place
Transfer is the looser of the two on purpose. Being strict there would throw away correct
families over target-side boundary noise, which is not what is being measured.
"""

import argparse
import json
from pathlib import Path

import polars as pl


def extract_accession(col: pl.Expr) -> pl.Expr:
    """UniProt FASTA names are sp|P12345|NAME_SPECIES; annotations key on the accession.
    Names already bare (Foldseek, after its filename cleanup) pass through untouched."""
    return (
        pl.when(col.str.contains(r"\|"))
        .then(col.str.split("|").list.get(1, null_on_oob=True))
        .otherwise(col)
    )


def load_regions(path: Path, direct: bool) -> pl.LazyFrame | None:
    """Normalize any tool's output to one schema. Returns None for an empty result, which
    is a real outcome (a combo that found nothing), not an error."""
    if path.stat().st_size == 0:
        return None

    if path.suffix == ".parquet":
        # kmerseek. region_start/region_end are query-side; target_start/target_end are
        # target-side. region_poisson_score is -log10 of the region's Poisson tail, so
        # bigger is better and it sorts the same direction as a bitscore.
        lf = pl.scan_parquet(path)
        names = lf.collect_schema().names()
        if not names:
            return None
        return lf.select(
            extract_accession(pl.col("query_name")).alias("query_acc"),
            extract_accession(pl.col("target_name")).alias("target_acc"),
            pl.col("region_start").cast(pl.Int64).alias("qstart"),
            pl.col("region_end").cast(pl.Int64).alias("qend"),
            pl.col("target_start").cast(pl.Int64).alias("tstart"),
            pl.col("target_end").cast(pl.Int64).alias("tend"),
            pl.col("region_poisson_score").cast(pl.Float64).alias("score"),
        )

    if direct:
        cols = ["query", "pfam_id", "qstart", "qend", "score", "evalue"]
        lf = pl.scan_csv(path, separator="\t", has_header=False, new_columns=cols)
        return lf.select(
            extract_accession(pl.col("query")).alias("query_acc"),
            # hmmscan reports versioned Pfam accessions (PF00001.24); the tables key on
            # the unversioned id.
            pl.col("pfam_id").str.split(".").list.get(0).alias("pfam_id"),
            pl.col("qstart").cast(pl.Int64),
            pl.col("qend").cast(pl.Int64),
            pl.col("score").cast(pl.Float64),
        )

    cols = ["query", "target", "qstart", "qend", "tstart", "tend", "score", "evalue"]
    lf = pl.scan_csv(path, separator="\t", has_header=False, new_columns=cols)
    return lf.select(
        extract_accession(pl.col("query")).alias("query_acc"),
        extract_accession(pl.col("target")).alias("target_acc"),
        pl.col("qstart").cast(pl.Int64),
        pl.col("qend").cast(pl.Int64),
        pl.col("tstart").cast(pl.Int64),
        pl.col("tend").cast(pl.Int64),
        pl.col("score").cast(pl.Float64),
    )


def overlap_expr(a_start: str, a_end: str, b_start: str, b_end: str) -> pl.Expr:
    lo = pl.max_horizontal(pl.col(a_start), pl.col(b_start))
    hi = pl.min_horizontal(pl.col(a_end), pl.col(b_end))
    return (hi - lo).clip(lower_bound=0)


def transfer_domains(regions: pl.LazyFrame, domain_map: pl.LazyFrame, min_overlap: float) -> pl.LazyFrame:
    """Label each region with every target-side Pfam domain it covers."""
    joined = regions.join(
        domain_map.select(
            pl.col("accession").alias("target_acc"),
            "pfam_id",
            pl.col("domain_start").alias("t_dom_start"),
            pl.col("domain_end").alias("t_dom_end"),
        ),
        on="target_acc",
        how="inner",
    )
    return (
        joined.with_columns(
            overlap_expr("tstart", "tend", "t_dom_start", "t_dom_end").alias("t_overlap"),
            (pl.col("t_dom_end") - pl.col("t_dom_start")).alias("t_dom_len"),
        )
        .filter(pl.col("t_overlap") >= min_overlap * pl.col("t_dom_len"))
        .select("query_acc", "pfam_id", "qstart", "qend", "score")
    )


def score_calls(calls: pl.LazyFrame, truth: pl.LazyFrame, min_overlap: float) -> pl.DataFrame:
    """Match each call to the best true instance of the same family on the same protein."""
    matched = (
        calls.join(
            truth.select(
                pl.col("accession").alias("query_acc"),
                "pfam_id",
                pl.col("domain_start").alias("true_start"),
                pl.col("domain_end").alias("true_end"),
            ),
            on=["query_acc", "pfam_id"],
            how="left",
        )
        .with_columns(overlap_expr("qstart", "qend", "true_start", "true_end").alias("ov"))
        .with_columns(
            # The null guard is load-bearing. A left join leaves true_start/true_end null
            # when the query protein has no instance of the transferred family -- the
            # definitive false positive. polars' max_horizontal/min_horizontal SKIP nulls
            # rather than propagating them, so without this branch the union collapses to
            # the call's own span, IoU comes out 1.0, and every call for a family the
            # protein does not have scores as a true positive. That inverts the metric.
            pl.when(pl.col("true_start").is_null() | pl.col("true_end").is_null())
            .then(pl.lit(0.0))
            .otherwise(
                pl.col("ov")
                / (
                    pl.max_horizontal("qend", "true_end")
                    - pl.min_horizontal("qstart", "true_start")
                )
            )
            .fill_null(0.0)
            .alias("iou")
        )
    )

    # One row per call: its best-matching true instance. A call whose family is absent from
    # the protein has iou 0 and stays a false positive.
    #
    # sort_by inside the aggregation, not a sort before group_by: polars does not promise
    # that a group_by preserves the input order within a group, so picking .first() off a
    # pre-sorted frame can silently return a non-best match.
    best = (
        matched.group_by(["query_acc", "pfam_id", "qstart", "qend"])
        .agg(
            # A call can appear once per target that transferred it; they are the same
            # call, so keep its strongest evidence rather than an arbitrary row.
            pl.col("score").max(),
            pl.col("iou").max(),
            pl.col("true_start").sort_by("iou", descending=True).first(),
            pl.col("true_end").sort_by("iou", descending=True).first(),
        )
        .with_columns((pl.col("iou") >= min_overlap).alias("is_tp"))
    )
    return best.collect(engine="streaming")


def compute_metrics(calls: pl.DataFrame, truth: pl.DataFrame, reachable: pl.DataFrame,
                    min_overlap: float) -> dict:
    n_calls = calls.height
    n_tp_calls = int(calls["is_tp"].sum()) if n_calls else 0

    # Recall counts distinct true instances found, not calls: several regions hitting one
    # domain is one recovery, not many.
    found = (
        calls.filter("is_tp")
        .select("query_acc", "pfam_id", "true_start", "true_end")
        .unique()
        .height
        if n_calls
        else 0
    )
    n_truth = truth.height
    n_reachable = reachable.height

    metrics = {
        "n_calls": n_calls,
        "n_tp_calls": n_tp_calls,
        "n_fp_calls": n_calls - n_tp_calls,
        "precision": n_tp_calls / n_calls if n_calls else 0.0,
        "n_truth_instances": n_truth,
        "n_reachable_instances": n_reachable,
        "n_instances_found": found,
        "recall": found / n_truth if n_truth else 0.0,
        # Recall against what was actually transferable. A human family absent from this
        # target proteome's annotations cannot be recovered by any search, so the raw
        # recall above understates every tool by the same species-specific amount.
        "recall_reachable": found / n_reachable if n_reachable else 0.0,
        "median_iou_tp": float(calls.filter("is_tp")["iou"].median()) if n_tp_calls else 0.0,
        "min_overlap": min_overlap,
    }
    denom = metrics["precision"] + metrics["recall_reachable"]
    metrics["f1_reachable"] = (
        2 * metrics["precision"] * metrics["recall_reachable"] / denom if denom else 0.0
    )
    metrics["auprc"] = average_precision(calls, n_reachable)
    return metrics


def average_precision(calls: pl.DataFrame, n_positives: int) -> float:
    """Area under the precision-recall curve over score-ranked calls.

    Recall is against reachable instances, so a tool that never reports a domain cannot
    reach precision 1.0 by reporting one lucky call.
    """
    if calls.height == 0 or n_positives == 0:
        return 0.0
    ranked = calls.sort("score", descending=True, nulls_last=True)
    # Credit a true instance once, at its highest-scoring correct call.
    #
    # Build the key first and mask non-TP rows to null, then take is_first_distinct over
    # the key. Doing it the other way -- is_first_distinct inside a when(is_tp) -- computes
    # distinctness across every row including false positives, so an FP sharing a key with
    # a later TP would consume the "first" flag and the real recovery would go uncredited.
    ranked = ranked.with_columns(
        pl.when("is_tp")
        .then(pl.struct("query_acc", "pfam_id", "true_start", "true_end"))
        .otherwise(None)
        .alias("tp_key")
    ).with_columns(
        (pl.col("tp_key").is_not_null() & pl.col("tp_key").is_first_distinct()).alias("novel_tp")
    )
    tp_cum = ranked["novel_tp"].cum_sum()
    rank = pl.arange(1, ranked.height + 1, eager=True)
    precision = tp_cum / rank
    delta_recall = ranked["novel_tp"].cast(pl.Float64) / n_positives
    return float((precision * delta_recall).sum())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--regions", required=True, type=Path)
    p.add_argument("--tool", required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--species", required=True)
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--domain-map", type=Path)
    p.add_argument("--direct-annotation", action="store_true")
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--calls-out", required=True, type=Path)
    p.add_argument("--metrics-out", required=True, type=Path)
    args = p.parse_args()

    if not args.direct_annotation and args.domain_map is None:
        raise SystemExit("--domain-map is required unless --direct-annotation is set")

    truth_lf = pl.scan_parquet(args.truth)
    truth = truth_lf.collect()

    regions = load_regions(args.regions, args.direct_annotation)

    if args.direct_annotation:
        # hmmscan names the family itself; every human family is reachable.
        reachable = truth
        calls_lf = regions
    else:
        map_lf = pl.scan_parquet(args.domain_map)
        target_families = map_lf.select("pfam_id").unique().collect()
        reachable = truth.join(target_families, on="pfam_id", how="inner")
        calls_lf = (
            transfer_domains(regions, map_lf, args.min_overlap) if regions is not None else None
        )

    if calls_lf is None:
        scored = pl.DataFrame(
            schema={
                "query_acc": pl.String, "pfam_id": pl.String, "qstart": pl.Int64,
                "qend": pl.Int64, "score": pl.Float64, "iou": pl.Float64,
                "true_start": pl.Int64, "true_end": pl.Int64, "is_tp": pl.Boolean,
            }
        )
    else:
        scored = score_calls(calls_lf, truth_lf, args.min_overlap)

    scored.write_parquet(args.calls_out, compression="zstd")

    metrics = compute_metrics(scored, truth, reachable, args.min_overlap)
    metrics.update({"tool": args.tool, "variant": args.variant, "species": args.species})
    pl.DataFrame([metrics]).write_parquet(args.metrics_out, compression="zstd")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
