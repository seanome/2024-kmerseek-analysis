#!/usr/bin/env python3
"""Rescore a finished run's Pfam-N whole-set metrics from its published call files.

Runs scored before restrictTruthToQueries (main.nf) divided Pfam-N recall and Fmax by
every human Pfam-N region (44,315 on 18,724 proteins), not by the regions on the proteins
the run searched. Which calls are correct does not depend on that: a call is checked only
against the truth on its own protein, and every call is on a searched protein. So the
published calls/pfamn.*.calls.parquet files hold everything needed, and the whole-set
numbers can be recomputed exactly with the scorer's own functions, without a Nextflow run.

For every call file this computes the same whole-set numbers twice:

  before  against the full truth file, as the run did. Compared with the run's own
          all_domain_metrics.parquet when --metrics is given: a mismatch means this
          script does not reproduce the scorer, and the "after" numbers cannot be trusted.
  after   against the truth cut to the run's query proteins.

Only the whole set ("all" split, "all" stratum) is recomputed. Strata and reachability
need inputs the call files do not carry; those come back with a full rescore.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cafa_metrics as cm  # noqa: E402
from restrict_truth_to_queries import query_accessions, restrict  # noqa: E402

KEY = ["query_acc", "pfam_id", "true_start", "true_end"]


def parse_name(path: Path) -> dict:
    """pfamn.<tool>.<variant>.<species>[.dedup].calls.parquet, as evaluate_domain_calls.py
    names them."""
    stem = path.name[: -len(".calls.parquet")]
    dedup = stem.endswith(".dedup")
    parts = (stem[: -len(".dedup")] if dedup else stem).split(".")
    return {
        "tool": parts[1],
        "variant": ".".join(parts[2:-1]),
        "species": parts[-1],
        "dedup_transfers": dedup,
    }


def whole_set(calls: pl.DataFrame, truth: pl.DataFrame) -> dict:
    """The whole-set numbers score_one() writes, for the "all" split and stratum."""
    proteins = truth.select(pl.col("accession").unique().alias("query_acc"))
    c = calls.join(proteins, on="query_acc", how="inner")
    found = c.filter("is_tp").select(KEY).unique().height if c.height else 0
    ic = cm.information_content(truth)
    out = {
        "n_truth_instances": truth.height,
        "n_stratum_proteins": truth["accession"].n_unique(),
        "n_instances_found": found,
        "recall": found / truth.height if truth.height else 0.0,
    }
    out.update(cm.cafa_scalars(cm.protein_centric_curve(c, truth, ic)))
    out.update(
        cm.cafa_scalars(
            cm.protein_centric_curve(c, truth, ic, level="family"), prefix="family_"
        )
    )
    return out


METRICS = [
    "n_truth_instances",
    "n_stratum_proteins",
    "n_instances_found",
    "recall",
    "fmax",
    "fmax_precision",
    "fmax_recall",
    "wfmax",
    "smin",
    "family_fmax",
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calls-dir", required=True, type=Path)
    p.add_argument(
        "--truth", required=True, type=Path, help="full human_pfamn_truth.parquet"
    )
    p.add_argument(
        "--annotations", required=True, type=Path, help="the run's --annotations"
    )
    p.add_argument("--metrics", type=Path, help="the run's all_domain_metrics.parquet")
    p.add_argument("--run", required=True, help="run label written on every row")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    full = pl.read_parquet(args.truth)
    cut = restrict(full, query_accessions(args.annotations))
    print(
        f"{args.run}: Pfam-N {full.height:,} regions on {full['accession'].n_unique():,} "
        f"proteins -> {cut.height:,} on {cut['accession'].n_unique():,}",
        file=sys.stderr,
    )

    files = sorted(args.calls_dir.glob("pfamn.*.calls.parquet"))
    if not files:
        raise SystemExit(f"no pfamn.*.calls.parquet under {args.calls_dir}")
    rows = []
    for i, f in enumerate(files, 1):
        calls = pl.read_parquet(f)
        ident = {"run": args.run, **parse_name(f)}
        b, a = whole_set(calls, full), whole_set(calls, cut)
        rows.append(
            {
                **ident,
                **{f"{k}_before": b[k] for k in METRICS},
                **{f"{k}_after": a[k] for k in METRICS},
            }
        )
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} call files", file=sys.stderr)
    df = pl.DataFrame(rows, infer_schema_length=None)

    if args.metrics:
        old = pl.read_parquet(args.metrics).filter(
            (pl.col("truth_set") == "pfamn")
            & (pl.col("split") == "all")
            & (pl.col("stratum_axis") == "all")
        )
        keys = ["tool", "variant", "species", "dedup_transfers"]
        j = df.join(
            old.select(
                *keys,
                pl.lit(True).alias("in_published_metrics"),
                *[pl.col(k).alias(f"{k}_published") for k in METRICS],
            ),
            on=keys,
            how="left",
        ).with_columns(pl.col("in_published_metrics").fill_null(False))
        n_matched = int(j["in_published_metrics"].sum())
        worst = {
            k: j.select(
                (pl.col(f"{k}_before") - pl.col(f"{k}_published")).abs().max()
            ).item()
            for k in METRICS
        }
        print(
            f"reproduces the published metrics: {n_matched} of {j.height} call files "
            f"matched a row; largest difference per metric: {worst}",
            file=sys.stderr,
        )
        # A call file with no published row is kept and flagged, not an error. The
        # midi-plus calls/ folder is shared with run-midi-plus-all-qfo (77 targets), so it
        # holds 7,560 Pfam-N call files where all_domain_metrics.parquet has 396 rows.
        bad = {k: v for k, v in worst.items() if v is not None and v > 1e-9}
        # abs().max() skips nulls, so a value present on one side only is counted here.
        one_sided = j.filter(
            pl.col("in_published_metrics")
            & pl.any_horizontal(
                pl.col(f"{k}_before").is_null() != pl.col(f"{k}_published").is_null()
                for k in METRICS
            )
        ).height
        if n_matched == 0 or bad or one_sided:
            raise SystemExit("the 'before' numbers do not reproduce the run's metrics")
        df = j.drop([f"{k}_published" for k in METRICS])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(args.out)
    print(f"wrote {df.height} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
