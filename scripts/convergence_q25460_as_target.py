"""Every kmerseek hit to mussel foot protein 1 (Q25460) in the ice-binding run.

The ice-binding run (nextflow-runs/convergence-afp) searched all of Swiss-Prot 2026_03
with 6 ice-binding queries and their null draws, so Q25460 is there only as a target.
This reads that run's per-setting best-hit tables (one row per query and target) and
keeps the rows whose target is Q25460.

Run on Sherlock inside the pipeline image, which has polars:
  apptainer exec -B $SCRATCH <image> python3 scripts/convergence_q25460_as_target.py \
      --hits <run>/results/hits/kmerseek --out q25460_as_target_in_ice_binding_run.tsv
"""

import argparse
import sys
from pathlib import Path

import polars as pl

TARGET = "Q25460"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--hits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    files = sorted(args.hits.glob("*.best.parquet"))
    if not files:
        sys.exit(f"no *.best.parquet in {args.hits}")
    hits = pl.concat(
        [pl.read_parquet(f).filter(pl.col("target_name") == TARGET) for f in files]
    )
    out = (
        hits.with_columns(
            pl.col("query_name").str.split("__").list.get(0).alias("query_accession"),
            pl.col("query_name")
            .str.split("__")
            .list.get(1)
            .str.replace(r"_\d+$", "")
            .alias("query_kind"),
        )
        .select(
            "setting_label",
            "query_name",
            "query_accession",
            "query_kind",
            "min_region_evalue",
            "max_region_ka_bits",
            "n_regions",
        )
        .sort("min_region_evalue", "setting_label", "query_name")
    )
    out.write_csv(args.out, separator="\t")
    print(
        f"{len(files)} settings read; {out.height} hits to {TARGET} in "
        f"{out['setting_label'].n_unique()} settings"
    )
    print(out.group_by("query_kind").len().sort("query_kind"))
    print(f"smallest E-value: {out['min_region_evalue'].min()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
