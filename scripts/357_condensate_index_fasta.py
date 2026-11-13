"""Write the target FASTA of the nb 357 condensate index from the Stop 0 region table.

One record per distinct indexed sequence (region plus 10-residue flanks) among the regions
kept for the index (`in_index`). Regions that share a sequence share one record; the name
table lists every region behind each record. Records are named `<accession>__<region_id>`
after the first region in (source, accession, region_id) order. PR 114's
make_window_decoys.py then adds the decoys (scripts/357_stop1_submit.sh says how).

Usage:
    python scripts/357_condensate_index_fasta.py REGION_TSV OUT_FASTA OUT_NAMES_TSV
"""

import sys
from pathlib import Path

import polars as pl


def main(regions: Path, out_fasta: Path, out_names: Path) -> None:
    kept = (
        pl.read_csv(regions, separator="\t", infer_schema_length=None)
        .filter(pl.col("in_index"))
        .sort("source", "accession", "region_id")
        .with_columns(
            (pl.col("accession") + "__" + pl.col("region_id")).alias("region_name")
        )
    )
    records = kept.group_by("indexed_sequence", maintain_order=True).agg(
        pl.col("region_name").first().alias("fasta_name"),
        pl.col("region_name").str.join(";").alias("regions"),
        pl.col("source").unique().sort().str.join(";").alias("sources"),
        pl.len().alias("n_regions"),
    )
    assert records["fasta_name"].n_unique() == records.height
    assert not records["fasta_name"].str.contains(r"\s").any()
    out_fasta.write_text(
        "".join(
            f">{name}\n{seq}\n"
            for name, seq in records.select(
                "fasta_name", "indexed_sequence"
            ).iter_rows()
        )
    )
    records.drop("indexed_sequence").write_csv(out_names, separator="\t")
    print(
        f"{kept.height} regions -> {records.height} records in {out_fasta}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main(*(Path(a) for a in sys.argv[1:4]))
