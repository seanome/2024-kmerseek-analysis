"""Query FASTA for notebook 342 Stop 3: every set A exon with 20 residues of flank on each
side, from the including protein, and one dipeptide-shuffled copy of each.

    python scripts/342_make_queries.py

Writes tables/342/342_setA_exon_queries_with_decoys.fasta (and its .md5) and
tables/342/342_setA_exon_queries.tsv, which gives each query's exon coordinates inside the
query (1-based, inclusive), so a kmerseek region can be tested for overlap with the exon.

The decoy is `dipeptide_shuffle` from nextflow-runs/shared/decoys.py (PR 138), which keeps
the first and last residue and the count of every dipeptide. One random.Random(342) is used
for all decoys, in the order of the sorted event ids, so the file is the same on every run.
"""

from __future__ import annotations

import hashlib
import random
import sys
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "nextflow-runs" / "shared"))
from decoys import dipeptide_shuffle  # noqa: E402

SEED = 342
TABLES = REPO / "tables" / "342"
LABELS = TABLES / "342_setA_exon_labels_kinds1to3.tsv"
EXON_TABLE = TABLES / "342_setA_exon_table_patowary2024_grch37.tsv"
FASTA = TABLES / "342_setA_exon_queries_with_decoys.fasta"
QUERY_TABLE = TABLES / "342_setA_exon_queries.tsv"
DECOY_PREFIX = "DECOY_"


def main() -> None:
    exons = pl.read_csv(EXON_TABLE, separator="\t").with_columns(
        pl.format(
            "{}_{}_{}_{}", "gene_name", "including_isoform", "exon_start", "exon_end"
        ).alias("event_id")
    )
    ids = pl.read_csv(LABELS, separator="\t", columns=["event_id"])
    assert (
        set(ids["event_id"]) == set(exons["event_id"])
        and exons["event_id"].n_unique() == exons.height
    )
    exons = exons.sort("event_id").with_columns(
        (
            pl.col("flank_before") + pl.col("exon_residues") + pl.col("flank_after")
        ).alias("query_residues"),
        (pl.col("flank_before").str.len_chars() + 1).alias("exon_start_in_query"),
        (
            pl.col("flank_before").str.len_chars()
            + pl.col("exon_residues").str.len_chars()
        ).alias("exon_end_in_query"),
    )
    assert not exons["event_id"].str.contains(",").any()
    rng = random.Random(SEED)
    lines = []
    decoys = []
    for event_id, residues in exons.select("event_id", "query_residues").iter_rows():
        decoy = dipeptide_shuffle(residues, rng)
        assert sorted(decoy) == sorted(residues)
        lines.append(f">{event_id}\n{residues}\n")
        decoys.append(f">{DECOY_PREFIX}{event_id}\n{decoy}\n")
    FASTA.write_text("".join(lines + decoys))
    md5 = hashlib.md5(FASTA.read_bytes()).hexdigest()
    FASTA.with_suffix(".fasta.md5").write_text(f"{md5}  {FASTA.name}\n")
    exons.select(
        "event_id",
        "gene_name",
        "outcome",
        "half",
        "exon_start_in_query",
        "exon_end_in_query",
        pl.col("query_residues").str.len_chars().alias("query_length"),
    ).write_csv(QUERY_TABLE, separator="\t")
    lengths = exons["query_residues"].str.len_chars()
    print(
        f"{exons.height} exon queries + {len(decoys)} decoys -> {FASTA.relative_to(REPO)} (md5 {md5})"
    )
    print(
        f"query length: median {lengths.median():.0f}, range {lengths.min()}-{lengths.max()} aa; "
        f"{lengths.sum() * 2:_} residues in all"
    )


if __name__ == "__main__":
    main()
