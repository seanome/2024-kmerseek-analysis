"""What a query-reference pair is, per dataset. Each dataset has its own rule; they are
never pooled.

relation values:
  self                the query's own entry (or an identical sequence): left out of every
                      metric, checked on its own (the self-hit check)
  positive            what the dataset asks a method to find
  same_origin         ice_binding only: an ice-binding protein of the query's own origin.
                      Homology, not convergence: neither positive nor negative.
  negative_annotated  another framework / target class: annotated, not shown not to bind
  negative_hard       composition-matched, same-family non-ice-binding, cysteine-rich
                      non-toxin
  background          random Swiss-Prot, unannotated
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

HARD_ROLES = {"hard_negative_composition", "hard_negative_same_family", "cys_rich_nontoxin"}
NEGATIVE = ("negative_annotated", "negative_hard", "background")


def pairs(data_dir: Path, dataset: str) -> pl.DataFrame:
    """Every query x reference pair with its relation."""
    q = pl.read_csv(data_dir / "queries.tsv", separator="\t", infer_schema_length=0)
    r = pl.read_csv(data_dir / "references.tsv", separator="\t", infer_schema_length=0)
    q = q.select(
        pl.col("id").alias("query"), pl.col("label").alias("q_label"),
        pl.col("group").alias("q_group"), pl.col("split"), pl.col("sha1").alias("q_sha1"),
    )  # fmt: skip
    r = r.select(
        pl.col("id").alias("reference"), pl.col("role"), pl.col("label").alias("r_label"),
        pl.col("group").alias("r_group"), pl.col("sha1").alias("r_sha1"),
    )  # fmt: skip
    p = q.join(r, how="cross")
    positive_role = "ice_binding" if dataset == "ice_binding" else "toxin"
    rel = (
        pl.when((pl.col("query") == pl.col("reference")) | (pl.col("q_sha1") == pl.col("r_sha1")))
        .then(pl.lit("self"))
        .when((pl.col("role") == positive_role) & (pl.col("r_label") == pl.col("q_label")))
        .then(pl.lit("same_origin" if dataset == "ice_binding" else "positive"))
        .when(pl.col("role") == positive_role)
        .then(pl.lit("positive" if dataset == "ice_binding" else "negative_annotated"))
        .when(pl.col("role").is_in(list(HARD_ROLES)))
        .then(pl.lit("negative_hard"))
        .otherwise(pl.lit("background"))
    )
    return p.with_columns(relation=rel)


def positive_sets(data_dir: Path, dataset: str) -> dict[str, set[str]]:
    p = pairs(data_dir, dataset).filter(
        pl.col("relation").is_in(["positive", "self", "same_origin"])
    )
    out: dict[str, set[str]] = {}
    for q, r in zip(p["query"], p["reference"]):
        out.setdefault(q, set()).add(r)
    return out
