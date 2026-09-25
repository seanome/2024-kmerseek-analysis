"""Loading, splits and controls for notebook 250 (ELM motif transfer).

Inputs are written by scripts/fetch_elm.py, scripts/prep_elm_inputs.py and
scripts/reduce_elm_landing.py. Coordinates are 0-based and end-exclusive everywhere, the
convention of kmerseek's region tables; hero_example_utils.format_alignment prints them
1-based.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import numpy as np
import polars as pl

ELM_DIR = Path(os.environ.get("ELM_DIR", "/Users/olga/data/elm-motif-transfer"))
LANDING_DIR = Path(os.environ.get("ELM_LANDING_DIR", ELM_DIR / "landing"))
HGNC_FILE = Path(
    os.environ.get(
        "HGNC_FILE", "/Users/olga/data/qfo-pfam-region-midi-plus/hgnc_complete_set.txt"
    )
)
HUMAN = "Homo sapiens"
#: Queries whose length is within this fraction of the case's query length form the
#: random-query control for that case.
LENGTH_MATCH = 0.10
#: A placement null or random-query share at or above this does not count as a call.
ALPHA = 0.05

#: The comparison tools, by the directory name the pipeline publishes them under.
COMPARISON_ARMS = {
    "foldseek": "Foldseek",
    "prostt5": "ProstT5",
    "reseek": "Reseek",
    "hmmer3_phmmer": "phmmer",
    "hmmer3_jackhmmer": "jackhmmer",
    "mmseqs2_seqseq": "MMseqs2",
    "mmseqs2_iterative": "MMseqs2 iterative",
    "hhblits": "HHblits",
}
STRUCTURE_TOOLS = {"foldseek", "prostt5", "reseek"}

_ARM_RE = re.compile(
    r"^kmerseek\.(?P<alpha>.+)_k(?P<k>\d+)_s(?P<s>\d+)_lc(?P<lc>True|False)$"
)


def split_half(unit: str) -> str:
    """ "choose" or "report", from the parity of the first byte of SHA-1(unit), as in nb 244."""
    return "choose" if hashlib.sha1(unit.encode()).digest()[0] % 2 == 0 else "report"


def arm_fields(arms: pl.Series) -> pl.DataFrame:
    rows = []
    for a in arms.unique().sort().to_list():
        m = _ARM_RE.match(a)
        if m:
            rows.append(
                {
                    "arm": a,
                    "alphabet": m["alpha"],
                    "k": int(m["k"]),
                    "scaled": int(m["s"]),
                    "mask_on": m["lc"] == "True",
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "arm": pl.String,
            "alphabet": pl.String,
            "k": pl.Int64,
            "scaled": pl.Int64,
            "mask_on": pl.Boolean,
        },
    )


def hgnc_units() -> pl.DataFrame:
    """accession -> HGNC symbol, gene group and split unit, from the HGNC complete set."""
    h = pl.read_csv(
        HGNC_FILE,
        separator="\t",
        infer_schema_length=0,
        columns=["symbol", "gene_group", "uniprot_ids"],
    )
    return (
        h.with_columns(pl.col("uniprot_ids").str.split("|"))
        .explode("uniprot_ids")
        .drop_nulls("uniprot_ids")
        .select(
            pl.col("uniprot_ids").alias("canonical"),
            pl.col("symbol").alias("hgnc_symbol"),
            pl.col("gene_group").alias("hgnc_gene_group"),
        )
        .unique("canonical", keep="first", maintain_order=True)
    )


def load_classes() -> pl.DataFrame:
    """ELM class table: class, functional site name (ELM's own grouping of classes), regex."""
    text = (ELM_DIR / "elm_classes.tsv").read_text()
    body = "\n".join(l for l in text.splitlines() if not l.startswith("#"))
    import io

    c = pl.read_csv(io.StringIO(body), separator="\t", infer_schema_length=0)
    return c.select(
        pl.col("ELMIdentifier").alias("elm_class"),
        pl.col("FunctionalSiteName").alias("functional_site"),
        pl.col("Regex").alias("regex"),
    )


def load_instances() -> pl.DataFrame:
    """Every true-positive ELM instance, with checks, covariates, splits and grouping."""
    inst = pl.read_parquet(ELM_DIR / "elm_instances_tp.parquet")
    checks = pl.read_parquet(ELM_DIR / "elm_instance_checks.parquet")
    cov = pl.read_parquet(ELM_DIR / "elm_instance_covariates.parquet").select(
        "elm_instance", "mean_plddt_motif", "frac_disordered_motif"
    )
    out = (
        inst.join(checks, on="elm_instance", how="left")
        .join(cov, on="elm_instance", how="left")
        .join(
            load_classes().select("elm_class", "functional_site"),
            on="elm_class",
            how="left",
        )
        .with_columns(
            is_human=pl.col("organism") == HUMAN,
            canonical=pl.col("accession").str.replace(r"-\d+$", ""),
        )
        .join(hgnc_units(), on="canonical", how="left")
        .with_columns(
            split_unit=pl.coalesce("hgnc_gene_group", "hgnc_symbol", "canonical"),
            functional_site=pl.coalesce("functional_site", "elm_class"),
        )
    )
    units = out["split_unit"].unique().drop_nulls().to_list()
    sites = out["functional_site"].unique().drop_nulls().to_list()
    return out.with_columns(
        half=pl.col("split_unit").replace_strict(
            {u: split_half(u) for u in units}, return_dtype=pl.String
        ),
        # The class held-out split is by functional site, so the numbered variants of one
        # motif (LIG_SH3_1 .. LIG_SH3_4) sit on the same side.
        class_split=pl.col("functional_site").replace_strict(
            {
                s: ("choose" if split_half("class:" + s) == "choose" else "held_out")
                for s in sites
            },
            return_dtype=pl.String,
        ),
    )


def load_landing(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    files = sorted(landing_dir.glob("*.instances.parquet"))
    frames = [pl.read_parquet(f) for f in files if f.stat().st_size > 0]
    frames = [f for f in frames if f.height]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def load_target_ranks(
    arm: str, target_set: str, landing_dir: Path = LANDING_DIR
) -> pl.DataFrame:
    p = landing_dir / f"{arm}.{target_set}.target_ranks.parquet"
    return pl.read_parquet(p) if p.exists() and p.stat().st_size else pl.DataFrame()


def load_length_checks(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    frames = [
        pl.read_parquet(f)
        for f in sorted(landing_dir.glob("*.length_checks.parquet"))
        if f.stat().st_size
    ]
    frames = [f for f in frames if f.height]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def random_query_pvalues(
    cases: pl.DataFrame, ranks: pl.DataFrame, queries: pl.DataFrame
) -> pl.DataFrame:
    """Random-query control for each landed case of one arm.

    For a case (query q, target t, t reached at rank r in q's list), the null queries are
    the other human queries whose length is within LENGTH_MATCH of q's, that sit in a
    different split unit (so no paralog of q) and carry no instance of the case's class.
    The p-value is (1 + number of null queries that reach t at rank <= r) / (1 + number of
    null queries). A null query whose list does not contain t does not reach it.

    `queries` has accession, protein_length, split_unit and a list column `classes`.
    """
    if cases.is_empty() or ranks.is_empty():
        return cases.with_columns(
            p_random_query=pl.lit(None, pl.Float64),
            n_null_queries=pl.lit(None, pl.Int64),
        )
    q = {r["accession"]: r for r in queries.iter_rows(named=True)}
    lens = np.array([r["protein_length"] or 0 for r in queries.iter_rows(named=True)])
    accs = queries["accession"].to_list()
    by_target: dict[str, dict[str, int]] = {}
    for r in ranks.iter_rows(named=True):
        by_target.setdefault(r["target_acc"], {})[r["query_acc"]] = r["best_rank"]
    ps, ns = [], []
    for c in cases.iter_rows(named=True):
        me = q.get(c["query_acc"])
        if me is None or c["land_rank"] is None:
            ps.append(None)
            ns.append(None)
            continue
        L = me["protein_length"]
        ok = np.abs(lens - L) <= LENGTH_MATCH * L
        null = [
            a
            for a, keep in zip(accs, ok)
            if keep
            and a != me["accession"]
            and q[a]["split_unit"] != me["split_unit"]
            and c["elm_class"] not in (q[a]["classes"] or [])
        ]
        reach = by_target.get(c["land_target_acc"], {})
        hits = sum(1 for a in null if reach.get(a, 10**9) <= c["land_rank"])
        ps.append((1 + hits) / (1 + len(null)))
        ns.append(len(null))
    return cases.with_columns(
        p_random_query=pl.Series(ps, dtype=pl.Float64),
        n_null_queries=pl.Series(ns, dtype=pl.Int64),
    )
