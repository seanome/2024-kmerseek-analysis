"""
disprot_tool_names.py

The one place the kmerseek tool name is taken apart. main.nf builds it as
kmerseek_<alphabet>_k<ksize>_lc<true|false>; every script that needs the alphabet or the
ksize back imports this rather than re-deriving the pattern.

The old single-arm names (kmerseek_k26, from the pre-sweep pipeline) parse too, with the
alphabet reported as "hp_thomas_dill2" because every such run used hp-thomas-dill.
"""

import re

import polars as pl

_SWEEP = re.compile(r"^kmerseek_(?P<alphabet>.+)_k(?P<ksize>\d+)_lc(?P<lowcomp>true|false)$")
_OLD = re.compile(r"^kmerseek_k(?P<ksize>\d+)$")

# Class count is the trailing number of every alphabet name (protein20, gbmr4,
# hp_lehninger_hpc3). Used to order alphabets from finest to coarsest.
_CLASSES = re.compile(r"(\d+)$")


def parse_tool(tool: str) -> dict | None:
    """alphabet, ksize, lowcomp for a kmerseek tool name; None for a baseline."""
    m = _SWEEP.match(tool)
    if m:
        return {
            "alphabet": m["alphabet"],
            "ksize": int(m["ksize"]),
            "lowcomp": m["lowcomp"] == "true",
        }
    m = _OLD.match(tool)
    if m:
        return {"alphabet": "hp_thomas_dill2", "ksize": int(m["ksize"]), "lowcomp": False}
    return None


def is_kmerseek(tool: str) -> bool:
    return parse_tool(tool) is not None


def alphabet_classes(alphabet: str) -> int:
    m = _CLASSES.search(alphabet)
    return int(m.group(1)) if m else 20


def annotate(df: pl.DataFrame) -> pl.DataFrame:
    """Add alphabet / ksize / lowcomp columns to any frame with a `tool` column.

    Baselines get nulls. Done row-wise in Python because the frame is small (one row per
    tool x species x stratum) and a regex in polars would be a second copy of the pattern.
    """
    tools = df["tool"].unique().to_list()
    parsed = {t: (parse_tool(t) or {}) for t in tools}
    key = pl.DataFrame(
        {
            "tool": tools,
            "alphabet": [parsed[t].get("alphabet") for t in tools],
            "ksize": [parsed[t].get("ksize") for t in tools],
            "lowcomp": [parsed[t].get("lowcomp") for t in tools],
        },
        schema={"tool": pl.String, "alphabet": pl.String, "ksize": pl.Int32, "lowcomp": pl.Boolean},
    )
    return df.join(key, on="tool", how="left")


BASELINES = ["mmseqs2", "foldseek"]


def headline_tools(metrics: pl.DataFrame, n: int) -> list[str]:
    """Baselines present in the run, then the n kmerseek combos with the highest mean
    recall@FDR5% on the disordered stratum (falling back to all queries when no query
    is in that stratum)."""
    if "alphabet" not in metrics.columns:
        metrics = annotate(metrics)
    present = metrics["tool"].unique().to_list()
    tools = [b for b in BASELINES if b in present]
    km = metrics.filter(pl.col("alphabet").is_not_null())
    for stratum in ("disordered", "all"):
        ranked = (
            km.filter(pl.col("disorder_category") == stratum)
            .group_by("tool")
            .agg(pl.col("recall_at_fdr05").mean().alias("r"))
            .drop_nulls("r")
            .sort(["r", "tool"], descending=[True, False])
        )
        if len(ranked):
            return tools + ranked["tool"].head(n).to_list()
    return tools
