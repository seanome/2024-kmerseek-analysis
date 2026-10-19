"""Write reports/350_stage_diagnostic.md: the hand-written decision section
(reports/350_stage_diagnostic_decision.md) followed by every number it rests on, read
from evaluate's tables, so the numbers cannot drift from the files."""

from __future__ import annotations

from pathlib import Path

import polars as pl

DATASETS = ["ice_binding", "toxin_scaffold", "toxin_target"]
NAMES = {
    "A": "A full-residue alignment, every reference",
    "B": "B reduced alignment, every reference",
    "C_B": "C kmerseek candidates (scaled 1), reduced alignment",
    "C_A": "C kmerseek candidates (scaled 1), full-residue alignment",
    "D_B": "D kmerseek candidates (scaled 10), reduced alignment",
    "E1": "E kmerseek end to end, scaled 1",
    "E10": "E kmerseek end to end, scaled 10",
}


def md(df: pl.DataFrame, floats: int = 2) -> str:
    cols = df.columns
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.iter_rows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append(f"{v:.{floats}f}")
            else:
                cells.append("" if v is None else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def run(cfg, res: Path, out: Path) -> None:
    h = cfg["evaluation"]["headline_budget"]
    parts = []
    decision = out.with_name("350_stage_diagnostic_decision.md")
    if decision.exists():
        parts.append(decision.read_text())
    parts.append("## Numbers\n\nEvery number below is read from `tables/350_stage_diagnostic/v1/`.")
    sel = pl.read_csv(res / "selection.tsv", separator="\t")
    parts.append(
        "### kmerseek setting chosen on the dev split\n\nHighest candidate recall at scaled 1 on "
        "dev queries; ties to fewer candidates. Only the chosen setting is evaluated on held-out "
        "queries.\n\n" + md(sel)
    )
    dec = pl.read_csv(res / "decisions.tsv", separator="\t")
    parts.append(
        f"### Decision rules (held-out queries, {h} negative allowed per query)\n\n"
        "Rules from `config/stage_diagnostic_v1.yaml`, written before any kmerseek result.\n\n"
        + md(dec)
    )
    for ds in DATASETS:
        d = res / ds
        r = pl.read_csv(d / "recall.tsv", separator="\t").filter(pl.col("split") == "heldout")
        r = r.with_columns(
            method=pl.col("method").replace_strict(NAMES),
            ci=pl.format("{}-{}", pl.col("ci_low").round(2), pl.col("ci_high").round(2)),
        ).select("method", "budget", "recall", "ci", "n_queries", "n_groups")
        c = pl.read_csv(d / "candidate_recall.tsv", separator="\t")
        loss = pl.read_csv(d / "loss_by_stage.tsv", separator="\t").sort("split", "path", "stage")
        col = pl.read_csv(d / "collisions.tsv", separator="\t")
        parts.append(
            f"### {ds}\n\nRecall on held-out queries (mean over queries; 95% interval from resampling {cfg['evaluation']['bootstrap_unit'][ds]}s):\n\n"
            + md(r)
        )
        parts.append(
            "Candidate recall (share of positives kmerseek emits as candidates):\n\n" + md(c)
        )
        parts.append("Positive pairs by the stage they were lost at:\n\n" + md(loss))
        parts.append(
            "Candidate pairs at scaled 1 by whether they also share a full-residue k-mer of the "
            "same length (false = the reduced alphabet alone made the seed match):\n\n" + md(col)
        )
        sf = d / "same_family_vs_not.tsv"
        if sf.exists():
            parts.append(
                "Positive pairs of the same toxin family against different families (share found at "
                f"{h} negative allowed):\n\n" + md(pl.read_csv(sf, separator="\t"))
            )
    sh = pl.read_csv(res / "self_hits.tsv", separator="\t", infer_schema_length=None)
    parts.append(
        "### Self-hit check\n\nEach query against its own entry in the reference panel.\n\n"
        + md(sh.group_by("dataset", "setting", "self_status").len().sort("dataset", "self_status"))
    )
    keep = [
        c
        for c in (
            "query",
            "length",
            "seeds_available",
            "seeds_retained_s1",
            "seeds_retained_s10",
            "C_cand",
            "D_cand",
            "E1_reported",
            "E10_reported",
            "self_status",
        )
        if c in sh.columns
    ]
    parts.append(
        "Ice-binding queries whose own entry is lost at some stage:\n\n"
        + md(
            sh.filter(
                (pl.col("dataset") == "ice_binding")
                & (pl.col("self_status") != "found at every stage")
            ).select(keep)
        )
    )
    fx = pl.read_csv(res / "fixtures.tsv", separator="\t")
    parts.append(
        f"### Fixtures\n\n{fx['passed'].sum()} of {fx.height} checks pass. They test the "
        "implementation, not biology.\n\n" + md(fx)
    )
    cost = pl.read_csv(res / "cost.tsv", separator="\t", infer_schema_length=None)
    parts.append("### Runtime, memory, candidate volume\n\n" + md(cost, 3))
    parts.append(
        "### Figures\n\n"
        + "\n".join(
            f"- [{p.stem}](../figures/350/{p.name})"
            for p in sorted(Path(out.parent.parent / "figures" / "350").glob("*.png"))
        )
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n".join(parts) + "\n")
    print(f"wrote {out}")
