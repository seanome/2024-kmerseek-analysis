#!/usr/bin/env python3
"""How many of the genes Domen et al. 2026 list as changed did this run also find?

The paper's genes (g1, g2, ...) and this run's genes (FUN_000001, ...) are different gene
models, so each paper gene is first matched to a FUN gene through its protein:

    mmseqs easy-search Bs_proteins.fa protein.fasta g_vs_fun.m8 tmp -s 7 \\
        --alignment-mode 3 --max-accept 5 \\
        --format-output query,target,pident,alnlen,qlen,tlen,qcov,tcov,evalue,bits

A paper gene is matched to the best-scoring FUN protein if the alignment has at least
MIN_IDENTITY percent identity over at least MIN_COVERAGE percent of the paper's protein.
These cutoffs were set before any overlap was counted.

Usage: compare_paper_genes.py <g_vs_fun.m8> <Dataset_S1.xlsx> <deseq2_tables_dir> <out_dir>
Writes paper_gene_overlap.tsv (counts for the paper's listed genes),
paper_gene_overlap_genes.tsv (one row per listed gene and window) and
paper_gene_agreement.tsv (every gene with a test in both, from the paper's full DESeq2 tables)
and paper_gene_agreement_all_windows.tsv (the same for every comparison against every window).
"""

import sys
from pathlib import Path

import openpyxl
import polars as pl
from scipy import stats

MIN_IDENTITY = 90
MIN_COVERAGE = 50
PADJ = 0.05
SHEET = "annotated DE all groups young"
# Column pairs in the sheet: (column holding g IDs, comparison, direction). The two
# group 5 columns have no "gi" header but hold g IDs like the others.
LISTS = [
    (0, "2 h", "up"),
    (2, "2 h", "down"),
    (4, "24 h", "up"),
    (6, "24 h", "down"),
    (8, "13-49 d", "up"),
    (10, "13-49 d", "down"),
    (12, "132 d", "up"),
    (14, "132 d", "down"),
]
WINDOWS = {
    "2 h": ["young_2h"],
    "24 h": ["young_24h"],
    "13-49 d": ["young_13_17d", "young_49_52d"],
    "132 d": ["young_104_132d"],
}
# The paper's full DESeq2 result for each young comparison.
FULL_SHEETS = {
    "2 h": "Group1_2Hr_young_DEseqres",
    "24 h": "group2_24Hr_Young_DEseqresult",
    "13-49 d": "Group4_13and49Days_Young_DEseq",
    "132 d": "group5_132Days_young_DEseq",
}


def paper_lists(xlsx):
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    rows = list(wb[SHEET].iter_rows(min_row=2, values_only=True))
    out = []
    for col, comparison, direction in LISTS:
        for row in rows:
            gi = row[col] if col < len(row) else None
            if isinstance(gi, str) and gi.strip():
                out.append((gi.strip(), comparison, direction))
    return pl.DataFrame(
        out, schema=["paper_gene", "comparison", "direction"], orient="row"
    )


def paper_full(xlsx):
    """One row per gene and comparison, from the paper's full DESeq2 sheets."""
    wb = openpyxl.load_workbook(xlsx, read_only=True)
    out = []
    for comparison, sheet in FULL_SHEETS.items():
        rows = wb[sheet].iter_rows(values_only=True)
        header = next(rows)
        gene, lfc, p, padj = (
            header.index(c) for c in ("Gene_ID", "log2FoldChange", "pvalue", "padj")
        )
        for row in rows:
            if isinstance(row[p], (int, float)) and isinstance(row[lfc], (int, float)):
                q = row[padj] if isinstance(row[padj], (int, float)) else None
                out.append(
                    (str(row[gene]), comparison, float(row[lfc]), float(row[p]), q)
                )
    return pl.DataFrame(
        out,
        schema={
            "paper_gene": pl.String,
            "comparison": pl.String,
            "paper_log2fc": pl.Float64,
            "paper_p": pl.Float64,
            "paper_padj": pl.Float64,
        },
        orient="row",
    )


def signed_evidence(lfc, p):
    """-log10(p) with the sign of the fold change, so a shrunken fold change cannot rank it."""
    return p.clip(1e-300, 1).log(10).neg() * lfc.sign()


def agreement(full, matched, results, windows=WINDOWS):
    """Agreement over every gene tested in both, not only the genes the paper listed."""
    both = (
        full.join(matched.filter(strong_match()), on="paper_gene")
        .with_columns(
            window=pl.col("comparison").replace_strict(
                windows, return_dtype=pl.List(pl.String)
            )
        )
        .explode("window")
        .join(results.filter(pl.col("pvalue").is_not_null()), on=["fun_gene", "window"])
        .with_columns(
            paper=signed_evidence(pl.col("paper_log2fc"), pl.col("paper_p")),
            here=signed_evidence(pl.col("log2FoldChange"), pl.col("pvalue")),
        )
    )
    rows = []
    for (comparison, window), g in both.group_by(
        "comparison", "window", maintain_order=True
    ):
        paper_hit = g.filter(pl.col("paper_p") < PADJ)
        here_share = (g["pvalue"] < PADJ).mean()
        same_hit = paper_hit.filter(
            (pl.col("pvalue") < PADJ)
            & (pl.col("paper_log2fc").sign() == pl.col("log2FoldChange").sign())
        ).height
        paper_up = g["paper_log2fc"] > 0
        here_up = g["log2FoldChange"] > 0
        rows.append(
            {
                "comparison": comparison,
                "window": window,
                "genes_in_both": g.height,
                "spearman": round(stats.spearmanr(g["paper"], g["here"]).statistic, 3),
                "paper_p_lt_0_05": paper_hit.height,
                "also_p_lt_0_05_here_same_direction": same_hit,
                # A gene passing here by chance, then pointing the paper's way half the time.
                "chance_also_p_lt_0_05_here_same_direction": round(
                    paper_hit.height * here_share * 0.5, 1
                ),
                "paper_padj_lt_0_05": g.filter(pl.col("paper_padj") < PADJ).height,
                "paper_padj_also_padj_here_same_direction": g.filter(
                    (pl.col("paper_padj") < PADJ)
                    & (pl.col("padj") < PADJ)
                    & (pl.col("paper_log2fc").sign() == pl.col("log2FoldChange").sign())
                ).height,
                "paper_padj_also_p_here_same_direction": g.filter(
                    (pl.col("paper_padj") < PADJ)
                    & (pl.col("pvalue") < PADJ)
                    & (pl.col("paper_log2fc").sign() == pl.col("log2FoldChange").sign())
                ).height,
                "share_up_paper": round(paper_up.mean(), 3),
                "share_up_here": round(here_up.mean(), 3),
            }
        )
    return pl.DataFrame(rows)


def strong_match():
    return (pl.col("pident") >= MIN_IDENTITY) & (pl.col("coverage") >= MIN_COVERAGE)


def gene_map(m8):
    hits = pl.read_csv(
        m8,
        separator="\t",
        has_header=False,
        new_columns=[
            "qseqid",
            "sseqid",
            "pident",
            "length",
            "qlen",
            "slen",
            "qcov",
            "scov",
            "evalue",
            "bitscore",
        ],
    )
    return (
        hits.filter(pl.col("qseqid").str.contains(r"^g\d+\.t\d+$"))
        .with_columns(
            paper_gene=pl.col("qseqid").str.replace(r"\.t\d+$", ""),
            # MMseqs2 gives coverage as a fraction.
            coverage=(pl.col("qcov") * 100).round(1),
            fun_gene=pl.col("sseqid")
            .str.extract(r"^FUN(\d+)_", 1)
            .map_elements(lambda n: f"FUN_{n}", return_dtype=pl.String),
        )
        # Ties on score go to the longer alignment, then the lower ID, so the map is stable.
        .sort(["bitscore", "length", "sseqid"], descending=[True, True, False])
        .unique(subset="paper_gene", keep="first", maintain_order=True)
        .select("paper_gene", "fun_gene", "pident", "coverage")
    )


def this_run(tables):
    frames = []
    for path in sorted(Path(tables).glob("*.deseq2.results.tsv")):
        window = path.name.split("_treated_vs_control")[0]
        frames.append(
            pl.read_csv(
                path, separator="\t", null_values=["NA"], infer_schema_length=None
            ).select(
                pl.col("gene_id").alias("fun_gene"),
                pl.lit(window).alias("window"),
                "log2FoldChange",
                "pvalue",
                "padj",
            )
        )
    return pl.concat(frames)


def main(m8, xlsx, tables, out_dir):
    listed = paper_lists(xlsx)
    matched = gene_map(m8)
    results = this_run(tables)
    tested_per_window = (
        results.filter(pl.col("pvalue").is_not_null()).group_by("window").len()
    )

    genes = (
        listed.join(matched, on="paper_gene", how="left")
        .with_columns(matched=strong_match())
        .with_columns(matched=pl.col("matched").fill_null(False))
        .with_columns(
            window=pl.col("comparison").replace_strict(
                WINDOWS, return_dtype=pl.List(pl.String)
            )
        )
        .explode("window")
        .join(results, on=["fun_gene", "window"], how="left")
        .with_columns(
            same_direction=pl.when(pl.col("direction") == "up")
            .then(pl.col("log2FoldChange") > 0)
            .otherwise(pl.col("log2FoldChange") < 0),
            tested=pl.col("matched") & pl.col("pvalue").is_not_null(),
        )
        .with_columns(
            padj_hit=pl.col("tested") & (pl.col("padj") < PADJ).fill_null(False),
            p_hit=pl.col("tested") & (pl.col("pvalue") < PADJ),
        )
        .with_columns(
            padj_same=pl.col("padj_hit") & pl.col("same_direction"),
            p_same=pl.col("p_hit") & pl.col("same_direction"),
            same_sign=pl.col("tested") & pl.col("same_direction"),
        )
    )

    counts = (
        genes.group_by("comparison", "direction", "window", maintain_order=True)
        .agg(
            paper_listed=pl.len(),
            matched_to_fun_gene=pl.col("matched").sum(),
            tested_here=pl.col("tested").sum(),
            padj_lt_0_05=pl.col("padj_hit").sum(),
            padj_lt_0_05_same_direction=pl.col("padj_same").sum(),
            p_lt_0_05=pl.col("p_hit").sum(),
            p_lt_0_05_same_direction=pl.col("p_same").sum(),
            same_sign_any_p=pl.col("same_sign").sum(),
        )
        .join(
            # What a random gene would give: the share of all tested genes in this window
            # that pass, one direction only for the same-direction expectation.
            results.filter(pl.col("pvalue").is_not_null())
            .group_by("window")
            .agg(
                all_tested=pl.len(),
                all_p_up=(
                    (pl.col("pvalue") < PADJ) & (pl.col("log2FoldChange") > 0)
                ).sum(),
                all_p_down=(
                    (pl.col("pvalue") < PADJ) & (pl.col("log2FoldChange") < 0)
                ).sum(),
                all_up=(pl.col("log2FoldChange") > 0).sum(),
                all_down=(pl.col("log2FoldChange") < 0).sum(),
            ),
            on="window",
        )
        .with_columns(
            chance_p_same_direction=(
                pl.col("tested_here")
                * pl.when(pl.col("direction") == "up")
                .then(pl.col("all_p_up"))
                .otherwise(pl.col("all_p_down"))
                / pl.col("all_tested")
            ).round(1),
            # Most genes lean one way in most windows, so half is not the chance level.
            # Genes with a fold change of exactly 0 point neither way and are left out.
            chance_same_sign=(
                pl.col("tested_here")
                * pl.when(pl.col("direction") == "up")
                .then(pl.col("all_up"))
                .otherwise(pl.col("all_down"))
                / (pl.col("all_up") + pl.col("all_down"))
            ).round(1),
        )
        .drop("all_p_up", "all_p_down", "all_up", "all_down")
    )
    full = paper_full(xlsx)
    full_agreement = agreement(full, matched, results)
    # Every paper comparison against every window: a time-matched pair should agree better
    # than a mismatched one. If it does not, the agreement is not about the treatment.
    all_windows = sorted(results["window"].unique())
    crossed = agreement(full, matched, results, {c: all_windows for c in WINDOWS})
    print(tested_per_window)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200):
        print(counts)
        print(full_agreement)
    out_dir = Path(out_dir)
    counts.write_csv(out_dir / "paper_gene_overlap.tsv", separator="\t")
    full_agreement.write_csv(out_dir / "paper_gene_agreement.tsv", separator="\t")
    crossed.write_csv(out_dir / "paper_gene_agreement_all_windows.tsv", separator="\t")
    genes.select(
        "paper_gene",
        "comparison",
        "direction",
        "fun_gene",
        "pident",
        "coverage",
        "matched",
        "window",
        "log2FoldChange",
        "pvalue",
        "padj",
    ).write_csv(out_dir / "paper_gene_overlap_genes.tsv", separator="\t")


if __name__ == "__main__":
    main(*sys.argv[1:5])
