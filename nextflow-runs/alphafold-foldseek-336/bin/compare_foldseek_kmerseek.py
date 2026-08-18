#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""
compare_foldseek_kmerseek.py

Compare FoldSeek (with AF2 structures) hits against KmerSeek assignments for
the 336 FoldSeek-missed SCOPe proteins.

The 336 are defined as: queries that KmerSeek detected at family level
(bonferroni < threshold) but that FoldSeek returned NO hit for.
FoldSeek only finds hits for ~1/3 of SCOPe-40; these 336 are among the ~2/3
that FoldSeek misses entirely.

For each query, we ask:
  - Does FoldSeek (with AF2 structures) find any hit in the same SCOP family?

Outputs:
  comparison.parquet   — per-query table
  summary.txt          — aggregate statistics

Usage:
  compare_foldseek_kmerseek.py \\
      --foldseek-hits  foldseek_336.tsv.gz \\
      --query-uniprot  query_uniprot_map.tsv \\
      --ref-uniprot    ref_uniprot_map.tsv \\
      --scope-lookup   data/foldseek-analysis/scop_lookup.fix.tsv \\
      --kmerseek-eval  scope_eval.hp_pbotc_1st_ed.k26.tsv.gz \\
      --foldseek-rocx  data/tea_scope40_rocx_files/foldseek.rocx \\
      --outdir         results/
"""

import argparse
import sys
from pathlib import Path

import polars as pl


BONFERRONI_THRESHOLD = 0.05


def load_scope_lookup(path: str) -> pl.DataFrame:
    """Load domain_id → scop_family mapping (all SCOPe-40 domains)."""
    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["domain_id", "scop_family"],
        infer_schema_length=0,
    )
    return df


def load_uniprot_map(path: str) -> pl.DataFrame:
    """Load domain_id <-> uniprot_acc mapping from fetch_uniprot_ids output."""
    df = pl.read_csv(path, separator="\t")
    return df.filter(pl.col("uniprot_acc") != "NONE")


def af2_name_to_uniprot(name: str) -> str:
    """AF-P12345-F1-model_v4 → P12345 (handles with or without .cif suffix)."""
    base = name.replace(".cif", "").replace(".pdb", "")
    parts = base.split("-")
    # AF-{ACC}-F1-model_v4 → index 1
    if len(parts) >= 2 and parts[0] == "AF":
        return parts[1]
    return base


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--foldseek-hits",  required=True)
    parser.add_argument("--query-uniprot",  required=True)
    parser.add_argument("--ref-uniprot",    required=True)
    parser.add_argument("--scope-lookup",   required=True)
    parser.add_argument("--kmerseek-eval",  required=True)
    parser.add_argument("--foldseek-rocx",  required=True,
                        help="TEA FoldSeek ROCx used to define in-benchmark queries")
    parser.add_argument("--outdir",         required=True)
    parser.add_argument("--bonferroni-threshold", type=float,
                        default=BONFERRONI_THRESHOLD)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load SCOP lookup (domain_id → scop_family) ──────────────────────────
    print("Loading SCOP lookup …", file=sys.stderr)
    scope_df = load_scope_lookup(args.scope_lookup)
    domain_to_family = dict(zip(scope_df["domain_id"], scope_df["scop_family"]))

    # ── Load UniProt maps ────────────────────────────────────────────────────
    print("Loading UniProt maps …", file=sys.stderr)
    q_map = load_uniprot_map(args.query_uniprot)
    r_map = load_uniprot_map(args.ref_uniprot)

    # domain_id → uniprot_acc (one-to-one; take first if duplicated)
    q_domain_to_uni = dict(zip(q_map["domain_id"], q_map["uniprot_acc"]))
    r_domain_to_uni = dict(zip(r_map["domain_id"], r_map["uniprot_acc"]))

    # uniprot_acc → domain_id (for mapping AF2 hit names back to domains)
    q_uni_to_domain = {v: k for k, v in q_domain_to_uni.items()}
    r_uni_to_domain = {v: k for k, v in r_domain_to_uni.items()}

    # ── Identify the 336 outside-benchmark query domains ────────────────────
    print("Identifying 336 outside-benchmark queries from KmerSeek eval …", file=sys.stderr)
    kmerseek = pl.read_csv(args.kmerseek_eval, separator="\t")
    kmerseek = kmerseek.with_columns(
        pl.col("bonferroni").alias("bonferroni_correct")
    )
    kmerseek_sig = kmerseek.filter(
        pl.col("bonferroni_correct") < args.bonferroni_threshold
    )

    foldseek_rocx = pl.read_csv(args.foldseek_rocx, separator="\t")
    ref_queries = set(foldseek_rocx["NAME"].to_list())

    # Queries detected by KmerSeek at family level (FAM > 0 equivalent)
    # Use query_domain column to get domain IDs
    kmerseek_detections = (
        kmerseek_sig
        .filter(pl.col("same_family"))
        .select("query_domain")
        .unique()
        ["query_domain"]
        .to_list()
    )

    foldseek_missed = [d for d in kmerseek_detections if d not in ref_queries]
    print(
        f"  KmerSeek detections at family level: {len(kmerseek_detections)}\n"
        f"  Missed by FoldSeek: {len(foldseek_missed)}",
        file=sys.stderr,
    )

    # ── Load FoldSeek hits ───────────────────────────────────────────────────
    print("Loading FoldSeek hits …", file=sys.stderr)
    fs = pl.read_csv(
        args.foldseek_hits,
        separator="\t",
        has_header=False,
        new_columns=["query_af2", "target_af2", "bits", "evalue"],
        infer_schema_length=0,
    )

    # Map AF2 names → domain IDs → SCOP families
    fs = fs.with_columns([
        pl.col("query_af2").map_elements(af2_name_to_uniprot,  return_dtype=pl.String).alias("query_uniprot"),
        pl.col("target_af2").map_elements(af2_name_to_uniprot, return_dtype=pl.String).alias("target_uniprot"),
    ])
    fs = fs.with_columns([
        pl.col("query_uniprot").map_elements(
            lambda u: q_uni_to_domain.get(u, ""), return_dtype=pl.String
        ).alias("query_domain"),
        pl.col("target_uniprot").map_elements(
            lambda u: r_uni_to_domain.get(u, ""), return_dtype=pl.String
        ).alias("target_domain"),
    ])
    fs = fs.with_columns([
        pl.col("query_domain").map_elements(
            lambda d: domain_to_family.get(d, ""), return_dtype=pl.String
        ).alias("q_scop_family"),
        pl.col("target_domain").map_elements(
            lambda d: domain_to_family.get(d, ""), return_dtype=pl.String
        ).alias("t_scop_family"),
        pl.col("bits").cast(pl.Float64, strict=False),
        pl.col("evalue").cast(pl.Float64, strict=False),
    ])
    fs = fs.with_columns(
        (
            (pl.col("q_scop_family") != "")
            & (pl.col("t_scop_family") != "")
            & (pl.col("q_scop_family") == pl.col("t_scop_family"))
        ).alias("same_family")
    )

    # ── Per-query comparison ─────────────────────────────────────────────────
    print("Building per-query comparison …", file=sys.stderr)

    rows = []
    for domain_id in sorted(foldseek_missed):
        uniprot = q_domain_to_uni.get(domain_id)
        scop_family = domain_to_family.get(domain_id, "")
        has_af2 = uniprot is not None

        if has_af2:
            q_hits = fs.filter(pl.col("query_domain") == domain_id)
            foldseek_same_family = q_hits.filter(pl.col("same_family")).height > 0
            n_foldseek_hits = q_hits.height
            best_evalue = q_hits["evalue"].min() if n_foldseek_hits > 0 else None
        else:
            foldseek_same_family = None
            n_foldseek_hits = 0
            best_evalue = None

        rows.append({
            "query_domain":         domain_id,
            "scop_family":          scop_family,
            "has_af2_model":        has_af2,
            "uniprot_acc":          uniprot or "",
            "n_foldseek_hits":      n_foldseek_hits,
            "foldseek_same_family": foldseek_same_family,
            "best_evalue":          best_evalue,
        })

    comparison = pl.DataFrame(rows)
    comparison.write_parquet(outdir / "comparison.parquet")

    # ── Summary statistics ───────────────────────────────────────────────────
    n_total        = len(foldseek_missed)
    n_has_af2      = comparison.filter(pl.col("has_af2_model")).height
    n_any_hit      = comparison.filter(pl.col("n_foldseek_hits") > 0).height
    n_same_family  = comparison.filter(pl.col("foldseek_same_family") == True).height
    n_no_hit       = n_has_af2 - n_any_hit

    summary_lines = [
        "=== FoldSeek (AF2) vs KmerSeek: 336 outside-benchmark comparison ===",
        "",
        f"Total KmerSeek detections missed by FoldSeek (family level): {n_total}",
        f"  With AF2 model:                       {n_has_af2}  ({100*n_has_af2/n_total:.1f}%)",
        f"  Without AF2 model (skipped):           {n_total - n_has_af2}",
        "",
        f"Of {n_has_af2} with AF2 structures:",
        f"  FoldSeek finds any hit:               {n_any_hit}  ({100*n_any_hit/max(n_has_af2,1):.1f}%)",
        f"  FoldSeek confirms same SCOP family:   {n_same_family}  ({100*n_same_family/max(n_has_af2,1):.1f}%)",
        f"  FoldSeek finds NO hit at all:         {n_no_hit}  ({100*n_no_hit/max(n_has_af2,1):.1f}%)",
        "",
        "Interpretation:",
        "  - 'FoldSeek confirms' = structural evidence agrees with KmerSeek family assignment",
        "  - 'No hit' = even with AF2 structure FoldSeek misses these — supports KmerSeek's",
        "    unique ability to detect remote homologs through HP k-mer conservation.",
        "",
        f"Output files written to: {outdir}",
    ]
    summary_text = "\n".join(summary_lines)
    (outdir / "summary.txt").write_text(summary_text)
    print(summary_text)


if __name__ == "__main__":
    main()
