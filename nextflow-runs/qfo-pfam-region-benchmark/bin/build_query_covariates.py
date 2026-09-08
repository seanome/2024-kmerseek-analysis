#!/usr/bin/env python3
"""Per-query-protein covariates, so benchmark results can be cut by biology.

Four axes, matching what the 200-series notebooks stratify on:

  HGNC gene group   Functional family of the human gene. HGNC's own table carries
                    uniprot_ids, so it joins straight onto the benchmark's accessions
                    with no symbol-mapping round trip.
  dN/dS (omega)     Selection pressure, from the human-mouse-dnds-omega pipeline.
  mean pLDDT        AlphaFold's per-protein confidence, parsed from the same .cif files
                    the Foldseek arm already needs -- no extra download.
  disorder          Fraction of residues with pLDDT < 50, the standard AlphaFold
                    disorder proxy. MobiDB's curated annotation is optional on top.

Coverage differs sharply between axes and is reported per axis rather than left to look
like missing data. dN/dS in particular covers ~1.3k genes against ~19.4k query proteins,
so any dN/dS-stratified result is a statement about that subset only.
"""

import argparse
import gzip
import json
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gene_sets as gs  # noqa: E402

# CA-atom B-factor holds pLDDT in AlphaFold models. Field 14 of the ATOM record, field 3
# is the atom name -- same parse as notebook 202's, kept identical so the two agree.
CA_ATOM_NAME_FIELD = 3
PLDDT_FIELD = 14


def parse_plddt(path: Path) -> tuple[int, float, float, float, float] | None:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt") as f:
            plddts = [
                float(parts[PLDDT_FIELD])
                for line in f
                if line.startswith("ATOM")
                and len(parts := line.split()) > PLDDT_FIELD
                and parts[CA_ATOM_NAME_FIELD] == "CA"
            ]
    except (OSError, ValueError):
        return None
    if not plddts:
        return None
    n = len(plddts)
    return (
        n,
        sum(plddts) / n,
        min(plddts),
        sum(1 for p in plddts if p < 50) / n,
        sum(1 for p in plddts if p < 70) / n,
    )


def load_plddt(struct_dir: Path, accessions: set[str]) -> pl.DataFrame:
    schema = {
        "accession": pl.String, "n_residues": pl.Int64, "mean_plddt": pl.Float64,
        "min_plddt": pl.Float64, "frac_plddt_lt50": pl.Float64,
        "frac_plddt_lt70": pl.Float64,
    }
    if not struct_dir.exists():
        return pl.DataFrame(schema=schema)

    rows = []
    for path in sorted(struct_dir.rglob("AF-*.cif*")):
        acc = path.name.split("-")[1]
        if acc not in accessions:
            continue
        parsed = parse_plddt(path)
        if parsed:
            rows.append((acc,) + parsed)
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=list(schema), orient="row").unique(
        subset="accession", keep="first"
    )


def load_hgnc(hgnc_file: Path) -> pl.DataFrame:
    df = pl.read_csv(
        hgnc_file, separator="\t", infer_schema_length=0, quote_char=None,
        null_values=[""],
    ).select(
        pl.col("symbol").alias("hgnc_symbol"),
        pl.col("gene_group").alias("hgnc_gene_group"),
        pl.col("locus_group"),
        pl.col("uniprot_ids"),
    ).filter(pl.col("uniprot_ids").is_not_null())

    # One HGNC row can list several UniProt accessions; explode so each accession gets
    # its own row and the join downstream stays one-to-one.
    return (
        df.with_columns(pl.col("uniprot_ids").str.split("|").alias("accession"))
        .explode("accession")
        .with_columns(pl.col("accession").str.strip_chars())
        .filter(pl.col("accession") != "")
        .unique(subset="accession", keep="first")
        .drop("uniprot_ids")
    )


def load_omega(omega_file: Path, hgnc: pl.DataFrame) -> pl.DataFrame:
    """dN/dS is keyed by human gene symbol; route it onto accessions through HGNC."""
    df = pl.read_csv(omega_file, separator="\t", infer_schema_length=10000)
    # The upstream file's dS column is corrupt and is deliberately not read.
    # nextflow-runs/human-mouse-dnds-omega/bin/compute_omega.py parses codeml output with
    # r"dS\s*=\s*(...)", and re.search finds that pattern inside "dN/dS=" first, so the
    # dS column holds a copy of omega. Verified: dS == omega byte-for-byte in all 1335
    # rows. dN and omega themselves parse correctly (omega's own pattern is anchored on
    # the full "dN/dS", and "dN\s*=" cannot match "dN/" ), and omega's median of 0.143
    # matches published human-mouse dN/dS.
    #
    # dS is recovered as dN/omega rather than dropped, which is exact rather than an
    # approximation, and gives a median of 0.62 -- plausible synonymous divergence for
    # this pair. Fix the parser upstream and this reconstruction can go away.
    df = df.filter(pl.col("status") == "ok").select(
        pl.col("human_gene").alias("hgnc_symbol"),
        pl.col("dN").cast(pl.Float64, strict=False),
        pl.col("omega").cast(pl.Float64, strict=False),
        pl.col("n_codons").cast(pl.Int64, strict=False),
    ).with_columns(
        pl.when(pl.col("omega") > 0)
        .then(pl.col("dN") / pl.col("omega"))
        .otherwise(None)
        .alias("dS_recovered")
    )
    return (
        df.join(hgnc.select("hgnc_symbol", "accession"), on="hgnc_symbol", how="inner")
        .unique(subset="accession", keep="first")
        .drop("hgnc_symbol")
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--truth", required=True, type=Path)
    p.add_argument("--hgnc", type=Path)
    p.add_argument("--omega", type=Path)
    p.add_argument("--structures", type=Path, help="per-species AlphaFold dir (human/)")
    p.add_argument("--mobidb", type=Path, help="optional cached MobiDB disorder parquet")
    # Sequence-based, so it shares neither of pLDDT's confounds: it needs no structure and
    # no alignment. See bin/predict_disorder_metapredict.py for why that matters.
    p.add_argument("--metapredict", type=Path,
                   help="optional metapredict disorder parquet from "
                        "predict_disorder_metapredict.py")
    # Which bucket of the query set each protein came from, written by
    # make_mini_testset.py. Optional so a run built before it existed still works.
    p.add_argument("--query-sets", type=Path,
                   help="optional accession -> query_set TSV from make_mini_testset.py")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--summary-out", required=True, type=Path)
    args = p.parse_args()

    truth = pl.read_parquet(args.truth)
    accessions = set(truth["accession"].unique().to_list())
    cov = pl.DataFrame({"accession": sorted(accessions)})
    summary = {"n_query_proteins": len(accessions)}

    # Joined first, before any other axis, so `query_set` is present even when HGNC,
    # structures and omega are all absent. Downstream cuts key on it to reconstruct the
    # original chr6 set exactly, and a run whose covariates lack it cannot be compared to
    # one that has it -- so a missing file is reported, not silently skipped.
    if args.query_sets and args.query_sets.exists():
        qs = pl.read_csv(args.query_sets, separator="\t", infer_schema_length=0)
        cov = cov.join(qs, on="accession", how="left")
        counts = (
            cov.group_by("query_set").len().sort("query_set").rows()
        )
        summary["query_sets"] = {
            "n_covered": int(cov["query_set"].is_not_null().sum()),
            "counts": {(k if k is not None else "unlabelled"): v for k, v in counts},
        }
    else:
        summary["query_sets"] = "skipped: no query-set labels supplied"

    hgnc = pl.DataFrame(schema={"hgnc_symbol": pl.String, "accession": pl.String})
    if args.hgnc and args.hgnc.exists():
        hgnc = load_hgnc(args.hgnc)
        cov = cov.join(hgnc, on="accession", how="left")
        summary["hgnc"] = {
            "n_covered": int(cov["hgnc_symbol"].is_not_null().sum()),
            "n_gene_groups": int(cov["hgnc_gene_group"].n_unique()),
        }
    else:
        summary["hgnc"] = "skipped: file not found"

    if args.omega and args.omega.exists() and hgnc.height:
        omega = load_omega(args.omega, hgnc)
        cov = cov.join(omega, on="accession", how="left")
        summary["omega"] = {
            "n_covered": int(cov["omega"].is_not_null().sum()),
            "note": "dN/dS covers a small fraction of query proteins; any omega-stratified "
                    "result describes that subset, not the proteome",
        }
    else:
        summary["omega"] = "skipped: file not found or HGNC unavailable"

    if args.structures:
        plddt = load_plddt(args.structures, accessions)
        if plddt.height:
            cov = cov.join(plddt, on="accession", how="left")
            # The AlphaFold disorder proxy. Named for what it is rather than as plain
            # "disorder", because it is a confidence-derived estimate and not a curated
            # annotation -- MobiDB below is the curated one when present.
            cov = cov.with_columns(
                pl.col("frac_plddt_lt50").alias("disorder_fraction_plddt")
            )
        summary["plddt"] = {"n_covered": int(plddt.height)}
    else:
        summary["plddt"] = "skipped: no structure dir"

    # ---- curated gene sets from the 200-series ----
    if "hgnc_symbol" in cov.columns:
        cov = cov.with_columns(
            pl.col("hgnc_symbol").replace_strict(gs.MHC_CLASSES, default=None).alias("mhc_class"),
            pl.col("hgnc_symbol").is_in(gs.MHC_CLASS_I_GENES).alias("is_mhc_class_i_heavy"),
            pl.col("hgnc_symbol").is_in(gs.ANTIVIRAL_RESTRICTION_FACTORS)
              .alias("is_antiviral_restriction_factor"),
            pl.col("hgnc_symbol").is_in(gs.IGSF_DECOYS).alias("is_igsf_decoy"),
        )

        # Fast-evolving anchors, matched on HGNC group the way notebook 206 does.
        if "hgnc_gene_group" in cov.columns:
            for label, pattern in gs.FAST_EVOLVING_GROUP_PATTERNS.items():
                cov = cov.with_columns(
                    (
                        pl.col("hgnc_gene_group").is_not_null()
                        & pl.col("hgnc_gene_group").str.contains(f"(?i){pattern}")
                    ).alias(f"is_{label}")
                )
            # c2h2 first, so the narrower label wins for genes matching both patterns.
            for label, pattern in gs.ZINC_FINGER_GROUP_PATTERNS.items():
                cov = cov.with_columns(
                    (
                        pl.col("hgnc_gene_group").is_not_null()
                        & pl.col("hgnc_gene_group").str.contains(f"(?i){pattern}")
                    ).alias(f"is_{label}")
                )
            cov = cov.with_columns(
                pl.any_horizontal(
                    [pl.col(f"is_{l}") for l in gs.FAST_EVOLVING_GROUP_PATTERNS]
                ).alias("is_fast_evolving_family"),
                # Repeat-driven, not homology-driven: notebook 206 excludes these, and they
                # are the largest HGNC group in the query set, so a groups sweep that keeps
                # them is dominated by the one family it should not trust.
                (
                    pl.col("hgnc_gene_group").is_not_null()
                    & pl.col("hgnc_gene_group").str.contains(
                        f"(?i){gs.HGNC_EXCLUDE_FAMILY_PATTERN}"
                    )
                ).alias("hgnc_group_excluded"),
            )

        summary["gene_sets"] = {
            "mhc": int(cov["mhc_class"].is_not_null().sum()),
            "mhc_class_i_heavy": int(cov["is_mhc_class_i_heavy"].sum()),
            "antiviral_restriction_factor": int(cov["is_antiviral_restriction_factor"].sum()),
            "igsf_decoy": int(cov["is_igsf_decoy"].sum()),
            "fast_evolving_family": int(cov.get_column("is_fast_evolving_family").sum())
                if "is_fast_evolving_family" in cov.columns else 0,
            # Excluded from the HGNC sweep, still cut as its own geneset stratum.
            "zinc_finger_c2h2": int(cov.get_column("is_zinc_finger_c2h2").sum())
                if "is_zinc_finger_c2h2" in cov.columns else 0,
            "zinc_finger_any": int(cov.get_column("is_zinc_finger_other").sum())
                if "is_zinc_finger_other" in cov.columns else 0,
        }

    if args.metapredict and args.metapredict.exists():
        mpred = pl.read_parquet(args.metapredict)
        cov = cov.join(mpred, on="accession", how="left")
        summary["metapredict"] = {
            "n_covered": int(cov["disorder_fraction_metapredict"].is_not_null().sum())
        }
    else:
        summary["metapredict"] = "skipped: no prediction supplied"

    if args.mobidb and args.mobidb.exists():
        mobi = pl.read_parquet(args.mobidb)
        cov = cov.join(mobi, on="accession", how="left")
        summary["mobidb"] = {"n_covered": int(cov["disorder_fraction_mobidb"].is_not_null().sum())}
    else:
        summary["mobidb"] = "skipped: no cache supplied"

    cov.write_parquet(args.out, compression="zstd")
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
