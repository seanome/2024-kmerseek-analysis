#!/usr/bin/env python3
"""Which proteins did every sequence arm fail to place into the reference?

The dark set is the denominator of the proteome-annotate claim. A protein phmmer already
places is not territory kmerseek can add anything in, so the only proteins worth arguing
about are the ones where every arm returned nothing.

Defined against the STRONGEST search available, not the cheapest: jackhmmer at 3 iterations
and mmseqs2 iterative both count. A protein reachable by iterative search is not dark, and
counting it as dark would inflate the headline in the direction the claim wants -- which is
exactly the kind of error that has to be designed out rather than checked for later.

Two cutoffs, deliberately different. Searches ran at --evalue_report (permissive, 10.0) so
the raw hits are kept for later re-thresholding; a protein counts as PLACED here only at
--evalue-call (1e-3). Re-running this script with a different call cutoff does not require
re-running any search.
"""

import argparse
import gzip
import json
from pathlib import Path

import polars as pl

# Every arm's tsv is written as: query, target, tstart, tend, bits, evalue
COLS = ["query", "target", "tstart", "tend", "bits", "evalue"]


def arm_of(path: Path) -> str:
    for arm in ("phmmer", "jackhmmer", "mmseqs2"):
        if arm in path.name:
            return arm
    return path.name


def read_hits(path: Path) -> pl.DataFrame:
    if path.stat().st_size == 0:
        return pl.DataFrame({c: [] for c in COLS}).with_columns(
            pl.lit(arm_of(path)).alias("arm"))
    df = pl.read_csv(path, separator="\t", has_header=False, infer_schema_length=0,
                     truncate_ragged_lines=True, new_columns=COLS)
    return df.with_columns(pl.lit(arm_of(path)).alias("arm"))


def query_accessions(fasta: Path) -> list[str]:
    opener = gzip.open if fasta.suffix == ".gz" else open
    accs = []
    with opener(fasta, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                # The accession is the join key to every other product, so it is taken
                # exactly as the FASTA writes it -- no splitting on '|', which would break
                # Botryllus's FUN000001_FUN000001 ids differently from UniProt's.
                accs.append(line[1:].strip().split()[0])
    return accs


def focus_proteins(registry: Path | None, species: str) -> dict[str, str]:
    """accession -> what it is, from the registry row's focus_proteins; {} without one."""
    if registry is None or not registry.exists():
        return {}
    row = json.loads(registry.read_text()).get(species) or {}
    return dict(row.get("focus_proteins") or {})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--hits", type=Path, nargs="+", required=True)
    ap.add_argument("--evalue-call", type=float, default=1e-3)
    ap.add_argument("--registry", type=Path, default=None,
                    help="species_metadata.json; its focus_proteins for --species are "
                         "followed through the report, so their per-arm best E-value "
                         "and placed/dark call are written into the summary")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    accs = query_accessions(args.query)
    if not accs:
        raise SystemExit(f"no records in {args.query}")
    proteome = pl.DataFrame({"accession": accs}).unique()

    frames = [read_hits(h) for h in args.hits]
    hits = pl.concat(frames, how="diagonal") if frames else None
    if hits is None or hits.height == 0:
        raise SystemExit(
            f"every arm returned an empty hit file for '{args.species}'.\n"
            f"That is not a dark proteome, it is a failed run: phmmer against 572_700 "
            f"reviewed sequences returns something for almost any real protein. Check the "
            f"reference actually built (reference.fasta non-empty) before reading this as "
            f"a result."
        )

    hits = hits.with_columns(pl.col("evalue").cast(pl.Float64, strict=False))
    called = hits.filter(pl.col("evalue") <= args.evalue_call)

    per_arm = {}
    for arm in sorted(hits["arm"].unique().to_list()):
        per_arm[arm] = int(called.filter(pl.col("arm") == arm)["query"].n_unique())

    placed = called.select("query").unique().rename({"query": "accession"})
    # Left join then null-check, never a horizontal min/max: a left join that misses leaves
    # nulls, and treating null as "no hit" has to be explicit or it silently becomes one.
    marked = proteome.join(placed.with_columns(pl.lit(True).alias("placed")),
                           on="accession", how="left")
    marked = marked.with_columns(
        pl.when(pl.col("placed").is_null()).then(False).otherwise(True).alias("placed"))

    dark = marked.filter(~pl.col("placed")).select("accession").with_columns(
        pl.lit(args.species).alias("species"))
    dark.write_parquet(args.out, compression="zstd")

    total = proteome.height
    n_dark = dark.height
    # The focus proteins: for each, whether it is placed, and every arm's best E-value
    # against it (None where the arm reported nothing at the permissive cutoff).
    focus = {}
    for acc, what in focus_proteins(args.registry, args.species).items():
        mine = hits.filter(pl.col("query") == acc)
        arms = {}
        for arm in sorted(hits["arm"].unique().to_list()):
            best = mine.filter(pl.col("arm") == arm)["evalue"].min()
            arms[arm] = {"best_evalue": (float(best) if best is not None else None),
                         "placed": bool(best is not None and best <= args.evalue_call)}
        focus[acc] = {"what": what, "in_proteome": acc in accs,
                      "placed": any(a["placed"] for a in arms.values()),
                      "per_arm": arms}

    summary = {
        "species": args.species,
        "proteins_in_proteome": total,
        "proteins_placed_by_any_arm": total - n_dark,
        "proteins_dark": n_dark,
        "fraction_dark": round(n_dark / total, 4),
        "evalue_call": args.evalue_call,
        "proteins_placed_per_arm": per_arm,
        "raw_hit_rows": hits.height,
        "focus_proteins": focus,
    }
    print(json.dumps(summary, indent=2))
    print(f"\n{args.species}: {n_dark} of {total} proteins ({100 * n_dark / total:.1f}%) "
          f"are dark to phmmer, jackhmmer and mmseqs2 at E<={args.evalue_call}")
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
