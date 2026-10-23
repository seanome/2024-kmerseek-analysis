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
import math
import re
from pathlib import Path

import polars as pl

# Every arm's tsv is written as: query, target, tstart, tend, bits, evalue
COLS = ["query", "target", "tstart", "tend", "bits", "evalue"]


ARMS = ("phmmer", "jackhmmer", "mmseqs2")


def arm_of(path: Path) -> str:
    match = re.search(r"(?:^|\.)(phmmer|jackhmmer|mmseqs2)\.tsv(?:\.gz)?$", path.name)
    if match is None:
        raise ValueError(f"unrecognised baseline arm filename: {path}")
    return match.group(1)


def validate_hit_files(paths: list[Path], expected_chunks: int | None = None) -> dict:
    """Require the same unique chunk outputs from all three baseline arms.

    A missing chunk shared by every arm can only be detected when expected_chunks is
    supplied from the pipeline's query split, not inferred from the available files.
    """
    chunks = {arm: set() for arm in ARMS}
    for path in paths:
        arm = arm_of(path)
        prefix = re.sub(r"(?:^|\.)" + arm + r"\.tsv(?:\.gz)?$", "", path.name)
        if prefix in chunks[arm]:
            raise ValueError(f"duplicate {arm} chunk output: {prefix}")
        chunks[arm].add(prefix)
    if not all(chunks.values()):
        raise ValueError("need output files for phmmer, jackhmmer and mmseqs2")
    if any(chunks[arm] != chunks[ARMS[0]] for arm in ARMS[1:]):
        raise ValueError("baseline arms have different chunk coverage")
    if expected_chunks is not None:
        if expected_chunks < 1 or len(chunks[ARMS[0]]) != expected_chunks:
            raise ValueError(
                f"expected {expected_chunks} chunks per arm; found {len(chunks[ARMS[0]])}"
            )
    return {
        "files_per_arm": {arm: len(ids) for arm, ids in chunks.items()},
        "expected_chunks": expected_chunks,
        "complete_chunk_count_checked": expected_chunks is not None,
    }


def read_hits(path: Path) -> pl.DataFrame:
    empty = pl.DataFrame(schema={c: pl.String for c in COLS})
    try:
        df = pl.read_csv(path, separator="\t", has_header=False, infer_schema_length=0)
    except pl.exceptions.NoDataError:
        df = empty
    if df.width != len(COLS):
        raise ValueError(f"{path}: expected exactly six hit columns, got {df.width}")
    df.columns = COLS
    if any(
        df[c].is_null().any() or (df[c].str.strip_chars() == "").any()
        for c in ("query", "target")
    ):
        raise ValueError(f"{path}: missing query or target identifier")
    ev = df["evalue"].cast(pl.Float64, strict=False)
    if (ev.is_null() | ~ev.is_finite() | (ev < 0)).any():
        raise ValueError(
            f"{path}: invalid E-value; expected finite nonnegative numbers"
        )
    return df.with_columns(ev.alias("evalue"), pl.lit(arm_of(path)).alias("arm"))


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


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--hits", type=Path, nargs="+", required=True)
    ap.add_argument("--evalue-call", type=float, default=1e-3)
    ap.add_argument(
        "--expected-chunks",
        type=int,
        default=None,
        help="query split chunk count; required to detect chunks missing from every arm",
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    if not math.isfinite(args.evalue_call) or args.evalue_call < 0:
        raise SystemExit("--evalue-call must be finite and nonnegative")
    validation = validate_hit_files(args.hits, args.expected_chunks)
    accs = query_accessions(args.query)
    if not accs:
        raise SystemExit(f"no records in {args.query}")
    if len(set(accs)) != len(accs):
        raise SystemExit("duplicate query accessions in FASTA")
    proteome = pl.DataFrame({"accession": accs})

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
    unknown = set(hits["query"].to_list()) - set(accs)
    if unknown:
        raise SystemExit(f"hit queries absent from FASTA: {sorted(unknown)[:3]}")
    called = hits.filter(pl.col("evalue") <= args.evalue_call)

    per_arm = {}
    for arm in ARMS:
        per_arm[arm] = int(called.filter(pl.col("arm") == arm)["query"].n_unique())

    placed = called.select("query").unique().rename({"query": "accession"})
    # Left join then null-check, never a horizontal min/max: a left join that misses leaves
    # nulls, and treating null as "no hit" has to be explicit or it silently becomes one.
    marked = proteome.join(
        placed.with_columns(pl.lit(True).alias("placed")), on="accession", how="left"
    )
    marked = marked.with_columns(
        pl.when(pl.col("placed").is_null()).then(False).otherwise(True).alias("placed")
    )

    dark = (
        marked.filter(~pl.col("placed"))
        .select("accession")
        .with_columns(pl.lit(args.species).alias("species"))
    )
    dark.write_parquet(args.out, compression="zstd")

    total = proteome.height
    n_dark = dark.height
    summary = {
        "species": args.species,
        "input_validation": validation,
        "proteins_in_proteome": total,
        "proteins_placed_by_any_arm": total - n_dark,
        "proteins_dark": n_dark,
        "fraction_dark": round(n_dark / total, 4),
        "evalue_call": args.evalue_call,
        "proteins_placed_per_arm": per_arm,
        "raw_hit_rows": hits.height,
    }
    print(json.dumps(summary, indent=2))
    print(
        f"\n{args.species}: {n_dark} of {total} proteins ({100 * n_dark / total:.1f}%) "
        f"are dark to phmmer, jackhmmer and mmseqs2 at E<={args.evalue_call}"
    )
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
