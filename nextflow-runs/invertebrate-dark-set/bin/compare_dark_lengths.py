#!/usr/bin/env python3
"""Is the dark set short? Protein length, dark vs placed.

The dark set is every protein phmmer, jackhmmer and mmseqs2 all failed to place. That
fraction is the headline of the proteome-annotate claim, and it currently conflates two
things that need separating: proteins whose homologs sequence search genuinely cannot
reach, and gene models that are not real proteins at all.

Botryllus is a 2026 annotation. New gene sets carry a tail of spurious models -- fragments,
mispredictions, ORFs called on non-coding sequence -- and nothing places a spurious model
into Swiss-Prot because there is nothing to place. Those land in the dark set for a reason
that has nothing to do with homology detection being hard.

Junk gene models are typically SHORT, so length is the cheapest available discriminator.
If dark proteins skew sharply shorter than placed ones, part of the dark fraction is
annotation noise and the headline needs qualifying by how much. If the two distributions
sit on top of each other, shortness-driven contamination is not what is inflating the
number. This script reports which, and does not decide in advance which it wants.

Length is a proxy and only a proxy. Short proteins are also genuinely harder for sequence
search -- fewer residues means less signal, so a real short protein is more likely to be
dark on the merits. A shortness skew is therefore consistent with contamination but does
not prove it; separating the two needs the annotation evidence (InterProScan, expression,
Chainsaw), not this file.

Reported both ways on purpose: a p-value on 45_339 proteins is significant at effect sizes
far too small to matter, so the effect size is what to read. The common-language effect
size is the probability that a randomly drawn dark protein is longer than a randomly drawn
placed one -- 0.5 is no difference, below 0.5 means dark proteins are shorter.
"""

import argparse
import gzip
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

# Two thresholds under which a eukaryotic gene model is more likely to be a fragment or a
# spurious call than a real protein. Neither is a rule; they are here so the tail can be
# read directly instead of inferred from a median.
SHORT_AA = (50, 100)

# The dark fraction recomputed over proteins of at least this many residues. The raw
# fraction counts a 60-aa gene model that nothing can align the same as a 400-aa protein
# nothing can place, and on the QfO ladder that mattered: 48% of ciona's dark set is under
# 100 aa against 16-18% for the other species, and the ciona fraction moved from 29% to
# 18% at >= 100 aa while nothing else moved more than four points. 0 is the raw number,
# kept in the same table so the two are read side by side.
MIN_AA = (0, 50, 100, 200)


def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sided Mann-Whitney U. Returns (U for x, p-value).

    Written out rather than imported from scipy because this runs inside
    params.kmerseek_image, which has polars and numpy but NOT scipy -- checked, not
    assumed. Adding a container directive to get scipy would pin the image in a second
    place and break the profile that already sets it.

    This is the normal approximation with tie correction and a continuity correction,
    which is what scipy.stats.mannwhitneyu(method="asymptotic") computes. The exact test
    is O(n1*n2) and would never finish on 45_339 proteins, and integer lengths are nothing
    but ties, so the approximation is the right one regardless of what is available.
    Verified against scipy to ~1e-12 relative on tied, untied and heavily-tied inputs.
    """
    n1, n2 = len(x), len(y)
    both = np.concatenate([x, y]).astype(np.float64)
    n = n1 + n2

    # Average ranks, ties shared. For a value whose run of duplicates ends at cumulative
    # position e, the ranks it occupies are e-c+1..e, averaging e - (c-1)/2.
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    end = np.cumsum(counts)
    ranks = (end - (counts - 1) / 2.0)[inverse]

    u_x = float(ranks[:n1].sum() - n1 * (n1 + 1) / 2.0)
    u_y = n1 * n2 - u_x

    # Tie correction: without it the variance is overstated and every p-value is too large.
    tie_term = float((counts.astype(np.float64) ** 3 - counts).sum())
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1.0))))
    if sigma == 0:
        # Every length identical: there is no difference to detect, and dividing by zero
        # would report one.
        return u_x, 1.0

    # Two-sided works off the larger U, with the 0.5 continuity correction pulling toward
    # the mean, then doubles the upper tail.
    z = (max(u_x, u_y) - n1 * n2 / 2.0 - 0.5) / sigma
    p = 2.0 * (0.5 * math.erfc(z / math.sqrt(2.0)))
    return u_x, min(max(p, 0.0), 1.0)


def read_lengths(fasta: Path) -> pl.DataFrame:
    """accession -> residue count, accession taken verbatim as the FASTA writes it."""
    opener = gzip.open if fasta.suffix == ".gz" else open
    accs: list[str] = []
    lengths: list[int] = []
    with opener(fasta, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                # First whitespace-delimited token, NOT split on '|'. Botryllus ids look
                # like FUN000001_FUN000001 and are the join key to the dark-set parquet;
                # splitting on '|' would mangle them and silently join nothing.
                accs.append(line[1:].strip().split()[0])
                lengths.append(0)
            elif accs:
                # '*' is a stop codon and '-' a gap, neither is a residue.
                lengths[-1] += len(line.strip().replace("*", "").replace("-", ""))
    return pl.DataFrame({"accession": accs, "length": lengths},
                        schema={"accession": pl.Utf8, "length": pl.Int64})


def group_stats(df: pl.DataFrame, group: str) -> dict:
    lens = df.filter(pl.col("group") == group)["length"]
    stats = {
        "n": lens.len(),
        "median": float(lens.median()),
        "mean": round(float(lens.mean()), 1),
        "p25": float(lens.quantile(0.25, interpolation="linear")),
        "p75": float(lens.quantile(0.75, interpolation="linear")),
        "min": int(lens.min()),
        "max": int(lens.max()),
    }
    for cut in SHORT_AA:
        n_short = int((lens < cut).sum())
        stats[f"fraction_under_{cut}aa"] = round(n_short / lens.len(), 4)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, required=True,
                    help="query proteome FASTA, the same one computeDarkSet counted")
    ap.add_argument("--dark", type=Path, required=True,
                    help="dark-set parquet with an 'accession' column")
    ap.add_argument("--species", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    proteome = read_lengths(args.query)
    if proteome.height == 0:
        raise SystemExit(
            f"no records in the query FASTA for '{args.species}': {args.query}. "
            f"An empty proteome cannot produce a length comparison, and an empty result "
            f"here would read downstream as 'no length difference'."
        )

    # A duplicated accession would be counted twice in one group and skew both the median
    # and the test, so dedupe on the join key exactly as computeDarkSet does.
    n_raw = proteome.height
    proteome = proteome.unique(subset="accession", keep="first")
    n_duplicate_accessions = n_raw - proteome.height

    dark_df = pl.read_parquet(args.dark)
    if "accession" not in dark_df.columns:
        raise SystemExit(
            f"the dark-set parquet for '{args.species}' has no 'accession' column "
            f"(found: {', '.join(dark_df.columns)}): {args.dark}"
        )
    dark_accs = dark_df.select("accession").unique()
    if dark_accs.height == 0:
        raise SystemExit(
            f"the dark set for '{args.species}' is empty, so there are no dark lengths to "
            f"compare against. Either every protein was placed by a sequence arm, or "
            f"computeDarkSet ran against a broken reference."
        )

    # Left join then an explicit null check, never a horizontal min/max: polars' horizontal
    # helpers skip nulls, which on this project silently turned every false positive into a
    # perfect score once already.
    marked = proteome.join(
        dark_accs.with_columns(pl.lit(True).alias("is_dark")), on="accession", how="left")
    marked = marked.with_columns(
        pl.when(pl.col("is_dark").is_null()).then(pl.lit("placed")).otherwise(pl.lit("dark"))
          .alias("group")
    ).drop("is_dark").with_columns(pl.lit(args.species).alias("species"))

    n_dark = int((marked["group"] == "dark").sum())
    # The join is the whole comparison. If the dark accessions are not the FASTA's
    # accessions -- a different proteome build, or an id that got split on '|' somewhere
    # upstream -- every protein lands in 'placed' and the result reads as a clean null.
    matched = dark_accs.join(proteome.select("accession"), on="accession", how="inner").height
    if matched == 0:
        raise SystemExit(
            f"none of the {dark_accs.height} dark accessions for '{args.species}' appear in "
            f"{args.query}. The dark set and the query FASTA are not the same proteome, or "
            f"the accessions were parsed differently. Example dark accession: "
            f"{dark_accs['accession'][0]!r}; example FASTA accession: "
            f"{proteome['accession'][0]!r}."
        )
    if matched < dark_accs.height:
        raise SystemExit(
            f"{dark_accs.height - matched} of {dark_accs.height} dark accessions for "
            f"'{args.species}' are missing from {args.query}. Every dark protein came from "
            f"this proteome, so a missing one means the two inputs disagree; refusing to "
            f"report a comparison over a subset."
        )

    n_placed = marked.height - n_dark
    if n_placed == 0:
        raise SystemExit(
            f"every protein in '{args.species}' is dark, so there is no placed group to "
            f"compare lengths against. That is a failed run, not a result."
        )

    # Sorted before writing: the join above does not preserve input order, so without this
    # two identical runs produce parquets that differ row-for-row. Every statistic below is
    # order-independent, but a file that changes for no reason is a file nobody can diff.
    marked = marked.select("accession", "species", "group", "length").sort("accession")
    marked.write_parquet(args.out, compression="zstd")

    dark_len = marked.filter(pl.col("group") == "dark")["length"].to_numpy()
    placed_len = marked.filter(pl.col("group") == "placed")["length"].to_numpy()

    u_stat, p_value = mann_whitney_u(dark_len, placed_len)
    # U counted for the dark group, so cles = P(dark > placed): 0.5 is no difference, below
    # 0.5 means dark proteins are shorter. Rank-biserial is the same number on -1..1.
    cles = u_stat / (n_dark * n_placed)

    dark_stats = group_stats(marked, "dark")
    placed_stats = group_stats(marked, "placed")

    by_min_length = []
    for cut in MIN_AA:
        kept = marked.filter(pl.col("length") >= cut)
        n_kept_dark = int((kept["group"] == "dark").sum())
        by_min_length.append({
            "min_aa": cut,
            "proteins": kept.height,
            "dark": n_kept_dark,
            "fraction_dark": round(n_kept_dark / kept.height, 4) if kept.height else None,
        })

    summary = {
        "species": args.species,
        "proteins_in_proteome": marked.height,
        "proteins_dark": n_dark,
        "proteins_placed": n_placed,
        "fraction_dark": round(n_dark / marked.height, 4),
        "dark_fraction_by_min_length": by_min_length,
        "duplicate_accessions_dropped": n_duplicate_accessions,
        "dark": dark_stats,
        "placed": placed_stats,
        "median_ratio_dark_over_placed": round(
            dark_stats["median"] / placed_stats["median"], 4) if placed_stats["median"] else None,
        "mann_whitney_u": {
            "u_statistic": u_stat,
            "p_value": p_value,
            "alternative": "two-sided",
            "common_language_effect_size": round(cles, 4),
            "rank_biserial_correlation": round(2 * cles - 1, 4),
            "effect_size_note": (
                "common_language_effect_size is P(a random dark protein is longer than a "
                "random placed one). 0.5 is no difference; below 0.5 means dark proteins "
                "are shorter. Read this, not the p-value: at this n the p-value is "
                "significant at effect sizes too small to matter."
            ),
        },
    }

    print(json.dumps(summary, indent=2))

    direction = ("shorter than" if dark_stats["median"] < placed_stats["median"]
                 else "longer than" if dark_stats["median"] > placed_stats["median"]
                 else "the same length as")
    print(f"\n{args.species}: {n_dark} dark proteins have a median length of "
          f"{dark_stats['median']:.0f} aa (IQR {dark_stats['p25']:.0f}-{dark_stats['p75']:.0f}), "
          f"{n_placed} placed proteins {placed_stats['median']:.0f} aa "
          f"(IQR {placed_stats['p25']:.0f}-{placed_stats['p75']:.0f}).")
    print(f"Dark proteins are {direction} placed proteins by median. "
          f"Mann-Whitney p={p_value:.3g}, "
          f"common-language effect size {cles:.3f} "
          f"(rank-biserial {2 * cles - 1:+.3f}).")
    for cut in SHORT_AA:
        print(f"Under {cut} aa: {100 * dark_stats[f'fraction_under_{cut}aa']:.1f}% of dark, "
              f"{100 * placed_stats[f'fraction_under_{cut}aa']:.1f}% of placed.")
    print("Dark fraction over proteins of at least: " + ", ".join(
        f"{r['min_aa']} aa {100 * r['fraction_dark']:.1f}% ({r['dark']}/{r['proteins']})"
        for r in by_min_length if r["fraction_dark"] is not None))

    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
