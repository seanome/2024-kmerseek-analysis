#!/usr/bin/env python3
"""Is the dark set more disordered than the proteins the sequence arms did place?

The dark set is every protein phmmer, jackhmmer and mmseqs2 all failed to place. This
script attaches a sequence-only disorder score to each of those proteins and to each
placed one, and asks whether the two distributions differ.

Why the question matters here. A known result on this project is that kmerseek's
HP-alphabet arms show a real accuracy dip at low pLDDT -- accuracy tracks disorder. If
the dark set turns out to be markedly more disordered than the placed set, that is two
things at once: an explanation for why sequence search misses these proteins, and a
warning that the territory kmerseek is being pointed at is exactly the territory where
its own accuracy is weakest. That is an uncomfortable result for the proteome-annotate
claim, which is why it is measured rather than assumed, and reported whichever way it
comes out. The reverse result -- dark proteins no more disordered than placed ones --
would mean the dark set is genuinely remote homology rather than junk, which is the
stronger case for the claim. Both are worth knowing; neither is worth guessing.

Disorder comes from metapredict (sequence only). It is deliberately NOT the pLDDT proxy:
pLDDT below 50 is a confidence measurement that correlates with disorder, and it also
drops when AlphaFold merely modelled a protein badly, which usually means a shallow MSA
-- and a shallow MSA is precisely what makes phmmer and jackhmmer fail too. Reading this
axis off pLDDT would therefore risk measuring the very thing that defines the dark set,
and calling the circularity a finding. Botryllus also has no usable AFDB coverage
(~1.6% of the query set), so pLDDT is not available here in any case.

The comparison is Mann-Whitney U, two-sided: mean-disorder-per-protein is bounded on
[0, 1] and strongly skewed, so a t-test's normality assumption does not hold. The
reported effect size is the rank-biserial correlation, which is a monotone function of U
and reads as "how often a random dark protein is more disordered than a random placed
one". A p-value on 45_339 proteins will be small for an effect of no practical size, so
the medians, the IQRs and the effect size are the numbers to read, not the p-value.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import polars as pl
from scipy.stats import mannwhitneyu

# Below this many proteins in either group, the U test is reported as null rather than
# computed. scipy will happily return a p-value for n=1 and it means nothing; a null with
# a stated reason is more honest than a number a reader would take at face value.
MIN_GROUP_FOR_TEST = 3


def query_accessions(fasta: Path) -> list[str]:
    """Accessions exactly as compute_dark_set.py reads them.

    Byte-identical logic to that script on purpose: these two lists are joined, so any
    divergence in how the key is derived turns into a silent all-miss join. The
    accession is the first whitespace-delimited token with NO splitting on '|', which
    would break Botryllus's FUN000001_FUN000001 ids differently from UniProt's.
    """
    opener = gzip.open if fasta.suffix == ".gz" else open
    accs = []
    with opener(fasta, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                accs.append(line[1:].strip().split()[0])
    return accs


def quartiles(s: pl.Series) -> dict:
    """n, median and IQR for one group, or nulls when the group is empty."""
    if s.len() == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "iqr": None,
                "mean": None}
    q25, q75 = float(s.quantile(0.25)), float(s.quantile(0.75))
    return {
        "n": s.len(),
        "median": float(s.median()),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "mean": float(s.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disorder", type=Path, required=True,
                    help="per-protein parquet from predict_disorder_metapredict.py")
    ap.add_argument("--dark", type=Path, required=True,
                    help="dark-set parquet from compute_dark_set.py")
    ap.add_argument("--query", type=Path, required=True,
                    help="the query proteome FASTA, which defines the denominator")
    ap.add_argument("--species", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, required=True)
    args = ap.parse_args()

    sp = args.species

    # Every input is checked by hand before anything is read, so a missing or truncated
    # upstream product fails naming the species instead of producing a confident-looking
    # table built on nothing.
    for label, path in (("query FASTA", args.query),
                        ("disorder parquet", args.disorder),
                        ("dark-set parquet", args.dark)):
        if not path.exists():
            raise SystemExit(f"{sp}: {label} does not exist: {path}")
        if path.stat().st_size == 0:
            raise SystemExit(f"{sp}: {label} is empty: {path}")

    proteome = query_accessions(args.query)
    if not proteome:
        raise SystemExit(f"{sp}: no records in the query FASTA {args.query}")

    disorder = pl.read_parquet(args.disorder)
    if disorder.height == 0:
        raise SystemExit(
            f"{sp}: metapredict scored 0 proteins from {args.query.name}. That is a "
            f"failed run, not a result -- check the FASTA reached the container intact."
        )

    dark = pl.read_parquet(args.dark)
    if dark.height == 0:
        raise SystemExit(
            f"{sp}: the dark-set parquet holds 0 proteins, so there is nothing to "
            f"compare the placed set against. compute_dark_set.py writes only the dark "
            f"accessions, so an empty file means every protein was placed by some arm -- "
            f"which for a real invertebrate proteome means the reference or the searches "
            f"are wrong, not that the proteome is fully annotated."
        )

    # Left join then null-check, never a horizontal min/max: a left join that misses
    # leaves nulls, and treating null as "not dark" has to be explicit or it silently
    # becomes one. This is the same guard compute_dark_set.py uses for `placed`.
    dark_keys = dark.select("accession").unique().with_columns(
        pl.lit(True).alias("is_dark"))
    labelled = disorder.join(dark_keys, on="accession", how="left").with_columns(
        pl.when(pl.col("is_dark").is_null()).then(False).otherwise(True).alias("is_dark")
    ).with_columns(pl.lit(sp).alias("species"))

    # A dark accession with no disorder row means the two scripts disagree about the key,
    # which is the failure mode the --accession-mode flag exists to prevent. It would
    # otherwise show up only as a dark group that is quietly too small.
    n_matched = int(labelled["is_dark"].sum())
    if n_matched != dark.select("accession").n_unique():
        missing = (dark_keys.join(disorder.select("accession"), on="accession", how="anti")
                   ["accession"].head(5).to_list())
        raise SystemExit(
            f"{sp}: {dark.select('accession').n_unique() - n_matched} of "
            f"{dark.select('accession').n_unique()} dark accessions have no disorder row. "
            f"The dark parquet and the disorder parquet are not keyed the same way. "
            f"First few unmatched: {missing}. predict_disorder_metapredict.py must run "
            f"with --accession-mode verbatim for this pipeline."
        )

    labelled.write_parquet(args.out, compression="zstd")

    col = "mean_disorder_metapredict"
    dark_vals = labelled.filter(pl.col("is_dark"))[col].drop_nulls()
    placed_vals = labelled.filter(~pl.col("is_dark"))[col].drop_nulls()

    stats = {"dark": quartiles(dark_vals), "placed": quartiles(placed_vals)}

    u_stat = p_value = effect = None
    note = None
    if dark_vals.len() >= MIN_GROUP_FOR_TEST and placed_vals.len() >= MIN_GROUP_FOR_TEST:
        res = mannwhitneyu(dark_vals.to_list(), placed_vals.to_list(),
                           alternative="two-sided")
        u_stat, p_value = float(res.statistic), float(res.pvalue)
        # Rank-biserial: U / (n1 * n2) is the probability a random dark protein outranks
        # a random placed one; 2p - 1 recentres it on 0 so the sign carries the direction.
        effect = 2.0 * (u_stat / (dark_vals.len() * placed_vals.len())) - 1.0
    else:
        note = (f"Mann-Whitney U not computed: one group has fewer than "
                f"{MIN_GROUP_FOR_TEST} proteins with a disorder score.")

    summary = {
        "species": sp,
        "proteins_in_proteome": len(proteome),
        "proteins_scored": labelled.height,
        "metric": col,
        "dark": stats["dark"],
        "placed": stats["placed"],
        "mannwhitneyu_u": u_stat,
        "mannwhitneyu_p": p_value,
        "rank_biserial": effect,
        "note": note,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    d, p = stats["dark"], stats["placed"]
    if d["median"] is not None and p["median"] is not None:
        direction = ("MORE disordered" if d["median"] > p["median"]
                     else "LESS disordered" if d["median"] < p["median"]
                     else "no different in median disorder")
        print(f"\n{sp}: the {d['n']} dark proteins are {direction} than the "
              f"{p['n']} placed ones -- median mean-disorder "
              f"{d['median']:.3f} (IQR {d['iqr']:.3f}) vs {p['median']:.3f} "
              f"(IQR {p['iqr']:.3f}).")
        if effect is not None:
            print(f"Rank-biserial {effect:+.3f}, Mann-Whitney p={p_value:.3g}. "
                  f"Read the medians and the effect size, not the p-value: at this n a "
                  f"p-value is small for an effect of no practical size.")


if __name__ == "__main__":
    main()
