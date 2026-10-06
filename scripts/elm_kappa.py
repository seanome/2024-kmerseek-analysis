#!/usr/bin/env python3
"""Kappa (chance-corrected class agreement) on the ELM ortholog pairs, per alphabet and stretch.

Question: the extension's mismatch penalty C comes from kappa measured on Pfam-A 38.2 seed
pairs at 20-30% identity (nextflow-runs/qfo-pfam-region-benchmark/assets/
kappa_by_alphabet.pfam_a_38.2_seed_pairs_20-30pct_identity.tsv, notebook 230). Does that
kappa hold on the pairs this benchmark is about?

Inputs (all written by scripts/elm_stage0_flanks.py):
  <stage0>/stage0_pairs.parquet       one row per (human ELM instance, species) with a 1:1
                                      OMA ortholog
  <stage0>/stage0_instances.parquet   motif start and end on the human protein
  <stage0>/alignments/<species>/<human>__<ortholog>.fasta   MAFFT alignment of the pair

Method: notebook 230's own function, notebooks/hp_conservation_utils.pair_stats, on the
alignment columns of one stretch of the human protein at a time:
  motif   the ELM instance's residues
  flanks  the 10 residues on each side (both sides together)
  rest    every other residue
For each stretch, pair_stats keeps the columns where both proteins have a residue and gives
  agree    = share of those columns whose two residues are in the same class
  expected = sum over classes of (share of the class in the human columns) x (share in the
             ortholog columns), the agreement a composition-keeping shuffle gives
  kappa    = (agree - expected) / (1 - expected)
Two summaries per species and stretch, both from that same function:
  pooled        the stretch's columns from every pair of the species joined into one string
                (pairs separated by a gap column, so no run crosses a pair boundary), then one
                pair_stats call. Used for the motif and flanks, which are about 6 and 20
                columns per pair, too few for a per-pair kappa.
  per-pair mean notebook 230's own summary: kappa per pair, averaged over pairs with at least
                50 aligned columns (its MIN_COLS). Only the rest-of-protein stretch has enough.
kappa is turned into the penalty with the formula in the Pfam table's header:
  C = -ln(1 - kappa) / ln(1 + kappa * classes - kappa)

Alphabets: the 15 in hp_conservation_utils.ALPHABET_CLUSTERS. mmseqs12, wass14 and hsdm17
are not in that module's tables, and funcgroups8 has no kappa anywhere, so those four are
left out and named in the output.

Output: <out>/elm_kappa_by_species.parquet and tables/250_elm_kappa_by_species.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
import hp_conservation_utils as hc  # noqa: E402

ELM_DIR = Path("/Users/olga/data/elm-motif-transfer")
FLANK = 10
MIN_COLS = 50  # notebook 230's MIN_COLS for a per-pair kappa
NOT_COVERED = ["mmseqs12", "wass14", "hsdm17", "funcgroups8"]


def read_alignment(path: Path) -> tuple[str, str]:
    recs = path.read_text().split(">")[1:]
    ha, ta = ["".join(r.splitlines()[1:]).upper() for r in recs]
    return ha, ta


def stretch(ha: str, ta: str, keep) -> tuple[str, str]:
    """The alignment columns whose human residue index i satisfies keep(i)."""
    qs, ts, i = [], [], 0
    for a, b in zip(ha, ta):
        if a != "-":
            if keep(i):
                qs.append(a)
                ts.append(b)
            i += 1
    return "".join(qs), "".join(ts)


def c_from_kappa(kappa: float, classes: int) -> float | None:
    if kappa is None or not (0 < kappa < 1):
        return None
    return -math.log(1 - kappa) / math.log(1 + kappa * classes - kappa)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage0", type=Path, default=ELM_DIR / "stage0")
    ap.add_argument("--out", type=Path, default=ELM_DIR / "stage0")
    ap.add_argument(
        "--table", type=Path, default=REPO / "tables" / "250_elm_kappa_by_species.csv"
    )
    args = ap.parse_args()

    pairs = pl.read_parquet(args.stage0 / "stage0_pairs.parquet").join(
        pl.read_parquet(args.stage0 / "stage0_instances.parquet").select(
            "elm_instance", "start", "end"
        ),
        on="elm_instance",
    )
    alphabets = list(hc.ALPHABET_CLUSTERS)
    rng = np.random.default_rng(250)
    rows, stretches = [], []
    for r in pairs.iter_rows(named=True):
        ha, ta = read_alignment(
            args.stage0
            / "alignments"
            / r["species"]
            / f"{r['accession']}__{r['ortholog']}.fasta"
        )
        s, e = r["start"], r["end"]
        windows = {
            "motif": lambda i: s <= i < e,
            "flanks": lambda i: (s - FLANK <= i < s) or (e <= i < e + FLANK),
            "rest": lambda i: not (s - FLANK <= i < e + FLANK),
        }
        for w, keep in windows.items():
            qs, ts = stretch(ha, ta, keep)
            stretches.append(((r["species"], r["mya"], w), qs, ts))
            for st in hc.pair_stats(qs, ts, alphabets, n_shuffle=1, rng=rng):
                rows.append(
                    {
                        "elm_instance": r["elm_instance"],
                        "species": r["species"],
                        "mya": r["mya"],
                        "window": w,
                        "alphabet": st["alphabet"],
                        "n_cols": st["n_cols"],
                        "agree": st["agree"],
                        "expected": st["expected"],
                        "kappa": st["kappa"],
                    }
                )
    per_pair = pl.DataFrame(rows)
    per_pair.write_parquet(args.out / "elm_kappa_per_pair.parquet")

    # Pooled: one pair_stats call per (species, stretch) on the joined columns.
    pooled = []
    cols: dict[tuple, list[tuple[str, str]]] = {}
    for key, qs, ts in stretches:
        cols.setdefault(key, []).append((qs, ts))
    for (species, mya, w), parts in cols.items():
        q = "-".join(a for a, _ in parts)
        t = "-".join(b for _, b in parts)
        for st in hc.pair_stats(q, t, alphabets, n_shuffle=1, rng=rng):
            pooled.append(
                {
                    "species": species,
                    "mya": mya,
                    "window": w,
                    "alphabet": st["alphabet"],
                    "n_pairs": len(parts),
                    "n_cols": st["n_cols"],
                    "agree": st["agree"],
                    "expected": st["expected"],
                    "kappa_pooled": st["kappa"],
                }
            )
    pooled = pl.DataFrame(pooled)
    per_pair_mean = (
        per_pair.filter(pl.col("n_cols") >= MIN_COLS)
        .with_columns(pl.col("kappa").fill_nan(None))
        .group_by("species", "window", "alphabet")
        .agg(n_pairs_ge_50_cols=pl.len(), kappa_per_pair_mean=pl.col("kappa").mean())
    )
    summ = (
        pooled.join(per_pair_mean, on=["species", "window", "alphabet"], how="left")
        .with_columns(
            classes=pl.col("alphabet").replace_strict(hc.SIZES, return_dtype=pl.Int64)
        )
        .with_columns(
            C_pooled=pl.struct("kappa_pooled", "classes").map_elements(
                lambda x: c_from_kappa(x["kappa_pooled"], x["classes"]),
                return_dtype=pl.Float64,
            ),
            C_per_pair_mean=pl.struct("kappa_per_pair_mean", "classes").map_elements(
                lambda x: c_from_kappa(x["kappa_per_pair_mean"], x["classes"]),
                return_dtype=pl.Float64,
            ),
        )
        .sort("mya", "window", "alphabet")
    )
    summ.write_parquet(args.out / "elm_kappa_by_species.parquet")
    args.table.parent.mkdir(parents=True, exist_ok=True)
    summ.with_columns(pl.col(pl.Float64).round(4)).write_csv(args.table)
    print(
        f"{per_pair['elm_instance'].n_unique()} human instances with an ortholog, {pairs.height} pairs, "
        f"{len(alphabets)} alphabets; not covered: {', '.join(NOT_COVERED)}"
    )
    print(
        summ.filter(pl.col("alphabet") == "hp_pbotc_1st_ed2")
        .select(
            "species",
            "window",
            "n_pairs",
            "n_cols",
            "agree",
            "expected",
            "kappa_pooled",
            "C_pooled",
            "n_pairs_ge_50_cols",
            "kappa_per_pair_mean",
            "C_per_pair_mean",
        )
        .with_columns(pl.col(pl.Float64).round(3))
    )
    print(f"wrote {args.table}")


if __name__ == "__main__":
    main()
