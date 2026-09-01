#!/usr/bin/env python3
"""Build the query-side answer key and the target-side domain-transfer maps.

Two products, both from results/pfam_benchmark/annotations:

  human_domain_truth.parquet    every Pfam domain instance on a human protein, with its
                                interval. This is what a domain call is scored against.
  <species>_domain_map.parquet  every Pfam domain instance on a target protein. An aligned
                                region's target interval is looked up here to decide which
                                Pfam family the region is claiming.

Only domains with resolved coordinates (has_position) are usable: a domain with no interval
cannot be matched against a region's interval in either direction.
"""

import argparse
import hashlib
import json
from pathlib import Path

import polars as pl

# domain_start/domain_end are the Pfam envelope; domain_ali_start/end are the aligned core.
# The envelope is the right target here -- a tool that finds the domain but trims the edges
# should not be penalised for missing residues Pfam itself marks as peripheral.
COLUMNS = ["accession", "pfam_id", "domain_start", "domain_end", "protein_length"]


def assign_split(df: pl.DataFrame, by: str, holdout_fraction: float, seed: int) -> pl.DataFrame:
    """Tag every domain instance selection/heldout, grouped so a unit never straddles.

    None of the tools here learn anything from this data, so there is nothing to overfit
    in the usual sense. The leak is elsewhere: picking the best of 113 alphabet x ksize
    combos on the same instances you then report is model selection, and reporting the
    winner's score on the data that chose it is optimistically biased. Tune on
    `selection`, report on `heldout`.

    Grouping by pfam_id is the default rather than by protein. Splitting on proteins lets
    the same family sit on both sides, so a ksize tuned on PF00001 in the selection half
    gets tested on PF00001 again -- the held-out score then measures memorised families,
    not generalisation. Grouping by family means the held-out half is families the sweep
    never saw.

    Hash-based, not random: the split is a pure function of the group key and the seed, so
    it reproduces across machines and across re-runs without carrying a state file.
    """
    key = "pfam_id" if by == "family" else "accession"

    def bucket(value: str) -> str:
        h = hashlib.sha256(f"{seed}:{value}".encode()).digest()
        # First 8 bytes as a fraction of the 64-bit space -- a uniform [0,1) draw keyed
        # on the group, so the holdout fraction is honoured in expectation.
        frac = int.from_bytes(h[:8], "big") / 2**64
        return "heldout" if frac < holdout_fraction else "selection"

    return df.with_columns(
        pl.col(key)
        .map_elements(bucket, return_dtype=pl.String)
        .alias("split")
    )


def load_domains(path: Path) -> pl.DataFrame:
    df = pl.read_parquet(path)
    df = df.filter(pl.col("has_position") & pl.col("domain_start").is_not_null())
    # Guard against zero/negative-length intervals, which would make every overlap
    # fraction either 0 or undefined downstream.
    df = df.filter(pl.col("domain_end") > pl.col("domain_start"))
    return df.select(COLUMNS).unique()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--truth-out", required=True, type=Path)
    p.add_argument("--map-outdir", required=True, type=Path)
    p.add_argument("--summary-out", required=True, type=Path)
    p.add_argument("--split-by", choices=["family", "protein"], default="family",
                   help="unit held out as a group; family is stricter, see assign_split")
    p.add_argument("--holdout-fraction", type=float, default=0.5)
    p.add_argument("--split-seed", type=int, default=20260818)
    args = p.parse_args()

    summary = {}

    human_path = args.annotations / "human_pfam_domains.parquet"
    if not human_path.exists():
        raise SystemExit(f"missing query-side annotations: {human_path}")

    human = load_domains(human_path)
    human = assign_split(human, args.split_by, args.holdout_fraction, args.split_seed)
    human.write_parquet(args.truth_out, compression="zstd")
    summary["human_truth"] = {
        "n_domain_instances": human.height,
        "n_proteins": human["accession"].n_unique(),
        "n_pfam_families": human["pfam_id"].n_unique(),
        "split_by": args.split_by,
        "holdout_fraction": args.holdout_fraction,
        "split_seed": args.split_seed,
        "splits": {
            row["split"]: {"n_instances": row["n"], "n_families": row["n_fam"]}
            for row in human.group_by("split")
            .agg(pl.len().alias("n"), pl.col("pfam_id").n_unique().alias("n_fam"))
            .to_dicts()
        },
    }

    args.map_outdir.mkdir(parents=True, exist_ok=True)
    for path in sorted(args.annotations.glob("*_pfam_domains.parquet")):
        species = path.name.replace("_pfam_domains.parquet", "")
        if species == "human":
            continue
        df = load_domains(path)
        df.write_parquet(args.map_outdir / f"{species}_domain_map.parquet", compression="zstd")
        summary[species] = {
            "n_domain_instances": df.height,
            "n_proteins": df["accession"].n_unique(),
            "n_pfam_families": df["pfam_id"].n_unique(),
            # Recall has a hard ceiling per species: a human domain family absent from the
            # target proteome's annotations can never be transferred, no matter how good
            # the search is. Record it so the ceiling is visible next to the score.
            "n_human_families_present": (
                df.select("pfam_id").unique().join(
                    human.select("pfam_id").unique(), on="pfam_id", how="inner"
                ).height
            ),
        }

    args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
