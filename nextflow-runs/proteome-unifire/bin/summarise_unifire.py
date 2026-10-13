#!/usr/bin/env python3
"""Turn UniFIRE's three prediction files into one parquet, and report the coverage number.

UniFIRE writes predictions_unirule.out, predictions_unirule-pirsr.out and
predictions_arba.out. This merges them, tagging each row with which rule system produced
it, because the three are not equivalent evidence: UniRule is manually curated, ARBA is
machine-learned, and mixing them into one "annotated" count overstates what curation
reaches.

The number this exists to produce is the last one printed: how many proteins in the
proteome received NO annotation from any rule system. UniRule conditions are taxon-scoped
and thin outside well-studied clades, so on a bivalve or a tunicate that fraction should be
large -- and it is the proteome-annotate premise measured rather than asserted.

Nothing here is an answer key. These are rule-based inferences, circular with every profile
arm and with Swiss-Prot itself.
"""

import argparse
import json
from pathlib import Path

import polars as pl

FILES = {
    "unirule": "predictions_unirule.out",
    "unirule_pirsr": "predictions_unirule-pirsr.out",
    "arba": "predictions_arba.out",
}


def load(path: Path, system: str) -> pl.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    df = pl.read_csv(path, separator="\t", infer_schema_length=0,
                     truncate_ragged_lines=True)
    if df.height == 0:
        return None
    return df.with_columns(pl.lit(system).alias("rule_system"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indir", type=Path, required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--fasta", type=Path, required=True,
                    help="the proteome, for the denominator")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    frames, missing = [], []
    for system, fname in FILES.items():
        df = load(args.indir / fname, system)
        if df is None:
            missing.append(fname)
        else:
            frames.append(df)

    if not frames:
        raise SystemExit(
            f"no UniFIRE predictions under {args.indir} for '{args.species}' "
            f"(looked for {', '.join(FILES.values())}).\n"
            f"An empty result here is ambiguous between 'the rules reached nothing' and "
            f"'the run did not happen', and those need opposite responses. If the run did "
            f"complete, check that the input FASTA carried OX={args.species}'s taxid -- "
            f"without it UniRule's taxon-scoped conditions cannot fire and every file is "
            f"legitimately empty."
        )

    merged = pl.concat(frames, how="diagonal")
    merged.write_parquet(args.out, compression="zstd")

    # The accession column UniFIRE echoes back is the first field; its name varies across
    # rule systems, so it is taken positionally rather than by a name that might not exist.
    acc_col = merged.columns[0]
    annotated = set(merged[acc_col].unique().to_list())

    total = 0
    with open(args.fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                total += 1

    per_system = {
        s: int(merged.filter(pl.col("rule_system") == s)[acc_col].n_unique())
        for s in merged["rule_system"].unique().to_list()
    }
    n_ann = len(annotated)
    summary = {
        "species": args.species,
        "proteins_in_proteome": total,
        "proteins_with_any_prediction": n_ann,
        "proteins_with_no_prediction": total - n_ann,
        "fraction_unannotated": round((total - n_ann) / total, 4) if total else None,
        "prediction_rows": merged.height,
        "proteins_by_rule_system": per_system,
        "files_absent_or_empty": missing,
    }
    print(json.dumps(summary, indent=2))
    print(f"\n{args.species}: {total - n_ann} of {total} proteins "
          f"({100 * (total - n_ann) / total:.1f}%) received no UniProt rule annotation"
          if total else "", flush=True)

    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
