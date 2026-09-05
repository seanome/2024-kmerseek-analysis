#!/usr/bin/env python3
"""Per-protein disorder from metapredict, as a covariate parquet.

Why this exists alongside the pLDDT proxy: pLDDT below 50 is not a disorder measurement,
it is a confidence measurement that CORRELATES with disorder. It also drops when AlphaFold
simply modelled a protein badly, which usually means a shallow MSA -- and a shallow MSA
independently hurts the profile baselines (jackhmmer, hhblits). So an apparent "accuracy
falls with disorder" effect read off pLDDT could partly be an MSA-depth effect hitting
several arms at once.

metapredict predicts disorder from sequence alone. It needs no structure and no alignment,
so it shares neither confound, and the two axes disagreeing is itself informative.

Verified against metapredict 3.0.1: predict_disorder_fasta returns
{header: [sequence, per_residue_scores]}, scores in 0-1. Sanity check on the way in --
P04637 (p53, a textbook IDP) scores 0.432 disordered and P69905 (haemoglobin alpha, a
compact globin) scores 0.000.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def accession_of(header: str) -> str:
    """`sp|P04637|P53_HUMAN ...` -> `P04637`, matching every other table's key.

    Falls back to the first whitespace-delimited token for headers that are not
    UniProt-style, rather than dropping the record: an unjoinable accession is a
    left-join miss downstream, which is visible, while a dropped record is not.
    """
    first = header.split()[0]
    parts = first.split("|")
    return parts[1] if len(parts) >= 2 and parts[1] else first


# Pyrrolysine, the 22nd amino acid, substituted by its standard analogue.
#
# It is a real residue, not sequencing damage: encoded by UAG in the methylamine
# methyltransferases of Methanosarcina. Across all 79 canonical QfO proteomes it occurs 7
# times, in 7 proteins, in exactly one proteome -- macetivorans (UP000002487_188937), whose
# MtmB1 (P58865) is what stopped this step on the run where the archaea first enter the
# target set.
#
# metapredict passes invalid_sequence_action='convert' to protfasta, which repairs some
# symbols and then REVALIDATES, failing on what is left. Measured in the pinned container
# (metapredict:2026-08-27) rather than assumed, one symbol at a time:
#
#     B -> N     Z -> Q     X -> G     U -> C        repaired by protfasta
#     J, O                                           rejected by protfasta
#
# So O is the only symbol that needs handling here, and it has to be handled before
# protfasta sees the file -- changing invalid_sequence_action cannot help, because
# 'convert' is already what metapredict passes and 'convert' is what rejects O.
#
# O -> K is the conventional mapping and the defensible one: pyrrolysine is a lysine
# derivative, so the substitution keeps the backbone and the rough polarity, which is all a
# per-residue disorder predictor reads. Disorder is a stratification covariate rather than a
# scored quantity, and this is 7 residues in one proteome, so it cannot move a metric -- but
# the count is recorded in the summary rather than applied silently.
#
# U is deliberately NOT here. protfasta already maps it to C, identically, so listing it
# would claim credit for a substitution this script does not make and would write a cleaned
# copy of the 21 selenoprotein-bearing proteomes -- human among them, at 36 U in 25
# proteins -- for no change in the result.
NONCANONICAL = {"O": "K"}

# What protfasta repairs on its own, measured as above. Left to it so this script does not
# take over a job the library already does, and so anything genuinely unexpected errors.
#
# J is NOT in this set, and that is the measurement rather than an oversight: protfasta
# rejects it exactly as it rejects O. No QfO proteome contains one, so leaving J unmapped
# means the guard below names it clearly if one ever appears, instead of this script waving
# it through into the same opaque protfasta traceback it exists to prevent.
PROTFASTA_HANDLES = set("BZXU")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: Path):
    """(header, sequence) pairs, header without the leading '>'."""
    header, buf = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(buf)
                header, buf = line[1:].rstrip("\n"), []
            else:
                buf.append(line.strip())
    if header is not None:
        yield header, "".join(buf)


def substitute_noncanonical(fasta: Path, out: Path) -> dict:
    """Rewrite `fasta` with pyrrolysine replaced. Per-symbol counts, {} if nothing changed.

    Writes the output only when a substitution was actually needed, so the clean proteomes
    -- most of the 79 -- pay one extra read and no write. Headers are copied verbatim,
    which is what keeps accession_of() and every join downstream working.
    """
    counts: dict[str, int] = {}
    unexpected: dict[str, str] = {}
    records = []
    for header, seq in read_fasta(fasta):
        up = seq.upper()
        for sym, repl in NONCANONICAL.items():
            n = up.count(sym)
            if n:
                counts[sym] = counts.get(sym, 0) + n
                up = up.replace(sym, repl)
        # Anything still outside the standard 20 that protfasta does not repair itself,
        # reported by accession and symbol rather than left to surface as a protfasta
        # traceback naming a 450-residue sequence and no species.
        leftover = set(up) - STANDARD_AA - PROTFASTA_HANDLES
        if leftover and header not in unexpected:
            unexpected[header] = "".join(sorted(leftover))
        records.append((header, up))

    if unexpected:
        lines = [f"    {h.split()[0]}: {syms}" for h, syms in list(unexpected.items())[:5]]
        raise SystemExit(
            f"{fasta.name} holds residue symbols this script does not know how to map, "
            f"and metapredict cannot score them:\n" + "\n".join(lines) +
            (f"\n    ... and {len(unexpected) - 5} more entries" if len(unexpected) > 5 else "") +
            f"\n\nKnown non-canonical residues are {sorted(NONCANONICAL)} "
            f"(mapped to {[NONCANONICAL[k] for k in sorted(NONCANONICAL)]}); "
            f"protfasta repairs {sorted(PROTFASTA_HANDLES)} on its own. Add the new symbol "
            f"to NONCANONICAL with a justified analogue rather than stripping it."
        )

    if not counts:
        return {}

    with open(out, "w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                fh.write(seq[i:i + 60] + "\n")
    return counts


def write_domain_disorder(domains: Path, out: Path,
                          by_accession: dict[str, list], thr: float) -> int:
    """One row per domain instance, disorder averaged over that instance's own residues.

    Truth coordinates are 1-based and inclusive, which is what the Pfam envelope files
    carry and what every other table in this pipeline joins on, so the slice is
    [start - 1, end). A domain whose protein has no prediction, or whose interval falls
    outside the predicted length, is written with nulls rather than dropped: a missing row
    would silently shrink the denominator of any figure drawn on this axis.
    """
    truth = pl.read_parquet(domains)
    key = ["accession", "domain_start", "domain_end"]
    rows = []
    for r in truth.select(key).unique().iter_rows(named=True):
        scores = by_accession.get(r["accession"])
        lo, hi = int(r["domain_start"]) - 1, int(r["domain_end"])
        window = scores[lo:hi] if scores and 0 <= lo < hi <= len(scores) else []
        n = len(window)
        rows.append({
            **r,
            "n_residues_region": n,
            "mean_disorder_region": (sum(window) / n) if n else None,
            "disorder_fraction_region":
                (sum(1 for s in window if s >= thr) / n) if n else None,
        })
    df = pl.DataFrame(rows)
    df.write_parquet(out, compression="zstd")
    return df.height


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fasta", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--summary-out", type=Path)
    # The per-protein table above answers "is this protein disordered". It cannot answer
    # "is this DOMAIN disordered", and those differ: an ordered domain sitting in a mostly
    # disordered protein takes its protein's score and reads as disordered when it is not.
    # Given the truth intervals, the same residue scores are sliced per domain instance.
    p.add_argument("--domains", type=Path,
                   help="human domain truth parquet; when given, --domains-out gets one "
                        "row per domain instance instead of one per protein")
    p.add_argument("--domains-out", type=Path,
                   help="where to write the per-domain table (requires --domains)")
    # metapredict's own default is used when this is not set, rather than a number copied
    # here that would silently diverge from the package's if it ever changed.
    p.add_argument("--threshold", type=float, default=None,
                   help="disorder score at or above which a residue counts as disordered; "
                        "default is whatever metapredict's own default is")
    args = p.parse_args()
    if bool(args.domains) != bool(args.domains_out):
        raise SystemExit("--domains and --domains-out go together or not at all")

    import metapredict as mp

    # Substituted into a sibling file rather than in place: params.qfo_dir is the shared
    # QfO release every run reads, and rewriting a proteome there would change an input
    # hash for every other process that stages it.
    cleaned = args.fasta.with_name(args.fasta.stem + ".canonical_aa.fasta")
    substitutions = substitute_noncanonical(args.fasta, cleaned)
    scored_fasta = cleaned if substitutions else args.fasta
    if substitutions:
        detail = ", ".join(f"{n} {sym}->{NONCANONICAL[sym]}"
                           for sym, n in sorted(substitutions.items()))
        print(f"NOTE: {args.fasta.name}: substituted {detail} before scoring")

    preds = mp.predict_disorder_fasta(str(scored_fasta), show_progress_bar=False)
    thr = args.threshold if args.threshold is not None else 0.5

    rows, by_accession = [], {}
    for header, value in preds.items():
        # [sequence, scores]. Guarded rather than unpacked blind: this shape is a property
        # of the metapredict version, and a silent misread would produce a plausible-looking
        # column of wrong numbers.
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            raise SystemExit(
                f"metapredict returned {type(value).__name__} of length "
                f"{len(value) if hasattr(value, '__len__') else '?'} for {header!r}; "
                "expected [sequence, scores]. The API changed -- check the version."
            )
        _seq, scores = value
        n = len(scores)
        if n == 0:
            continue
        by_accession[accession_of(header)] = scores
        rows.append({
            "accession": accession_of(header),
            "n_residues_metapredict": n,
            "disorder_fraction_metapredict": sum(1 for s in scores if s >= thr) / n,
            "mean_disorder_metapredict": sum(scores) / n,
        })

    cov = pl.DataFrame(rows)
    cov.write_parquet(args.out, compression="zstd")

    n_domains = None
    if args.domains:
        n_domains = write_domain_disorder(args.domains, args.domains_out,
                                          by_accession, thr)

    summary = {
        "n_proteins": cov.height,
        "threshold": thr,
        "metapredict_version": getattr(mp, "__version__", "unknown"),
        "median_disorder_fraction":
            float(cov["disorder_fraction_metapredict"].median()) if cov.height else None,
        "n_mostly_disordered":
            int((cov["disorder_fraction_metapredict"] >= 0.5).sum()) if cov.height else 0,
        "n_domains": n_domains,
        # Empty for a proteome that needed none, so the field is always present and a
        # reader never has to tell "no substitutions" apart from "an older run that did
        # not record them".
        "noncanonical_substitutions": substitutions,
    }
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
