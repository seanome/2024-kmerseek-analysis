#!/usr/bin/env python3
"""Convert Folddisco motif hits into this pipeline's normalized region rows.

Folddisco is not an aligner. Its query is a set of motifs drawn from the query structure,
and a hit is a *discontinuous* set of residues -- "A56,A99,A195" -- not a contiguous
alignment. Every other arm here reports an interval, so the residue set is reduced to its
envelope (first matched residue to last).

That reduction is lossy in one direction only, and it matters: a 3-residue motif spanning
positions 56..195 produces a 139-residue envelope while actually touching 3 residues. The
envelope therefore *overstates* Folddisco's footprint. Two consequences, both handled
rather than hidden:

  - The count of genuinely matched residues is carried through as a 9th column, so the
    envelope's density is visible downstream instead of being assumed.
  - Scoring a motif envelope by interval IoU, the way an alignment is scored, would
    penalise Folddisco for the reduction rather than for its predictions. The evaluator
    takes --interval-semantics motif for this arm, which asks whether the envelope covers
    the true domain rather than whether it coincides with it.

Folddisco emits no E-value. `idf` (inverse document frequency; rarer motif, higher score)
goes in the score column, matching every other arm's bigger-is-better convention, and
`rmsd` goes in the E-value slot, which is likewise lower-is-better.
"""

import argparse
import re
from pathlib import Path

# Residue labels carry a chain prefix: "A56" and occasionally "A-12" for negative
# numbering. Capture the trailing signed integer and ignore the chain.
RESIDUE_RE = re.compile(r"(-?\d+)\s*$")


def residue_span(field: str) -> tuple[int, int, int] | None:
    """(start, end, n_matched) for a comma-separated residue list."""
    positions = []
    for token in field.split(","):
        token = token.strip()
        if not token:
            continue
        m = RESIDUE_RE.search(token)
        if m:
            positions.append(int(m.group(1)))
    if not positions:
        return None
    return min(positions), max(positions), len(positions)


def accession_from_tid(tid: str) -> str:
    """Folddisco names targets by structure path; reduce to the UniProt accession."""
    name = Path(tid.strip()).name
    if name.startswith("AF-"):
        return name.split("-")[1]
    for suffix in (".pdb", ".cif", ".mmcif", ".gz", ".ent"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hits", required=True, type=Path,
                   help="folddisco query stdout: tid node_count idf rmsd matching_residues query_residues")
    p.add_argument("--query-accession", required=True,
                   help="folddisco does not echo the query name; it is one structure per run")
    p.add_argument("--out", required=True, type=Path, help="append normalized rows here")
    args = p.parse_args()

    written = 0
    with open(args.hits) as fh, open(args.out, "a") as out:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 6:
                continue
            tid, node_count, idf, rmsd, matching, query_res = parts[:6]

            t_span = residue_span(matching)
            q_span = residue_span(query_res)
            if t_span is None or q_span is None:
                continue

            try:
                score = float(idf)
                rmsd_val = float(rmsd)
            except ValueError:
                continue

            # n_matched comes from the query side: it is how many of the query's residues
            # were placed, which is what node_count reports.
            out.write(
                f"{args.query_accession}\t{accession_from_tid(tid)}\t"
                f"{q_span[0]}\t{q_span[1]}\t{t_span[0]}\t{t_span[1]}\t"
                f"{score}\t{rmsd_val}\t{q_span[2]}\n"
            )
            written += 1

    print(f"{args.query_accession}: {written} rows")


if __name__ == "__main__":
    main()
