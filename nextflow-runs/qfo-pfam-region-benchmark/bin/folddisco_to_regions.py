#!/usr/bin/env python3
"""Convert Folddisco motif hits into this pipeline's normalized region rows.

Folddisco is not an aligner. Its query is a set of motifs drawn from the query structure,
and a hit is a *discontinuous* set of residues. "A56,A99,A195". not a contiguous
alignment. Every other arm here reports an interval, so the residue set is reduced to its
envelope (first matched residue to last).

That reduction is lossy in one direction only, and it matters: a 3-residue motif spanning
positions 56..195 produces a 139-residue envelope while touching 3 residues. The
envelope therefore *overstates* Folddisco's footprint. Two consequences, both handled
rather than hidden:

  - The count of matched residues is carried through as a 9th column, so the
    envelope's density is visible downstream instead of being assumed.
  - Scoring a motif envelope by interval IoU, the way an alignment is scored, would
    penalise Folddisco for the reduction rather than for its predictions. The evaluator
    takes --interval-semantics motif for this arm, which asks whether the envelope covers
    the true domain rather than whether it coincides with it.

Folddisco emits no E-value. `idf` (inverse document frequency; rarer motif, higher score)
goes in the score column, matching every other arm's bigger-is-better convention, and
`rmsd` goes in the E-value slot, which is likewise lower-is-better.

The two residue columns are one motif read from two structures, so entry i of
`matching_residues` (target side) answers entry i of `query_residues` (query side) and
they are the same length. Only the pairs where both sides are present count: the query
list is the whole motif that was asked for, so its full envelope would be the same
interval on every hit, and for the whole-structure queries this pipeline runs that
interval is the whole protein.

This depends on `folddisco query` being given -q. Folddisco has no other channel for
query-side positions -- `query_residues` is the -q argument echoed back, identical on
every row -- and query_pdb.rs echoes the ORIGINAL argument, so omitting -q searches the
whole structure but leaves that column blank. Every hit then converts to nothing, which is
how this arm scored zero calls across nine species while its tasks reported success.
"""

import argparse
import re
import sys
from pathlib import Path

# Residue labels carry a chain prefix: "A56" and occasionally "A-12" for negative
# numbering. Capture the trailing signed integer and ignore the chain.
RESIDUE_RE = re.compile(r"(-?\d+)\s*$")


def residue_positions(field: str) -> list[int | None]:
    """Positions of a comma-separated folddisco residue list, None where it wrote `_`.

    Position is load-bearing, so unmatched nodes keep their slot rather than being dropped:
    the target list and the query list are the same motif read from two structures, and
    entry i of one corresponds to entry i of the other.
    """
    out: list[int | None] = []
    for token in field.split(","):
        token = token.strip()
        m = RESIDUE_RE.search(token) if token else None
        out.append(int(m.group(1)) if m else None)
    return out


# AlphaFold models proteins over 2700 aa as overlapping 1400-residue fragments on a
# 200-residue stride, numbering each fragment's residues from 1. A hit on F<n> therefore
# sits (n-1)*200 before its true position in the full sequence Pfam annotates. Verified
# directly on AF-A0A087WUL8-F2: auth_seq_id 1..1400, SIFTS xref UniProt 201..1600.
AF_FRAGMENT_STRIDE = 200
_AF_FRAGMENT = re.compile(r"-F(\d+)")


def accession_and_offset(tid: str) -> tuple[str, int]:
    """Folddisco names targets by structure path; reduce to accession plus fragment offset."""
    name = Path(tid.strip()).name
    if name.startswith("AF-"):
        m = _AF_FRAGMENT.search(name)
        offset = (int(m.group(1)) - 1) * AF_FRAGMENT_STRIDE if m else 0
        return name.split("-")[1], offset
    for suffix in (".pdb", ".cif", ".mmcif", ".gz", ".ent"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name, 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hits", required=True, type=Path,
                   help="folddisco query stdout: tid node_count idf rmsd matching_residues query_residues")
    p.add_argument("--query-accession", required=True,
                   help="folddisco does not echo the query name; it is one structure per run")
    p.add_argument("--out", required=True, type=Path, help="append normalized rows here")
    args = p.parse_args()

    n_lines = 0
    written = 0
    reasons: dict[str, int] = {}

    def drop(reason: str):
        reasons[reason] = reasons.get(reason, 0) + 1

    with open(args.hits) as fh, open(args.out, "a") as out:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            n_lines += 1
            parts = line.split("\t")
            if len(parts) < 6:
                drop(f"fewer than 6 columns (saw {len(parts)})")
                continue
            tid, node_count, idf, rmsd, matching, query_res = parts[:6]

            if not query_res.strip():
                drop("query_residues column is empty")
                continue

            t_pos = residue_positions(matching)
            q_pos = residue_positions(query_res)
            if len(t_pos) != len(q_pos):
                drop(f"matching_residues has {len(t_pos)} entries, "
                     f"query_residues has {len(q_pos)}")
                continue

            # Only the nodes that were actually placed. The query list is the whole motif
            # asked for, so taking its full envelope would report the same interval for
            # every hit -- for a whole-structure query, the whole protein. Zipping keeps
            # the envelope to the part of the query the target answered.
            matched = [(q, t) for q, t in zip(q_pos, t_pos)
                       if q is not None and t is not None]
            if not matched:
                drop("no residue matched on both sides")
                continue

            try:
                score = float(idf)
                rmsd_val = float(rmsd)
            except ValueError:
                drop("idf or rmsd is not a number")
                continue

            q_hit = [q for q, _ in matched]
            t_hit = [t for _, t in matched]
            # n_matched is the count of placed nodes, which is what folddisco's node_count
            # reports; it is recomputed here rather than read off column 2 so the number
            # and the interval can never disagree.
            t_acc, t_off = accession_and_offset(tid)
            out.write(
                f"{args.query_accession}\t{t_acc}\t"
                f"{min(q_hit)}\t{max(q_hit)}\t"
                f"{min(t_hit) + t_off}\t{max(t_hit) + t_off}\t"
                f"{score}\t{rmsd_val}\t{len(matched)}\n"
            )
            written += 1

    print(f"{args.query_accession}: {written} rows")

    # A hit file with rows in it that converts to nothing is a parse failure, not a
    # no-hit result: folddisco was not run at all when the file is empty, because the
    # caller skips this script then. Saying so here is the difference between finding this
    # in the task log and finding it in a report weeks later as a bar of length zero.
    # Omitting -q used to land exactly here, silently, on every row of every chunk.
    if n_lines and not written:
        detail = "; ".join(f"{v} x {k}" for k, v in sorted(reasons.items()))
        hint = (
            " An empty query_residues column means `folddisco query` ran without -q, "
            "which leaves no query-side coordinates to report."
            if "query_residues column is empty" in reasons
            else " The two residue lists are one motif read from two structures, so they "
                 "are the same length by construction; a mismatch means the column order "
                 "has moved and --format-output should pin it."
        )
        raise SystemExit(
            f"{args.hits}: {n_lines} folddisco hit rows converted to 0 regions "
            f"({detail}).{hint}"
        )
    if reasons:
        detail = "; ".join(f"{v} x {k}" for k, v in sorted(reasons.items()))
        print(f"{args.query_accession}: dropped {detail}", file=sys.stderr)


if __name__ == "__main__":
    main()
