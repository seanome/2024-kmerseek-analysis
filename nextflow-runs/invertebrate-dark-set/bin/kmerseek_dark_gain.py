#!/usr/bin/env python3
"""What does kmerseek reach that no sequence arm could?

The only number this pipeline exists for. The dark set is the proteins phmmer, jackhmmer and
mmseqs2 all failed to place; this counts how many of those kmerseek puts a region on, split
by alphabet, ksize, and -- the part that decides whether any of it is real -- whether the
low-complexity mask was on.

Read the mask pair, never one side alone. BHF's flagship matches included polar-biased
low-complexity segments, so a rescue present with the mask off and absent with it on is an
artifact of composition, not a homology signal. This reports both and the difference; a
large mask-off-only number is a negative result wearing a positive one's clothes.

This is REACH, not accuracy. A region on a dark protein says kmerseek found something
there, not that the family label is right -- that needs the structural key. A number here
is necessary for the claim and nowhere near sufficient.

The PLACED set is counted next to the dark one, for the same combos, because reach alone
saturates. On Botryllus, hp_thomas_dill2 k23 put a region on 20_448 of 20_448 dark
proteins -- and on all 45_339 proteins of the proteome, which is not a rescue, it is an arm
that reports something for everything. An arm that reaches 95% of the placed proteins and
10% of the dark ones is discriminating; one that reaches 100% of both is saturated, and
its dark number means nothing. The two columns side by side are what tell those apart,
and across a ksize sweep the k where the placed fraction starts to fall is the k where
the dark fraction starts to mean something.

kmerseek 0.4 adds two more ways to make "reached" mean something. An `extend` arm grows
each exact run through mismatches and gives the region a Karlin-Altschul E-value, so a
dark protein can be counted as reached only when some region on it has E <= a cutoff
(--evalue-max, one row per cutoff, `evalue_max` set). And --scaled N keeps 1/N of the
k-mers, which is the same question asked with a smaller index. Each (alphabet, ksize,
mask, scaled, extension, cutoff) is its own row and its own `arm` label.
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl


def query_accessions(fasta: Path) -> set[str]:
    """First whitespace token of every header, verbatim -- the key every arm reports."""
    accs = set()
    with open(fasta) as fh:
        for line in fh:
            if line.startswith(">"):
                accs.add(line[1:].strip().split()[0])
    return accs

# <chunk>.<alphabet>.k<ksize>.s<scaled>.lc<true|false>.<exact|extend-c<C>>.queries.tsv
# The two-column TSV holds every query with any region and the smallest region_evalue
# among its regions ("inf" on an exact arm).
NAME = re.compile(r"^(?P<chunk>[^.]+)\.(?P<alphabet>.+)\.k(?P<ksize>\d+)\.s(?P<scaled>\d+)"
                  r"\.lc(?P<lc>true|false)\.(?P<ext>exact|extend-c[0-9.]+)\.queries\.tsv$")


def arm_label(alphabet: str, scaled: int, ext: str, evalue_max) -> str:
    """One string a reader can tell arms apart by: alphabet, then only the settings that
    differ from the plain exact search at scaled 1."""
    parts = [alphabet]
    if scaled != 1:
        parts.append(f"scaled {scaled}")
    if ext != "exact":
        # extend-c1.63 -> "extend C=1.63": the mismatch penalty is part of the arm.
        parts.append(f"extend C={ext[len('extend-c'):]}")
    if evalue_max is not None:
        parts.append(f"E<={evalue_max:g}")
    return " ".join(parts)


def read_queries(f: Path) -> dict[str, float] | None:
    """query accession -> smallest region E-value, from one search chunk's TSV. None when
    the file starts with a #nofit line: the search was refused because the index holds no
    Karlin-Altschul fit for that arm's penalty, so the arm was not searched at all."""
    out: dict[str, float] = {}
    with open(f) as fh:
        header = fh.readline().rstrip("\n").split("\t")
        if header and header[0] == "#nofit":
            return None
        if header[:2] != ["query_name", "min_region_evalue"]:
            raise SystemExit(f"{f.name}: expected a query_name/min_region_evalue header, got {header}")
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            q, e = line.split("\t")
            ev = float(e)
            if q not in out or ev < out[q]:
                out[q] = ev
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True,
                    help="the proteome every arm searched; the placed set is this minus --dark")
    ap.add_argument("--species", required=True)
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--evalue-max", default="",
                    help="comma list of region_evalue cutoffs; an extend arm is counted once "
                         "per cutoff on top of the any-region count")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    dark = set(pl.read_parquet(args.dark)["accession"].to_list())
    proteome = query_accessions(args.query)
    stray = dark - proteome
    if stray:
        raise SystemExit(
            f"{len(stray)} dark accessions are not in {args.query} (e.g. "
            f"{sorted(stray)[:3]}): the dark set and the proteome disagree on the key.")
    placed = proteome - dark
    if not dark:
        raise SystemExit(
            f"the dark set for '{args.species}' is empty, so there is nothing for kmerseek "
            f"to be measured on. Either every protein was placed by a sequence arm, or "
            f"computeDarkSet ran against a broken reference."
        )

    cutoffs = [float(x) for x in args.evalue_max.split(",") if x.strip()]

    # combo -> {query accession: smallest E-value}, unioned across chunks
    combos: dict[tuple, dict[str, float]] = {}
    unfitted: set[tuple] = set()
    unparsed = []
    for f in args.queries:
        m = NAME.match(f.name)
        if not m:
            unparsed.append(f.name)
            continue
        key = (m["alphabet"], int(m["ksize"]), m["lc"] == "true", int(m["scaled"]), m["ext"])
        got = read_queries(f)
        if got is None:
            unfitted.add(key)
            continue
        acc = combos.setdefault(key, {})
        for q, ev in got.items():
            if q not in acc or ev < acc[q]:
                acc[q] = ev
    if unparsed:
        raise SystemExit(f"could not parse combo from: {', '.join(unparsed[:5])}")
    # A fit is a property of the index, so every chunk of an unfitted arm is refused
    # alike; a mix would mean two chunks saw two different indexes.
    mixed = unfitted & set(combos)
    if mixed:
        raise SystemExit(f"arms both refused and searched across chunks: {sorted(mixed)[:3]}")
    unfitted_arms = [
        {"alphabet": a, "ksize": k, "low_complexity_mask": lc, "scaled": sc, "extension": ext,
         "arm": arm_label(a, sc, ext, None),
         "reason": "no Karlin-Altschul fit stored in the index for this penalty (too few score bins at index time)"}
        for a, k, lc, sc, ext in sorted(unfitted)
    ]

    rows = []
    for (alphabet, ksize, lc, scaled, ext), evalues in sorted(combos.items()):
        # The any-region count, then one row per E-value cutoff for an arm that has them.
        views = [(None, set(evalues))]
        if ext != "exact":
            views += [(c, {q for q, ev in evalues.items() if ev <= c}) for c in cutoffs]
        for evalue_max, hit in views:
            rescued = hit & dark
            seen_placed = hit & placed
            rows.append({
                "species": args.species, "alphabet": alphabet, "ksize": ksize,
                "low_complexity_mask": lc,
                "scaled": scaled, "extension": ext, "evalue_max": evalue_max,
                "arm": arm_label(alphabet, scaled, ext, evalue_max),
                "dark_proteins": len(dark),
                "dark_reached": len(rescued),
                "fraction_dark_reached": round(len(rescued) / len(dark), 4),
                "placed_proteins": len(placed),
                "placed_reached": len(seen_placed),
                "fraction_placed_reached": (round(len(seen_placed) / len(placed), 4)
                                            if placed else None),
                "queries_with_any_region": len(hit),
            })
    df = pl.DataFrame(rows)
    df.write_parquet(args.out, compression="zstd")

    # The paired difference is the headline, not either column on its own.
    paired = []
    for arm, ksize in sorted({(r["arm"], r["ksize"]) for r in rows}):
        on = next((r for r in rows if r["arm"] == arm and r["ksize"] == ksize
                   and r["low_complexity_mask"]), None)
        off = next((r for r in rows if r["arm"] == arm and r["ksize"] == ksize
                    and not r["low_complexity_mask"]), None)
        if on and off:
            paired.append({
                "alphabet": on["alphabet"], "arm": arm, "ksize": ksize,
                "dark_reached_mask_on": on["dark_reached"],
                "dark_reached_mask_off": off["dark_reached"],
                "lost_to_masking": off["dark_reached"] - on["dark_reached"],
            })

    summary = {"species": args.species, "dark_proteins": len(dark),
               "placed_proteins": len(placed),
               "by_combo": rows, "mask_pairs": paired, "unfitted_arms": unfitted_arms}
    print(json.dumps(summary, indent=2))
    if unfitted_arms:
        print(f"\n{len(unfitted_arms)} arm(s) not searched, no Karlin-Altschul fit in the index:")
        for u in unfitted_arms:
            print(f"  {u['arm']} k{u['ksize']}")
    if paired:
        print("\nsurviving the low-complexity mask (the number that counts):")
        for q in paired:
            print(f"  {q['arm']} k{q['ksize']}: {q['dark_reached_mask_on']} of "
                  f"{len(dark)} dark proteins reached with the mask ON "
                  f"({q['lost_to_masking']} lost when it is applied)")
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
