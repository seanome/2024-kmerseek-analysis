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

# <chunk>.<alphabet>.k<ksize>.lc<true|false>.queries.txt
NAME = re.compile(r"^(?P<chunk>[^.]+)\.(?P<alphabet>.+)\.k(?P<ksize>\d+)\.lc(?P<lc>true|false)\.queries\.txt$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True,
                    help="the proteome every arm searched; the placed set is this minus --dark")
    ap.add_argument("--species", required=True)
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
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

    # combo -> set of query accessions with any region, unioned across chunks
    combos: dict[tuple, set] = {}
    unparsed = []
    for f in args.queries:
        m = NAME.match(f.name)
        if not m:
            unparsed.append(f.name)
            continue
        key = (m["alphabet"], int(m["ksize"]), m["lc"] == "true")
        got = {ln.strip() for ln in f.read_text().splitlines() if ln.strip()}
        combos.setdefault(key, set()).update(got)
    if unparsed:
        raise SystemExit(f"could not parse combo from: {', '.join(unparsed[:5])}")

    rows = []
    for (alphabet, ksize, lc), hit in sorted(combos.items()):
        rescued = hit & dark
        seen_placed = hit & placed
        rows.append({
            "species": args.species, "alphabet": alphabet, "ksize": ksize,
            "low_complexity_mask": lc,
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
    for alphabet, ksize in sorted({(r["alphabet"], r["ksize"]) for r in rows}):
        on = next((r for r in rows if r["alphabet"] == alphabet and r["ksize"] == ksize
                   and r["low_complexity_mask"]), None)
        off = next((r for r in rows if r["alphabet"] == alphabet and r["ksize"] == ksize
                    and not r["low_complexity_mask"]), None)
        if on and off:
            paired.append({
                "alphabet": alphabet, "ksize": ksize,
                "dark_reached_mask_on": on["dark_reached"],
                "dark_reached_mask_off": off["dark_reached"],
                "lost_to_masking": off["dark_reached"] - on["dark_reached"],
            })

    summary = {"species": args.species, "dark_proteins": len(dark),
               "placed_proteins": len(placed),
               "by_combo": rows, "mask_pairs": paired}
    print(json.dumps(summary, indent=2))
    if paired:
        print("\nsurviving the low-complexity mask (the number that counts):")
        for q in paired:
            print(f"  {q['alphabet']} k{q['ksize']}: {q['dark_reached_mask_on']} of "
                  f"{len(dark)} dark proteins reached with the mask ON "
                  f"({q['lost_to_masking']} lost when it is applied)")
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
