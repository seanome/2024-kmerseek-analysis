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
"""

import argparse
import json
import re
from pathlib import Path

import polars as pl

# <chunk>.<alphabet>.k<ksize>.lc<true|false>.queries.txt
NAME = re.compile(r"^(?P<chunk>[^.]+)\.(?P<alphabet>.+)\.k(?P<ksize>\d+)\.lc(?P<lc>true|false)\.queries\.txt$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dark", type=Path, required=True)
    ap.add_argument("--species", required=True)
    ap.add_argument("--queries", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    dark = set(pl.read_parquet(args.dark)["accession"].to_list())
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
        rows.append({
            "species": args.species, "alphabet": alphabet, "ksize": ksize,
            "low_complexity_mask": lc,
            "dark_proteins": len(dark),
            "dark_reached": len(rescued),
            "fraction_dark_reached": round(len(rescued) / len(dark), 4),
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
