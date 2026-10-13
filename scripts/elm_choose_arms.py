#!/usr/bin/env python3
"""Choose one kmerseek arm per alphabet from the chicken ELM cover scores.

For each alphabet, the arm that put the most chicken motifs on position on the ortholog
(n_ortholog_on_position in reduce_elm_cover.py's summaries). Ties go to the larger scaled,
because it is the cheaper search, then to the mask off, then to the arm name, so the choice
is the same on every run. Only arms whose scoring finished (status "ok") are eligible.

Chicken is the set these arms are chosen on. The other target species are where the chosen
arms are measured, so a chicken number for a chosen arm is optimistic and a number from any
other species is not.

Writes assets/elm_cover_chosen_arms.tsv, which `make run-elm-search-all` turns into pinned
--kmerseek_combos entries (alphabet:k:s<scaled>:lc<true|false>).

    python scripts/elm_choose_arms.py --cover /Users/olga/data/elm-motif-transfer/cover/chicken
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import polars as pl

ARM = re.compile(r"^(?P<alphabet>.+)\.k(?P<k>\d+)(?:\.s(?P<scaled>\d+))?\.lc(?P<mask>true|false)$")
OUT = Path(__file__).resolve().parents[1] / "nextflow-runs" / "qfo-pfam-region-benchmark" / "assets" / "elm_cover_chosen_arms.tsv"


def kmerseek_arms(cover: Path) -> pl.DataFrame:
    rows = []
    for f in sorted(cover.glob("*.summary.json")):
        s = json.loads(f.read_text())
        m = ARM.match(s["arm"])
        if m is None or s.get("status") != "ok":
            continue
        rows.append({"alphabet": m["alphabet"], "k": int(m["k"]), "scaled": int(m["scaled"] or 1),
                     "mask": m["mask"] == "true", "arm": s["arm"],
                     "on_position": s["n_ortholog_on_position"], "of": s["n_with_projection"]})
    return pl.DataFrame(rows)


def choose(arms: pl.DataFrame) -> pl.DataFrame:
    return (arms.sort(["alphabet", "on_position", "scaled", "mask", "arm"],
                      descending=[False, True, True, False, False])
            .group_by("alphabet", maintain_order=True).first()
            .sort("on_position", "alphabet", descending=[True, False]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cover", type=Path, required=True, help="reduce_elm_cover.py's output for chicken")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    arms = kmerseek_arms(args.cover)
    chosen = choose(arms)
    chosen.select("alphabet", "k", "scaled", "mask", "arm", "on_position", "of").write_csv(args.out, separator="\t")
    print(f"{arms.height} scored kmerseek arms on {arms['alphabet'].n_unique()} alphabets; chose {chosen.height}")
    print(chosen)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
