#!/usr/bin/env python3
"""Does a decoy keep a transmembrane protein's hydrophobic stretch? Window vs dipeptide shuffle.

For 2_000 random non-mammal Swiss-Prot proteins with a TRANSMEM feature, finds the most
hydrophobic 19-residue stretch (share of AFILMVWC) in the real sequence, in its
10-residue-window shuffle (make_window_decoys.py) and in its dipeptide shuffle (Altschul
and Erickson 1985; the function from scripts/make_decoy_queries.py on
olgabot/midi-plus-0.4-decoy, copied below unchanged). A decoy that loses the stretch
cannot match a transmembrane query by composition, so a shuffle that loses it in whole
proteins but not in short regions-index entries gives the two indexes unequal decoys.

    python3 tools/compare_decoy_shuffles.py --results <build outdir>
"""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "bin"))
from make_window_decoys import window_shuffle  # noqa: E402

HYDROPHOBIC = set("AFILMVWC")
STRETCH = 19


def dipeptide_shuffle(seq: str, rng: random.Random) -> str:
    """Copied unchanged from scripts/make_decoy_queries.py (olgabot/midi-plus-0.4-decoy)."""
    if len(seq) < 3:
        return seq
    out = defaultdict(list)
    for a, b in zip(seq, seq[1:]):
        out[a].append(b)
    last = seq[-1]
    while True:
        final = {v: rng.choice(ends) for v, ends in out.items() if v != last}
        ok = True
        for v in final:
            seen, u = set(), v
            while u != last:
                if u in seen:
                    ok = False
                    break
                seen.add(u)
                u = final[u]
            if not ok:
                break
        if ok:
            break
    order = {}
    for v, ends in out.items():
        rest = list(ends)
        if v in final:
            rest.remove(final[v])
        rng.shuffle(rest)
        order[v] = rest + ([final[v]] if v in final else [])
    walk, u = [seq[0]], seq[0]
    for _ in range(len(seq) - 1):
        u = order[u].pop(0)
        walk.append(u)
    return "".join(walk)


def best_stretch(seq: str) -> float:
    a = np.array([c in HYDROPHOBIC for c in seq], float)
    if len(a) < STRETCH:
        return float(a.mean())
    return float((np.convolve(a, np.ones(STRETCH), "valid") / STRETCH).max())


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    feats = pl.read_parquet(args.results / "swissprot/sprot_features.parquet").filter(
        (pl.col("feature_type") == "TRANSMEM") & ~pl.col("is_mammal")
    )
    seqs = pl.read_parquet(args.results / "swissprot/sprot_sequences.parquet")
    seq_of = dict(seqs.select("accession", "sequence").iter_rows())
    rng = random.Random(args.seed)
    accs = sorted(set(feats["accession"]))
    rng.shuffle(accs)
    res = {"real": [], "window10": [], "dipeptide": []}
    for acc in accs[: args.n]:
        s = seq_of[acc]
        res["real"].append(best_stretch(s))
        res["window10"].append(best_stretch(window_shuffle(s, rng, 10)))
        res["dipeptide"].append(best_stretch(dipeptide_shuffle(s, rng)))
    print(
        f"proteins with a TRANSMEM: {min(args.n, len(accs))}; most hydrophobic "
        f"{STRETCH}-residue stretch (share of AFILMVWC)"
    )
    for k, v in res.items():
        v = np.array(v)
        print(
            f"  {k:10s} median {np.median(v):.2f}   share >= 0.8: {np.mean(v >= 0.8):.2f}"
        )


if __name__ == "__main__":
    main()
