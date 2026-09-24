#!/usr/bin/env python3
"""The dark proteins with their residues shuffled: a null for kmerseek reach.

The shuffle preserves k-let counts, k=2 by default: every adjacent pair of residues occurs
as often in the shuffled copy as in the real one. A uniform shuffle would NOT be a fair
null here, and the difference decides the whole comparison.

Why: a reduced-alphabet index reaches by long runs of one class. Under an independent
shuffle the chance of the next residue sharing a class is just its composition, so runs
collapse to what independent draws give. Under a dipeptide-preserving shuffle the
class-to-class transition rate is whatever the real protein had -- around 6 in 10 for a
two-letter alphabet -- and long runs survive at their real rate. A uniform shuffle
therefore scores far below the real proteins for a reason that has nothing to do with
homology, and the report would read that gap as signal.

k=2 preserves adjacent pairs (Altschul-Erickson 1985, the Euler-path shuffle). k=3
preserves triples and is a stricter null. Stricter is not automatically better: on a
20-letter alphabet most triples in a single protein are unique, the graph has almost no
choice left, and the shuffle returns something close to the original -- a null that cannot
lose. On a two-letter alphabet there are only eight triples and k=3 is safe. Pick it for
what the ARM encodes to, not for the amino-acid sequence it starts from.

Accessions are kept verbatim so the reach can be counted against the same dark set. The
output is split into chunks of the same size the real proteome was, so each search is one
job of the same shape.
"""
import argparse
import random
from pathlib import Path

import polars as pl


def read_fasta(path: Path):
    acc, seq = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if acc is not None:
                    yield acc, "".join(seq)
                acc, seq = line[1:].strip().split()[0], []
            else:
                seq.append(line.strip())
    if acc is not None:
        yield acc, "".join(seq)


def klet_shuffle(seq: str, k: int, rng: random.Random, tries: int = 100) -> str:
    """Shuffle preserving every k-let count exactly (Altschul-Erickson).

    The sequence is an Eulerian path through a graph whose vertices are (k-1)-lets and
    whose edges are the k-lets. Any other Eulerian path over the same edge multiset is a
    sequence with identical k-let counts, so the job is to walk a random one.

    A random permutation of each vertex's out-edges does NOT generally give an Eulerian
    path -- the walk can strand edges. The fix is the one Altschul and Erickson gave: pick
    one out-edge per vertex to go LAST, check those chosen edges form a tree rooted at the
    final vertex (every vertex reaches the root by following them), shuffle the rest
    freely, and append the last edge. A last-edge set that is not such a tree is redrawn.
    """
    if k < 2 or len(seq) <= k:
        return seq
    verts = [seq[i:i + k - 1] for i in range(len(seq) - k + 2)]
    root = verts[-1]
    out: dict[str, list[str]] = {}
    for a, b in zip(verts, verts[1:]):
        out.setdefault(a, []).append(b)
    out.setdefault(root, [])

    for _ in range(tries):
        last: dict[str, str] = {}
        for v, es in out.items():
            if v != root and es:
                last[v] = rng.choice(es)
        # Every vertex must reach the root by following its chosen last edge, or some
        # edges are unreachable and the walk strands.
        ok = True
        for v in out:
            if v == root:
                continue
            seen, cur = set(), v
            while cur != root:
                if cur in seen or cur not in last:
                    ok = False
                    break
                seen.add(cur)
                cur = last[cur]
            if not ok:
                break
        if not ok:
            continue

        order: dict[str, list[str]] = {}
        for v, es in out.items():
            rest = list(es)
            if v in last:
                rest.remove(last[v])
            rng.shuffle(rest)
            order[v] = rest + ([last[v]] if v in last else [])

        walk = [verts[0]]
        cur = verts[0]
        for _ in range(len(verts) - 1):
            if not order.get(cur):
                break
            cur = order[cur].pop(0)
            walk.append(cur)
        if len(walk) == len(verts):
            return walk[0] + "".join(v[-1] for v in walk[1:])
    # Every redraw failed. Returning the original is the safe answer: it keeps the k-let
    # counts exactly right, and an unshuffled control is visible in the numbers, where a
    # silently uniform one would not be.
    return seq

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--query", type=Path, required=True,
                    help="the proteome every arm searched, bare-accession headers")
    ap.add_argument("--dark", type=Path, required=True, help="<species>_dark_set.parquet")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--chunk-size", type=int, required=True)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--also-placed", type=int, default=0, metavar="N",
                    help="also shuffle N randomly chosen PLACED proteins. They are the "
                         "positive control: a protein the sequence arms did place should "
                         "beat its own shuffle, and if it does not, the arm is reading "
                         "composition and the dark number means nothing. 0 (default) "
                         "shuffles the dark set only, and the control cannot be computed.")
    ap.add_argument("--preserve-klet", type=int, default=2,
                    help="preserve counts of every k-let (default 2, adjacent pairs). "
                         "3 is a stricter null; see the note at the top on when it is "
                         "too strict to lose.")
    args = ap.parse_args()

    dark = set(pl.read_parquet(args.dark)["accession"].to_list())
    rng = random.Random(args.seed)

    # Which accessions to shuffle: the dark set, plus a sample of the placed proteins when
    # the positive control is wanted. Sampled rather than exhaustive because the placed set
    # is the bigger half and the control only needs enough proteins to have an opinion.
    wanted = set(dark)
    if args.also_placed:
        placed = [acc for acc, _ in read_fasta(args.query) if acc not in dark]
        n = min(args.also_placed, len(placed))
        wanted |= set(rng.sample(placed, n))
        print(f"including {n} placed proteins as the positive control")
    args.outdir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    chunk_i = 0
    fh = None
    for acc, seq in read_fasta(args.query):
        if acc not in wanted:
            continue
        if n_written % args.chunk_size == 0:
            if fh:
                fh.close()
            fh = open(args.outdir / f"chunk_{chunk_i:04d}.fasta", "w")
            chunk_i += 1
        fh.write(f">{acc}\n{klet_shuffle(seq, args.preserve_klet, rng)}\n")
        n_written += 1
    if fh:
        fh.close()
    missing = len(wanted) - n_written
    if missing:
        raise SystemExit(f"{missing} accessions are not in {args.query}: the dark set "
                         f"and the proteome disagree on the key.")
    print(f"shuffled {n_written} proteins into {chunk_i} chunk(s) of "
          f"{args.chunk_size} under {args.outdir}, preserving {args.preserve_klet}-let counts")


if __name__ == "__main__":
    main()
