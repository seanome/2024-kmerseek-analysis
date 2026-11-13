"""Chance match rate of every linked-pair key, measured on Swiss-Prot (notebook 294).

Encodes every Swiss-Prot protein in hp_thomas_dill2 and counts, for each setting of
notebooks/linked_pairs_utils.py, how often two keys drawn at random from Swiss-Prot are
equal (the sum over keys of p(key)^2, also called the collision probability). Its -log2 is
the measured number of bits a key carries; the naive prediction is one bit per HP letter
plus log2 of the number of gap values the key stores. Single words (one anchor alone and
notebook 232's 10-mer seed) are measured the same way.

From the rate it also gives the expected number of chance key matches between two unrelated
sequences of length L (the median domain length of notebook 232's pairs), counting every
pair of positions, not only one diagonal: n_keys(L)^2 * rate, where n_keys(L) counts only
the keys that fit inside L residues (a word of a letters fits L - a + 1 times; an anchor
pair with gap g fits L - 2a - g + 1 times; a randstrobe key needs room for its nearest
partner, L - 2a + 1 positions), times the share of positions with a valid first word.

With --cost-out it also writes one row per key design of notebook 294 section 4d
(linked_pairs_utils.DESIGNS: single words of 8-16 letters, two 6-8-letter words up to 32
apart with the gap stored exactly or in 7-residue bins, the chosen key and the randstrobe),
counted the same way.

Usage:
    python scripts/linked_pairs_key_spectrum.py \
        --fasta /Users/olga/data/uniprot/uniprot_sprot.fasta.gz \
        --pairs /Users/olga/data/pfam/294_linked_pairs_scores.parquet \
        --out /Users/olga/data/pfam/294_linked_pairs_key_spectrum_swissprot.parquet
    python scripts/linked_pairs_key_spectrum.py \
        --fasta /Users/olga/data/uniprot/uniprot_sprot.fasta.gz \
        --pairs /Users/olga/data/pfam/294_linked_pairs_scores.parquet \
        --cost-out /Users/olga/data/pfam/294_key_cost_by_design.parquet
"""

from __future__ import annotations

import argparse
import gzip
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))
import linked_pairs_utils as lp  # noqa: E402
from hp_conservation_utils import GAP, TABLES  # noqa: E402

SEPARATOR = b"*"  # not in the HP table, so no word crosses from one protein to the next
CHUNK = 2_000_000  # residues per randstrobe chunk (memory: CHUNK x (gap_max + 1) int64)


def read_swissprot(path: str) -> tuple[np.ndarray, int]:
    seqs, cur = [], []
    with gzip.open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b">"):
                if cur:
                    seqs.append(b"".join(cur))
                cur = []
            else:
                cur.append(line.strip())
    if cur:
        seqs.append(b"".join(cur))
    raw = SEPARATOR.join(seqs) + SEPARATOR
    cls = TABLES[lp.ALPHABET][np.frombuffer(raw, dtype=np.uint8)]
    return cls, len(seqs)


def collision(counts: np.ndarray) -> float:
    p = counts / counts.sum()
    return float((p * p).sum())


def offset_collisions(codes: np.ndarray, ok: np.ndarray, a: int, gmax: int):
    """C[g, g'] = Pr(a random key of two a-letter words at gap g equals a random key at
    gap g'), from the joint word distribution at each gap 0..gmax counted on Swiss-Prot.
    """
    P = np.zeros((gmax + 1, 1 << (2 * a)))
    n = codes.size
    for g in range(gmax + 1):
        j = a + g
        both = ok[: n - j] & ok[j:]
        joint = (codes[: n - j][both] << a) | codes[j:][both]
        cnt = np.bincount(joint, minlength=1 << (2 * a)).astype(float)
        P[g] = cnt / cnt.sum()
    return P @ P.T


def gap_mask(G: int, treatment: str) -> np.ndarray:
    """Which (gap, gap') pairs a key with this gap rule treats as equal."""
    gi = np.arange(G + 1)
    if treatment == "exact":
        return gi[:, None] == gi[None, :]
    if treatment.startswith("binned"):
        b = gi // int(treatment[len("binned") :])
        return b[:, None] == b[None, :]
    return np.ones((G + 1, G + 1), bool)


def randstrobe_key_counts(cls: np.ndarray, a: int, G: int):
    """Counts of every randstrobe key (one per first-anchor position) under each gap
    rule, and the number of keys."""
    nb = G // lp.GAP_BIN + 1
    cnt = {tr: None for tr in lp.TREATMENTS}
    n_keys = 0
    for start, ch in chunks_at_separators(cls):
        sel = lp.randstrobe_partner(ch, a, G)
        cc, _ = lp.word_codes(ch, a)
        i = np.flatnonzero(sel >= 0)
        if i.size == 0:
            continue
        n_keys += i.size
        pair = (cc[i] << a) | cc[sel[i]]
        g = sel[i] - i - a
        for tr, key, size in [
            ("exact", pair * (G + 1) + g, (1 << (2 * a)) * (G + 1)),
            ("binned3", pair * nb + g // lp.GAP_BIN, (1 << (2 * a)) * nb),
            ("none", pair, 1 << (2 * a)),
        ]:
            b = np.bincount(key, minlength=size)
            cnt[tr] = b if cnt[tr] is None else cnt[tr] + b
    return cnt, n_keys


def keys_that_fit(mode: str, a: int, G: int, L: float) -> float:
    """Key positions inside one sequence of L residues (before the valid-word share): a
    word of a letters fits L - a + 1 times; an anchor pair with gap g fits L - 2a - g + 1
    times; a randstrobe key needs room for its nearest partner, L - 2a + 1 positions."""
    if mode == "word":
        return max(0.0, L - a + 1)
    if mode == "all":
        return sum(max(0.0, L - 2 * a - g + 1) for g in range(G + 1))
    if mode == "rand":
        return max(0.0, L - 2 * a + 1)
    raise ValueError(mode)


def design_costs(cls: np.ndarray, n_res: int, L: float, t0: float) -> pl.DataFrame:
    """Chance key matches between two unrelated L-residue proteins for every design of
    section 4d (lp.DESIGNS), counted the same way as main() counts every setting:
    keys_per_sequence^2 * rate, with keys_per_sequence = valid-word share x keys_that_fit.
    """
    rate, valid = {}, {}
    G = lp.DESIGN_GAP_MAX
    for k in sorted({ln for _, ln, _, _ in lp.DESIGNS}):
        codes, ok = lp.word_codes(cls, k)
        valid[k] = ok.sum() / n_res
        for d in lp.DESIGNS:
            m, ln, g_, tr = d
            if ln != k:
                continue
            if m == "word":
                rate[d] = collision(np.bincount(codes[ok]))
        if any(m == "all" and ln == k for m, ln, _, _ in lp.DESIGNS):
            C = offset_collisions(codes, ok, k, G)
            for d in lp.DESIGNS:
                m, ln, g_, tr = d
                if m == "all" and ln == k:
                    rate[d] = float(C[gap_mask(G, tr)].sum() / (G + 1) ** 2)
            del C
        del codes, ok
        print(f"  design costs: length {k} done {time.time() - t0:.0f} s", flush=True)
    for d in lp.DESIGNS:
        m, ln, g_, tr = d
        if m == "rand":
            cnt, _ = randstrobe_key_counts(cls, ln, g_)
            rate[d] = collision(cnt[tr])
            print(
                f"  design costs: randstrobe done {time.time() - t0:.0f} s", flush=True
            )
    out = []
    for d in lp.DESIGNS:
        m, ln, g_, tr = d
        n_keys = valid[ln] * keys_that_fit(m, ln, g_, L)
        out.append(
            {
                "design": lp.design_name(*d),
                "mode": m,
                "length": ln,
                "gap_max": g_,
                "treatment": tr,
                "rate": rate[d],
                "measured_bits": float(-np.log2(rate[d])),
                "valid_word_share": float(valid[ln]),
                "keys_per_sequence": float(n_keys),
                "median_length": L,
                "chance_matches": float(n_keys**2 * rate[d]),
            }
        )
    return pl.DataFrame(out)


def chunks_at_separators(cls: np.ndarray):
    sep = np.flatnonzero(cls == GAP)
    start = 0
    while start < cls.size:
        stop = min(start + CHUNK, cls.size)
        if stop < cls.size:
            stop = int(sep[np.searchsorted(sep, stop)]) + 1
        yield start, cls[start:stop]
        start = stop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", help="every setting's key rate (sections 2-5)")
    ap.add_argument(
        "--cost-out",
        help="chance key matches for every section 4d design (lp.DESIGNS)",
    )
    args = ap.parse_args()
    if not (args.out or args.cost_out):
        ap.error("give --out, --cost-out or both")
    t0 = time.time()

    pairs = pl.read_parquet(args.pairs).filter(pl.col("kind") == "real")
    L = float(np.median(np.concatenate([pairs["q_len"], pairs["t_len"]])))
    cls, n_prot = read_swissprot(args.fasta)
    n_res = int((cls != GAP).sum())
    print(
        f"{n_prot:_} proteins, {n_res:_} HP residues, median pair length {L}",
        flush=True,
    )

    if args.cost_out:
        cost = design_costs(cls, n_res, L, t0)
        cost.write_parquet(args.cost_out)
        print(cost)
        if not args.out:
            return

    rows = []
    # cross-check against the stored Swiss-Prot spectrum (k=15 total k-mers in its header)
    c15, ok15 = lp.word_codes(cls, 15)
    print(f"valid HP 15-mers: {int(ok15.sum()):_}", flush=True)
    rows.append(
        {
            "mode": "check",
            "anchor": 15,
            "gap_max": 0,
            "treatment": "n_words",
            "rate": float(ok15.sum()),
            "predicted_bits": np.nan,
            "keys_per_residue": np.nan,
            "median_length": L,
        }
    )
    del c15, ok15

    valid = {}  # share of positions whose a-letter word is valid, per a
    for a in sorted(set(lp.ANCHOR_LENGTHS) | set(lp.BASELINE_SEEDS.values())):
        codes, ok = lp.word_codes(cls, a)
        valid[a] = ok.sum() / n_res
        rows.append(
            {
                "mode": "word",
                "anchor": a,
                "gap_max": 0,
                "treatment": "single",
                "rate": collision(np.bincount(codes[ok], minlength=1 << a)),
                "predicted_bits": float(a),
                "keys_per_residue": ok.sum() / n_res,
                "median_length": L,
            }
        )
        if a not in lp.ANCHOR_LENGTHS:
            continue
        gmax = max(lp.GAP_MAXES)
        # joint word distribution at each offset g (every-pair keys)
        C = offset_collisions(codes, ok, a, gmax)
        frac_first = ok.sum() / n_res
        for G in lp.GAP_MAXES:
            sub = C[: G + 1, : G + 1]
            gi = np.arange(G + 1)
            bins = gi // lp.GAP_BIN
            n_bins = bins.max() + 1
            for tr, mask, pred_extra in [
                ("exact", gi[:, None] == gi[None, :], np.log2(G + 1)),
                ("binned3", bins[:, None] == bins[None, :], np.log2(n_bins)),
                ("none", np.ones((G + 1, G + 1), bool), 0.0),
            ]:
                rows.append(
                    {
                        "mode": "all",
                        "anchor": a,
                        "gap_max": G,
                        "treatment": tr,
                        "rate": float(sub[mask].sum() / (G + 1) ** 2),
                        "predicted_bits": 2 * a + float(pred_extra),
                        "keys_per_residue": frac_first * (G + 1),
                        "median_length": L,
                    }
                )
        del C
        # randstrobe keys: one per first-anchor position
        for G in lp.GAP_MAXES:
            nb = G // lp.GAP_BIN + 1
            cnt, n_keys = randstrobe_key_counts(cls, a, G)
            for tr, pred_extra in [
                ("exact", np.log2(G + 1)),
                ("binned3", np.log2(nb)),
                ("none", 0.0),
            ]:
                rows.append(
                    {
                        "mode": "rand",
                        "anchor": a,
                        "gap_max": G,
                        "treatment": tr,
                        "rate": collision(cnt[tr]),
                        "predicted_bits": 2 * a + float(pred_extra),
                        "keys_per_residue": n_keys / n_res,
                        "median_length": L,
                    }
                )
            print(f"  a={a} G={G} done {time.time() - t0:.0f} s", flush=True)

    out = pl.DataFrame(rows)
    n_keys = []
    for r in out.iter_rows(named=True):
        a, G = r["anchor"], r["gap_max"]
        if r["mode"] == "check":
            n_keys.append(np.nan)
            continue
        n_keys.append(valid[a] * keys_that_fit(r["mode"], a, G, L))
    out = out.with_columns(pl.Series("keys_per_sequence", n_keys)).with_columns(
        (-pl.col("rate").log(2)).alias("measured_bits"),
        (pl.col("keys_per_sequence").pow(2) * pl.col("rate")).alias(
            "expected_chance_matches"
        ),
    )
    out.write_parquet(args.out)
    print(out.filter(pl.col("mode") != "check"))
    print(f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
