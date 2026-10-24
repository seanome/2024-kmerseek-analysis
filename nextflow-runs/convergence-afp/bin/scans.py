#!/usr/bin/env python3
"""Rank every reference entry for every query with the methods that need no search.

Each ranked list orders all reference entries for one query, best first:

  composition_aa20         Euclidean distance between the 20 amino-acid frequencies of
                           query and target (frequencies sum to 1). Smaller is better.
  composition_<alphabet>   the same in one kmerseek alphabet's classes, one list per
                           alphabet in assets/alphabet_classes.tsv. The notebook reads the
                           one that matches the chosen kmerseek setting.
  period_hp_autocorr       cosine distance between the query's and the target's
                           autocorrelation of the two-letter HP string (hp_pbotc_1st_ed2,
                           hydrophobic +1, polar -1) at lags 1 to 30. A lag at or past a
                           sequence's length counts as 0. A target whose 30 values are all
                           0 gets distance 1.
  period_hp_moment         cosine distance between hydrophobic moment spectra: for each
                           angle from 80 to 180 degrees in 5-degree steps, |sum_n h_n
                           exp(i n angle)| / N, with h the Eisenberg consensus scale
                           (assets/eisenberg_scale.tsv, copied from Biopython).
  window_kd_match_w<W>     the hydropathy window scan. Each sequence becomes a profile:
                           the mean Kyte-Doolittle hydropathy (assets/kyte_doolittle_scale.tsv)
                           of every window of W residues. The shorter profile is slid
                           along the longer one, fully inside it, and the score is the
                           smallest mean squared difference over all offsets, negated so
                           that larger is better. A sequence shorter than W, which has no
                           window, is ranked last. One list per --window-lengths value.
                           One list per minimum overlap in OVERLAP_CUTOFFS, named
                           window_kd_match_w<W>_ov<cutoff>: a comparison that used fewer
                           windows than the cutoff is dropped before ranking. A short
                           target matches a long query's best stretch easily, since a
                           9-residue entry is one window at W=9 and reaches a mean squared
                           difference of 0, and the cutoff is how the scan is given its
                           best shot. The `overlap` column holds how many windows each
                           comparison used.
  window_kd_spread_w<W>    the same window scores without their order: cosine distance
                           between histograms of the window values over fixed bins from
                           -4.5 to 4.5 (the scale's range) in 0.5 steps.
  motif_regex              for a query whose lineage has a published repeat consensus
                           (assets/known_motifs.tsv): residues covered by matches of the
                           consensus regex, larger is better; entries with no match share
                           the last rank. The regex does not read the query, so null
                           draws get no row.

Ties share the best rank of the group (rank "min"). Output keeps, per list and query,
the top --top-n entries and every truth entry, with each one's rank, so recall at the
first false positive and the area under the precision-recall curve can be computed from
it. n_ranked is the length of the full list.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

HP_ALPHABET = "hp_pbotc_1st_ed2"
LAGS = np.arange(1, 31)
ANGLES_DEG = np.arange(80, 181, 5)
AA20 = "ACDEFGHIKLMNPQRSTVWY"
# Bin edges for window_kd_spread: the Kyte-Doolittle scale runs from -4.5 (arginine) to
# 4.5 (isoleucine), so a window mean cannot fall outside this range.
SPREAD_BINS = np.arange(-4.5, 4.51, 0.5)
# The match lists are emitted once per minimum overlap, because a short target matches a
# long query's best stretch over very few windows. The cutoff is applied BEFORE the top
# rows are kept, so each list is ranked among the comparisons it allows. Applied
# afterwards it emptied the list of everything but the truth entries, which are always
# kept, and left no false positive to compare against.
OVERLAP_CUTOFFS = [0, 10, 20, 50]
# Overlap-save block size for the profile match. One FFT per block, all blocks in one
# batched call.
FFT_BLOCK = 1 << 16


def read_fasta(path: Path) -> tuple[list[str], list[str]]:
    names, seqs, chunks = [], [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if chunks or names:
                    seqs.append("".join(chunks))
                names.append(line[1:].split()[0])
                chunks = []
            else:
                chunks.append(line.strip())
    seqs.append("".join(chunks))
    return names, seqs


class Concat:
    """All sequences as one uint8 array of residue codes, with protein boundaries."""

    def __init__(self, seqs: list[str]):
        self.lengths = np.array([len(s) for s in seqs], dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]])
        self.codes = np.frombuffer("".join(seqs).encode(), dtype=np.uint8)
        self.protein = np.repeat(np.arange(len(seqs)), self.lengths)
        self.pos = np.arange(len(self.codes)) - np.repeat(self.starts, self.lengths)

    def segment_sum(
        self, values: np.ndarray, which: np.ndarray | None = None
    ) -> np.ndarray:
        idx = self.protein if which is None else self.protein[which]
        return np.bincount(idx, weights=values, minlength=len(self.lengths))


def lookup(table: dict[str, float], default: float = 0.0) -> np.ndarray:
    lut = np.full(256, default)
    for aa, v in table.items():
        lut[ord(aa)] = v
    return lut


def composition(c: Concat, classes: dict[str, int], n_classes: int) -> np.ndarray:
    lut = np.full(256, -1, dtype=np.int64)
    for aa, k in classes.items():
        lut[ord(aa)] = k
    cls = lut[c.codes]
    keep = cls >= 0  # X, B, Z, U, O and other non-standard letters are not counted
    flat = np.bincount(
        c.protein[keep] * n_classes + cls[keep], minlength=len(c.lengths) * n_classes
    )
    m = flat.reshape(len(c.lengths), n_classes).astype(float)
    tot = m.sum(axis=1, keepdims=True)
    return np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)


def hp_autocorr(c: Concat, hydrophobic: set[str]) -> np.ndarray:
    x = np.where(
        np.isin(
            c.codes,
            np.frombuffer("".join(sorted(hydrophobic)).encode(), dtype=np.uint8),
        ),
        1.0,
        -1.0,
    )
    out = np.zeros((len(c.lengths), len(LAGS)))
    for j, lag in enumerate(LAGS):
        prod = x[:-lag] * x[lag:]
        same = c.protein[:-lag] == c.protein[lag:]
        s = np.bincount(
            c.protein[:-lag][same], weights=prod[same], minlength=len(c.lengths)
        )
        n = c.lengths - lag
        out[:, j] = np.divide(s, n, out=np.zeros_like(s), where=n > 0)
    return out


def hp_moment(c: Concat, scale: dict[str, float]) -> np.ndarray:
    h = lookup(scale)[c.codes]
    out = np.zeros((len(c.lengths), len(ANGLES_DEG)))
    for j, deg in enumerate(ANGLES_DEG):
        theta = np.deg2rad(deg) * c.pos
        re_ = c.segment_sum(h * np.cos(theta))
        im_ = c.segment_sum(h * np.sin(theta))
        out[:, j] = np.hypot(re_, im_) / np.maximum(c.lengths, 1)
    return out


def euclid(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.sqrt(((t - q) ** 2).sum(axis=1))


def cosine_distance(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    qn = np.linalg.norm(q)
    tn = np.linalg.norm(t, axis=1)
    d = np.ones(len(t))
    ok = (tn > 0) & (qn > 0)
    d[ok] = 1.0 - (t[ok] @ q) / (tn[ok] * qn)
    return d


def window_profiles(c, scale, window):
    """Mean hydropathy of every window of `window` residues, in concatenated coordinates.

    Position p holds the mean over residues p..p+window-1 when that whole window lies
    inside one protein, and 0 where it does not. `valid` marks the real windows.
    """
    h = lookup(scale)[c.codes]
    cs = np.concatenate([[0.0], np.cumsum(h)])
    prof = (cs[window:] - cs[:-window]) / window
    prof = np.concatenate([prof, np.zeros(window - 1)])
    valid = c.pos <= (c.lengths[c.protein] - window)
    prof = np.where(valid, prof, 0.0)
    return prof, valid


def profile_lengths(c, window):
    """Windows per protein: length - window + 1, or 0 for a protein shorter than window."""
    return np.maximum(c.lengths - window + 1, 0)


def block_fft(signal, kernel_len, block=FFT_BLOCK):
    """Overlap-save setup: the FFT of every block of `signal`, with its block size.

    Every kernel is zero-padded to `kernel_len`, so the block layout does not depend on
    the query and these FFTs are computed once and reused for every query. The block size
    travels with the FFTs, because a correlation done at a different one is wrong.
    """
    if block <= kernel_len:
        raise ValueError(
            f"block size {block} must exceed the kernel length {kernel_len}"
        )
    step = block - kernel_len + 1
    n_blocks = int(np.ceil(len(signal) / step))
    padded = np.zeros(step * n_blocks + kernel_len - 1)
    padded[: len(signal)] = signal
    blocks = np.lib.stride_tricks.sliding_window_view(padded, block)[::step]
    return {
        "fft": np.fft.rfft(blocks, axis=1),
        "block": block,
        "kernel_len": kernel_len,
    }


def correlate_saved(plan, kernel, n_out):
    """sum_j kernel[j] * signal[p+j] for every p, from precomputed block FFTs."""
    block = plan["block"]
    if len(kernel) > plan["kernel_len"]:
        raise ValueError(
            f"kernel of {len(kernel)} is longer than the {plan['kernel_len']} "
            "the block FFTs were built for"
        )
    k = np.zeros(block)
    k[: plan["kernel_len"]] = np.r_[kernel, np.zeros(plan["kernel_len"] - len(kernel))][
        ::-1
    ]
    out = np.fft.irfft(plan["fft"] * np.fft.rfft(k), n=block, axis=1)
    keep = out[:, plan["kernel_len"] - 1 :].reshape(-1)
    return keep[:n_out]


def best_profile_match(tc, qc, scale, window, plan, short_groups):
    """-min mean squared difference between each target's and the query's profile.

    The shorter profile is slid fully inside the longer one. Targets whose profile is at
    least as long as the query's go through the precomputed block FFTs; shorter ones are
    done by sliding the target inside the query, grouped by profile length so each group
    is one matrix product. A sequence with no window at all scores -inf.
    """
    t_prof, t_valid = window_profiles(tc, scale, window)
    q_prof, q_valid = window_profiles(qc, scale, window)
    q_len = profile_lengths(qc, window)
    t_len = profile_lengths(tc, window)
    n_targets = len(tc.lengths)
    out = np.full((len(qc.lengths), n_targets), -np.inf)
    overlap = np.zeros((len(qc.lengths), n_targets), dtype=np.int64)
    # One cumulative sum of the squared profile, for every query's window sums. Caching a
    # moving sum per query length instead would hold one array the size of the whole
    # reference for each of the ~250 distinct query lengths.
    cs2 = np.concatenate([[0.0], np.cumsum(t_prof**2)])
    for qi in range(len(qc.lengths)):
        lp = int(q_len[qi])
        if lp <= 0:
            continue
        q = q_prof[qc.starts[qi] : qc.starts[qi] + lp]
        q2 = float((q**2).sum())

        # Targets at least as long as the query profile: slide the query inside them.
        corr = correlate_saved(plan, q, len(t_prof))
        n_pos = len(t_prof) - lp + 1
        msd = np.full(len(t_prof), np.inf)
        if n_pos > 0:
            msd[:n_pos] = (cs2[lp:] - cs2[:n_pos] + q2 - 2.0 * corr[:n_pos]) / lp
        start_ok = t_valid
        end_ok = np.zeros(len(t_prof), dtype=bool)
        if lp <= len(t_prof):
            end_ok[: len(t_prof) - lp + 1] = t_valid[lp - 1 :]
            same = np.zeros(len(t_prof), dtype=bool)
            same[: len(t_prof) - lp + 1] = (
                tc.protein[lp - 1 :] == tc.protein[: len(t_prof) - lp + 1]
            )
        else:
            same = np.zeros(len(t_prof), dtype=bool)
        cand = np.flatnonzero(start_ok & end_ok & same)
        if len(cand):
            pid = tc.protein[cand]
            first = np.flatnonzero(np.r_[True, pid[1:] != pid[:-1]])
            hit = pid[first]
            out[qi, hit] = -np.minimum.reduceat(msd[cand], first)
            overlap[qi, hit] = lp

        # Targets shorter than the query profile: slide each inside the query.
        for lt, (idx, mat, sq) in short_groups.items():
            if lt >= lp:
                continue
            n_off = lp - lt + 1
            qwin = np.lib.stride_tricks.sliding_window_view(q, lt)[:n_off]
            d = sq[:, None] + (qwin**2).sum(axis=1)[None, :] - 2.0 * (mat @ qwin.T)
            out[qi, idx] = -(d.min(axis=1) / lt)
            overlap[qi, idx] = lt
    return out, overlap


def short_profile_groups(tc, scale, window, max_len):
    """Profiles of targets with fewer than max_len windows, grouped by window count."""
    t_prof, _ = window_profiles(tc, scale, window)
    t_len = profile_lengths(tc, window)
    groups = {}
    wanted = np.flatnonzero((t_len > 0) & (t_len < max_len))
    for lt in np.unique(t_len[wanted]):
        idx = wanted[t_len[wanted] == lt]
        mat = np.stack([t_prof[tc.starts[i] : tc.starts[i] + lt] for i in idx])
        groups[int(lt)] = (idx, mat, (mat**2).sum(axis=1))
    return groups


def window_spread(c, scale, window):
    """Histogram of each protein's window values over SPREAD_BINS."""
    prof, valid = window_profiles(c, scale, window)
    n_bins = len(SPREAD_BINS) - 1
    b = np.clip(np.digitize(prof, SPREAD_BINS) - 1, 0, n_bins - 1)
    keep = valid
    flat = np.bincount(
        c.protein[keep] * n_bins + b[keep], minlength=len(c.lengths) * n_bins
    )
    return flat.reshape(len(c.lengths), n_bins).astype(float)


def ranks_min(score: np.ndarray) -> np.ndarray:
    """Rank 1 = highest score; ties share the best rank of their group."""
    order = np.argsort(-score, kind="stable")
    s = score[order]
    r = np.empty(len(score), dtype=np.int64)
    first = np.r_[True, s[1:] != s[:-1]]
    group_rank = np.maximum.accumulate(np.where(first, np.arange(1, len(s) + 1), 0))
    r[order] = group_rank
    return r


def keep_rows(
    method, list_name, qid, score, names, truth_idx, top_n, n_total, overlap=None
):
    r = ranks_min(score)
    keep = np.unique(
        np.concatenate([np.argsort(-score, kind="stable")[:top_n], truth_idx])
    )
    return {
        "method": [method] * len(keep),
        "list_name": [list_name] * len(keep),
        "query_id": [qid] * len(keep),
        "target": [names[i] for i in keep],
        "score": score[keep].tolist(),
        "rank": r[keep].tolist(),
        "n_ranked": [n_total] * len(keep),
        "overlap": [None] * len(keep) if overlap is None else overlap[keep].tolist(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--draws", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--alphabet-classes", type=Path, required=True)
    ap.add_argument("--motifs", type=Path, required=True)
    ap.add_argument(
        "--eisenberg", type=Path, required=True, help="assets/eisenberg_scale.tsv"
    )
    ap.add_argument(
        "--kyte-doolittle",
        type=Path,
        required=True,
        help="assets/kyte_doolittle_scale.tsv",
    )
    ap.add_argument(
        "--window-lengths",
        default="9,19",
        help="window lengths for the hydropathy scan, comma separated",
    )
    ap.add_argument("--top-n", type=int, default=2000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    names, tseqs = read_fasta(args.reference)
    qnames, qseqs = read_fasta(args.queries)
    t, q = Concat(tseqs), Concat(qseqs)
    n_total = len(names)
    index = {n: i for i, n in enumerate(names)}
    truth_idx = np.array(
        sorted(index[a] for a in pl.read_csv(args.truth, separator="\t")["Entry"])
    )
    draws = pl.read_csv(args.draws, separator="\t")
    lineage_of = dict(zip(draws["query_id"], draws["lineage"]))
    kind_of = dict(zip(draws["query_id"], draws["kind"]))

    cls = pl.read_csv(args.alphabet_classes, separator="\t")
    hp = set(
        cls.filter((pl.col("alphabet") == HP_ALPHABET) & (pl.col("class_index") == 0))[
            "residue"
        ]
    )

    es = pl.read_csv(args.eisenberg, separator="\t", comment_prefix="#")
    eisenberg = dict(zip(es["residue"], es["hydrophobicity"]))

    kds = pl.read_csv(args.kyte_doolittle, separator="\t", comment_prefix="#")
    kyte_doolittle = dict(zip(kds["residue"], kds["hydropathy"]))

    feats = {"composition_aa20": ({aa: i for i, aa in enumerate(AA20)}, 20)}
    for a in sorted(cls["alphabet"].unique()):
        sub = cls.filter(pl.col("alphabet") == a)
        feats[f"composition_{a}"] = (
            dict(zip(sub["residue"], sub["class_index"])),
            int(sub["class_index"].max()) + 1,
        )

    chunks = []
    for list_name, (classes, k) in feats.items():
        tm, qm = composition(t, classes, k), composition(q, classes, k)
        for i, qid in enumerate(qnames):
            chunks.append(
                keep_rows(
                    "composition",
                    list_name,
                    qid,
                    -euclid(qm[i], tm),
                    names,
                    truth_idx,
                    args.top_n,
                    n_total,
                )
            )
        print(f"{list_name}: done", file=sys.stderr)
    for list_name, fn in [
        ("period_hp_autocorr", lambda c: hp_autocorr(c, hp)),
        ("period_hp_moment", lambda c: hp_moment(c, eisenberg)),
    ]:
        tm, qm = fn(t), fn(q)
        for i, qid in enumerate(qnames):
            chunks.append(
                keep_rows(
                    "period",
                    list_name,
                    qid,
                    -cosine_distance(qm[i], tm),
                    names,
                    truth_idx,
                    args.top_n,
                    n_total,
                )
            )
        print(f"{list_name}: done", file=sys.stderr)

    for window in [int(w) for w in args.window_lengths.split(",")]:
        q_len_max = int(profile_lengths(q, window).max())
        if q_len_max <= 0:
            print(f"window {window}: every query is shorter; skipped", file=sys.stderr)
            continue
        t_prof, _ = window_profiles(t, kyte_doolittle, window)
        plan = block_fft(t_prof, q_len_max)
        groups = short_profile_groups(t, kyte_doolittle, window, q_len_max)
        match, overlap = best_profile_match(t, q, kyte_doolittle, window, plan, groups)
        del plan, groups
        for cut in OVERLAP_CUTOFFS:
            for i, qid in enumerate(qnames):
                # A comparison below the cutoff is pushed to the bottom rather than
                # removed, so the kept rows are the best of what the cutoff allows.
                scored = np.where(overlap[i] >= max(cut, 1), match[i], -np.inf)
                chunks.append(
                    keep_rows(
                        "window",
                        f"window_kd_match_w{window}_ov{cut}",
                        qid,
                        scored,
                        names,
                        truth_idx,
                        args.top_n,
                        n_total,
                        overlap[i],
                    )
                )
        del match, overlap
        ts, qs = window_spread(t, kyte_doolittle, window), window_spread(
            q, kyte_doolittle, window
        )
        for i, qid in enumerate(qnames):
            chunks.append(
                keep_rows(
                    "window",
                    f"window_kd_spread_w{window}",
                    qid,
                    -cosine_distance(qs[i], ts),
                    names,
                    truth_idx,
                    args.top_n,
                    n_total,
                )
            )
        print(f"window {window}: done", file=sys.stderr)

    motifs = pl.read_csv(args.motifs, separator="\t")
    for m in motifs.iter_rows(named=True):
        rx = re.compile(m["regex"])
        cover = np.array(
            [sum(len(x.group(0)) for x in rx.finditer(s)) for s in tseqs], dtype=float
        )
        for qid in qnames:
            if lineage_of[qid] == m["lineage"] and kind_of[qid] == "real":
                chunks.append(
                    keep_rows(
                        "motif",
                        "motif_regex",
                        qid,
                        cover,
                        names,
                        truth_idx,
                        args.top_n,
                        n_total,
                    )
                )
        print(
            f"motif {m['lineage']}: {int((cover > 0).sum())} entries match",
            file=sys.stderr,
        )

    # overlap is only set by the window scan, so its column is all nulls for every other
    # list; without the override polars gives those frames a Null dtype and refuses the
    # concat.
    out = pl.concat(
        [pl.DataFrame(c, schema_overrides={"overlap": pl.Int64}) for c in chunks]
    )
    out.write_parquet(args.out, compression="zstd")
    print(
        out.group_by("list_name")
        .agg(pl.col("query_id").n_unique().alias("n_queries"), pl.len().alias("rows"))
        .sort("list_name")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
