"""Per-column class agreement and exact-run statistics for aligned homolog pairs.

Given a pair of gapped alignment strings, every alphabet here is applied to both
sequences and three numbers are read off the columns where both have a residue:

* ``agree``: fraction of columns where the two residues fall in the same class.
* ``expected``: the same fraction under a composition-preserving shuffle of one
  partner, computed analytically as sum_c f_q(c) f_t(c).
* ``kappa``: (agree - expected) / (1 - expected), so alphabets with different class
  counts (and hence different chance agreement) sit on one scale.

And one number read off the whole alignment:

* ``longest_run``: the longest stretch of consecutive columns with no gap in either
  sequence and the same class in both. This is the longest k-mer the pair shares
  exactly in that alphabet, which is what an exact-match k-mer search can see.
  ``longest_run_null`` is the same after shuffling the target's residues.

And three numbers from the same shuffles, measured rather than computed from shares:

* ``agree_null``: fraction of the same aligned columns with the same class after the
  target's residues are shuffled, mean over shuffles.
* ``evidence``: bits of evidence per aligned column that the pair is related, the
  Kullback-Leibler divergence between agree/disagree at rate ``agree`` and at rate
  ``expected`` (see ``bernoulli_kl_bits``).
* ``evidence_null``: the same for each shuffle, against that shuffle's own ``expected``,
  mean over shuffles. It is above zero because a finite alignment never hits its chance
  rate exactly.

Alphabet tables are copied from kmerseek's ``src/rust/alphabets.rs`` (dayhoff6 from
sourmash), so the classes match what the search engine indexes. hsdm17, wass14, mmseqs12
and funcgroups8 were added 2026-10-02 from kmerseek v0.4.0, as cached in
``tables/274_alphabet_classes_kmerseek_v0.4.0.csv``, so all 19 kmerseek alphabets are here.
"""

from __future__ import annotations

import textwrap

import numpy as np
import polars as pl

# ---------------------------------------------------------------------------
# Alphabets, as residue clusters. Order within a list is arbitrary.
# ---------------------------------------------------------------------------
ALPHABET_CLUSTERS: dict[str, list[str]] = {
    "protein20": list("ACDEFGHIKLMNPQRSTVWY"),
    "hsdm17": list("ADRNTSQYFMCWHGP") + ["KE", "LIV"],
    "wass14": [
        "WM",
        "DI",
        "P",
        "C",
        "AV",
        "K",
        "T",
        "RE",
        "G",
        "L",
        "Y",
        "SH",
        "F",
        "NQ",
    ],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    "uniprot18": [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "EP",
        "G",
        "HL",
        "I",
        "K",
        "M",
        "F",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ],
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    "gbmr7": ["DN", "AEFIKLMQRVWY", "CH", "T", "S", "G", "P"],
    "dayhoff6": ["C", "AGPST", "DENQ", "FWY", "HKR", "ILMV"],
    "wwmj5": ["CMFILVWY", "ATH", "GP", "DE", "SNQRK"],
    "gbmr4": ["ADKERNTSQ", "YFLIVMCWH", "G", "P"],
    "polarity4": ["GAVLIFWMP", "STCYNQ", "DE", "HKR"],
    "hp_lehninger_hpc3": ["AFGILMPVWY", "DEHKNQRST", "C"],
    "hp_lehninger2": ["AFGILMPVWY", "CDEHKNQRST"],
    "hp_thomas_dill2": ["ACFILMVWY", "DEGHKNPQRST"],
    "hp_kyte_doolittle2": ["ACFILMV", "DEGHKNPQRSTWY"],
    "hp_thomas_dill_no_c2": ["AFILMVWY", "CDEGHKNPQRST"],
    "hp_lehninger_c_nonpolar2": ["ACFGILMPVWY", "DEHKNQRST"],
    "hp_pbotc_1st_ed2": ["ACFILMPVWY", "DEGHKNQRST"],
}

#: The two-class alphabets, which is what the HP hypothesis is about.
HP2 = [a for a in ALPHABET_CLUSTERS if a.startswith("hp_") and a.endswith("2")]

#: A short panel for figures where all 15 would be unreadable.
PANEL = [
    "protein20",
    "sdm12",
    "dayhoff6",
    "polarity4",
    "gbmr4",
    "hp_pbotc_1st_ed2",
    "hp_thomas_dill2",
    "hp_lehninger2",
]

GAP = 255


def _table(clusters: list[str]) -> np.ndarray:
    t = np.full(256, GAP, dtype=np.uint8)
    for i, cl in enumerate(clusters):
        for r in cl:
            t[ord(r)] = i
            t[ord(r.lower())] = i
    return t


TABLES: dict[str, np.ndarray] = {
    name: _table(cl) for name, cl in ALPHABET_CLUSTERS.items()
}
SIZES: dict[str, int] = {name: len(cl) for name, cl in ALPHABET_CLUSTERS.items()}
BITS: dict[str, float] = {name: float(np.log2(n)) for name, n in SIZES.items()}


def longest_true_run(mask: np.ndarray) -> int:
    if mask.size == 0 or not mask.any():
        return 0
    padded = np.concatenate(([False], mask, [False]))
    d = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    return int((ends - starts).max())


def bernoulli_kl_bits(agree, chance):
    """Bits of evidence per aligned column: agree*log2(agree/chance) + (1-agree)*log2((1-agree)/(1-chance)).

    `agree` is the measured share of columns with the same class, `chance` the share
    expected from the two sequences' class compositions. Works on scalars and arrays.
    Zero when agree == chance; nan when chance is 0 or 1.
    """
    a = np.asarray(agree, dtype=float)
    c = np.asarray(chance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(a > 0, a * np.log2(a / c), 0.0)
        t0 = np.where(a < 1, (1 - a) * np.log2((1 - a) / (1 - c)), 0.0)
        out = np.where((c > 0) & (c < 1), t1 + t0, np.nan)
    return out if out.ndim else float(out)


def pair_stats(
    qaln: str,
    taln: str,
    alphabets: list[str] | None = None,
    n_shuffle: int = 2,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """One row per alphabet for a single aligned pair."""
    alphabets = alphabets or list(ALPHABET_CLUSTERS)
    rng = rng or np.random.default_rng(0)
    q = np.frombuffer(qaln.encode(), dtype=np.uint8)
    t = np.frombuffer(taln.encode(), dtype=np.uint8)
    prot = TABLES["protein20"]
    qc, tc = prot[q], prot[t]
    both = (qc != GAP) & (tc != GAP)
    n = int(both.sum())
    if n == 0:
        return []
    # Shuffle the target's residues among its own residue positions (composition kept,
    # gap pattern kept), once per replicate, shared across alphabets.
    t_res_idx = np.flatnonzero(tc != GAP)
    shuffles = [t[rng.permutation(t_res_idx)] for _ in range(n_shuffle)]

    rows = []
    for name in alphabets:
        tab = TABLES[name]
        qa, ta = tab[q], tab[t]
        same = (qa == ta) & both
        agree = same.sum() / n
        k = SIZES[name]
        fq = np.bincount(qa[both], minlength=k)[:k] / n
        ft = np.bincount(ta[both], minlength=k)[:k] / n
        expected = float((fq * ft).sum())
        kappa = (agree - expected) / (1 - expected) if expected < 1 else np.nan
        run = longest_true_run(same)
        null_runs, null_agree, null_evidence = [], [], []
        for ts in shuffles:
            ta_s = ta.copy()
            ta_s[t_res_idx] = tab[ts]
            same_s = (qa == ta_s) & both
            null_runs.append(longest_true_run(same_s))
            agree_s = same_s.sum() / n
            ft_s = np.bincount(ta_s[both], minlength=k)[:k] / n
            null_agree.append(agree_s)
            null_evidence.append(bernoulli_kl_bits(agree_s, float((fq * ft_s).sum())))
        rows.append(
            {
                "alphabet": name,
                "n_cols": n,
                "agree": float(agree),
                "expected": expected,
                "kappa": float(kappa),
                "longest_run": run,
                "longest_run_null": float(np.mean(null_runs)),
                "agree_null": float(np.mean(null_agree)),
                "evidence": bernoulli_kl_bits(agree, expected),
                "evidence_null": float(np.mean(null_evidence)),
            }
        )
    return rows


def compute_pair_table(
    df: pl.DataFrame,
    key_cols: list[str],
    alphabets: list[str] | None = None,
    n_shuffle: int = 2,
    seed: int = 0,
) -> pl.DataFrame:
    """Long table: one row per (pair, alphabet). `key_cols` are copied from `df`."""
    rng = np.random.default_rng(seed)
    out = []
    for row in df.select(key_cols + ["qaln", "taln"]).iter_rows(named=True):
        keys = {k: row[k] for k in key_cols}
        for r in pair_stats(row["qaln"], row["taln"], alphabets, n_shuffle, rng):
            out.append({**keys, **r})
    return pl.DataFrame(out)


# ---------------------------------------------------------------------------
# Many pairs at once: one entry per aligned position, so per-pair and per-half numbers are
# bincounts instead of a Python loop per pair. Same definitions as pair_stats.
# ---------------------------------------------------------------------------
AA20 = "".join(ALPHABET_CLUSTERS["protein20"])


def aligned_codes(df: pl.DataFrame, min_cols: int = 0, return_columns: bool = False):
    """Flatten aligned pairs into one entry per aligned position.

    An aligned position is a column where both sequences have one of the 20 standard amino
    acids (X, B, Z, U and O count as gaps, as in pair_stats). Returns five arrays:
    query and target residue codes (0-19, in the order of AA20), the pair index (0..n-1 over
    the kept pairs), the position index within the pair (0, 1, 2, ...), and the row numbers
    of `df` that were kept. Pairs with fewer than `min_cols` aligned positions are dropped.
    With return_columns=True a sixth array gives each position's column in the gapped
    alignment, so callers can tell where a gap sat between two aligned positions.
    """
    tab = TABLES["protein20"]
    qs, ts, ps, js, cs, kept = [], [], [], [], [], []
    for i, (qa, ta) in enumerate(zip(df["qaln"], df["taln"])):
        q = tab[np.frombuffer(qa.encode(), np.uint8)]
        t = tab[np.frombuffer(ta.encode(), np.uint8)]
        m = (q != GAP) & (t != GAP)
        n = int(m.sum())
        if n == 0 or n < min_cols:
            continue
        k = len(kept)
        kept.append(i)
        qs.append(q[m])
        ts.append(t[m])
        ps.append(np.full(n, k, np.int64))
        js.append(np.arange(n, dtype=np.int64))
        if return_columns:
            cs.append(np.flatnonzero(m))
    out = (
        np.concatenate(qs).astype(np.int64),
        np.concatenate(ts).astype(np.int64),
        np.concatenate(ps),
        np.concatenate(js),
        np.array(kept),
    )
    return out + (np.concatenate(cs),) if return_columns else out


def class_index(alphabet: str) -> np.ndarray:
    """For each protein20 code (order of AA20), its class number in `alphabet`."""
    return TABLES[alphabet][np.frombuffer(AA20.encode(), np.uint8)].astype(np.int64)


def grouped_evidence(q, t, group, n_groups: int, alphabet: str = "protein20"):
    """Agreement, chance and evidence I for groups of aligned positions.

    `group` gives each aligned position's group (a pair, or one half of a pair). Returns
    (n, agree, chance, I) per group: n aligned positions; agree, the share in the same class;
    chance, sum over classes of the group's own query share times its target share; and I,
    bernoulli_kl_bits(agree, chance). For whole pairs this equals pair_stats.
    """
    cl = class_index(alphabet)
    qc, tc = cl[q], cl[t]
    k = int(cl.max()) + 1
    n = np.bincount(group, minlength=n_groups).astype(float)
    same = np.bincount(group, weights=(qc == tc).astype(float), minlength=n_groups)
    fq = np.bincount(group * k + qc, minlength=n_groups * k).reshape(n_groups, k)
    ft = np.bincount(group * k + tc, minlength=n_groups * k).reshape(n_groups, k)
    with np.errstate(divide="ignore", invalid="ignore"):
        agree = same / n
        chance = (fq * ft).sum(1) / n**2
    return n, agree, chance, bernoulli_kl_bits(agree, chance)


def pair_joint_counts(q, t, group, n_groups: int) -> np.ndarray:
    """Per group, the 20 x 20 counts of (query residue, target residue), flattened to 400."""
    return (
        np.bincount(group * 400 + q * 20 + t, minlength=n_groups * 400)
        .reshape(n_groups, 400)
        .astype(float)
    )


def two_class_splits(size: int = 9) -> np.ndarray:
    """Every way to put `size` of the 20 amino acids in one class and the rest in the other.

    Returns a boolean array, one row per split, one column per amino acid (order of AA20);
    True marks the `size`-letter class. For size 9 there are 167,960 rows.
    """
    from itertools import combinations

    idx = np.array(list(combinations(range(20), size)))
    out = np.zeros((len(idx), 20), bool)
    out[np.arange(len(idx))[:, None], idx] = True
    return out


def split_name(row: np.ndarray) -> str:
    """The letters of a split's smaller class, e.g. 'ACFILMVWY'."""
    return "".join(a for a, s in zip(AA20, row) if s)


def mean_split_evidence(
    joint: np.ndarray, splits: np.ndarray, chunk: int = 4000
) -> np.ndarray:
    """Mean over pairs of I for every two-class split, from pair_joint_counts.

    For one split and one pair, agree is the share of aligned positions whose two residues fall
    in the same class, and chance is h_query h_target + p_query p_target from that pair's own
    class shares, as pair_stats computes them for a named two-class alphabet.
    """
    n = joint.sum(1)
    jf = joint / n[:, None]
    j3 = jf.reshape(-1, 20, 20)
    q_share, t_share = j3.sum(2), j3.sum(1)
    out = np.empty(len(splits))
    for st in range(0, len(splits), chunk):
        s = splits[st : st + chunk].astype(float)
        same = (s[:, :, None] == s[:, None, :]).reshape(len(s), 400).astype(float)
        agree = jf @ same.T
        h_q, h_t = q_share @ s.T, t_share @ s.T
        chance = h_q * h_t + (1 - h_q) * (1 - h_t)
        out[st : st + chunk] = np.nanmean(bernoulli_kl_bits(agree, chance), axis=0)
    return out


def longest_runs(
    q, t, group, n_groups: int, columns, alphabet: str = "protein20"
) -> np.ndarray:
    """Per group, the longest run of adjacent alignment columns whose residues share a class.

    This is the longest k-mer the two sequences share exactly on the alignment's diagonal.
    `columns` is each position's column in the gapped alignment (aligned_codes with
    return_columns=True); a gap or a non-standard letter between two positions breaks the run,
    as in pair_stats. Positions must be in alignment order and each group's positions
    contiguous in the arrays, as aligned_codes returns them.
    """
    cl = class_index(alphabet)
    same = cl[q] == cl[t]
    start = np.r_[True, (group[1:] != group[:-1]) | (columns[1:] != columns[:-1] + 1)]
    idx = np.arange(same.size)
    # index of the last position that broke a run (a disagreement or a group start)
    last_break = np.where(~same | start, idx, 0)
    np.maximum.accumulate(last_break, out=last_break)
    # run length ending here: positions since the last break, plus the break itself when it
    # was a group start that agrees
    run = np.where(same, idx - last_break + (start[last_break] & same[last_break]), 0)
    out = np.zeros(n_groups, np.int64)
    np.maximum.at(out, group, run)
    return out


def alignment_lines(qaln: str, taln: str, alphabet: str = "hp_thomas_dill2") -> dict:
    """Per-column strings for showing one aligned pair in protein20 and a 2-class alphabet.

    ``query_class``/``target_class`` write each residue as H (first class) or P (second
    class), ``-`` for a gap. ``identical`` and ``same_class`` hold ``|`` where both columns
    have a residue and they agree. ``aligned`` is True where neither sequence has a gap,
    the columns ``pair_stats`` counts.
    """
    h, p = ALPHABET_CLUSTERS[alphabet]
    cls = {r: "H" for r in h} | {r: "P" for r in p}

    def letter(c):
        return cls.get(c.upper(), "-")

    qc = "".join(letter(c) for c in qaln)
    tc = "".join(letter(c) for c in taln)
    aligned = [a != "-" and b != "-" for a, b in zip(qc, tc)]
    identical = "".join(
        "|" if ok and a.upper() == b.upper() else " "
        for ok, a, b in zip(aligned, qaln, taln)
    )
    same_class = "".join(
        "|" if ok and a == b else " " for ok, a, b in zip(aligned, qc, tc)
    )
    return {
        "query": qaln,
        "target": taln,
        "query_class": qc,
        "target_class": tc,
        "aligned": aligned,
        "identical": identical,
        "same_class": same_class,
        "n_aligned": sum(aligned),
    }


def print_alignment(rows: list[tuple[str, str]], width: int = 60) -> None:
    """Print (label, line) rows in blocks of `width` columns, with the column range."""
    n = len(rows[0][1])
    pad = max(len(name) for name, _ in rows) + 2
    for s in range(0, n, width):
        for name, line in rows:
            print(f"{name:<{pad}}{line[s:s + width]}")
        print(f"columns {s + 1}-{min(s + width, n)}\n")


def draw_alignment_rows(
    ax,
    rows: list[tuple[str, str, str]],
    aligned: list[bool],
    h_color: str,
    p_color: str,
    agree_color: str = "black",
    differ_color: str = "#9a9a9a",
    gap_color: str = "#c8c8c8",
    x_max: int | None = None,
) -> None:
    """Draw an aligned pair one row per line, one column per alignment column.

    Each row is (label, kind, line). kind "class": a coloured box per H or P residue.
    kind "agree": a square where the line has ``|``, a dot where an aligned column
    differs, an x at a gap column, and "n of N (per 100)" at the right.
    """
    from matplotlib.patches import Rectangle

    n_col = len(rows[0][2])
    n_al = sum(aligned)
    cols = np.arange(1, n_col + 1)
    gap = [x for x, ok in zip(cols, aligned) if not ok]
    x_max = x_max or n_col
    for y, (_, kind, line) in enumerate(rows):
        if kind == "class":
            for x, c in zip(cols, line):
                if c in "HP":
                    ax.add_patch(
                        Rectangle(
                            (x - 0.5, y - 0.35),
                            1,
                            0.7,
                            lw=0,
                            color=h_color if c == "H" else p_color,
                        )
                    )
        else:
            hit = [x for x, ok, c in zip(cols, aligned, line) if ok and c == "|"]
            miss = [x for x, ok, c in zip(cols, aligned, line) if ok and c != "|"]
            ax.scatter(hit, [y] * len(hit), marker="s", s=14, color=agree_color, lw=0)
            ax.scatter(
                miss, [y] * len(miss), marker=".", s=10, color=differ_color, lw=0
            )
            ax.scatter(gap, [y] * len(gap), marker="x", s=10, color=gap_color, lw=0.8)
            ax.text(
                x_max + 2,
                y,
                f"{len(hit)} of {n_al} ({100 * len(hit) / n_al:.0f} per 100)",
                va="center",
                ha="left",
                fontsize=9,
            )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.5, x_max + 0.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)


TICK_BASE = 1.55  # every tick rises from here (just above the target strip), in rows
TICK_TALL = 1.1  # height of the identical-residue tick
TICK_SHORT = 0.4  # height of the tick for different residues, both H or both P
TICK_WIDTH = 0.5  # tick width, in alignment columns


def draw_hp_pair(
    ax,
    lines: dict,
    h_color: str,
    p_color: str,
    gap_color: str,
    tick_color: str = "black",
    x_max: int | None = None,
) -> dict:
    """Draw one aligned pair as query H/P strip, match line, target H/P strip.

    `lines` comes from `alignment_lines`. Rows: y=0 query, y=1 match line, y=2 target.
    Match line: ticks rising from one baseline, tall where the residues are identical,
    short where they differ but are both H or both P, none where one is H and one is P.
    A column with a gap in either sequence gets one band of `gap_color` across all three
    rows and no tick.
    Returns the counts drawn, so a caller can put them in a title.
    """
    from matplotlib.patches import Rectangle

    qc, tc = lines["query_class"], lines["target_class"]
    aligned, identical = lines["aligned"], lines["identical"]
    n_col = len(qc)
    x_max = x_max or n_col
    n_tall = n_short = 0
    for i in range(n_col):
        x = i + 1
        if not aligned[i]:
            ax.add_patch(
                Rectangle((x - 0.52, -0.5), 1.04, 3, lw=0, color=gap_color, zorder=0)
            )
        for y, c in ((0, qc[i]), (2, tc[i])):
            if c in "HP":
                # 1.04 wide so neighbouring boxes overlap and no white seam shows
                ax.add_patch(
                    Rectangle(
                        (x - 0.52, y - 0.35),
                        1.04,
                        0.7,
                        lw=0,
                        color=h_color if c == "H" else p_color,
                        zorder=1,
                    )
                )
        if aligned[i] and qc[i] == tc[i]:
            height = TICK_TALL if identical[i] == "|" else TICK_SHORT
            n_tall += identical[i] == "|"
            n_short += identical[i] != "|"
            ax.add_patch(
                Rectangle(
                    (x - TICK_WIDTH / 2, TICK_BASE - height),
                    TICK_WIDTH,
                    height,
                    lw=0,
                    color=tick_color,
                    zorder=2,
                )
            )
    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_ylim(2.5, -0.5)
    ax.set_yticks([0, 1, 2], ["query H/P", "agreement", "target H/P"])
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    n_al = lines["n_aligned"]
    return {
        "gap_free": n_al,
        "identical": n_tall,
        "same_class": n_tall + n_short,
        "gap_columns": n_col - n_al,
    }


IDENTITY_BINS = [0.0, 0.2, 0.3, 0.4, 0.6, 1.01]
IDENTITY_LABELS = ["<20%", "20-30%", "30-40%", "40-60%", ">=60%"]


def add_identity_bin(df: pl.DataFrame, col: str = "seqid_ali") -> pl.DataFrame:
    return df.with_columns(
        pl.col(col)
        .cut(IDENTITY_BINS[1:-1], labels=IDENTITY_LABELS)
        .cast(pl.Enum(IDENTITY_LABELS))
        .alias("identity_bin")
    )


def bootstrap_mean_ci(
    x: np.ndarray, n_boot: int = 500, seed: int = 0
) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (
        float(x.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def summarise(
    df: pl.DataFrame, by: list[str], value: str, n_boot: int = 500
) -> pl.DataFrame:
    """Mean with a bootstrap 95% CI of `value`, grouped by `by`."""
    rows = []
    for keys, g in df.group_by(by, maintain_order=True):
        m, lo, hi = bootstrap_mean_ci(g[value].to_numpy(), n_boot)
        rows.append(
            {
                **dict(zip(by, keys)),
                "n": g.height,
                f"{value}_mean": m,
                f"{value}_lo": lo,
                f"{value}_hi": hi,
            }
        )
    return pl.DataFrame(rows)


def frac_run_at_least(
    df: pl.DataFrame, by: list[str], ks: list[int], col: str = "longest_run"
) -> pl.DataFrame:
    """Fraction of pairs whose longest exact run reaches each k, grouped by `by`."""
    return (
        df.group_by(by, maintain_order=True)
        .agg(
            [pl.len().alias("n")]
            + [(pl.col(col) >= k).mean().alias(f"k{k}") for k in ks]
        )
        .unpivot(index=by + ["n"], variable_name="k", value_name="frac")
        .with_columns(pl.col("k").str.strip_prefix("k").cast(pl.Int64))
    )


# ---------------------------------------------------------------------------
# Figure stamping, same convention as mhc_region_utils.finish_figure.
# ---------------------------------------------------------------------------
NO_TOOL = "none (no search result: computed from alignments and alphabet tables only)"


# One implementation for every notebook; see figure_utils.finish_figure.
from figure_utils import finish_figure  # noqa: E402,F401 (re-exported)

# ---------------------------------------------------------------------------
# Exact chance of a long class-identical run, for one pair at a time.
# ---------------------------------------------------------------------------


def aligned_stretches(qaln: str, taln: str) -> np.ndarray:
    """Lengths of the gap-free stretches of an alignment: runs of columns where both have a residue."""
    q = TABLES["protein20"][np.frombuffer(qaln.encode(), dtype=np.uint8)]
    t = TABLES["protein20"][np.frombuffer(taln.encode(), dtype=np.uint8)]
    d = np.diff(np.concatenate(([0], ((q != GAP) & (t != GAP)).astype(np.int8), [0])))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def pr_longest_run_at_least(
    pr_agree: np.ndarray, stretch_lengths: list[np.ndarray], ks: list[int] | np.ndarray
) -> np.ndarray:
    """Pr(longest run of agreeing positions >= k), exactly, for every pair and every k.

    Each pair i is a set of stretches (lengths in ``stretch_lengths[i]``). Every position
    agrees independently with probability ``pr_agree[i]``; a run cannot cross from one
    stretch to the next. Pass one stretch per pair to ignore gaps.

    Within one stretch of n positions, Q_n = Pr(no run >= k) follows
    Q_n = 1 for n < k, Q_k = 1 - p^k, Q_n = Q_(n-1) - (1 - p) p^k Q_(n-k-1) for n > k
    (the first run of k ends at position n, after a disagreeing position n - k).
    Stretches are independent, so the pair's Pr(no run >= k) is the product over them.

    Returns an array of shape (len(ks), n_pairs).
    """
    p = np.asarray(pr_agree, dtype=float)
    n_pairs = p.size
    n_max = max(int(s.max()) for s in stretch_lengths)
    pair_idx, length, count = [], [], []
    for i, s in enumerate(stretch_lengths):
        u, c = np.unique(s, return_counts=True)
        pair_idx.append(np.full(u.size, i))
        length.append(u)
        count.append(c)
    pair_idx, length, count = (np.concatenate(x) for x in (pair_idx, length, count))
    out = np.empty((len(ks), n_pairs))
    for ki, k in enumerate(ks):
        q = np.ones((n_max + 1, n_pairs))
        pk = p**k
        if k <= n_max:
            q[k] = 1 - pk
        for n in range(k + 1, n_max + 1):
            q[n] = q[n - 1] - (1 - p) * pk * q[n - k - 1]
        log_none = np.zeros(n_pairs)
        np.add.at(
            log_none,
            pair_idx,
            count * np.log(np.clip(q[length, pair_idx], 1e-300, 1.0)),
        )
        out[ki] = 1 - np.exp(log_none)
    return out
