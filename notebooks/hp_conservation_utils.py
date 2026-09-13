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

Alphabet tables are copied from kmerseek's ``src/rust/alphabets.rs`` (dayhoff6 from
sourmash), so the classes match what the search engine indexes.
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
        null_runs = []
        for ts in shuffles:
            ta_s = ta.copy()
            ta_s[t_res_idx] = tab[ts]
            null_runs.append(longest_true_run((qa == ta_s) & both))
        rows.append(
            {
                "alphabet": name,
                "n_cols": n,
                "agree": float(agree),
                "expected": expected,
                "kappa": float(kappa),
                "longest_run": run,
                "longest_run_null": float(np.mean(null_runs)),
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


def finish_figure(
    fig,
    path,
    tools: str,
    hypothesis: str,
    conclusion: str,
    title: str | None = None,
    *,
    footer_y: float = -0.01,
    header_y: float = 1.005,
    dpi: int = 200,
    wrap: int | None = None,
):
    """Stamp TOOLS / hypothesis / conclusion on `fig`, then save it to `path`."""
    try:
        fig.tight_layout()
    except Exception:  # noqa: BLE001
        pass
    w_in, h_in = fig.get_size_inches()
    wrap = wrap or max(70, int(w_in * 12))
    line = lambda pt: pt / 72.0 / h_in * 1.45
    is_km = "kmerseek" in tools and not tools.startswith("no kmerseek")
    tool_lines = textwrap.wrap("TOOLS: " + tools, wrap)
    y = header_y
    fig.text(
        0.0,
        y,
        "\n".join(tool_lines),
        ha="left",
        va="bottom",
        fontsize=9.5,
        fontweight="bold",
        color="#8B1A1A" if is_km else "#1F3B73",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="#FBEAEA" if is_km else "#E8EEF8",
            edgecolor="none",
        ),
    )
    y += line(9.5) * len(tool_lines) + line(9.5) * 0.9
    if title:
        fig.text(
            0.5, y, title, ha="center", va="bottom", fontsize=12.5, fontweight="bold"
        )
    foot = textwrap.wrap("Hypothesis: " + hypothesis, wrap) + textwrap.wrap(
        "Conclusion: " + conclusion, wrap
    )
    fig.text(
        0.0,
        footer_y,
        "\n".join(foot),
        ha="left",
        va="top",
        fontsize=9,
        color="#222222",
        linespacing=1.35,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#F6F6F6", edgecolor="#DDDDDD"),
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
