"""BHF's low-complexity stretches, found with kmerseek's own mask rule, and the copies of
BHF that keep or remove them (notebook 283).

kmerseek's low-complexity mask (`--remove-low-complexity`, on for every index of notebooks
241, 242 and 256) is not SEG. At kmerseek 982a055, the build those notebooks used, it drops
one k-mer at a time (`src/rust/sketch.rs`, `src/rust/kmer.rs::is_homopolymer_kmer`):

1. if its k residues are all the same amino acid (a raw run such as KKKKK), for every
   alphabet; or
2. if its k letters, after encoding with the alphabet, are all the same class, for every
   alphabet that has a class table in `src/rust/alphabets.rs` (all but protein20 and
   dayhoff6), and for hp_lehninger2.

A stretch, for one alphabet and k, is a run of residues covered by masked k-mers: the union
of every masked window, with windows that touch or overlap merged. So the stretches depend
on the setting (alphabet and k), and this module computes them per setting.

The class tables are read from the kmerseek source file itself, not typed in here, and
`check_against_binary` compares the number of masked windows with the count the kmerseek
binary prints when it indexes BHF alone.
"""

from __future__ import annotations

import os
import random
import re
import subprocess
import tempfile
from pathlib import Path

# kmerseek at 982a055: the source the class tables are read from, and the binary built from it.
KMERSEEK_SRC = Path(
    os.environ.get("KMERSEEK_SRC", "/Users/olga/code/kmerseek-ka-lambda-region")
)
KMERSEEK = KMERSEEK_SRC / "target/release/kmerseek"

# Alphabet name in kmerseek -> the name of its table in alphabets.rs (Alphabet::partition).
_TABLE_NAME = {
    "hp_lehninger2": "LEHNINGER_HP",
    "hp_thomas_dill2": "THOMAS_DILL_HP",
    "hp_kyte_doolittle2": "KYTE_DOOLITTLE_HP",
    "hp_thomas_dill_no_c2": "THOMAS_DILL_NO_C_HP",
    "hp_lehninger_c_nonpolar2": "LEHNINGER_C_NONPOLAR_HP",
    "hp_lehninger_hpc3": "LEHNINGER_HPC",
    "hp_pbotc_1st_ed2": "PBOTC_1ST_ED_HP",
    "gbmr4": "GBMR4",
    "polarity4": "POLARITY4",
    "wwmj5": "WWMJ5",
    "gbmr7": "GBMR7",
    "funcgroups8": "FUNCGROUPS8",
    "sdm12": "SDM12",
    "mmseqs12": "MMSEQS12",
    "wass14": "WASS14",
    "hsdm17": "HSDM17",
    "uniprot18": "UNIPROT18",
}
# No class check: kmerseek hands these to sourmash's own encoder and only checks raw runs.
RAW_ONLY = {"protein20", "dayhoff6"}


def alphabet_classes(
    src: Path = KMERSEEK_SRC / "src/rust/alphabets.rs",
) -> dict[str, dict[str, int]]:
    """Residue -> class number for every alphabet with a class table, read from kmerseek's
    alphabets.rs. Two residues share a number when the alphabet puts them in one class.
    """
    text = src.read_text()
    out = {}
    for alphabet, name in _TABLE_NAME.items():
        m = re.search(
            rf"static {name}: LazyLock<HashMap<u8, u8>> =\s*LazyLock::new\(\|\| (\w+)\(([^;]*?)\)\);",
            text,
        )
        if not m:
            raise ValueError(f"table {name} not found in {src}")
        builder, args = m.group(1), m.group(2)
        if builder in ("build_hp", "build_hpc"):
            groups = re.findall(r'b"([A-Z]+)"', args)
        elif builder == "build_reduced":
            consts = re.search(
                rf"const {args.strip()}: &\[&str\] =\s*&\[(.*?)\];", text, re.S
            )
            groups = re.findall(r'"([A-Z]+)"', consts.group(1))
        else:
            raise ValueError(f"unknown table builder {builder} for {name}")
        residues = "".join(groups)
        if sorted(residues) != sorted("ACDEFGHIKLMNPQRSTVWY"):
            raise ValueError(
                f"{name}: classes do not cover the 20 amino acids once each: {groups}"
            )
        out[alphabet] = {aa: i for i, g in enumerate(groups) for aa in g}
    return out


CLASSES = alphabet_classes()


def masked_windows(seq: str, alphabet: str, k: int) -> list[int]:
    """0-based start of every k-mer of `seq` that kmerseek's mask drops for this setting.

    Only for the 20 canonical residues in upper case, which is all BHF and its copies hold.
    kmerseek treats the others differently (lowercase is folded, B/J/Z are expanded into
    both readings, the index stops at the first `*`), and this copy does not."""
    if set(seq) - set("ACDEFGHIKLMNPQRSTVWY"):
        raise ValueError(
            f"non-canonical residues {sorted(set(seq) - set('ACDEFGHIKLMNPQRSTVWY'))}: not handled"
        )
    table = None if alphabet in RAW_ONLY else CLASSES[alphabet]
    starts = []
    for i in range(len(seq) - k + 1):
        w = seq[i : i + k]
        if len(set(w)) == 1 or (table is not None and len({table[c] for c in w}) == 1):
            starts.append(i)
    return starts


def stretches(seq: str, alphabet: str, k: int) -> list[tuple[int, int]]:
    """The residues covered by masked k-mers, as merged (start, end) intervals, 0-based and
    end-exclusive. Two windows that overlap or touch end to end make one stretch."""
    out: list[list[int]] = []
    for s in masked_windows(seq, alphabet, k):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], s + k)
        else:
            out.append([s, s + k])
    return [(a, b) for a, b in out]


def check_against_binary(seq: str, alphabet: str, k: int) -> tuple[int, int]:
    """(masked windows here, k-mers the kmerseek binary reports removed) for `seq` alone."""
    with tempfile.TemporaryDirectory() as d:
        fa = Path(d) / "q.fa"
        fa.write_text(f">q\n{seq}\n")
        r = subprocess.run(
            [
                str(KMERSEEK),
                "index",
                "-i",
                str(fa),
                "-o",
                str(Path(d) / "q.rocksdb"),
                "-k",
                str(k),
                "-s",
                "1",
                "-a",
                alphabet,
                "--remove-low-complexity",
                "--extend-mismatch-penalty",
                "0",
            ],
            capture_output=True,
            text=True,
        )
        m = re.search(
            r"Removed (\d+) of (\d+) k-mer windows as low-complexity",
            r.stderr + r.stdout,
        )
        if r.returncode != 0 or not m:
            raise RuntimeError(
                f"{alphabet} k={k}: rc={r.returncode}\n{r.stderr[-2000:]}"
            )
        removed = int(m.group(1).replace(",", ""))
    return len(masked_windows(seq, alphabet, k)), removed


def _segments(n: int, keep: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The intervals of [0, n) between the kept stretches."""
    out, pos = [], 0
    for a, b in keep:
        if a > pos:
            out.append((pos, a))
        pos = b
    if pos < n:
        out.append((pos, n))
    return out


def stretches_kept(
    seq: str, keep: list[tuple[int, int]], rng: random.Random, shuffle
) -> str:
    """A copy of `seq` with every stretch in `keep` verbatim at its own position and each
    segment between stretches dipeptide-shuffled on its own (`shuffle(segment, rng)`).
    """
    pieces = sorted(
        [(a, b, True) for a, b in keep]
        + [(a, b, False) for a, b in _segments(len(seq), keep)]
    )
    out = "".join(seq[a:b] if kept else shuffle(seq[a:b], rng) for a, b, kept in pieces)
    assert len(out) == len(seq), (len(out), len(seq))
    for a, b in keep:
        assert out[a:b] == seq[a:b]
    return out


def stretches_removed(seq: str, keep: list[tuple[int, int]]) -> str:
    """`seq` with every stretch deleted and the pieces between them joined."""
    return "".join(seq[a:b] for a, b in _segments(len(seq), keep))


# ---------------------------------------------------------------------------------------
# Notebook 283: BHF against each of the three sets of copies.

DATA = Path("/Users/olga/data/botryllus/alphabet-ranking-three-cases")
NULLS = {
    "null 1: whole protein shuffled (notebook 256)": DATA / "null_bhf_dipeptide",
    "null 2: stretches kept in place": DATA / "283" / "null_kept",
    "null 3: stretches removed": DATA / "283" / "null_removed",
    # Real Botryllus proteins of BHF's length (added 2026-10-20): they do not share
    # BHF's composition, as nulls 1-3 do.
    "null 4: other Botryllus proteins, no conventional match": DATA
    / "283"
    / "null_dark_proteins",
    "null 5: other Botryllus proteins, any": DATA / "283" / "null_any_proteins",
}
# The same scores PR 83 reported; the bit score ranks exactly as the E-value, and the
# Poisson p-value as the Poisson score, so each pair is one row.
SCORES = [
    "E-value",
    "tf-idf",
    "containment",
    "shared k-mers",
    "mean IDF",
    "protein enrichment",
    "protein Poisson p-value",
    "Poisson score",
    "enrichment",
]
ALPHA = (
    0.05  # BHF "beats 95% of its copies": (1 + copies at least as good) / 301 <= 0.05
)


def run_finished(run: Path) -> tuple[int, int]:
    """(search tables written, tables expected) for one run folder."""
    import bhf_shuffle_null_utils as su

    if not (run / "run.json").exists():
        return 0, 1
    before, su.RUN = su.RUN, run
    try:
        return su.progress()
    finally:
        su.RUN = before


def best_score_p_all(names: list[str] | None = None) -> "pl.DataFrame":
    """One row per (null, score, setting): BHF's best value against its copies' best
    values, with p = (1 + copies at least as good) / (1 + copies), from notebook 256's code.
    `names` limits it to those nulls (default: all of NULLS).
    """
    import polars as pl
    import bhf_shuffle_null_utils as su

    out, before = [], su.RUN
    for name, run in NULLS.items():
        if names is not None and name not in names:
            continue
        su.RUN = run
        out.append(su.best_score_p(su.load()).with_columns(pl.lit(name).alias("null")))
    su.RUN = before
    return pl.concat(out, how="diagonal").filter(pl.col("metric").is_in(SCORES))


def wins_by_null(p: "pl.DataFrame") -> "pl.DataFrame":
    """Per null and score: settings where BHF has a score, settings where it beats 95% of
    its copies, the count chance predicts (5%), and the decision line (twice chance)."""
    import polars as pl

    return (
        p.group_by("null", "metric")
        .agg(
            pl.len().alias("n_settings"),
            (pl.col("p") <= ALPHA).sum().alias("n_beats_95pct"),
        )
        .with_columns(
            (ALPHA * pl.col("n_settings")).round(1).alias("expected_by_chance"),
            (2 * ALPHA * pl.col("n_settings")).round(1).alias("twice_chance"),
            (100 * pl.col("n_beats_95pct") / pl.col("n_settings"))
            .round(1)
            .alias("pct_settings"),
        )
        .with_columns(
            pl.col("metric")
            .replace_strict({m: i for i, m in enumerate(SCORES)})
            .alias("_s"),
            pl.col("null")
            .replace_strict({m: i for i, m in enumerate(NULLS)})
            .alias("_n"),
        )
        .sort("_s", "_n")
        .drop("_s", "_n")
    )


def fig_wins_by_null(
    w: "pl.DataFrame", path_stem: Path, hypothesis: str, conclusion: str
):
    """For each score, the share of settings where BHF beats 95% of its copies, under each
    null, against the 5% chance predicts and the 10% decision line."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import pubfig as pf

    pf.use_style()
    colours = [
        pf.OKABE_ITO["blue"],
        pf.OKABE_ITO["vermillion"],
        pf.OKABE_ITO["bluish_green"],
        pf.OKABE_ITO["orange"],
        pf.OKABE_ITO["reddish_purple"],
    ]
    shapes = ["o", "s", "^", "D", "v"]
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 95)
    rows = {m: i for i, m in enumerate(SCORES)}
    for y in rows.values():
        ax.axhline(y, color="#dddddd", lw=0.3, zorder=0)
    ax.axvline(100 * ALPHA, color=pf.GREY, lw=1, zorder=1)
    ax.axvline(200 * ALPHA, color=pf.GREY, lw=1, ls="--", zorder=1)
    present = set(w["null"].to_list())
    for j, (null, c, s) in enumerate(zip(NULLS, colours, shapes)):
        if null not in present:
            continue
        d = w.filter(w["null"] == null)
        ys = [rows[m] + (j - (len(NULLS) - 1) / 2) * 0.16 for m in d["metric"]]
        ax.scatter(
            d["pct_settings"], ys, color=c, marker=s, s=18, zorder=3, edgecolor="none"
        )
    ax.set_yticks(
        list(rows.values()),
        [
            f"{m}"
            + (
                " (and bit score)"
                if m == "E-value"
                else " (and Poisson p-value)" if m == "Poisson score" else ""
            )
            for m in rows
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Settings where BHF beats 95% of the null's 300 queries (% of the settings where BHF has that score)"
    )
    ax.set_xlim(0, max(30, float(w["pct_settings"].max()) + 3))
    handles = [
        Line2D([], [], color=c, marker=s, ls="none", label=n)
        for n, c, s in zip(NULLS, colours, shapes)
        if n in present
    ]
    handles += [
        Line2D([], [], color=pf.GREY, lw=1, label="chance: 5%"),
        Line2D(
            [],
            [],
            color=pf.GREY,
            lw=1,
            ls="--",
            label="decision line: 10% (twice chance)",
        ),
    ]
    # Above the axis, before the marks in reading order; with five nulls it no longer
    # fits inside the plot above the first row.
    ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(-0.3, 1.01),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.01,
        1.14,
        "kmerseek 0.4.0 (982a055): BHF and 300 copies or other proteins per null searched against 19_732 GENCODE v49 "
        "human proteins at the 152 settings (alphabet, k) of notebook 241.",
        ha="left",
        va="top",
    )
    fig.text(0.01, 1.10, "Hypothesis: " + hypothesis, ha="left", va="top")
    fig.text(0.01, 1.06, "Conclusion: " + conclusion, ha="left", va="top")
    pf.save(fig, str(path_stem))
    return fig


def fig_stretches(
    seq: str, rows: list[tuple[str, list[tuple[int, int]]]], path_stem: Path
):
    """BHF as a line with one row per rule, each stretch a box (1-based labels)."""
    from matplotlib.patches import Rectangle
    import pubfig as pf

    pf.use_style()
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 14 + 8 * len(rows))
    for i, (label, iv) in enumerate(rows):
        ax.plot([1, len(seq)], [i, i], color="black", lw=0.6, zorder=1)
        for a, b in iv:
            ax.add_patch(
                Rectangle(
                    (a + 0.5, i - 0.22),
                    b - a,
                    0.44,
                    color=pf.OKABE_ITO["vermillion"],
                    zorder=2,
                )
            )
            ax.text(
                (a + b) / 2 + 0.5, i - 0.26, f"{a + 1}-{b}", ha="center", va="bottom"
            )
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_ylim(len(rows) - 0.4, -0.9)
    ax.set_xlim(0, len(seq) + 1)
    ax.set_xlabel("BHF residue (1-252)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.01,
        1.0,
        "BHF (252 aa) is the black line; each box is a stretch the rule on that row calls "
        "low-complexity.",
        ha="left",
        va="bottom",
    )
    pf.save(fig, str(path_stem))
    return fig


def class_string(s: str, alphabet: str) -> str:
    """`s` written in the alphabet's classes: h/p for the two-letter alphabets, otherwise
    the first residue of each class (as kmerseek's class tables name them)."""
    t = CLASSES[alphabet]
    if alphabet.startswith("hp_") and len(set(t.values())) == 2:
        return "".join("hp"[t[c]] if c in t else "?" for c in s)
    first = {}
    for aa, i in sorted(t.items(), key=lambda x: x[1]):
        first.setdefault(i, aa)
    return "".join(first[t[c]] if c in t else "?" for c in s)


def show_alignment(
    q: str,
    qs: int,
    t: str,
    ts: int,
    gene: str,
    alphabet: str,
    hp: str = "hp_pbotc_1st_ed2",
) -> str:
    """An ungapped kmerseek match as text: BHF and human residues with 1-based
    coordinates, a match line, "n of m identical", then the class strings of the setting's
    alphabet and of the paper's H/P alphabet, each with its own class match line."""
    assert len(q) == len(t), "kmerseek regions are ungapped: both sides have one length"
    n = len(q)
    same = sum(a == b for a, b in zip(q, t))
    lines = [
        f"BHF   {qs + 1:>5} {q} {qs + n}",
        f"{'':12}{''.join('|' if a == b else ' ' for a, b in zip(q, t))}",
        f"{gene:<6}{ts + 1:>5} {t} {ts + n}",
        f"{same} of {n} residues identical",
    ]
    for a in dict.fromkeys([alphabet, hp]):
        if a in RAW_ONLY:
            continue
        cq, ct = class_string(q, a), class_string(t, a)
        lines += [
            f"{a} classes:",
            f"BHF   {'':>5} {cq}",
            f"{'':12}{''.join('|' if x == y else ' ' for x, y in zip(cq, ct))}",
            f"{gene:<6}{'':>5} {ct}",
            f"{sum(x == y for x, y in zip(cq, ct))} of {n} classes the same",
        ]
    return "\n".join(lines)
