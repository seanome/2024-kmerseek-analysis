"""Notebook 359: kmerseek calls on alternative exons that IsoAnnotLite leaves blank.

Pure functions shared by the Sherlock scripts (scripts/359_isoform_blank_exons/), the
notebook and the tests. Every interval here is 0-based and end-exclusive unless a name
says otherwise. Python 3.9 or later; polars only where a function says so.

The rule these functions serve is frozen in notebooks/359_data/preregistered.md.
"""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Exon to protein coordinates
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Exon:
    """One exon of a transcript, in genomic coordinates (1-based inclusive, as in GTF)."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def transcript_order(exons: list[Exon], strand: str) -> list[Exon]:
    """Exons in 5' to 3' order of the transcript."""
    ordered = sorted(exons, key=lambda e: e.start)
    return ordered if strand == "+" else ordered[::-1]


def exon_protein_intervals(
    exons: list[Exon], strand: str, cds_start: int, n_codons: int
) -> list[tuple[int, int] | None]:
    """Protein interval coded by each exon, in transcript order.

    `cds_start` is the 1-based transcript position of the first base of the start codon
    (SQANTI3's CDS_start). `n_codons` is the number of amino acids in the ORF (stop codon
    not counted). Codon i covers transcript bases cds_start-1+3i .. cds_start-1+3i+2
    (0-based). A codon split by a splice junction goes to the exon that holds two of its
    three bases. Returns, per exon in transcript order, the 0-based end-exclusive range of
    codons assigned to it, or None when the exon holds no whole-or-majority codon. A codon
    whose three bases lie in three different exons goes to none of them.
    """
    ordered = transcript_order(exons, strand)
    ends = []  # transcript 0-based end (exclusive) of each exon
    pos = 0
    for e in ordered:
        pos += e.length
        ends.append(pos)
    out: list[tuple[int, int] | None] = [None] * len(ordered)
    for i in range(n_codons):
        first = cds_start - 1 + 3 * i
        holders = [bisect.bisect_right(ends, b) for b in (first, first + 1, first + 2)]
        k = max(set(holders), key=holders.count)
        if holders.count(k) < 2:
            continue  # three bases in three exons: no exon holds two, so no exon gets it
        cur = out[k]
        out[k] = (i, i + 1) if cur is None else (cur[0], i + 1)
    return out


def usable_orf(orf_seq: str, cds_start, cds_end) -> str | None:
    """The protein SQANTI3's ORF_seq gives, or None when it cannot be used: the CDS length
    must be 3 x protein length (plus 3 for a stop codon) and the protein has no internal
    stop. One filter for build_exons.py and write_orfs.py, so InterProScan and the exon
    table cover the same isoforms."""
    prot = orf_seq.rstrip("*")
    s, e = int(float(cds_start)), int(float(cds_end))
    if (e - s + 1) not in (3 * len(prot), 3 * len(prot) + 3) or "*" in prot:
        return None
    return prot


def exon_transcript_intervals(exons: list[Exon], strand: str) -> list[tuple[int, int]]:
    """0-based end-exclusive transcript interval of each exon, in transcript order."""
    out, pos = [], 0
    for e in transcript_order(exons, strand):
        out.append((pos, pos + e.length))
        pos += e.length
    return out


def protein_to_transcript(seg: tuple[int, int], cds_start: int) -> tuple[int, int]:
    """Protein interval (0-based, end-exclusive) to the transcript bases of its codons."""
    return (cds_start - 1 + 3 * seg[0], cds_start - 1 + 3 * seg[1])


def protein_to_genomic(
    seg: tuple[int, int], exons: list[Exon], strand: str, cds_start: int
) -> list[tuple[int, int]]:
    """Genomic pieces (1-based inclusive) holding the codons of a protein interval."""
    t0, t1 = protein_to_transcript(seg, cds_start)
    pieces = []
    pos = 0
    for e in transcript_order(exons, strand):
        a, b = max(t0, pos), min(t1, pos + e.length)
        if a < b:
            if strand == "+":
                pieces.append((e.start + (a - pos), e.start + (b - pos) - 1))
            else:
                pieces.append((e.end - (b - pos) + 1, e.end - (a - pos)))
        pos += e.length
    return sorted(pieces)


# --------------------------------------------------------------------------------------
# Overlap and landing
# --------------------------------------------------------------------------------------


def overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Residues shared by two 0-based end-exclusive intervals."""
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def lands(
    call: tuple[int, int], segment: tuple[int, int], min_cover: float = 0.5
) -> bool:
    """A call lands on an exon when it covers at least `min_cover` of the exon's protein
    segment. The fraction is of the EXON segment, not of the call."""
    seg_len = segment[1] - segment[0]
    return seg_len > 0 and overlap(call, segment) >= min_cover * seg_len


# --------------------------------------------------------------------------------------
# Decoy
# --------------------------------------------------------------------------------------


def shuffle_segment(protein: str, segment: tuple[int, int], rng: random.Random) -> str:
    """The protein with only the residues of `segment` shuffled in place: same length,
    same composition, every residue outside the segment unchanged."""
    a, b = segment
    piece = list(protein[a:b])
    rng.shuffle(piece)
    return protein[:a] + "".join(piece) + protein[b:]


# --------------------------------------------------------------------------------------
# Threshold at a fixed decoy rate
# --------------------------------------------------------------------------------------


def landed_at(best: dict[str, float], t: float) -> set[str]:
    """Exons whose best landing score is at or below t (lower = stronger)."""
    return {k for k, v in best.items() if v <= t}


def threshold_at_decoy_rate(
    decoy_best: dict[str, float], n_blank: int, max_rate: float = 0.01
) -> float:
    """Loosest threshold t (scores <= t count) at which decoy landings are at most
    floor(max_rate * n_blank) exons. `decoy_best` maps each decoy exon that has any
    landing call to its best (lowest) score. Returns +inf when at most `cap` decoys have
    any landing call (every real landing counts); otherwise the float just below the
    (cap+1)-th smallest decoy score. That value is finite, never -inf: real exons with a
    score below it still land."""
    cap = math.floor(max_rate * n_blank)
    scores = sorted(decoy_best.values())
    if len(scores) <= cap:
        return math.inf
    # Every score equal to the (cap+1)-th smallest must be left out, ties included, so
    # the threshold sits just below it.
    return math.nextafter(scores[cap], -math.inf)


def union_thresholds(
    decoy_best_by_setting: dict[str, dict[str, float]],
    n_blank: int,
    max_rate: float = 0.01,
    steps: int = 2000,
) -> tuple[dict[str, float], float]:
    """Thresholds for the best-of-settings union at one shared per-setting quantile.

    Each setting's threshold is set at its own decoy-rate cap q (the loosest threshold at
    which that setting alone lands on at most floor(q * n_blank) decoy exons). q is the
    largest value on a grid of `steps` from 0 to max_rate at which the UNION of decoy
    exons over all settings stays at or below floor(max_rate * n_blank). Returns
    ({setting: threshold}, q).
    """
    cap = math.floor(max_rate * n_blank)
    best_q = 0.0
    best_t = {
        s: threshold_at_decoy_rate(d, n_blank, 0.0)
        for s, d in decoy_best_by_setting.items()
    }
    for i in range(steps + 1):
        q = max_rate * i / steps
        ts = {
            s: threshold_at_decoy_rate(d, n_blank, q)
            for s, d in decoy_best_by_setting.items()
        }
        union = set()
        for s, d in decoy_best_by_setting.items():
            union |= landed_at(d, ts[s])
        if len(union) <= cap:
            best_q, best_t = q, ts
        else:
            break
    return best_t, best_q


# --------------------------------------------------------------------------------------
# Kyte-Doolittle scan (notebook 231 and 244's scan, threshold left free)
# --------------------------------------------------------------------------------------

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}  # fmt: skip


def window_means(protein: str, window: int = 19) -> list[float]:
    """Mean hydropathy of every full window; entry i covers residues i .. i+window-1."""
    h = [KYTE_DOOLITTLE.get(c, 0.0) for c in protein.upper()]
    if len(h) < window:
        return []
    out, s = [], sum(h[:window])
    out.append(s / window)
    for i in range(window, len(h)):
        s += h[i] - h[i - window]
        out.append(s / window)
    return out


def scan_calls(
    means: list[float], threshold: float, window: int = 19
) -> list[tuple[int, int]]:
    """Segments (0-based end-exclusive residues) covered by consecutive windows whose mean
    is above `threshold`, as swissprot_control_utils.kd_transmem_calls draws them."""
    calls, start = [], None
    for i, m in enumerate(means + [-math.inf]):
        if m > threshold and start is None:
            start = i
        elif m <= threshold and start is not None:
            calls.append((start, i - 1 + window))
            start = None
    return calls


def scan_best_landing(
    means: list[float], segment: tuple[int, int], window: int = 19
) -> float:
    """The highest threshold at which the scan still lands on the segment, as a score
    where lower is stronger (its negative), so it reads like an E-value in
    threshold_at_decoy_rate. +inf when the scan cannot land at any threshold.

    A scan call at threshold t is a run of windows with mean > t. Lowering t only merges
    and grows runs, so landing is monotone in t and a bisection over the observed window
    means finds the highest landing threshold exactly.
    """
    if not means:
        return math.inf
    levels = sorted(set(means), reverse=True)

    def lands_at(t: float) -> bool:
        return any(lands(c, segment) for c in scan_calls(means, t, window))

    # thresholds just below each observed mean; t below the minimum makes one whole call
    cands = [math.nextafter(v, -math.inf) for v in levels]
    lo, hi = 0, len(cands) - 1
    if not lands_at(cands[hi]):
        return math.inf
    while lo < hi:
        mid = (lo + hi) // 2
        if lands_at(cands[mid]):
            hi = mid
        else:
            lo = mid + 1
    return -cands[lo]


# --------------------------------------------------------------------------------------
# Regions-index entry names
# --------------------------------------------------------------------------------------


def parse_region_name(name: str) -> dict:
    """`accession|feature type|description|start-end` from the regions-index FASTA.
    Descriptions have spaces as underscores and "none" when empty."""
    acc, ftype, desc, span = name.split("|")
    a, b = span.split("-")
    return {
        "accession": acc,
        "feature_type": ftype,
        "description": "" if desc == "none" else desc.replace("_", " "),
        "start": int(a),
        "end": int(b),
    }


def is_named(feature_type: str, description: str) -> bool:
    """A feature with a type AND a description, other than REGION "Disordered" alone."""
    if not feature_type or not description:
        return False
    return not (feature_type == "REGION" and description.lower() == "disordered")


# --------------------------------------------------------------------------------------
# Figures (pubfig style; one colour per source, the same in every panel)
# --------------------------------------------------------------------------------------

SOURCE_COLOR = {"real": "#0072B2", "decoy": "#999999", "scan": "#E69F00"}
SOURCE_LABEL = {
    "real": "kmerseek, real exons",
    "decoy": "kmerseek, same exons shuffled",
    "scan": "Kyte-Doolittle scan, real exons",
}


def fig_gate(gate: dict, path_stem: str):
    """SF-A: share of IsoAnnotLite's Pfam features that InterProScan finds again."""
    import pubfig as pf

    pf.use_style()
    fig, ax = pf.figure(pf.ONE_COLUMN_MM / 2, 50)
    r = gate["recall_by_overlap"] or 0.0
    ax.bar([0], [100 * r], color=SOURCE_COLOR["real"], width=0.6)
    ax.axhline(90, color="black", linestyle="--", linewidth=0.5)
    ax.text(0.45, 90, "gate 90%", va="bottom", ha="right")
    ax.set_xticks(
        [0],
        [
            f"{gate['recovered_by_overlap']:_} of\n{gate['isoannot_pfam_features']:_}".replace(
                "_", ","
            )
        ],
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("IsoAnnotLite Pfam features\nfound by InterProScan (%)")
    pf.save(fig, path_stem)
    return fig


def fig_landing(summary, path_stem: str):
    """SF-B: landing rate on the blank set per kmerseek setting and the union; real,
    decoy and scan bars in every group. a: all blank exons. b: split by whether the exon
    is in a GENCODE basic isoform with a CCDS tag."""
    import matplotlib.patches as mpatches
    import polars as pl
    import pubfig as pf

    pf.use_style()
    km = summary.filter(pl.col("setting") != "kyte_doolittle_scan")
    scan = summary.filter(pl.col("setting") == "kyte_doolittle_scan")
    settings = km.filter(pl.col("stratum") == "all")["setting"].to_list()
    fig, axs = pf.figure(
        pf.TWO_COLUMN_MM, 75, ncols=2, gridspec_kw={"width_ratios": [1.4, 1]}
    )
    handles = [
        mpatches.Patch(color=SOURCE_COLOR[s], label=SOURCE_LABEL[s])
        for s in ("real", "decoy", "scan")
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False)
    w = 0.27
    ax = axs[0]
    for i, s in enumerate(settings):
        row = km.filter((pl.col("setting") == s) & (pl.col("stratum") == "all")).row(
            0, named=True
        )
        sc = scan.filter(pl.col("stratum") == "all").row(0, named=True)
        n = row["n_blank"]
        vals = [row["real_landings"], row["decoy_landings"], sc["real_landings"]]
        for j, (src, v) in enumerate(zip(("real", "decoy", "scan"), vals)):
            ax.bar(i + (j - 1) * w, 100 * v / n if n else 0, w, color=SOURCE_COLOR[src])
        ax.text(
            i,
            100 * max(vals) / n + 0.3 if n else 0,
            f"net {100 * row['net_fraction']:.1f}%",
            ha="center",
            va="bottom",
        )
    ax.axhline(10, color="black", linestyle="--", linewidth=0.5)
    ax.set_xticks(
        range(len(settings)),
        [
            s.replace("_of_fitted_settings", " of settings").replace("_k", "\nk")
            for s in settings
        ],
        rotation=0,
    )
    ax.set_ylabel("Blank exons landed on (%)")
    ax = axs[1]
    strata = ["basic_ccds", "not_basic_ccds"]
    names = {
        "basic_ccds": "in a GENCODE basic\nCCDS isoform",
        "not_basic_ccds": "not in one",
    }
    union = km.filter(pl.col("setting") == "union_of_fitted_settings")
    for i, st in enumerate(strata):
        row = union.filter(pl.col("stratum") == st).row(0, named=True)
        sc = scan.filter(pl.col("stratum") == st).row(0, named=True)
        n = row["n_blank"]
        vals = [row["real_landings"], row["decoy_landings"], sc["real_landings"]]
        for j, (src, v) in enumerate(zip(("real", "decoy", "scan"), vals)):
            ax.bar(i + (j - 1) * w, 100 * v / n if n else 0, w, color=SOURCE_COLOR[src])
        ax.text(
            i,
            -0.12,
            f"n = {n:_}".replace("_", ","),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
        )
    ax.set_xticks(range(len(strata)), [names[s] for s in strata])
    ax.set_ylabel("Blank exons landed on (%), union of settings")
    for ax, letter in zip(axs, "ab"):
        pf.panel_label(ax, letter)
    pf.save(fig, path_stem)
    return fig


def fig_examples(survivors, exon_ids: list[str], path_stem: str):
    """SF-C: the randomly picked surviving landings, residues shown: exon segment (query)
    over the Swiss-Prot region entry (target), with a bar where the residues are equal.
    """
    import polars as pl
    import pubfig as pf

    pf.use_style()
    rows = (
        survivors.filter(pl.col("exon_id").is_in(exon_ids)).sort("exon_id").to_dicts()
    )
    fig, ax = pf.figure(pf.TWO_COLUMN_MM, 20 + 22 * max(1, len(rows)))
    ax.axis("off")
    y = 1.0
    step = 1.0 / max(1, len(rows))
    for r in rows:
        head = (
            f"{r['gene']} {r['exon_id']} (isoform {r['rep_isoform']}, exon residues {r['exon_p_start'] + 1}-{r['exon_p_end']}); "
            f"kmerseek {r['setting']}, E = {r['evalue']:.2g}; target {r['target']} "
            f"({r['feature_type']}: {r['description'] or 'no description'}); "
            f"{r['n_identical']} of {r['n_aligned']} identical"
        )
        ax.text(0, y, head, va="top", transform=ax.transAxes)
        q0, t0 = r["qstart"] + 1, r["tstart"] + 1
        block = "\n".join(
            [
                f"isoform  {q0:>5} {r['qseq'][:90]}",
                f"               {r['match'][:90]}",
                f"Swiss-Prot {t0:>3} {r['tseq'][:90]}",
            ]
        )
        ax.annotate(
            block,
            (0, y),
            xycoords="axes fraction",
            xytext=(0, -10),
            textcoords="offset points",
            va="top",
            family="monospace",
        )
        y -= step
    pf.save(fig, path_stem)
    return fig
