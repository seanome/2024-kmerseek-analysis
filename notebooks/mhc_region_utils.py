#!/usr/bin/env python3
"""Shared definitions for the MHC-region notebooks (220-228) built on the QfO midi-plus run.

The midi-plus run (see `RUN`) searches 998 human proteins -- every chromosome 6 protein
plus B2M, CD1A-E, MR1 and the KIR/LILR receptors -- against nine whole proteomes that are
scored, and a tenth (Botryllus schlosseri) that is searched but has no Pfam annotation to
score against. Notebooks 210-216 asked a different question of a different run -- human vs
mouse, ortholog pairs, one alphabet -- so their helpers in `ortholog_analysis_utils` (`ou`)
stay the source of truth for anything about MHC gene taxonomy or domain architecture. This
module adds only what the region run needs on top: where a gene sits on chr6, which Pfam
family plays which structural role, how to load the extracted tables, and how to name a
tool on a figure so a reader never has to guess whether a panel is kmerseek.

Vocabulary used across the notebooks, in plain words:

* **arm**: one tool run with one setting. For kmerseek that is one alphabet at one k, so
  ``kmerseek hp_pbotc_1st_ed2 k=19`` is one arm and ``kmerseek protein20 k=10`` is another.
  Every other tool has one arm in this run.
* **comparison tools**: the nine tools kmerseek is measured against, named in
  `COMPARISON_TOOLS`. Eight are scored on IoU; Folddisco is scored on coverage.
* **found** (``covered`` in the tables): a hit overlaps at least half of the true domain.
  Asks whether the search reached the domain at all.
* **boundary matched** (``boundary_matched`` in the tables): the hit's interval and the true
  domain agree at IoU >= 0.5, IoU being the overlap divided by the union of the two
  intervals. Asks whether the hit's start and end are right. This is the run's own
  true-positive rule.

Nothing here re-defines a constant `ou` already carries. `ou.MHC_CLASSES`,
`ou.MHC_CLASS_COLORS`, `ou.ARD_I`, `ou.IG_I`, `ou.CONTACT_PROT` and `ou.mhc_gene_arch` are
imported and re-exported so a notebook needs one import, not two.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

import ortholog_analysis_utils as ou
from ortholog_analysis_utils import (  # noqa: F401  (re-exported for the notebooks)
    ARD_I,
    CONTACT_PROT,
    DOMAINS_CLASS_I,
    IG_I,
    MHC_ARCH,
    MHC_CLASS_COLORS,
    MHC_CLASS_I_GENES,
    MHC_CLASS_ORDER,
    MHC_CLASSES,
    TM_I,
    mhc_gene_arch,
)

# ---------------------------------------------------------------------------
# The run these notebooks read, stated once so every intro can quote the same facts.
#
# Numbers come from the run's own trace (`traces/midi-plus.trace.txt`) and its
# `truth/covariates_summary.json`, not from memory. The Botryllus searches were added on
# 2026-09-04 and ran inside the all-QfO run that shares this results directory, which is
# why they carry only that run's alphabets (five non-HP alphabets, no low-complexity mask)
# and why Botryllus has no scored metrics: it has no Pfam annotation to score against.
# ---------------------------------------------------------------------------
RUN = {
    "name": "midi-plus",
    "queries": 998,
    "query_sets": {"chr6": 964, "b2m": 1, "cd1_mr1": 6, "kir_lilr": 27},
    "truth_instances": 2_435,
    "truth_families": 1_089,
    "scored_targets": 9,
    "searched_only_targets": ["botryllus"],
    "started": "2026-09-01",
    "finished": "2026-09-02",
    "tasks": 4_723,
    "kmerseek_search_tasks": 3_656,
    "botryllus_searched": "2026-09-04 to 2026-09-10",
    "supersedes": "midi (964 chr6 queries, nine targets, 2026-08-25 to 08-27)",
}

RUN_SUMMARY = (
    "Run: QfO Pfam-region benchmark, midi-plus (2026-09-01 to 09-02, 4,723 tasks). "
    "998 human queries: 964 chromosome 6 proteins plus B2M, CD1A-E, MR1 and 27 KIR/LILR "
    "receptors. Targets: mouse, chicken, zebrafish, ciona, fly, worm, yeast, Arabidopsis "
    "and E. coli, all scored against Pfam; Botryllus schlosseri searched but unscored."
)

# ---------------------------------------------------------------------------
# Where the extracted midi-plus tables live.
#
# The run itself stays on Sherlock: `results/kmerseek` alone is hundreds of GB and the
# per-arm call files are 5.4 GB across 22,411 files. `scripts/extract_mhc.py` collapses
# that on the cluster to the five tables below, which together are small enough to sit on
# a laptop. The earlier midi extract is kept next door in `qfo-pfam-region-midi/`.
# ---------------------------------------------------------------------------
MIDI_DIR = Path("/Users/olga/data/qfo-pfam-region-midi-plus")

TRUTH_DIR = MIDI_DIR / "truth"
IDENTITY_DIR = MIDI_DIR / "identity"
EXTRACT_DIR = MIDI_DIR / "extract"

HUMAN_TRUTH = TRUTH_DIR / "human_domain_truth.parquet"
HUMAN_COVARIATES = TRUTH_DIR / "human_query_covariates.parquet"
QUERY_GENE_MAP = MIDI_DIR / "query_gene_map.parquet"
CHR6_GENE_MAP = (
    QUERY_GENE_MAP  # older name, kept so notebooks written on midi still run
)
CHR6_GENCODE = MIDI_DIR / "gencode_v50_chr6_genes.parquet"

GENE_LEVEL_ALL_ARMS = EXTRACT_DIR / "mhc_gene_level_all_arms.parquet"
DOMAIN_LEVEL_ALL_ARMS = EXTRACT_DIR / "mhc_domain_level_all_arms.parquet"
CHR6_CALLS_FOCUS = EXTRACT_DIR / "chr6_calls_focus_arms.parquet"
KMERSEEK_REGIONS = EXTRACT_DIR / "mhc_kmerseek_regions.parquet"
BASELINE_REGIONS = EXTRACT_DIR / "mhc_baseline_regions.parquet"
ALL_DOMAIN_METRICS = EXTRACT_DIR / "all_domain_metrics.parquet"
KSIZE_SWEEP = EXTRACT_DIR / "mhc_region_ksize_sweep.parquet"

# ---------------------------------------------------------------------------
# The extended MHC on GRCh38.
#
# Boundaries are Horton et al. 2004's xMHC partition, the standard one an immunologist will
# expect to see, in GRCh38 coordinates. They are bp intervals rather than
# `ou.region_by_anchor_genes` anchors on purpose: that function re-derives a *single* span
# from two anchor genes, which is the right tool for "where does the classical MHC start and
# stop", but the five-way class I / class III / class II split needs four internal
# boundaries that no pair of anchor genes defines.
#
# Assignment uses the gene midpoint, so a gene straddling a boundary lands on the side
# holding most of it instead of being dropped or double-counted.
# ---------------------------------------------------------------------------
XMHC_START = 25_726_063
XMHC_END = 33_400_644

MHC_SUBREGIONS: list[tuple[str, int, int]] = [
    ("extended class I", 25_726_063, 29_722_774),
    ("class I", 29_722_775, 31_371_356),
    ("class III", 31_371_357, 32_145_873),
    ("class II", 32_146_874, 33_080_775),
    ("extended class II", 33_080_776, 33_400_644),
]

MHC_SUBREGION_ORDER: list[str] = [name for name, _, _ in MHC_SUBREGIONS]

#: Telomere-to-centromere ordering, so a legend reads in the same direction as the map.
MHC_SUBREGION_COLORS: dict[str, str] = {
    "extended class I": "#F2C57C",
    "class I": "#D65F5F",
    "class III": "#8C6BB1",
    "class II": "#4878CF",
    "extended class II": "#9EC8E8",
}

#: Landmark genes worth labelling on a positional figure. Chosen for what an immunologist
#: uses to orient inside the region, not for how they score.
MHC_LANDMARKS: dict[str, str] = {
    "HFE": "extended class I",
    "BTN3A1": "extended class I",
    "TRIM26": "extended class I",
    "HLA-F": "class I",
    "HLA-G": "class I",
    "HLA-A": "class I",
    "MICA": "class I",
    "HLA-C": "class I",
    "HLA-B": "class I",
    "MICB": "class I",
    "TNF": "class III",
    "LTA": "class III",
    "C4A": "class III",
    "CFB": "class III",
    "C2": "class III",
    "NOTCH4": "class III",
    "HLA-DRA": "class II",
    "HLA-DRB1": "class II",
    "HLA-DQB1": "class II",
    "TAP1": "class II",
    "PSMB9": "class II",
    "HLA-DPB1": "class II",
    "COL11A2": "extended class II",
    "RXRB": "extended class II",
}


def assign_subregion(midpoint: pl.Expr) -> pl.Expr:
    """chr6 midpoint -> xMHC sub-region name, null outside the extended MHC.

    Built by folding the interval list in reverse so the first-listed (most telomeric)
    interval ends up as the outermost `when`, which keeps the generated expression's branch
    order the same as `MHC_SUBREGIONS` reads on the page.
    """
    expr = pl.lit(None, dtype=pl.String)
    for name, lo, hi in reversed(MHC_SUBREGIONS):
        expr = (
            pl.when((midpoint >= lo) & (midpoint <= hi))
            .then(pl.lit(name))
            .otherwise(expr)
        )
    return expr


# ---------------------------------------------------------------------------
# Target species.
#
# Divergence times are the pipeline's own (`main.nf` SPECIES), repeated rather than parsed
# so a notebook can order an axis without the Nextflow file being on the path. They are
# round numbers for axis ordering, not literature point estimates.
# ---------------------------------------------------------------------------
SPECIES_MYA: dict[str, int] = {
    "mouse": 100,
    "chicken": 300,
    "zebrafish": 430,
    "ciona": 550,
    "fly": 600,
    "worm": 650,
    "yeast": 900,
    "arabidopsis": 1500,
    "ecoli": 2000,
}

#: The nine targets that have Pfam annotations and therefore scored metrics.
SPECIES_ORDER: list[str] = list(SPECIES_MYA)

#: Botryllus schlosseri, a colonial tunicate at the same 550 Mya node as ciona. Searched
#: by every tool that needs only sequence, never scored: it is not in UniProt and has no
#: Pfam annotation in this run. Its hits are read in notebook 228 only.
BOTRYLLUS = "botryllus"
SPECIES_MYA_ALL: dict[str, int] = {**SPECIES_MYA, BOTRYLLUS: 550}
SEARCHED_SPECIES: list[str] = SPECIES_ORDER + [BOTRYLLUS]

SPECIES_LABELS: dict[str, str] = {
    "mouse": "mouse\n100",
    "chicken": "chicken\n300",
    "zebrafish": "zebrafish\n430",
    "ciona": "ciona\n550",
    "fly": "fly\n600",
    "worm": "worm\n650",
    "yeast": "yeast\n900",
    "arabidopsis": "arabidopsis\n1500",
    "ecoli": "E. coli\n2000",
    "botryllus": "Botryllus\n550 (unscored)",
}

#: Where adaptive immunity's own components stop existing, which is not where homology
#: detection stops. Jawed vertebrates (gnathostomes) invented MHC class I and II; the split
#: from the lamprey lineage is ~500 Mya, so mouse/chicken/zebrafish are the only three
#: targets that can hold a true MHC ortholog. Everything from ciona out can only hold a
#: more distant relative of the fold -- an IgSF or GRP94/HSP70-family protein -- and a hit
#: there is a statement about the fold, not about an MHC molecule.
GNATHOSTOME_SPECIES: list[str] = ["mouse", "chicken", "zebrafish"]
PRE_MHC_SPECIES: list[str] = ["ciona", "fly", "worm", "yeast", "arabidopsis", "ecoli"]

# ---------------------------------------------------------------------------
# Pfam families and the structural role they play inside an MHC molecule.
#
# Verified against this run's own `human_domain_truth.parquet`, not recalled: every class I
# and class II gene in the query set carries exactly two scored domains, a platform and a
# C1-set Ig. That pairing is what makes the within-molecule contrast a controlled one --
# same protein, same search, same species, two domains under very different selection.
# ---------------------------------------------------------------------------
PFAM_PLATFORM: dict[str, str] = {
    "PF00129": "class I α1/α2",
    "PF00993": "class II α1",
    "PF00969": "class II β1",
}

#: CD1's platform is its own Pfam family (MHC_I_3). MR1 keeps PF00129. Both are class I-like
#: molecules added in midi-plus; they are not in `ou.MHC_CLASSES` and are kept out of the
#: curated core, but their platforms are still platforms for the reachability figures.
PFAM_PLATFORM_LIKE: dict[str, str] = {"PF16497": "CD1 α1/α2"}

#: The Ig domains the KIR/LILR receptors carry (Ig, Ig_2, Ig_3). Receptors, not MHC
#: molecules; listed so `domain_role` can name them on a figure.
PFAM_IG_RECEPTOR: dict[str, str] = {
    "PF00047": "Ig",
    "PF13895": "Ig_2",
    "PF13927": "Ig_3",
}

PFAM_IG_C1 = "PF07654"
PFAM_CLASS_I_TAIL = "PF06623"

#: TAP1/TAP2's two domains. The peptide transporter is inside the class II region but is an
#: ABC transporter, a fold shared with bacteria -- so it is the positive control for "the
#: pipeline can still find things at 2,000 Mya", separating "this species is too far" from
#: "this domain is too fast-evolving".
PFAM_TAP: dict[str, str] = {
    "PF00664": "ABC membrane",
    "PF00005": "ABC ATPase",
}

DOMAIN_ROLE_COLORS: dict[str, str] = {
    "peptide-binding platform": "#D65F5F",
    "Ig C1-set": "#4878CF",
    "class I tail": "#cccccc",
    "ABC transporter": "#6ACC65",
    "CD1 platform": "#E1917F",
    "Ig (receptor)": "#9EC8E8",
}


def domain_role(pfam_id: pl.Expr) -> pl.Expr:
    """Pfam accession -> the role it plays in an MHC molecule, null for anything else."""
    return (
        pl.when(pfam_id.is_in(list(PFAM_PLATFORM)))
        .then(pl.lit("peptide-binding platform"))
        .when(pfam_id == PFAM_IG_C1)
        .then(pl.lit("Ig C1-set"))
        .when(pfam_id == PFAM_CLASS_I_TAIL)
        .then(pl.lit("class I tail"))
        .when(pfam_id.is_in(list(PFAM_TAP)))
        .then(pl.lit("ABC transporter"))
        .when(pfam_id.is_in(list(PFAM_PLATFORM_LIKE)))
        .then(pl.lit("CD1 platform"))
        .when(pfam_id.is_in(list(PFAM_IG_RECEPTOR)))
        .then(pl.lit("Ig (receptor)"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


# ---------------------------------------------------------------------------
# Tool display names and grouping.
# ---------------------------------------------------------------------------
TOOL_FAMILY: dict[str, str] = {
    "hmmer3_phmmer": "sequence",
    "hmmer3_jackhmmer": "sequence",
    "mmseqs2_seqseq": "sequence",
    "mmseqs2_iterative": "sequence",
    "hhblits": "profile",
    "prostt5": "predicted structure",
    "foldseek": "structure",
    "reseek": "structure",
    "folddisco": "structure",
    "kmerseek": "kmerseek",
}

TOOL_LABELS: dict[str, str] = {
    "hmmer3_phmmer": "phmmer",
    "hmmer3_jackhmmer": "jackhmmer",
    "mmseqs2_seqseq": "MMseqs2",
    "mmseqs2_iterative": "MMseqs2 iterative",
    "hhblits": "HHblits",
    "prostt5": "ProstT5",
    "foldseek": "Foldseek",
    "reseek": "Reseek",
    "folddisco": "Folddisco",
    "kmerseek": "kmerseek",
}

#: What each comparison tool is, spelled out once for figure footers and intros.
TOOL_FULL_NAMES: dict[str, str] = {
    "hmmer3_phmmer": "phmmer (HMMER3 single-sequence search)",
    "hmmer3_jackhmmer": "jackhmmer (HMMER3, 3 iterations)",
    "mmseqs2_seqseq": "MMseqs2 sequence-sequence (-s 7)",
    "mmseqs2_iterative": "MMseqs2 iterative profile search (-s 7)",
    "hhblits": "HHblits (single-sequence profile HMM)",
    "prostt5": "ProstT5 (3Di predicted from sequence, searched with Foldseek)",
    "foldseek": "Foldseek (3Di + amino acid, AlphaFold structures)",
    "reseek": "Reseek (very sensitive, AlphaFold structures)",
    "folddisco": "Folddisco (structural motif search, AlphaFold structures)",
    "kmerseek": "kmerseek (reduced-alphabet k-mer containment)",
}

#: The nine tools kmerseek is compared against, in the order the figures use.
COMPARISON_TOOLS: list[str] = [t for t in TOOL_LABELS if t != "kmerseek"]
#: The eight of those the pipeline scores on IoU. Folddisco is scored on coverage.
IOU_SCORED_TOOLS: list[str] = [t for t in COMPARISON_TOOLS if t != "folddisco"]

COMPARISON_TOOLS_TEXT: str = ", ".join(TOOL_LABELS[t] for t in COMPARISON_TOOLS)
IOU_SCORED_TOOLS_TEXT: str = ", ".join(TOOL_LABELS[t] for t in IOU_SCORED_TOOLS)

#: For figures that draw no search result at all.
NO_TOOL: str = (
    "none (no search result: drawn from the query set and Pfam annotations only)"
)

#: The designated best kmerseek arm, see feedback_best_ksize: hp_pbotc_1st_ed k=19.
BEST_KMERSEEK: tuple[str, int] = ("hp_pbotc_1st_ed2", 19)


def kmerseek_label(
    alphabet: str, ksize: int, lc: bool = True, short: bool = False
) -> str:
    """One kmerseek arm, named so alphabet and k are never implicit.

    ``short`` gives the axis form ``kmerseek hp_pbotc_1st_ed2 k=19``; the default gives the
    footer form with the low-complexity mask spelled out.
    """
    if short:
        return f"kmerseek {alphabet} k={ksize}" + ("" if lc else " (no LC mask)")
    mask = "low-complexity mask on" if lc else "no low-complexity mask"
    return f"kmerseek (alphabet {alphabet}, k={ksize}, {mask})"


def kmerseek_labels(arms: list[tuple[str, int]] | list[str], lc: bool = True) -> str:
    """Several kmerseek arms on one line: ``kmerseek: hp_pbotc_1st_ed2 k=19, k=21; ...``.

    Accepts ``(alphabet, k)`` pairs or variant strings like ``hp_pbotc_1st_ed2_k19_lcTrue``.
    Consecutive arms of one alphabet share the alphabet name.
    """
    pairs = []
    for a in arms:
        if isinstance(a, str):
            m = re.match(r"(.+?)_k(\d+)_lc(True|False)$", a)
            if m:
                pairs.append((m.group(1), int(m.group(2))))
            else:
                pairs.append((a, None))
        else:
            pairs.append(tuple(a))
    parts, last = [], None
    for alpha, k in pairs:
        kk = "" if k is None else f"k={k}"
        if alpha == last:
            parts[-1] += f", {kk}"
        else:
            parts.append(f"{alpha} {kk}".strip())
        last = alpha
    mask = "" if lc else " (no low-complexity mask)"
    return "kmerseek: " + "; ".join(parts) + mask


def arm_pretty(arm: str) -> str:
    """``tool.variant`` -> the label used on an axis, kmerseek keeping alphabet and k."""
    tool, _, variant = arm.partition(".")
    if tool == "kmerseek":
        m = re.match(r"(.+?)_k(\d+)_lc(True|False)$", variant)
        if m:
            return kmerseek_label(
                m.group(1), int(m.group(2)), m.group(3) == "True", short=True
            )
        return f"kmerseek {variant}"
    return TOOL_LABELS.get(tool, tool)


def tools_text(
    comparison: list[str] | None = None,
    kmerseek=None,
    lc: bool = True,
    note: str | None = None,
) -> str:
    """Build the TOOLS line for a figure footer.

    ``comparison`` is a list of pipeline tool ids (``hmmer3_phmmer`` ...), ``kmerseek`` a
    list of ``(alphabet, k)`` pairs or variant strings, or a free string such as
    ``"all 404 arms"``. Whichever is absent is said to be absent, so a reader can tell a
    kmerseek-free panel from one where kmerseek is merely unlabelled.
    """
    bits = []
    if comparison:
        bits.append(", ".join(TOOL_LABELS[t] for t in comparison))
    if kmerseek:
        bits.append(
            kmerseek if isinstance(kmerseek, str) else kmerseek_labels(kmerseek, lc=lc)
        )
    elif comparison:
        bits.append("no kmerseek in this figure")
    if not bits:
        bits.append(NO_TOOL)
    text = " | ".join(bits)
    return f"{text} ({note})" if note else text


TOOL_FAMILY_COLORS: dict[str, str] = {
    "sequence": "#999999",
    "profile": "#6ACC65",
    "predicted structure": "#B47CC7",
    "structure": "#4878CF",
    "kmerseek": "#D65F5F",
}

# ---------------------------------------------------------------------------
# One mark per meaning, shared by every notebook that draws the same thing.
#
# Three colour sets that never overlap inside one figure: tool families above; the five
# census classes below; and a single teal for the two scoring outcomes, told apart by
# fill rather than hue so the pair survives greyscale. Nothing else is green or teal.
# ---------------------------------------------------------------------------
#: The five census classes of notebook 226, in stacking order.
CENSUS_CLASS_ORDER: list[str] = [
    "family absent from target",
    "present, inaccessible by transfer",
    "reachable via length fraction",
    "reachable via single-domain target",
    "reachable via both",
]
CENSUS_CLASS_COLORS: dict[str, str] = {
    "family absent from target": "#333333",
    "present, inaccessible by transfer": "#D65F5F",
    "reachable via length fraction": "#F2C57C",
    "reachable via single-domain target": "#9EC8E8",
    "reachable via both": "#4878CF",
}

OUTCOME_TEAL: str = "#1B7F79"
#: Legend text for the two outcomes, with the definition in the label itself.
FOUND_LABEL: str = "found: at least half the true domain lies inside a call"
BOUNDARY_LABEL: str = "boundary matched: a call overlaps the true domain at IoU >= 0.5"


def outcome_style(kind: str) -> dict:
    """Bar style for ``"found"`` (hollow teal) or ``"boundary matched"`` (filled teal)."""
    if kind == "found":
        return dict(facecolor="none", edgecolor=OUTCOME_TEAL, linewidth=1.6)
    return dict(facecolor=OUTCOME_TEAL, edgecolor=OUTCOME_TEAL, linewidth=0)


def legend_above(
    ax, ncol: int = 2, title_pad: float | None = None, fontsize: float = 8.5, **kw
):
    """Put the legend between the axes title and the plot, so it is read before the marks.

    ``title_pad`` lifts the title clear of the legend; one legend row needs about 16 pt,
    two rows about 30 pt. Call after ``ax.set_title``.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    rows = -(-len(labels) // ncol)
    if title_pad is None:
        title_pad = 12 + 14 * rows
    ax.set_title(ax.get_title(), pad=title_pad)
    return ax.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.0),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        borderaxespad=0.0,
        handlelength=1.6,
        **kw,
    )


def arm_label(tool: pl.Expr, variant: pl.Expr) -> pl.Expr:
    """`tool`/`variant` -> the short string used on axes, e.g. ``kmerseek hp_pbotc_1st_ed2 k=19``.

    kmerseek keeps its alphabet and k because they *are* the arm; every other tool has one
    variant in this run, so repeating it on an axis is noise.
    """
    alphabet = variant.str.replace(r"_k\d+_lc(True|False)$", "")
    k = variant.str.extract(r"_k(\d+)_lc", 1)
    mask = (
        pl.when(variant.str.ends_with("lcFalse"))
        .then(pl.lit(" (no LC mask)"))
        .otherwise(pl.lit(""))
    )
    return (
        pl.when(tool == "kmerseek")
        .then(pl.lit("kmerseek ") + alphabet + pl.lit(" k=") + k + mask)
        .otherwise(tool.replace_strict(TOOL_LABELS, default=tool))
    )


# ---------------------------------------------------------------------------
# Figure finishing: every saved figure states which tool it shows, the hypothesis it
# tests and the conclusion it supports, on the image itself, so a panel pasted into a
# talk carries its own provenance.
#
# The text is placed outside the axes area in figure coordinates (above 1 and below 0)
# and relies on `savefig.bbox = "tight"` (set in every notebook's rcParams) and the inline
# backend's own tight bbox to be included. That keeps it out of the way of tight_layout
# and of colorbars added with `fig.colorbar(ax=...)`.
# ---------------------------------------------------------------------------
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
    layout: bool = True,
    dpi: int = 200,
    wrap: int | None = None,
) -> None:
    """Stamp TOOLS / hypothesis / conclusion on `fig`, then save it to `path`.

    ``tools`` is the string from `tools_text` (or `kmerseek_label`, `NO_TOOL`). ``title``
    replaces `fig.suptitle`: pass the figure's title here so it stacks cleanly under the
    tools line. ``footer_y`` moves the footer down when a legend already hangs below the
    axes. ``layout=False`` skips tight_layout for figures whose colorbars dislike it.
    """
    import textwrap

    if layout:
        try:
            fig.tight_layout()
        except Exception:  # noqa: BLE001  (a colorbar layout warning is not a failure)
            pass
    w_in, h_in = fig.get_size_inches()
    wrap = wrap or max(70, int(w_in * 12))
    line = lambda pt: pt / 72.0 / h_in * 1.45  # one text line, in figure fraction

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


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------
#: Labels for the query sets midi-plus added on top of chr6, in the order figures use.
QUERY_SET_LABELS: dict[str, str] = {
    "chr6": "chromosome 6",
    "b2m": "B2M (light chain, chr15)",
    "cd1_mr1": "CD1A-E + MR1 (class I-like, chr1)",
    "kir_lilr": "KIR + LILR receptors (chr19)",
}


def load_query_gene_map() -> pl.DataFrame:
    """All 998 midi-plus query proteins: covariates, query set, chromosome, xMHC sub-region.

    Chromosome 6 queries carry GENCODE v50 coordinates; seven of them have no GENCODE chr6
    gene record under their HGNC symbol and carry null coordinates but are kept, so counts
    still add up to the query-set size the run reports. The 34 off-chr6 additions carry the
    chromosome from their HGNC cytoband and no coordinates. `mhc_class_ext` extends
    `ou.MHC_CLASSES` with the two added groups so a heatmap can show them as their own block.
    """
    gm = pl.read_parquet(QUERY_GENE_MAP)
    return gm.with_columns(
        mhc_class_ext=pl.when(pl.col("mhc_class").is_not_null())
        .then(pl.col("mhc_class"))
        .when(pl.col("query_set") == "cd1_mr1")
        .then(pl.lit("I-like (CD1/MR1)"))
        .when(pl.col("query_set") == "kir_lilr")
        .then(pl.lit("receptor (KIR/LILR)"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def load_chr6_gene_map() -> pl.DataFrame:
    """Older name for `load_query_gene_map`; returns all 998 queries, not only chr6."""
    return load_query_gene_map()


def load_mhc_extras(gene_map: pl.DataFrame | None = None) -> pl.DataFrame:
    """The 34 queries midi-plus added off chromosome 6 (B2M, CD1A-E, MR1, KIR, LILR)."""
    gm = load_query_gene_map() if gene_map is None else gene_map
    return gm.filter(pl.col("query_set") != "chr6")


def load_mhc_window(gene_map: pl.DataFrame | None = None) -> pl.DataFrame:
    """The chr6 queries inside the extended MHC."""
    gm = load_chr6_gene_map() if gene_map is None else gene_map
    return gm.filter(pl.col("mhc_subregion").is_not_null())


def load_human_truth() -> pl.DataFrame:
    return pl.read_parquet(HUMAN_TRUTH)


def load_gene_level() -> pl.DataFrame:
    """One row per (arm, species, gene) for the 221 MHC-window genes plus the 34 extras."""
    return pl.read_parquet(GENE_LEVEL_ALL_ARMS)


def load_domain_level() -> pl.DataFrame:
    """One row per (arm, species, gene, Pfam family) for the 25 core genes plus CD1/MR1."""
    return pl.read_parquet(DOMAIN_LEVEL_ALL_ARMS)


def load_target_domain_map(species: str) -> pl.DataFrame:
    """Pfam domain instances on a target proteome -- what is *there* to be found."""
    return pl.read_parquet(TRUTH_DIR / f"{species}_domain_map.parquet")


def reachable_families(species: str, pfam_ids: list[str]) -> set[str]:
    """Which of `pfam_ids` exist at all in `species`.

    A recall number that ignores this punishes a tool for not finding a domain the target
    proteome does not contain. Class I and class II platforms are absent outside jawed
    vertebrates by definition, so almost every MHC recall figure needs this denominator.
    """
    dm = load_target_domain_map(species)
    return set(
        dm.filter(pl.col("pfam_id").is_in(pfam_ids))["pfam_id"].unique().to_list()
    )


# ---------------------------------------------------------------------------
# Target-side identity and position, for the synteny notebook.
#
# The region tables carry a target *protein* (`sp|ACC|NAME description`), and synteny needs
# a target *locus*. The bridge is the QfO FASTA header's own `GN=` field: UniProt already
# resolved each accession to a gene symbol, so no ID-mapping service has to be called and
# nothing depends on a web API being up. Symbols are then looked up in Ensembl 116.
#
# Only mouse, chicken and zebrafish get coordinates, and that is a scoping decision rather
# than an omission: notebook 221 shows no MHC molecule is detected in any target beyond
# zebrafish, so a syntenic block cannot exist to be found further out.
# ---------------------------------------------------------------------------
TARGET_DIR = MIDI_DIR / "targets"

#: QfO reference-proteome accession per species, from the midi run's own `qfo/` directory.
PROTEOME_ID: dict[str, str] = {
    "human": "UP000005640_9606",
    "mouse": "UP000000589_10090",
    "chicken": "UP000000539_9031",
    "zebrafish": "UP000000437_7955",
    "ciona": "UP000008144_7719",
    "fly": "UP000000803_7227",
    "worm": "UP000001940_6239",
    "yeast": "UP000002311_559292",
    "arabidopsis": "UP000006548_3702",
    "ecoli": "UP000000625_83333",
}

#: Species with an Ensembl 116 gene-coordinate table pulled locally.
ENSEMBL_SPECIES: dict[str, str] = {
    "mouse": "mus_musculus",
    "chicken": "gallus_gallus",
    "zebrafish": "danio_rerio",
}

#: Where each species keeps its MHC. Used only to orient a figure, never to select data.
MHC_LOCUS: dict[str, tuple[str, str]] = {
    "mouse": ("17", "H2 complex"),
    "chicken": ("16", "B locus (minimal essential MHC)"),
    "zebrafish": ("19", "class I core; class II is on other chromosomes"),
}


def uniprot_acc(col: pl.Expr) -> pl.Expr:
    """`sp|ACC|NAME description` -> ACC, leaving a bare accession untouched.

    Coalesce rather than when/then, because polars evaluates both arms of a when/then over
    every row before selecting, and the extract arm throws on rows with no pipe.
    """
    return pl.coalesce(col.str.extract(r"^[^|]*\|([^|]+)\|", 1), col)


def load_accession_to_gene(species: str) -> pl.DataFrame:
    """UniProt accession -> gene symbol, parsed from the QfO FASTA `GN=` field.

    Entries with no `GN=` keep an empty symbol and are dropped here; they cannot be placed
    on a chromosome, and silently carrying them would make an unplaceable hit look like a
    hit that landed nowhere interesting.
    """
    path = TARGET_DIR / "genemap" / f"{PROTEOME_ID[species]}.tsv"
    return (
        pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=["target_acc", "gene_symbol"],
            infer_schema_length=0,
        )
        .filter(pl.col("gene_symbol").is_not_null() & (pl.col("gene_symbol") != ""))
        .unique(subset="target_acc")
    )


def load_target_gene_coordinates(species: str) -> pl.DataFrame:
    """Ensembl 116 gene records for a target species.

    Read entirely as text and cast afterwards: seqnames include X, Y, MT and scaffold
    names, so inferring the column type from the first rows commits to i64 and then dies
    on chromosome X.
    """
    path = TARGET_DIR / "gtf" / f"{ENSEMBL_SPECIES[species]}.genes.tsv"
    cols = ["chrom", "start", "end", "strand", "gene_id", "gene_name", "biotype"]
    return (
        pl.read_csv(
            path,
            separator="\t",
            has_header=False,
            new_columns=cols,
            infer_schema_length=0,
        )
        .with_columns(pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64))
        .filter(pl.col("gene_name").is_not_null())
        .with_columns(midpoint=(pl.col("start") + pl.col("end")) // 2)
    )


def target_locus_table(species: str) -> pl.DataFrame:
    """target accession -> (gene symbol, chrom, start, end, midpoint) for one species.

    Symbol matching is case-insensitive: UniProt writes mouse symbols as `H2-D1` and
    zebrafish as `mhc1uba`, and Ensembl's capitalisation does not always agree.
    """
    genes = load_target_gene_coordinates(species).with_columns(
        _key=pl.col("gene_name").str.to_lowercase()
    )
    acc = load_accession_to_gene(species).with_columns(
        _key=pl.col("gene_symbol").str.to_lowercase()
    )
    return acc.join(genes.unique(subset="_key"), on="_key", how="inner").select(
        "target_acc",
        "gene_symbol",
        "chrom",
        "start",
        "end",
        "midpoint",
        "strand",
        "biotype",
    )


# ---------------------------------------------------------------------------
# Notebook 223/224 additions: per-call geometry and the reachability denominator.
#
# The three collapsed tables above answer "how much did this arm recover". They cannot
# answer "where exactly did the call land", because `mhc_domain_level_all_arms` reports
# `qstart` as a MIN and `qend` as a MAX over every call an arm made on that (gene, family)
# -- a union of intervals, not one call. Anything about boundary geometry has to be read
# off the raw call rows.
#
# One property of the raw table drives all of this: `true_start`/`true_end` are stamped
# ONLY on calls the pipeline assigned to a truth instance, and assignment already requires
# clearing the IoU cut (see `assign_instances` in evaluate_domain_calls.py). Filtering the
# raw table on `true_start.is_not_null()` therefore selects the calls that passed, which is
# exactly the wrong sample for asking why the others failed. Every function below joins the
# human truth table on itself instead.
# ---------------------------------------------------------------------------
PFAM_NAMES_TSV = MIDI_DIR / "pfam_a_names.tsv"


def load_pfam_names() -> pl.DataFrame:
    """Pfam accession -> (short name, description), parsed from the local Pfam-A HMM library.

    Built once by pulling the ACC/NAME/DESC triples out of `Pfam-A.hmm`; the accession is
    stripped of its version suffix so it joins against the run's `pfam_id`. To regenerate::

        grep -E "^(NAME|ACC|DESC) " ~/data/pfam/Pfam-A.hmm \
          | awk '{k=$1; $1=""; sub(/^ +/,""); if(k=="NAME") n=$0; else if(k=="ACC") a=$0;
                  else {split(a,p,"."); print p[1]"\t"n"\t"$0}}' \
          > ~/data/qfo-pfam-region-midi/pfam_a_names.tsv
    """
    return pl.read_csv(
        PFAM_NAMES_TSV,
        separator="\t",
        has_header=False,
        new_columns=["pfam_id", "pfam_name", "pfam_desc"],
        infer_schema_length=0,
    )


def pfam_label(pfam_ids: list[str]) -> dict[str, str]:
    """`{"PF00129": "PF00129 MHC_I"}` for axis ticks, falling back to the bare accession."""
    names = dict(load_pfam_names().select("pfam_id", "pfam_name").iter_rows())
    return {p: f"{p} {names[p]}" if p in names else p for p in pfam_ids}


def focus_calls() -> pl.LazyFrame:
    """Lazy handle on the 89M raw call rows. Always filter before collecting."""
    return pl.scan_parquet(CHR6_CALLS_FOCUS)


def focus_arms() -> pl.DataFrame:
    """The (tool, variant) pairs the raw call table carries."""
    return (
        focus_calls()
        .select("tool", "variant")
        .unique()
        .collect()
        .sort("tool", "variant")
    )


def core_truth() -> pl.DataFrame:
    """The true (gene, Pfam family) pairs on the curated MHC molecules, with role labels.

    Curated means `ou.MHC_CLASSES`: 24 chr6 genes plus B2M, which midi-plus added as its
    own query set (it is on chr15). CD1/MR1 and KIR/LILR are not in here; see
    `extras_truth`.
    """
    core = load_query_gene_map().filter(pl.col("mhc_class").is_not_null())
    truth = load_human_truth()
    return (
        truth.join(
            core.select(
                accession="accession", hgnc_symbol="hgnc_symbol", mhc_class="mhc_class"
            ),
            on="accession",
            how="inner",
        )
        .rename({"accession": "query_acc"})
        .with_columns(role=domain_role(pl.col("pfam_id")))
    )


def extras_truth() -> pl.DataFrame:
    """The true (gene, Pfam family) pairs on the 34 off-chr6 additions, with role labels."""
    extras = load_mhc_extras()
    return (
        load_human_truth()
        .join(
            extras.select(
                accession="accession",
                hgnc_symbol="hgnc_symbol",
                query_set="query_set",
                mhc_class_ext="mhc_class_ext",
            ),
            on="accession",
            how="inner",
        )
        .rename({"accession": "query_acc"})
        .with_columns(role=domain_role(pl.col("pfam_id")))
    )


def window_truth() -> pl.DataFrame:
    """Every true domain on a gene inside the extended MHC window."""
    win = load_mhc_window().select(accession="accession", hgnc_symbol="hgnc_symbol")
    return (
        load_human_truth()
        .join(win, on="accession", how="inner")
        .rename({"accession": "query_acc"})
    )


def domain_grid(
    truth: pl.DataFrame, species: list[str], arms: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Zero-filled (true domain x arm x species) grid with the best IoU and cover reached.

    The raw call table has no row at all for a (gene, family, arm) an arm never called on,
    so a plain group-by silently drops those cells and shrinks the recall denominator to
    the domains the arm already found. The cross join puts every cell back and fills the
    misses as 0, which is what "not recovered" means.
    """
    arms = focus_arms() if arms is None else arms
    accs = truth["query_acc"].unique().to_list()
    grid = truth.join(arms, how="cross").join(
        pl.DataFrame({"species": species}), how="cross"
    )
    best = (
        focus_calls()
        .filter(pl.col("query_acc").is_in(accs) & pl.col("species").is_in(species))
        .group_by("species", "tool", "variant", "query_acc", "pfam_id")
        .agg(
            best_iou=pl.col("iou").max(),
            best_cover=pl.col("cover").max(),
            n_calls=pl.len(),
        )
        .collect()
    )
    return grid.join(
        best, on=["species", "tool", "variant", "query_acc", "pfam_id"], how="left"
    ).with_columns(
        best_iou=pl.col("best_iou").fill_null(0.0),
        best_cover=pl.col("best_cover").fill_null(0.0),
        n_calls=pl.col("n_calls").fill_null(0),
        arm=pl.col("tool") + "." + pl.col("variant"),
    )


def call_offsets(species: str, accessions: list[str] | None = None) -> pl.DataFrame:
    """Per-call boundary error against the true domain the call overlaps.

    One row per raw call that (a) carries a Pfam family the query protein genuinely has and
    (b) overlaps that domain by at least one residue. `d_start` is `qstart - true_start` and
    `d_end` is `qend - true_end`, both in residues, so a call wider than the domain on both
    sides is negative then positive.

    A query can carry several instances of one family; the call is scored against the
    instance it overlaps most, not the first one the join happens to emit.
    """
    truth = load_human_truth().select(
        query_acc="accession",
        pfam_id="pfam_id",
        t_start="domain_start",
        t_end="domain_end",
        prot_len="protein_length",
    )
    calls = focus_calls().filter(pl.col("species") == species)
    if accessions is not None:
        calls = calls.filter(pl.col("query_acc").is_in(accessions))
    joined = (
        calls.select(
            "query_acc",
            "pfam_id",
            "qstart",
            "qend",
            "score",
            "iou",
            "cover",
            "is_gray",
            "tool",
            "variant",
        )
        .join(truth.lazy(), on=["query_acc", "pfam_id"], how="inner")
        .with_columns(
            overlap=(
                pl.min_horizontal("qend", "t_end")
                - pl.max_horizontal("qstart", "t_start")
            ).clip(lower_bound=0)
        )
        .filter(pl.col("overlap") > 0)
        # Total sort key. `overlap` alone leaves ties to polars' arbitrary row order, so a
        # call overlapping two instances of one family equally picks a different one each
        # run, and every d_start/d_end derived from it moves. (t_start, t_end) identifies
        # the instance, so adding them makes the choice deterministic.
        .sort(["overlap", "t_start", "t_end"], descending=[True, False, False])
        .group_by("tool", "variant", "query_acc", "pfam_id", "qstart", "qend")
        .agg(pl.all().first())
    )
    return joined.collect().with_columns(
        d_start=pl.col("qstart") - pl.col("t_start"),
        d_end=pl.col("qend") - pl.col("t_end"),
        arm=pl.col("tool") + "." + pl.col("variant"),
    )


def arm_ksize(variant: pl.Expr) -> pl.Expr:
    """kmerseek variant string -> its k, null for a variant with no `_k<N>_` field."""
    return variant.str.extract(r"_k(\d+)_", 1).cast(pl.Int64)


def target_family_counts(
    pfam_ids: list[str], species: list[str] | None = None
) -> pl.DataFrame:
    """How many proteins in each target proteome carry each family.

    Zero means the family is not annotated in that proteome, so no tool can score a true
    positive against it there no matter how well it searches. This is the denominator the
    cross-species figures need.
    """
    species = SPECIES_ORDER if species is None else species
    rows = []
    for sp in species:
        dm = load_target_domain_map(sp)
        cnt = dict(
            dm.filter(pl.col("pfam_id").is_in(pfam_ids))
            .group_by("pfam_id")
            .agg(pl.col("accession").n_unique().alias("n"))
            .iter_rows()
        )
        rows += [
            {"species": sp, "pfam_id": p, "n_target_proteins": cnt.get(p, 0)}
            for p in pfam_ids
        ]
    return pl.DataFrame(rows)
