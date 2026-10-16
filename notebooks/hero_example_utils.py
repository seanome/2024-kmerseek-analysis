"""Loaders and helpers for notebook 244 (hero example candidates on the Swiss-Prot key).

Every table here comes from one of three places, named at each loader:

* ``scripts/reduce_swissprot_instance_landing.py``: one row per (arm, species, human
  Swiss-Prot instance) with the best call, the best landing call and its target protein.
  Built on Sherlock from the midi-plus region tables with the pipeline's own
  ``load_regions`` / ``dedup_fragment_regions`` / transfer rule.
* ``scripts/prep_244_instance_covariates.py``: feature notes, pLDDT, metapredict disorder
  and the Kyte-Doolittle scan, per human instance.
* The truth key ``truth_swissprot/human_swissprot_truth.parquet`` that notebook 231 reads
  (``bin/build_swissprot_truth.py``).

Coordinates. Truth intervals are 1-based and inclusive. kmerseek region coordinates are
0-based and end-exclusive (checked in notebook 244: under that reading the query and target
class strings are identical at every position of all 300 regions tested). The landing
fractions use the pipeline's own arithmetic (``overlap_expr``), which compares the two
directly, as the benchmark does. Residues printed and drawn use each convention correctly.
"""

from __future__ import annotations

import glob
import hashlib
import math
import re
from pathlib import Path

import numpy as np
import polars as pl

import mhc_region_utils as mu
import swissprot_control_utils as su

MIDI = mu.MIDI_DIR
EXTRACT = MIDI / "extract"
TRUTH = MIDI / "truth_swissprot" / "human_swissprot_truth.parquet"
LANDING_DIR = MIDI / "landing_swissprot"
QFO_DIR = Path(
    "/Users/olga/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
)

SPECIES = mu.SPECIES_ORDER
SPECIES_MYA = mu.SPECIES_MYA

#: The landing rule of criterion 2, as in reduce_swissprot_instance_landing.py.
INSIDE_MIN = 0.8
COVER_MIN = 0.3

#: Comparison arms. The first three search structure (AlphaFold models or 3Di strings).
STRUCTURE_ARMS = {
    "foldseek.3di_aa": "Foldseek",
    "prostt5.3di_from_seq": "ProstT5",
    "reseek.verysensitive": "Reseek",
}
SEQUENCE_ARMS = {
    "hmmer3_phmmer.default": "phmmer",
    "mmseqs2_seqseq.s7": "MMseqs2",
    "mmseqs2_iterative.s7": "MMseqs2 iterative",
}
COMPARISON_ARMS = {**STRUCTURE_ARMS, **SEQUENCE_ARMS}

#: Notebook 231's composition types (swissprot_control_utils.COMPOSITION_TYPES). Used only
#: as a label on each row, never as a filter.
COMPOSITION_TYPES = su.COMPOSITION_TYPES

#: Criterion 6. Feature types a biologist knows by name, in the order they rank. REGION
#: is split by its note: a named region ranks with the others, "Disordered" and
#: "Interaction with ..." rank last.
NAMED_TYPE_ORDER = [
    "ZN_FING",
    "MOTIF",
    "DOMAIN",
    "REPEAT",
    "DNA_BIND",
    "COILED",
    "TRANSMEM",
    "INTRAMEM",
    "REGION (named)",
]


# ---------------------------------------------------------------------------
# Arms.
# ---------------------------------------------------------------------------
_ARM_RE = re.compile(r"^kmerseek\.(?P<alpha>.+)_k(?P<k>\d+)_lc(?P<lc>True|False)$")


def alphabet_size(alphabet: str) -> int:
    """Number of residue classes: the number at the end of the alphabet's name
    (hp_pbotc_1st_ed2 -> 2, dayhoff6 -> 6, protein20 -> 20, hsdm17 -> 17)."""
    return int(re.search(r"(\d+)$", alphabet).group(1))


def arm_fields(arms: pl.Series) -> pl.DataFrame:
    """kmerseek arm name -> alphabet, k, mask on/off, bits per position and per seed.

    Bits per seed is k x log2(number of classes), the axis notebook 240 uses: how much a
    single exact k-mer match says about the sequence.
    """
    rows = []
    for a in arms.unique().to_list():
        m = _ARM_RE.match(a)
        if not m:
            continue
        n = alphabet_size(m["alpha"])
        k = int(m["k"])
        rows.append(
            {
                "arm": a,
                "alphabet": m["alpha"],
                "k": k,
                "mask_on": m["lc"] == "True",
                "n_classes": n,
                "bits_per_seed": round(k * math.log2(n), 2),
            }
        )
    return pl.DataFrame(rows)


def arm_short(arm: str) -> str:
    """kmerseek.hp_pbotc_1st_ed2_k19_lcTrue -> 'hp_pbotc_1st_ed2 k19'; mask off gets a tag."""
    m = _ARM_RE.match(arm)
    if not m:
        return COMPARISON_ARMS.get(arm, arm)
    tag = "" if m["lc"] == "True" else " (mask off)"
    return f"{m['alpha']} k{m['k']}{tag}"


# ---------------------------------------------------------------------------
# Query halves.
# ---------------------------------------------------------------------------
def query_halves() -> pl.DataFrame:
    """Split the 998 query proteins into a "choose" half and a "report" half.

    Notebook 231 has no query split (its held-out split is by feature type), so this one is
    new. The unit is the HGNC gene group, so paralogs in one group (the HLA class I genes,
    the KIR receptors, the butyrophilins) never sit on both sides; a protein with no group
    is its own unit. The half is the parity of the first byte of the SHA-1 of the group
    name, so it is the same on every run and on every machine.
    """
    q = pl.read_parquet(mu.QUERY_GENE_MAP).select(
        "accession", "hgnc_symbol", "hgnc_gene_group"
    )
    unit = q.with_columns(
        split_unit=pl.coalesce(
            pl.col("hgnc_gene_group"), pl.col("hgnc_symbol"), pl.col("accession")
        )
    )
    halves = {
        u: ("choose" if hashlib.sha1(u.encode()).digest()[0] % 2 == 0 else "report")
        for u in unit["split_unit"].unique().to_list()
    }
    return unit.with_columns(
        half=pl.col("split_unit").replace_strict(halves, return_dtype=pl.String)
    )


# ---------------------------------------------------------------------------
# Instances.
# ---------------------------------------------------------------------------
def load_instances() -> pl.DataFrame:
    """Every human range instance in 231's truth key, with the columns 244 ranks on.

    Point features (is_point) are dropped: a 1-2 residue instance cannot hold 80% of any
    call. The rest carry the Swiss-Prot note, the query half, 231's feature-type split as
    a label, pLDDT and metapredict disorder over the instance, and whether the
    Kyte-Doolittle scan landed on it.
    """
    key = ["accession", "pfam_id", "domain_start", "domain_end"]
    t = pl.read_parquet(TRUTH).filter(~pl.col("is_point"))
    # Two Swiss-Prot features can share a type and coordinates and differ in their note
    # (two ligands on one binding site); the key keeps one instance, so their notes merge.
    notes = (
        pl.read_parquet(EXTRACT / "244_swissprot_feature_notes.parquet")
        .group_by(key)
        .agg(pl.col("note").drop_nulls().unique().sort().str.join(" / "))
        .with_columns(
            pl.when(pl.col("note") == "")
            .then(None)
            .otherwise(pl.col("note"))
            .alias("note")
        )
    )
    plddt = pl.read_parquet(EXTRACT / "244_swissprot_region_plddt.parquet")
    dis = pl.read_parquet(EXTRACT / "244_swissprot_region_disorder.parquet")
    kd = pl.read_parquet(EXTRACT / "244_kd_scan_instances.parquet").rename(
        {
            "query_acc": "accession",
            "true_start": "domain_start",
            "true_end": "domain_end",
        }
    )
    h = query_halves().select("accession", "hgnc_symbol", "split_unit", "half")
    loc = ["accession", "domain_start", "domain_end"]
    out = (
        t.join(notes, on=key, how="left")
        .join(h, on="accession", how="left")
        .join(
            plddt.select(loc + ["mean_plddt_region", "frac_plddt_lt70_region"]),
            on=loc,
            how="left",
        )
        .join(
            dis.select(loc + ["disorder_fraction_region", "mean_disorder_region"]),
            on=loc,
            how="left",
        )
        .join(kd, on=key, how="left")
        .with_columns(
            feature_length=pl.col("domain_end") - pl.col("domain_start") + 1,
            kd_landed=pl.col("kd_landed").fill_null(False),
            type_split=pl.when(pl.col("pfam_id").is_in(COMPOSITION_TYPES))
            .then(pl.lit("composition"))
            .otherwise(pl.lit("held out")),
        )
    )
    assert out.height == t.height, (out.height, t.height)
    return out.with_columns(
        feature_label=feature_label(pl.col("pfam_id"), pl.col("note"))
    )


def feature_label(ftype: pl.Expr, note: pl.Expr) -> pl.Expr:
    """Criterion 6 class of a feature: its type, with REGION split into named and not."""
    unnamed = (
        note.is_null()
        | (note == "Disordered")
        | note.str.starts_with("Interaction with")
    )
    return (
        pl.when((ftype == "REGION") & unnamed)
        .then(pl.lit("REGION (unnamed)"))
        .when(ftype == "REGION")
        .then(pl.lit("REGION (named)"))
        .otherwise(ftype)
    )


def type_rank(label: pl.Expr) -> pl.Expr:
    """Rank of a feature_label under criterion 6; lower is better, unnamed REGION last."""
    order = {name: i for i, name in enumerate(NAMED_TYPE_ORDER)}
    return (
        pl.when(label == "REGION (unnamed)")
        .then(pl.lit(99))
        .otherwise(label.replace_strict(order, default=50, return_dtype=pl.Int64))
    )


# ---------------------------------------------------------------------------
# Landing tables.
# ---------------------------------------------------------------------------
def load_landing(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    """All <arm>.<species>.instances.parquet files from the reduction, stacked.

    The instance key is renamed to the truth key's names (accession, pfam_id,
    domain_start, domain_end). Missing (arm, species) files are reported, not skipped
    silently.
    """
    files = sorted(Path(landing_dir).glob("*.instances.parquet"))
    parts = [pl.read_parquet(f) for f in files]
    parts = [p for p in parts if p.height]
    d = pl.concat(parts, how="diagonal_relaxed")
    return d.rename(
        {
            "query_acc": "accession",
            "true_start": "domain_start",
            "true_end": "domain_end",
        }
    )


def load_calls_by_type(landing_dir: Path = LANDING_DIR) -> pl.DataFrame:
    files = sorted(Path(landing_dir).glob("*.calls_by_type.parquet"))
    parts = [pl.read_parquet(f) for f in files]
    return pl.concat([p for p in parts if p.height], how="diagonal_relaxed")


def classify_comparison(km_iou: pl.Expr, prefix: str = "") -> pl.Expr:
    """Criterion 3 category of one comparison arm on one instance.

    Expects the landing table's columns under ``prefix``: best_iou (highest IoU of any call
    of that type overlapping the instance), any_inside_half (some call has >= 50% of its
    length inside the instance), n_overlapping_calls.

      no call              no call of that type overlaps the instance
      spills               every overlapping call has < 50% of its length inside
      inside, lower IoU    a call sits mostly inside, but its IoU is below kmerseek's
      equal or higher IoU  some call reaches kmerseek's IoU (criterion 3 fails)
    """
    n = pl.col(f"{prefix}n_overlapping_calls").fill_null(0)
    iou = pl.col(f"{prefix}best_iou").fill_null(0.0)
    inside = pl.col(f"{prefix}any_inside_half").fill_null(False)
    return (
        pl.when(n == 0)
        .then(pl.lit("no call"))
        .when(iou >= km_iou)
        .then(pl.lit("equal or higher IoU"))
        .when(inside)
        .then(pl.lit("inside, lower IoU"))
        .otherwise(pl.lit("spills"))
    )


# ---------------------------------------------------------------------------
# Sequences.
# ---------------------------------------------------------------------------
_SEQ_CACHE: dict[str, dict[str, str]] = {}


def proteome_fasta(species: str) -> Path:
    hits = glob.glob(str(QFO_DIR / "*" / f"{mu.PROTEOME_ID[species]}.fasta"))
    if len(hits) != 1:
        raise FileNotFoundError(f"{species}: expected one QfO FASTA, found {hits}")
    return Path(hits[0])


def sequences(species: str, accessions: set[str]) -> dict[str, str]:
    """Protein sequences from the QfO release the run searched (human included)."""
    cache = _SEQ_CACHE.setdefault(species, {})
    need = set(accessions) - set(cache)
    if need:
        cache.update(su.read_fasta(proteome_fasta(species), need))
    return {a: cache[a] for a in accessions if a in cache}


def region_residues(
    q_seq: str, t_seq: str, qstart: int, qend: int, tstart: int, tend: int
) -> tuple[str, str]:
    """The query and target residues of a kmerseek region (0-based, end-exclusive)."""
    return q_seq[qstart:qend], t_seq[tstart:tend]


def identity(a: str, b: str) -> tuple[int, int]:
    """(identical positions, length) of a gapless pair."""
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])), n


# ---------------------------------------------------------------------------
# Alignment printout.
# ---------------------------------------------------------------------------
HP_CLASSES = {"hp_pbotc_1st_ed2": ["ACFILMPVWY", "DEGHKNQRST"]}


def class_string(
    seq: str, alphabet: str = "hp_pbotc_1st_ed2", letters: str = "HP"
) -> str:
    groups = HP_CLASSES[alphabet]
    table = {r: letters[i] for i, g in enumerate(groups) for r in g}
    return "".join(table.get(c, "?") for c in seq)


def longest_equal_run(a: str, b: str) -> tuple[int, int]:
    """(start, length) of the longest stretch where a and b agree position by position."""
    best, cur, start, best_start = 0, 0, 0, 0
    for i, (x, y) in enumerate(zip(a, b)):
        if x == y:
            if cur == 0:
                start = i
            cur += 1
            if cur > best:
                best, best_start = cur, start
        else:
            cur = 0
    return best_start, best


def format_alignment(
    q_name: str,
    t_name: str,
    q_seq: str,
    t_seq: str,
    qstart: int,
    qend: int,
    tstart: int,
    tend: int,
    width: int = 60,
) -> str:
    """The residues of a gapless region, one above the other, then the HP class strings.

    Coordinates printed are 1-based and inclusive on both proteins. The match line puts
    '|' under an identical residue. The class block marks the longest run of identical
    classes with '^' underneath.
    """
    qa, ta = region_residues(q_seq, t_seq, qstart, qend, tstart, tend)
    n_id, n = identity(qa, ta)
    qc, tc = class_string(qa), class_string(ta)
    run_start, run_len = longest_equal_run(qc, tc)
    lab = max(len(q_name), len(t_name), len("hp_pbotc_1st_ed2")) + 1
    out = [
        f"{n_id} of {n} residues identical ({100 * n_id / n:.0f}%); "
        f"query {qstart + 1}-{qend}, target {tstart + 1}-{tend}"
    ]
    for i in range(0, n, width):
        qs, ts = qa[i : i + width], ta[i : i + width]
        match = "".join("|" if x == y else " " for x, y in zip(qs, ts))
        out += [
            f"{q_name:<{lab}}{qstart + 1 + i:>6} {qs} {qstart + i + len(qs)}",
            f"{'':<{lab}}{'':>6} {match}",
            f"{t_name:<{lab}}{tstart + 1 + i:>6} {ts} {tstart + i + len(ts)}",
            "",
        ]
    out.append(
        f"hp_pbotc_1st_ed2 classes (H = ACFILMPVWY, P = DEGHKNQRST); longest run of "
        f"identical classes: {run_len} positions, query {qstart + 1 + run_start}-"
        f"{qstart + run_start + run_len}"
    )
    for i in range(0, n, width):
        qs, ts = qc[i : i + width], tc[i : i + width]
        match = "".join("|" if x == y else " " for x, y in zip(qs, ts))
        mark = "".join(
            "^" if run_start <= i + j < run_start + run_len else " "
            for j in range(len(qs))
        )
        out += [
            f"{q_name:<{lab}}{qstart + 1 + i:>6} {qs}",
            f"{'':<{lab}}{'':>6} {match}",
            f"{t_name:<{lab}}{tstart + 1 + i:>6} {ts}",
            f"{'':<{lab}}{'':>6} {mark}",
            "",
        ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Candidate figure.
# ---------------------------------------------------------------------------
#: One colour and one hatch per call category. Colour alone is not the only cue: each
#: category also has its own hatch, and "no call" is a cross with no bar.
CATEGORY_STYLE = {
    "kmerseek landed": dict(facecolor="#8B1A1A", edgecolor="#8B1A1A", hatch=""),
    "inside, lower IoU": dict(facecolor="#6BAED6", edgecolor="#2B6C9E", hatch=""),
    "spills": dict(facecolor="white", edgecolor="#D98C1F", hatch="////"),
    "equal or higher IoU": dict(facecolor="#555555", edgecolor="#222222", hatch="xx"),
}
CATEGORY_LABEL = {
    "kmerseek landed": "kmerseek: call lands on the feature (>= 80% of call inside, >= 30% covered)",
    "inside, lower IoU": "call mostly inside the feature, lower IoU than kmerseek",
    "spills": "call overlaps the feature but < 50% of the call is inside it",
    "equal or higher IoU": "call reaches kmerseek's IoU or higher",
    "no call": "no call of this feature type overlaps the feature",
}
#: A tool row with no bar gets one of three marks, each with one meaning.
NO_BAR_MARKS = {
    "no call": dict(marker="x", color="#777777", ms=6, mew=1.5),
    "not on this protein": dict(marker="o", mfc="none", mec="#777777", ms=5.5, mew=1.2),
    "does not search": dict(marker="_", color="#777777", ms=9, mew=1.5),
}
NO_BAR_TEXT = {
    "no call": "no call",
    "not on this protein": "its best call is on another target protein",
    "does not search": "no target: the scan reads the human sequence only",
}
FEATURE_COLOR = "#E8E1C8"
FEATURE_EDGE = "#8A7F57"
TARGET_FEATURE_COLOR = "#F2D98C"


def _feature_rows(features: pl.DataFrame, lo: int, hi: int) -> list[tuple]:
    """Stack overlapping feature boxes on separate tiers so no two labels collide."""
    f = features.filter(
        (pl.col("domain_end") >= lo) & (pl.col("domain_start") <= hi)
    ).sort("domain_start", "domain_end")
    tiers_end: list[float] = []
    rows = []
    for r in f.iter_rows(named=True):
        # A label needs ~7 px per character; at these widths reserve its length in aa too.
        label_end = max(
            r["domain_end"],
            r["domain_start"] + 1.6 * len(r["pfam_id"]) * (hi - lo) / 120,
        )
        for i, e in enumerate(tiers_end):
            if r["domain_start"] > e + 2:
                tiers_end[i] = label_end
                break
        else:
            tiers_end.append(label_end)
            i = len(tiers_end) - 1
        rows.append((r["domain_start"], r["domain_end"], r["pfam_id"], i))
    return rows


def draw_protein_panel(
    ax,
    name: str,
    length: int,
    lo: int,
    hi: int,
    features: pl.DataFrame,
    instance: tuple[int, int] | None,
    bars: list[dict],
    side: str,
):
    """One protein as a thin line with its Swiss-Prot features as boxes, and one bar per
    tool underneath for that tool's call on this protein.

    ``bars`` items: label, category (a CATEGORY_STYLE key or "no call" / "not on this
    protein" / "does not search"), and ``side`` -> (start, end) in 1-based residues or None.
    """
    from matplotlib.patches import Rectangle

    feats = _feature_rows(features, lo, hi)
    n_tiers = max([t for *_, t in feats], default=-1) + 1
    y_line = 0.0
    ax.plot(
        [max(1, lo), min(length, hi)],
        [y_line, y_line],
        color="#333333",
        lw=1.2,
        zorder=1,
    )
    for s, e, ftype, tier in feats:
        y = y_line + 0.25 + tier * 0.75
        is_inst = instance is not None and (s, e) == instance
        ax.add_patch(
            Rectangle(
                (s - 0.5, y - 0.25),
                e - s + 1,
                0.5,
                facecolor=TARGET_FEATURE_COLOR if is_inst else FEATURE_COLOR,
                edgecolor="#222222" if is_inst else FEATURE_EDGE,
                lw=1.6 if is_inst else 0.8,
                zorder=2,
            )
        )
        ax.plot(
            [s - 0.5, s - 0.5], [y_line, y - 0.25], color=FEATURE_EDGE, lw=0.5, zorder=1
        )
        ax.text(
            max(s, lo) + 0.5,
            y,
            ftype,
            va="center",
            ha="left",
            fontsize=7.5,
            zorder=3,
            clip_on=True,
        )
    y0 = -0.9
    labels = []
    for i, b in enumerate(bars):
        y = y0 - i * 0.8
        iou = b.get("iou")
        show_iou = side == "human" and iou is not None and b["category"] != "no call"
        labels.append((y, f"{b['label']} (IoU {iou:.2f})" if show_iou else b["label"]))
        iv = b.get(side)
        cat = b["category"]
        if iv is None or cat in NO_BAR_MARKS:
            key = cat if cat in NO_BAR_MARKS else "no call"
            x = (instance[0] + instance[1]) / 2 if instance else (lo + hi) / 2
            ax.plot([x], [y], ls="", zorder=3, **NO_BAR_MARKS[key])
            ax.text(
                x + (hi - lo) * 0.015,
                y,
                NO_BAR_TEXT[key],
                va="center",
                fontsize=7,
                color="#555555",
            )
            continue
        s, e = iv
        st = CATEGORY_STYLE[cat]
        ax.add_patch(
            Rectangle((s - 0.5, y - 0.28), e - s + 1, 0.56, lw=1.2, zorder=3, **st)
        )
    ax.set_yticks(
        [y for y, _ in labels] + [y_line],
        [lab for _, lab in labels] + [name],
        fontsize=8,
    )
    ax.set_ylim(y0 - len(bars) * 0.8 + 0.2, 0.25 + max(n_tiers, 1) * 0.75 + 0.2)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(
        f"residue position on {name} (aa; protein is {length} aa, window shown)",
        fontsize=8.5,
    )
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)


def category_legend(fig, y: float = 1.0):
    """The call categories, placed above the panels so they are read before the marks."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    handles = [
        Patch(label=CATEGORY_LABEL[k], **{kk: v for kk, v in CATEGORY_STYLE[k].items()})
        for k in (
            "kmerseek landed",
            "inside, lower IoU",
            "spills",
            "equal or higher IoU",
        )
    ]
    for key, text in (
        ("no call", CATEGORY_LABEL["no call"]),
        ("not on this protein", "the tool's best call is on another target protein"),
        (
            "does not search",
            "Kyte-Doolittle scan: no target protein, it searches nothing",
        ),
    ):
        handles.append(Line2D([], [], ls="", label=text, **NO_BAR_MARKS[key]))
    handles.append(
        Patch(
            facecolor=TARGET_FEATURE_COLOR,
            edgecolor="#222222",
            lw=1.6,
            label="the Swiss-Prot feature being scored (human) / "
            "transferred from (target)",
        )
    )
    handles.append(
        Patch(
            facecolor=FEATURE_COLOR,
            edgecolor=FEATURE_EDGE,
            label="other Swiss-Prot features, labelled by type",
        )
    )
    fig.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.0, y),
        ncol=2,
        fontsize=7.5,
        frameon=False,
    )


def kd_category(inside: float | None, iou: float | None) -> str:
    """Criterion 3 category of the Kyte-Doolittle scan segment on an instance. The scan
    never carries a label other than TRANSMEM, so it is placed by position only."""
    if iou is None:
        return "no call"
    return "inside, lower IoU" if (inside or 0) >= 0.5 else "spills"


def candidate_bars(r: dict, other: dict | None) -> list[dict]:
    """The tool rows of one candidate figure, in the order of the brief.

    ``r`` is a row of the candidate table; ``other`` is the landing-table row of the best
    arm of a different alphabet. kmerseek coordinates are shifted from 0-based
    end-exclusive to 1-based inclusive; aligner and scan coordinates are 1-based already.
    """
    bars = [
        dict(
            label=f"kmerseek {arm_short(r['chosen_arm'])}, chosen for {r['pfam_id']}",
            category="kmerseek landed",
            iou=r["land_iou"],
            human=(r["land_qstart"] + 1, r["land_qend"]),
            target=(r["land_tstart"] + 1, r["land_tend"]),
        )
    ]
    if other is not None:
        same = other["land_target_acc"] == r["land_target_acc"]
        bars.append(
            dict(
                label=f"kmerseek {arm_short(other['arm'])}, best other alphabet",
                category="kmerseek landed",
                iou=other["land_iou"],
                human=(other["land_qstart"] + 1, other["land_qend"]),
                target=(other["land_tstart"] + 1, other["land_tend"]) if same else None,
                elsewhere=not same,
            )
        )
    for arm in [
        "hmmer3_phmmer.default",
        "mmseqs2_seqseq.s7",
        "foldseek.3di_aa",
        "prostt5.3di_from_seq",
        "reseek.verysensitive",
    ]:
        lab = COMPARISON_ARMS[arm]
        cat = r[f"{lab}|category"]
        if cat == "no call":
            bars.append(dict(label=lab, category="no call"))
            continue
        same = r[f"{lab}|best_target_acc"] == r["land_target_acc"]
        bars.append(
            dict(
                label=lab,
                category=cat,
                iou=r[f"{lab}|best_iou"],
                human=(r[f"{lab}|best_qstart"], r[f"{lab}|best_qend"]),
                target=(
                    (r[f"{lab}|best_tstart"], r[f"{lab}|best_tend"]) if same else None
                ),
                elsewhere=not same,
            )
        )
    kd = kd_category(r.get("kd_inside"), r.get("kd_best_iou"))
    bars.append(
        dict(
            label="Kyte-Doolittle scan",
            category=kd,
            iou=r.get("kd_best_iou"),
            human=(r["kd_qstart"], r["kd_qend"]) if kd != "no call" else None,
            target=None,
            scan=True,
        )
    )
    return bars


def target_side(bars: list[dict]) -> list[dict]:
    """The same rows for the target protein: a call on a different target protein gets the
    open-circle mark, the scan gets its own mark, a missing call stays a cross."""
    out = []
    for b in bars:
        b = dict(b)
        if b.get("scan"):
            b["category"] = "does not search"
        elif b["category"] != "no call" and (
            b.get("elsewhere") or b.get("target") is None
        ):
            b["category"] = "not on this protein"
        out.append(b)
    return out


def draw_candidate(r: dict, other: dict | None, notes: pl.DataFrame, pad: int = 60):
    """Human protein on top, target protein below, tool rows under each. Returns fig."""
    import matplotlib.pyplot as plt

    hseq = sequences("human", {r["accession"]})[r["accession"]]
    tseq = sequences(r["species"], {r["land_target_acc"]})[r["land_target_acc"]]
    bars = candidate_bars(r, other)
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9.2), height_ratios=[1.15, 1])
    sym = r.get("hgnc_symbol") or r["accession"]
    draw_protein_panel(
        axes[0],
        f"human {sym} ({r['accession']})",
        len(hseq),
        max(1, r["domain_start"] - pad),
        min(len(hseq), r["domain_end"] + pad),
        notes.filter(pl.col("accession") == r["accession"]),
        (r["domain_start"], r["domain_end"]),
        bars,
        "human",
    )
    draw_protein_panel(
        axes[1],
        f"{r['species']} {r['land_target_acc']}",
        len(tseq),
        max(1, r["land_tstart"] - pad),
        min(len(tseq), r["land_tend"] + pad),
        notes.filter(pl.col("accession") == r["land_target_acc"]),
        (r["land_t_feat_start"], r["land_t_feat_end"]),
        target_side(bars),
        "target",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    category_legend(fig, 0.89)
    return fig


def short_note(note: str | None) -> str:
    """A Swiss-Prot /note for a title: the part before '; Name=' or '; Note='."""
    if not note:
        return ""
    return re.split(r"; (?:Name|Note)=", note)[0]


def protein_name(species: str, accession: str) -> str:
    """The protein's name from its FASTA header in the QfO release (text before OS=)."""
    with open(proteome_fasta(species)) as fh:
        for line in fh:
            if line.startswith(">") and f"|{accession}|" in line:
                head = line[1:].split(" OS=")[0]
                return head.split(" ", 1)[1] if " " in head else head
    return ""
