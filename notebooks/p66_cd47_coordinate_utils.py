"""Tables and figures for notebook 245: where the P66/CD47 shared k-mers sit.

Everything here reads the four CSVs written by 245_p66_cd47_pair_runs.py and
scripts/fetch_p66_cd47_annotations.py. No number is recomputed from the pair JSONs.

Numbering. P66 is UniProt H7C7N8, a 618-residue precursor whose signal peptide is
1-21, so the mature chain is 597 aa and mature = UniProt - 21. CD47 is UniProt
Q08722, a 323-residue precursor with signal peptide 1-18, mature 305 aa and
mature = UniProt - 18. Every column in every table carries both numberings and so
does every axis here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hp_conservation_utils import finish_figure  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TABLES = REPO / "tables"
FIG = REPO / "figures"

ANNOTATIONS = TABLES / "245_p66_cd47_annotations.csv"
KMERS = TABLES / "245_p66_cd47_shared_kmers.csv"
REGIONS = TABLES / "245_p66_cd47_regions.csv"
CONTROLS = TABLES / "245_p66_cd47_controls.csv"
CONTROL_COUNTS = TABLES / "245_p66_cd47_control_counts.csv"

TOOLS = (
    "kmerseek pair (kmerseek-ka-lambda-region, the binary notebook 241 used), "
    "P66 (UniProt H7C7N8, mature chain) against CD47 (UniProt Q08722, GENCODE v49 "
    "canonical protein), every alphabet and k of the notebook 241 sweep that shares "
    "at least one k-mer; UniProt REST for the features; PDB 2JJS (Hatherley et al. 2008) "
    "for the CD47 residues that contact SIRP-alpha, measured at 4 A; Ristow et al. 2015 "
    "and Defoe and Coburn 2001 for the P66 loop; the 300 length-matched human control "
    "proteins of PR #44"
)

P66_OFFSET, CD47_OFFSET = 21, 18
P66_LEN_UNIPROT, CD47_LEN_UNIPROT = 618, 323
P66_LOOP = (202, 208)
P66_PEPTIDE = (203, 209)
CD47_CONTACT_SPAN = (115, 124)

# Alphabets grouped by how many classes they have, so the colours run in families
# instead of cycling. The reader finds an alphabet by its group first.
SIZE_GROUPS = [
    ("2 to 3 classes", ["hp_lehninger2", "hp_lehninger_c_nonpolar2", "hp_pbotc_1st_ed2",
                        "hp_thomas_dill2", "hp_thomas_dill_no_c2", "hp_kyte_doolittle2",
                        "hp_lehninger_hpc3"]),
    ("4 to 8 classes", ["gbmr4", "polarity4", "wwmj5", "dayhoff6", "gbmr7", "funcgroups8"]),
    ("12 to 18 classes", ["sdm12", "mmseqs12", "wass14", "hsdm17", "uniprot18"]),
    ("20 classes", ["protein20"]),
]
# One colour per alphabet. Blues for the 2-3 class alphabets, warm for 4-8, purples
# for 12-18, black for protein20. Nothing else in these figures uses these colours:
# the named stretches are grey boxes with a black outline and a black tick.
GROUP_COLORS = {
    "2 to 3 classes": ["#08306B", "#2171B5", "#4292C6", "#6BAED6", "#9ECAE1", "#C6DBEF", "#08519C"],
    "4 to 8 classes": ["#99000D", "#E6550D", "#FD8D3C", "#FDBE85", "#8C510A", "#F16913"],
    "12 to 18 classes": ["#3F007D", "#6A51A3", "#9E9AC8", "#BCBDDC", "#54278F"],
    "20 classes": ["#000000"],
}
ALPHABET_COLOR = {a: GROUP_COLORS[g][i]
                  for g, members in SIZE_GROUPS
                  for i, a in enumerate(members)}

NAMED_EDGE = "#000000"       # the outline of a stretch named in the literature
NAMED_FILL = "#D9D9D9"       # its fill
BACKBONE = "#4D4D4D"         # the protein line
FEATURE_FILL = "#FFFFFF"     # a UniProt feature box
LINK_ALPHA = 0.75


def contact_positions(annotations: pl.DataFrame) -> list[int]:
    """Every CD47 residue within 4 A of SIRP-alpha in PDB 2JJS, in UniProt numbering."""
    return sorted((annotations
                   .filter((pl.col("protein") == "CD47")
                           & (pl.col("feature_type") == "CONTACT"))
                   )["start_uniprot"].to_list())


def load() -> dict[str, pl.DataFrame]:
    return {
        "annotations": pl.read_csv(ANNOTATIONS),
        "kmers": pl.read_csv(KMERS),
        "regions": pl.read_csv(REGIONS),
        "controls": pl.read_csv(CONTROLS),
        "control_counts": pl.read_csv(CONTROL_COUNTS),
    }


def class_letters(kmers: pl.DataFrame, alphabet: str) -> dict[str, str]:
    """letter -> the residues that alphabet puts in that class, read back from the
    k-mers themselves: every shared k-mer carries both the real residues and the class
    letters kmerseek encoded them to, so the mapping is measured, not assumed."""
    out: dict[str, set[str]] = {}
    sub = kmers.filter(pl.col("alphabet") == alphabet)
    for row in sub.iter_rows(named=True):
        for letter, q, t in zip(row["kmer_letters"], row["p66_residues"], row["cd47_residues"]):
            out.setdefault(letter, set()).update([q, t])
    return {k: "".join(sorted(v)) for k, v in sorted(out.items())}


def residue_block(kmers: pl.DataFrame, row: dict) -> str:
    """The real residues of one shared k-mer, both sequences, with a match line, the
    class letters underneath and the classes those letters stand for."""
    q, t = row["p66_residues"], row["cd47_residues"]
    match = "".join("|" if x == y else " " for x, y in zip(q, t))
    letters = class_letters(kmers, row["alphabet"])
    seen = sorted(set(row["kmer_letters"]))
    legend = "; ".join(f"{c} = {letters.get(c, '?')}" for c in seen)
    tag = lambda name, lo_u, hi_u, lo_m, hi_m: (
        f"  {name:<5} UniProt {lo_u:>3}-{hi_u:<3} (mature {lo_m:>3}-{hi_m:<3})  ")
    p66_tag = tag("P66", row["p66_start_uniprot"], row["p66_end_uniprot"],
                  row["p66_start_mature"], row["p66_end_mature"])
    cd47_tag = tag("CD47", row["cd47_start_uniprot"], row["cd47_end_uniprot"],
                   row["cd47_start_mature"], row["cd47_end_mature"])
    pad = max(len(p66_tag), len(cd47_tag))
    return "\n".join([
        f"{row['alphabet']} k={row['ksize']}  ({row['bits']:.1f} bits per seed)",
        f"{p66_tag:<{pad}}{q}",
        f"{'':<{pad}}{match}",
        f"{cd47_tag:<{pad}}{t}",
        f"{'  classes':<{pad}}{row['kmer_letters']}",
        f"  {row['n_identical_residues']} of {row['ksize']} residues identical; covers "
        f"{row['n_cd47_contacts_covered_anywhere']} CD47 residues that contact "
        f"SIRP-alpha and {row['n_p66_loop_residues_covered']} of the 7 P66 loop residues",
        f"  {legend}",
    ])


def merged_spans(df: pl.DataFrame, start_col: str, end_col: str) -> str:
    """Overlapping or touching spans joined into one, as "202-208, 447-465"."""
    spans = sorted(df.select(start_col, end_col).unique().iter_rows())
    out: list[list[int]] = []
    for lo, hi in spans:
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return ", ".join(f"{lo}-{hi}" for lo, hi in out)


def best_per_alphabet(kmers: pl.DataFrame) -> pl.DataFrame:
    """For each alphabet, the k-mer that covers the most named residues, breaking ties
    on identical residues and then on k, so the row is the same on every run."""
    return (kmers
            .with_columns((pl.col("n_cd47_contacts_covered_anywhere")
                           + pl.col("n_p66_loop_residues_covered")).alias("n_named_covered"))
            .sort(["n_named_covered", "n_identical_residues", "ksize",
                   "cd47_start_uniprot", "p66_start_uniprot"],
                  descending=[True, True, True, False, False])
            .group_by("alphabet", maintain_order=True)
            .first())


def _protein_axis(ax, y: float, length_uniprot: int, offset: int, label: str,
                  features: pl.DataFrame, height: float = 0.16) -> None:
    """One protein: a thin line for the chain, open boxes for its UniProt features."""
    ax.plot([1, length_uniprot], [y, y], color=BACKBONE, lw=1.6, zorder=2,
            solid_capstyle="butt")
    for r in features.iter_rows(named=True):
        w = r["end_uniprot"] - r["start_uniprot"] + 1
        ax.add_patch(Rectangle((r["start_uniprot"], y - height / 2), w, height,
                               facecolor=FEATURE_FILL, edgecolor=BACKBONE, lw=0.9, zorder=3))
        if w >= 0.045 * length_uniprot:
            ax.text(r["start_uniprot"] + w / 2, y, r["short"], ha="center", va="center",
                    fontsize=7.0, color=BACKBONE, zorder=4)
    ax.text(1, y + height * 1.9, label, ha="left", va="bottom", fontsize=10,
            fontweight="bold", color=BACKBONE)


def _short_name(feature_type: str, name: str) -> str:
    if feature_type == "Transmembrane":
        return "TM"
    if feature_type == "Topological domain":
        return {"Extracellular": "outside", "Cytoplasmic": "inside"}.get(name, name)
    if feature_type == "Domain":
        return "Ig-like V-type"
    if feature_type == "Signal":
        return "signal"
    if feature_type == "Compositional bias":
        return "low complexity"
    if feature_type == "Region":
        return name.lower()
    return feature_type.lower()


def _features_for(annotations: pl.DataFrame, protein: str, keep: list[str]) -> pl.DataFrame:
    df = (annotations
          .filter((pl.col("protein") == protein) & pl.col("feature_type").is_in(keep))
          .with_columns(pl.struct("feature_type", "name")
                        .map_elements(lambda s: _short_name(s["feature_type"], s["name"] or ""),
                                      return_dtype=pl.Utf8).alias("short")))
    return df


def figure_map(data: dict[str, pl.DataFrame], path: Path,
               hypothesis: str, conclusion: str) -> None:
    """P66 and CD47 as lines with their UniProt features as boxes, every shared k-mer
    drawn as a link between the two, coloured by alphabet.

    A k-mer that touches one of the two stretches named in the literature is drawn
    thicker and solid and carries a square at each end, so it is found without reading
    its colour."""
    ann, kmers = data["annotations"], data["kmers"]
    contacts = contact_positions(ann)
    in_span = [p for p in contacts if CD47_CONTACT_SPAN[0] <= p <= CD47_CONTACT_SPAN[1]]
    present = [a for _, members in SIZE_GROUPS for a in members
               if a in set(kmers["alphabet"].unique())]

    fig, ax = plt.subplots(figsize=(14.0, 8.6))
    # Each protein is drawn in its own UniProt numbering, scaled to the same width, so
    # a position is a fraction of its own protein.
    fx = lambda pos, length: pos / length

    Y_P66_RULER, Y_P66_NOTE, Y_P66 = 1.52, 1.26, 1.00
    Y_CD47, Y_CD47_NOTE, Y_CD47_RULER = 0.00, -0.26, -0.52

    p66_feats = _features_for(ann, "P66", ["Signal", "Compositional bias"])
    cd47_feats = _features_for(ann, "CD47", ["Signal", "Domain", "Transmembrane"])

    for feats, length, y, label in (
        (p66_feats, P66_LEN_UNIPROT, Y_P66, "P66  Borreliella burgdorferi, UniProt H7C7N8, 618 aa"),
        (cd47_feats, CD47_LEN_UNIPROT, Y_CD47, "CD47  human, UniProt Q08722, 323 aa"),
    ):
        ax.plot([0, 1], [y, y], color=BACKBONE, lw=1.6, zorder=2, solid_capstyle="butt")
        for r in feats.iter_rows(named=True):
            x0, x1 = fx(r["start_uniprot"], length), fx(r["end_uniprot"], length)
            ax.add_patch(Rectangle((x0, y - 0.05), max(x1 - x0, 0.003), 0.10,
                                   facecolor=FEATURE_FILL, edgecolor=BACKBONE,
                                   lw=0.9, zorder=3))
            # Feature names sit outside the protein line, on the side away from the
            # links, with a leader line, so nothing is written over a box or a link.
            side = 1 if y == Y_P66 else -1
            ax.annotate(r["short"], xy=((x0 + x1) / 2, y + side * 0.05),
                        xytext=((x0 + x1) / 2, y + side * 0.135),
                        ha="center", va="bottom" if side > 0 else "top", fontsize=7.5,
                        color=BACKBONE, zorder=6,
                        arrowprops=dict(arrowstyle="-", color=BACKBONE, lw=0.6))
        ax.text(0, y + (0.23 if y == Y_P66 else -0.23), label, ha="left",
                va="bottom" if y == Y_P66 else "top", fontsize=11,
                fontweight="bold", color=BACKBONE)

    # The two stretches named in the literature.
    for (lo, hi), length, y, y_note, text in (
        (P66_LOOP, P66_LEN_UNIPROT, Y_P66, Y_P66_NOTE,
         "loop required for integrin binding, proposed to bind SIRP-alpha\n"
         "UniProt 202-208 = mature 181-187 (QENDKDT); deleting it cuts\n"
         "integrin binding ~1370-fold (Ristow et al. 2015)"),
        (CD47_CONTACT_SPAN, CD47_LEN_UNIPROT, Y_CD47, Y_CD47_NOTE,
         f"the {len(in_span)} CD47 residues that contact SIRP-alpha in one run,\n"
         f"UniProt 115-124 = mature 97-106 (of {len(contacts)} contacts in all,\n"
         "measured at 4 A in PDB 2JJS, Hatherley et al. 2008)"),
    ):
        x0, x1 = fx(lo, length), fx(hi, length)
        ax.add_patch(Rectangle((x0, y - 0.065), max(x1 - x0, 0.005), 0.13,
                               facecolor=NAMED_FILL, edgecolor=NAMED_EDGE, lw=1.5, zorder=5))
        ax.annotate(text, xy=((x0 + x1) / 2, y + (0.065 if y == Y_P66 else -0.065)),
                    xytext=(0.5, y_note), ha="center",
                    va="bottom" if y == Y_P66 else "top", fontsize=8.5, color=NAMED_EDGE,
                    zorder=6, arrowprops=dict(arrowstyle="-", color=NAMED_EDGE, lw=1.0))

    # The contacts outside that run, one triangle each. Black, like the other marks for
    # residues named in the literature; no alphabet uses black.
    outside = [p for p in contacts if p not in in_span]
    ax.scatter([fx(p, CD47_LEN_UNIPROT) for p in outside],
               [Y_CD47 - 0.075] * len(outside), marker="^", s=26, color=NAMED_EDGE,
               zorder=6)

    # Every shared k-mer: a link from its P66 span to its CD47 span. The ones that
    # touch a named stretch are drawn last, thicker, with a square at each end.
    touching = (pl.col("overlaps_p66_loop") | pl.col("overlaps_cd47_contact_span"))
    for sub, lw, alpha, mark in ((kmers.filter(~touching), 0.7, 0.35, False),
                                 (kmers.filter(touching), 1.8, 1.0, True)):
        for row in sub.iter_rows(named=True):
            xq = fx((row["p66_start_uniprot"] + row["p66_end_uniprot"]) / 2, P66_LEN_UNIPROT)
            xt = fx((row["cd47_start_uniprot"] + row["cd47_end_uniprot"]) / 2, CD47_LEN_UNIPROT)
            col = ALPHABET_COLOR[row["alphabet"]]
            ax.plot([xq, xt], [Y_P66 - 0.06, Y_CD47 + 0.06], color=col, lw=lw,
                    alpha=alpha, zorder=7 if mark else 1,
                    marker="s" if mark else None, markersize=4 if mark else 0)

    # Rulers: both numberings, UniProt above and mature in brackets below.
    for length, offset, y, name in ((P66_LEN_UNIPROT, P66_OFFSET, Y_P66_RULER, "P66"),
                                    (CD47_LEN_UNIPROT, CD47_OFFSET, Y_CD47_RULER, "CD47")):
        ax.plot([0, 1], [y, y], color=BACKBONE, lw=0.8)
        for t in range(0, length + 1, 100):
            x = fx(max(t, 1), length)
            ax.plot([x, x], [y, y - 0.02], color=BACKBONE, lw=0.8)
            ax.text(x, y - 0.035, f"{t}\n({t - offset})", ha="center", va="top",
                    fontsize=7.5, color=BACKBONE)
        ax.text(1.004, y, f"{name} residue: UniProt\n(mature)", ha="left", va="center",
                fontsize=8, color=BACKBONE)

    handles = [Patch(facecolor=NAMED_FILL, edgecolor=NAMED_EDGE,
                     label="stretch named in the literature"),
               Line2D([], [], color="none", marker="^", markersize=6,
                      markerfacecolor=NAMED_EDGE, markeredgecolor=NAMED_EDGE,
                      label=f"CD47 residue contacting SIRP-alpha, outside that run "
                            f"({len(outside)} of {len(contacts)})"),
               Patch(facecolor=FEATURE_FILL, edgecolor=BACKBONE, label="UniProt feature"),
               Line2D([], [], color=BACKBONE, lw=1.8, marker="s", markersize=4,
                      label="k-mer touching a named stretch (square ends)"),
               Line2D([], [], color=BACKBONE, lw=0.7, alpha=0.35,
                      label="k-mer touching neither")]
    for group, members in SIZE_GROUPS:
        members_here = [a for a in members if a in present]
        if not members_here:
            continue
        handles.append(Line2D([], [], color="none", label=f"$\\bf{{{group.replace(' ', chr(92) + ' ')}}}$"))
        handles += [Line2D([], [], color=ALPHABET_COLOR[a], lw=2,
                           label=f"{a}  ({int(kmers.filter(pl.col('alphabet') == a).height)} k-mers)")
                    for a in members_here]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=4,
              fontsize=8.5, frameon=False, handlelength=1.8,
              title="one line per shared k-mer, coloured by alphabet",
              title_fontsize=9.5, alignment="left",
              bbox_transform=fig.transFigure)

    ax.set_xlim(-0.012, 1.13)
    ax.set_ylim(Y_CD47_RULER - 0.17, Y_P66_RULER + 0.02)
    ax.axis("off")
    fig.subplots_adjust(left=0.008, right=0.992, top=0.815, bottom=0.055)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=0.995, footer_y=0.004, tight=False)


def figure_controls(data: dict[str, pl.DataFrame], path: Path,
                    hypothesis: str, conclusion: str) -> None:
    """The two controls, one row per arm: how many shared k-mers touch each named
    stretch against how many chance gives, and where CD47 sits among the 300
    length-matched human proteins."""
    c = data["controls"]
    order = [a for _, members in SIZE_GROUPS for a in members]
    c = (c.with_columns(pl.col("alphabet").map_elements(
            lambda a: order.index(a), return_dtype=pl.Int64).alias("group_rank"))
         .sort(["group_rank", "ksize"]))

    # A gap between alphabet-size groups, so the rows are grouped by what they are.
    group_of = {a: g for g, members in SIZE_GROUPS for a in members}
    ys, labels, prev = [], [], None
    y = 0.0
    for row in c.iter_rows(named=True):
        g = group_of[row["alphabet"]]
        if prev is not None and g != prev:
            y += 1.0
        ys.append(y)
        labels.append(f"{row['alphabet']} k={row['ksize']}")
        prev = g
        y += 1.0

    fig, axes = plt.subplots(1, 4, figsize=(17.5, 8.4), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.1], "wspace": 0.11})
    for ax in axes:
        for yy in ys:
            ax.axhline(yy, color="#E6E6E6", lw=0.7, zorder=0)
        ax.set_yticks(ys)
        ax.set_yticklabels(labels, fontsize=8)
    # The y axis is shared, so inverting it once per panel would undo itself on an even
    # number of panels. Invert it once.
    axes[0].invert_yaxis()

    for ax, obs_col, exp_col, span in (
        (axes[0], "n_touching_p66_loop", "expected_touching_p66_loop",
         "P66 loop, UniProt 202-208"),
        (axes[1], "n_touching_cd47_contact_span", "expected_touching_cd47_contact_span",
         "8 CD47 contacts in UniProt 115-124"),
        (axes[2], "n_touching_any_cd47_contact", "expected_touching_any_cd47_contact",
         "any of the 20 CD47 contacts"),
    ):
        ax.barh(ys, c[exp_col].to_list(), height=0.62, color="#BDBDBD",
                edgecolor="none", label="expected from chance placement", zorder=1)
        obs = c[obs_col].to_list()
        hit = [(x, yy) for x, yy in zip(obs, ys) if x > 0]
        none = [yy for x, yy in zip(obs, ys) if x == 0]
        if hit:
            ax.scatter([x for x, _ in hit], [yy for _, yy in hit], s=34, marker="o",
                       color="#8B1A1A", zorder=3, label="shared k-mers that touch it")
        if none:
            ax.scatter([0] * len(none), none, s=42, marker="x", color="#8B1A1A",
                       zorder=4, label="no shared k-mer touches it")
        ax.set_xlabel(f"shared k-mers touching {span}\n(number of k-mers; higher than the\ngrey bar means they concentrate there)", fontsize=8.5)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=8, frameon=False)

    ax = axes[3]
    ax.barh(ys, c["cd47_percentile_among_controls"].to_list(), height=0.62,
            color="#2171B5", edgecolor="none", zorder=1,
            label="CD47's place among the 300 control proteins")
    ax.axvline(50, color="#4D4D4D", lw=1.0, ls=":", zorder=2,
               label="50th percentile: an ordinary protein")
    ax.axvline(95, color="#000000", lw=1.0, ls="--", zorder=2,
               label="95th percentile")
    ax.set_xlim(0, 100)
    ax.set_xlabel("percentile of CD47's shared-k-mer count among 300 human proteins\n"
                  "of similar length (242-404 aa); higher means CD47 stands out more",
                  fontsize=8.5)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), fontsize=8, frameon=False)

    fig.subplots_adjust(left=0.135, right=0.995, top=0.865, bottom=0.10)
    finish_figure(fig, path, tools=TOOLS, hypothesis=hypothesis, conclusion=conclusion,
                  header_y=0.995, footer_y=0.005, tight=False)
