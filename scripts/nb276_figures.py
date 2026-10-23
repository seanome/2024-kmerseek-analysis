#!/usr/bin/env python3
"""Figures for notebook 276, the structural test of the P66 / CD47 mimicry claim.

Reads only the tables scripts/nb276_p66_cd47_structure.py wrote, so no figure can
disagree with the analysis. Four figures:

  276a  P66's model confidence along the chain, with the claimed loop marked.
  276b  P66 and CD47 superposed by US-align, the loop and the binding face coloured.
  276c  Whole-fold TM-score: the claimed pair against 20 immunoglobulin domains.
  276d  Loop-to-face backbone RMSD: the claimed loop against 10 other P66 loops.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 scripts/nb276_figures.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import pubfig  # noqa: E402

DATA = Path("/Users/olga/data/nb276-p66-cd47-structure")
FIGDIR = Path(__file__).resolve().parent.parent / "figures"
PYMOL = Path("/Users/olga/anaconda3/envs/pymol-env/bin/pymol")
USALIGN = Path("/Users/olga/anaconda3/envs/usalign/bin/USalign")

#: One colour per meaning, used the same way in every panel of this figure set.
C_LOOP = pubfig.OKABE_ITO["vermillion"]  # the P66 loop named in the claim
C_FACE = pubfig.OKABE_ITO["blue"]  # CD47's SIRPalpha-binding face
C_P66 = "#555555"  # the rest of P66
C_CD47 = "#BBBBBB"  # the rest of CD47
C_CLAIM = pubfig.OKABE_ITO["vermillion"]  # the claimed pair's measurement
C_BG = pubfig.GREY  # background comparisons

#: TM-score at or above this is normally read as the same fold.
SAME_FOLD_TM = 0.5
#: pLDDT below this is AlphaFold's "very low" band.
VERY_LOW_PLDDT = 50


def loop_span() -> tuple[int, int]:
    row = pl.read_csv(DATA / "01_loop_numbering.csv").row(0, named=True)
    return int(row["full_start"]), int(row["full_end"])


# ---------------------------------------------------------------------------
# 276a: model confidence
# ---------------------------------------------------------------------------


def figure_plddt() -> dict:
    plddt = pl.read_csv(DATA / "02_p66_plddt.csv")
    start, end = loop_span()
    model = pl.read_csv(DATA / "02_alphafold_models.csv").filter(
        pl.col("protein") == "P66"
    ).row(0, named=True)

    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=52)
    ax.axhspan(0, VERY_LOW_PLDDT, color=C_BG, alpha=0.18, lw=0,
               label=f"AlphaFold's very low band (pLDDT < {VERY_LOW_PLDDT})")
    ax.plot(plddt["full_position"], plddt["plddt"], color=C_P66, lw=0.6,
            label="P66, per residue")
    inside = plddt.filter(pl.col("full_position").is_between(start, end))
    ax.plot(inside["full_position"], inside["plddt"], color=C_LOOP, lw=2,
            solid_capstyle="butt",
            label=f"the loop in the claim ({start}-{end})")
    ax.set_xlabel("P66 residue (UniProt H7C7N8 numbering)")
    ax.set_ylabel("pLDDT")
    ax.set_ylim(0, 100)
    ax.set_xlim(1, int(plddt["full_position"].max()))
    pubfig.shared_legend(fig, ncol=1, fontsize=5)
    pubfig.save(fig, FIGDIR / "276a_p66_model_confidence")
    return {
        "model": f"{model['entry_id']} v{model['version']}",
        "loop_mean": float(inside["plddt"].mean()),
        "loop_min": float(inside["plddt"].min()),
        "loop_max": float(inside["plddt"].max()),
        "chain_median": float(plddt["plddt"].median()),
        "fraction_chain_below_50": float((plddt["plddt"] < VERY_LOW_PLDDT).mean()),
    }


# ---------------------------------------------------------------------------
# 276b: the superposition
# ---------------------------------------------------------------------------


def figure_superposition() -> dict:
    """Render the US-align superposition in PyMOL, then label it in matplotlib.

    PyMOL draws the structures; matplotlib puts the legend above them, so every
    colour in the picture is named before it is seen.
    """
    import matplotlib.pyplot as plt

    af = pl.read_csv(DATA / "02_alphafold_models.csv").filter(
        pl.col("protein") == "P66"
    ).row(0, named=True)
    p66_pdb = Path(af["pdb_path"])
    cd47_pdb = DATA / "cache" / "cd47_igv.pdb"
    start, end = loop_span()
    face = pl.read_csv(DATA / "04_cd47_sirpalpha_face.csv")
    face_sel = "+".join(str(p) for p in face["cd47_position"].to_list())

    work = DATA / "cache" / "superposition"
    work.mkdir(parents=True, exist_ok=True)
    # -o writes the superposed coordinates; the rotation is US-align's, not ours.
    subprocess.run(
        [str(USALIGN), str(p66_pdb), str(cd47_pdb), "-o", str(work / "sup")],
        check=True, capture_output=True, text=True,
    )
    moved_p66 = work / "sup.pdb"  # chain 1, rotated onto chain 2
    if not moved_p66.exists():
        raise RuntimeError(f"US-align wrote no superposition to {moved_p66}")

    def rgb(hex_colour):
        return (
            f"[{int(hex_colour[1:3], 16) / 255:.3f}, "
            f"{int(hex_colour[3:5], 16) / 255:.3f}, "
            f"{int(hex_colour[5:7], 16) / 255:.3f}]"
        )

    png = work / "superposition.png"
    # PyMOL will not take a "#rrggbb" string, so every colour is defined by name
    # from its RGB triple first. Named colours also keep the legend and the
    # picture reading from the same four values.
    script = f"""
set_color c_p66, {rgb(C_P66)}
set_color c_cd47, {rgb(C_CD47)}
set_color c_loop, {rgb(C_LOOP)}
set_color c_face, {rgb(C_FACE)}
load {moved_p66}, p66
load {cd47_pdb}, cd47
hide everything
bg_color white
set ray_opaque_background, 1
set cartoon_trace_atoms, 0
show ribbon, p66
set ribbon_width, 1.2
show cartoon, cd47
set cartoon_transparency, 0.25, cd47
color c_p66, p66
color c_cd47, cd47
show cartoon, p66 and resi {start}-{end}
set cartoon_transparency, 0.0, p66
color c_loop, p66 and resi {start}-{end}
color c_face, cd47 and resi {face_sel}
show spheres, p66 and resi {start}-{end} and name CA
show spheres, cd47 and resi {face_sel} and name CA
set sphere_scale, 0.42
show spheres, p66 and resi {start}-{end} and not name N+C+O
set stick_radius, 0.3
show sticks, p66 and resi {start}-{end}
color c_loop, p66 and resi {start}-{end}
orient cd47
turn x, -10
# The view must hold both marks, or the figure cannot show where the loop lands.
zoom (cd47) or (p66 and resi {start}-{end}), 5
set antialias, 2
set ray_trace_mode, 0
png {png}, width=1500, height=1200, dpi=450, ray=1
"""
    pml = work / "render.pml"
    pml.write_text(script)
    result = subprocess.run([str(PYMOL), "-cq", str(pml)], capture_output=True,
                            text=True)
    if not png.exists():
        raise RuntimeError(
            f"PyMOL wrote no image to {png}: {result.stdout[-600:]} "
            f"{result.stderr[-600:]}"
        )

    # How far the loop lands from the face, in the same superposition the picture
    # shows. A loop that mimics the face would sit on it, not across the cell.
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa

    def centroid(pdb_path, positions):
        chain = next(iter(PDBParser(QUIET=True).get_structure("s", pdb_path)[0]))
        coords = [
            r["CA"].coord
            for r in chain
            if is_aa(r, standard=True) and r.id[1] in positions and "CA" in r
        ]
        return np.mean(coords, axis=0)

    loop_centroid = centroid(moved_p66, set(range(start, end + 1)))
    face_centroid = centroid(cd47_pdb, set(face["cd47_position"].to_list()))
    centroid_gap = float(np.linalg.norm(loop_centroid - face_centroid))

    # The domain's own size, so that gap can be read against something.
    cd47_chain = next(iter(PDBParser(QUIET=True).get_structure("c", cd47_pdb)[0]))
    cd47_ca = np.array(
        [r["CA"].coord for r in cd47_chain if is_aa(r, standard=True) and "CA" in r]
    )
    cd47_width = float(
        np.max(np.linalg.norm(cd47_ca[:, None, :] - cd47_ca[None, :, :], axis=-1))
    )

    claimed = pl.read_csv(DATA / "05_wholefold_claimed.csv").row(0, named=True)
    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=78)
    ax.imshow(plt.imread(png))
    ax.axis("off")
    handles = [
        plt.Line2D([], [], color=C_LOOP, lw=3,
                   label=f"P66 loop in the claim ({start}-{end})"),
        plt.Line2D([], [], color=C_FACE, lw=3,
                   label=f"CD47 face that touches SIRPalpha ({face.height} residues)"),
        plt.Line2D([], [], color=C_P66, lw=3, label="rest of P66"),
        plt.Line2D([], [], color=C_CD47, lw=3, label="rest of CD47"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=5)
    ax.set_title(
        f"US-align superposition: TM-score {claimed['tm_score_norm_target']:.2f} "
        f"normalised by CD47,\naligned {claimed['aligned_length']} residues, "
        f"RMSD {claimed['rmsd_angstrom']:.2f} A.\nThe loop's centre sits "
        f"{centroid_gap:.0f} A from the centre of the binding face.",
        fontsize=5, pad=2,
    )
    pubfig.save(fig, FIGDIR / "276b_p66_cd47_superposition")
    pl.DataFrame(
        [
            {
                "loop_to_face_centroid_angstrom": centroid_gap,
                "cd47_igv_widest_ca_ca_angstrom": cd47_width,
                **claimed,
            }
        ]
    ).write_csv(DATA / "09_superposition_summary.csv")
    return {
        "png": str(png),
        "loop_to_face_centroid_angstrom": centroid_gap,
        "cd47_igv_widest_ca_ca_angstrom": cd47_width,
        **claimed,
    }


# ---------------------------------------------------------------------------
# 276c and 276d: the two backgrounds
# ---------------------------------------------------------------------------


def _strip(ax, values, claimed_value, claimed_label, bg_label, rng):
    jitter = rng.uniform(-0.09, 0.09, size=len(values))
    ax.scatter(np.zeros(len(values)) + jitter, values, s=7, color=C_BG,
               edgecolor="none", zorder=2, label=bg_label)
    ax.scatter([0], [claimed_value], s=28, marker="D", color=C_CLAIM,
               edgecolor="black", linewidth=0.3, zorder=3, label=claimed_label)
    ax.set_xlim(-0.35, 0.35)
    ax.set_xticks([])


def figure_wholefold_background() -> dict:
    claimed = pl.read_csv(DATA / "05_wholefold_claimed.csv").row(0, named=True)
    bg = pl.read_csv(DATA / "05_wholefold_background.csv")
    values = bg["tm_score_norm_target"].to_list()
    rng = np.random.default_rng(276)

    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=58)
    ax.axhline(SAME_FOLD_TM, color="black", lw=0.5, ls=":", zorder=1,
               label=f"TM-score {SAME_FOLD_TM}, the usual same-fold line")
    _strip(ax, values, claimed["tm_score_norm_target"],
           "P66 against CD47 (the claim)",
           f"{len(values)} immunoglobulin domains against CD47", rng)
    ax.set_ylabel("TM-score against CD47's IgV domain\n(higher means more alike)")
    ax.set_ylim(0, 1)
    pubfig.shared_legend(fig, ncol=1, fontsize=5)
    pubfig.save(fig, FIGDIR / "276c_wholefold_tmscore_background")
    n_above = sum(v > claimed["tm_score_norm_target"] for v in values)
    return {
        "claimed_tm": claimed["tm_score_norm_target"],
        "n_background": len(values),
        "background_median": float(np.median(values)),
        "background_min": float(min(values)),
        "background_max": float(max(values)),
        "n_background_above_claim": n_above,
    }


def figure_loop_background() -> dict:
    claimed = pl.read_csv(DATA / "07_claimed_loop_vs_face.csv").sort(
        "backbone_rmsd_angstrom"
    ).row(0, named=True)
    bg = pl.read_csv(DATA / "07_background_loops_vs_face.csv")
    values = bg["backbone_rmsd_angstrom"].to_list()
    rng = np.random.default_rng(276)

    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=58)
    _strip(ax, values, claimed["backbone_rmsd_angstrom"],
           "the loop in the claim", f"{len(values)} other P66 loops, same length", rng)
    ax.set_ylabel("backbone RMSD on the binding face, A\n(lower means a better fit)")
    ax.set_ylim(0, max(max(values), claimed["backbone_rmsd_angstrom"]) * 1.25)
    pubfig.shared_legend(fig, ncol=1, fontsize=5)
    pubfig.save(fig, FIGDIR / "276d_loop_to_face_rmsd_background")
    n_better = sum(v < claimed["backbone_rmsd_angstrom"] for v in values)
    return {
        "claimed_rmsd": claimed["backbone_rmsd_angstrom"],
        "claimed_window": f"{claimed['window_start']}-{claimed['window_end']}",
        "n_background": len(values),
        "background_median": float(np.median(values)),
        "background_best": float(min(values)),
        "n_background_better_than_claim": n_better,
    }


def figure_claim_diagram() -> dict:
    """The claim drawn out: which stretch of which protein, and against what.

    Each protein is a line with boxes for its features, the way InterPro draws
    one. Both numbering systems are on the figure, because the claim uses mature
    numbering and every file used here uses UniProt numbering.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    acc = pl.read_csv(DATA / "01_accessions.csv")
    p66 = acc.filter(pl.col("role") == "P66 query").row(0, named=True)
    cd47 = acc.filter(pl.col("role") == "CD47").row(0, named=True)
    start, end = loop_span()
    face = pl.read_csv(DATA / "04_cd47_sirpalpha_face.csv")
    face_uniprot = face["cd47_uniprot_position"].to_list()

    fig, ax = pubfig.figure(width_mm=pubfig.TWO_COLUMN_MM, height_mm=52)
    longest = max(p66["length"], cd47["length"])

    def draw(y, name, accession, length, signal_end, label):
        ax.hlines(y, 1, length, color="black", lw=0.8, zorder=2)
        ax.add_patch(Rectangle((1, y - 0.11), signal_end, 0.22, facecolor="white",
                               edgecolor="black", lw=0.5, zorder=3))
        # Below the line: on CD47 the face ticks sit above it.
        ax.text(signal_end + 6, y - 0.3, f"signal peptide 1-{signal_end}",
                fontsize=4.2, va="top")
        ax.text(-0.012 * longest, y, f"{name}\n{accession}\n{length} aa",
                fontsize=5, ha="right", va="center")
        ax.text(length + 8, y, label, fontsize=4.2, va="center")

    draw(1, "P66", p66["accession"], p66["length"], p66["signal_peptide_end"],
         "Borreliella burgdorferi\nouter membrane")
    draw(0, "CD47", cd47["accession"], cd47["length"], cd47["signal_peptide_end"],
         "human\ncell surface")

    ax.add_patch(Rectangle((start, 1 - 0.19), end - start + 1, 0.38,
                           facecolor=C_LOOP, edgecolor="none", zorder=4))
    ax.annotate(
        f"the loop in the claim\nUniProt {start}-{end}, mature 181-187\nQENDKDT, "
        f"D{start + 3} and D{start + 5}",
        xy=(end, 1.2), xytext=(end + 60, 1.72), fontsize=4.6, color=C_LOOP,
        arrowprops=dict(arrowstyle="-", color=C_LOOP, lw=0.5),
    )
    for pos in face_uniprot:
        ax.vlines(pos, -0.19, 0.19, color=C_FACE, lw=0.9, zorder=4)
    ax.annotate(
        f"the face CD47 uses on SIRPalpha\n{len(face_uniprot)} residues within 4.5 A, "
        f"measured from PDB 2JJS\nUniProt {min(face_uniprot)}-{max(face_uniprot)}",
        xy=(float(np.median(face_uniprot)), -0.22), xytext=(220, -0.95),
        fontsize=4.6, color=C_FACE,
        arrowprops=dict(arrowstyle="-", color=C_FACE, lw=0.5),
    )
    ax.text(
        0.5, 1.02,
        "The claim: this loop makes P66 bind SIRPalpha the way CD47 does.\n"
        "Nothing in the claim names CD47 residues, so the comparison uses the face "
        "CD47 itself uses.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=5,
    )
    ax.set_xlim(-0.22 * longest, longest * 1.22)
    ax.set_ylim(-1.35, 2.15)
    ax.set_yticks([])
    ax.set_xlabel("residue (UniProt numbering)")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    pubfig.save(fig, FIGDIR / "276h_claim_diagram")
    return {
        "p66": f"{p66['accession']} {p66['length']} aa",
        "cd47": f"{cd47['accession']} {cd47['length']} aa",
        "loop_uniprot": f"{start}-{end}",
        "face_uniprot_range": f"{min(face_uniprot)}-{max(face_uniprot)}",
    }


def figure_binding_face_map() -> dict:
    """Where the 21 face residues sit along CD47's IgV domain."""
    import matplotlib.pyplot as plt

    face = pl.read_csv(DATA / "04_cd47_sirpalpha_face.csv")
    domain = pl.read_csv(DATA / "05_cd47_igv_domain.csv").row(0, named=True)
    first, last = int(domain["first_position"]), int(domain["last_position"])

    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=38)
    ax.hlines(0, first, last, color=C_CD47, lw=4, zorder=1,
              label=f"CD47 IgV domain, 2JJS chain {domain['chain']} ({first}-{last})")
    ax.vlines(face["cd47_position"], -0.35, 0.35, color=C_FACE, lw=1.1, zorder=2,
              label=f"within 4.5 A of SIRPalpha ({face.height} residues)")
    for pos, dist in zip(face["cd47_position"], face["min_distance_angstrom"]):
        ax.plot([pos], [0.52], marker="v", ms=2.2, color=C_FACE, zorder=3)
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([])
    ax.set_xlabel("CD47 residue (2JJS numbering; add 18 for UniProt Q08722)")
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)
    pubfig.shared_legend(fig, ncol=1, fontsize=5)
    pubfig.save(fig, FIGDIR / "276e_cd47_binding_face_map")
    runs, current = [], [face["cd47_position"][0]]
    for prev, pos in zip(face["cd47_position"][:-1], face["cd47_position"][1:]):
        if pos == prev + 1:
            current.append(pos)
        else:
            runs.append(current)
            current = [pos]
    runs.append(current)
    longest = max(runs, key=len)
    return {
        "n_face_residues": face.height,
        "n_runs": len(runs),
        "longest_run": f"{longest[0]}-{longest[-1]} ({len(longest)} residues)",
        "closest_contact_angstrom": float(face["min_distance_angstrom"].min()),
    }


def figure_foldseek() -> dict:
    """Foldseek's hits for P66 in the PDB, and where CD47 does not appear."""
    hits = pl.read_csv(DATA / "06_foldseek_vs_pdb.csv")
    pair = pl.read_csv(DATA / "06_foldseek_vs_cd47_model.csv").row(0, named=True)

    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=52)
    ranked = hits.sort("evalue")
    ax.plot(range(1, ranked.height + 1), ranked["evalue"], marker="o", ms=2,
            lw=0.5, color=C_BG, label=f"P66 against the PDB ({ranked.height} hits)")
    ax.axhline(pair["evalue"], color=C_CLAIM, lw=1, ls="--",
               label=f"P66 against CD47's model, forced (E = {pair['evalue']:.2f})")
    ax.axhline(0.001, color="black", lw=0.5, ls=":",
               label="E = 0.001, a confident structural hit")
    ax.set_yscale("log")
    ax.set_xlabel("hit, ranked by E-value")
    ax.set_ylabel("Foldseek E-value (lower is a stronger hit)")
    pubfig.shared_legend(fig, ncol=1, fontsize=5)
    pubfig.save(fig, FIGDIR / "276f_foldseek_p66_vs_pdb")
    return {
        "n_pdb_hits": hits.height,
        "best_pdb_evalue": float(hits["evalue"].min()),
        "n_below_0.001": int((hits["evalue"] < 0.001).sum()),
        "cd47_model_evalue": pair["evalue"],
        "cd47_alignment": f"P66 {pair['qstart']}-{pair['qend']} on CD47 "
                          f"{pair['tstart']}-{pair['tend']}",
    }


#: Settings PR 96 swept, read off its committed tables/245_p66_cd47_controls.csv.
#: They decide which cells of the reachability map that PR's result speaks for.
PR96_SETTINGS = [
    ("dayhoff6", 7), ("funcgroups8", 6), ("funcgroups8", 7), ("gbmr4", 10),
    ("gbmr7", 8), ("gbmr7", 10), ("hp_kyte_doolittle2", 17), ("hp_lehninger2", 16),
    ("hp_lehninger2", 17), ("hp_lehninger2", 19), ("hp_lehninger2", 20),
    ("hp_lehninger_c_nonpolar2", 16), ("hp_lehninger_c_nonpolar2", 20),
    ("hp_lehninger_hpc3", 14), ("hp_lehninger_hpc3", 18), ("hp_lehninger_hpc3", 21),
    ("hp_pbotc_1st_ed2", 16), ("hp_thomas_dill2", 17), ("hp_thomas_dill_no_c2", 17),
    ("hsdm17", 5), ("hsdm17", 6), ("mmseqs12", 5), ("polarity4", 9),
    ("polarity4", 11), ("sdm12", 5), ("sdm12", 6), ("wwmj5", 7),
]


def figure_loop_class_strings() -> dict:
    """The loop's residues and the class each becomes, in every alphabet.

    Letters in the grid are class indices relabelled A, B, C..., not residues. In
    protein20 that relabelling happens to be one-to-one with the residues.
    """
    reach = pl.read_csv(DATA / "09_loop_kmer_reach_all_alphabets.csv")
    per = (
        reach.group_by("alphabet", "n_classes")
        .agg(
            pl.col("loop_class_string").first(),
            pl.col("n_distinct_classes_in_loop").first(),
        )
        .sort("n_classes", "alphabet")
    )
    start, end = loop_span()
    residues = pl.read_csv(DATA / "01_loop_numbering.csv")["residues"][0]
    positions = list(range(start, end + 1))

    fig, ax = pubfig.figure(width_mm=pubfig.ONE_COLUMN_MM, height_mm=96)
    for row_i, row in enumerate(per.iter_rows(named=True)):
        uniform = row["n_distinct_classes_in_loop"] == 1
        for col_i, letter in enumerate(row["loop_class_string"]):
            ax.text(col_i, -row_i, letter, ha="center", va="center", fontsize=5.5,
                    color="black")
        ax.text(-0.9, -row_i, f"{row['alphabet']} ({row['n_classes']})",
                ha="right", va="center", fontsize=4.6)
        ax.text(len(residues) - 0.4, -row_i,
                "one repeated class" if uniform
                else f"{row['n_distinct_classes_in_loop']} classes",
                ha="left", va="center", fontsize=4.4,
                color=C_LOOP if uniform else C_BG)
    for col_i, (pos, res) in enumerate(zip(positions, residues)):
        ax.text(col_i, 1.3, res, ha="center", va="center", fontsize=7,
                fontweight="bold", color=C_LOOP)
        ax.text(col_i, 0.72, str(pos), ha="center", va="center", fontsize=4.2,
                color=C_BG, rotation=90)
    ax.text(-0.9, 1.3, "P66 residue", ha="right", va="center", fontsize=4.6)
    ax.set_xlim(-8.2, len(residues) + 4.2)
    ax.set_ylim(-per.height + 0.4, 2.1)
    ax.axis("off")
    ax.set_title(
        "Grid letters are class numbers relabelled A, B, C, not residues.\n"
        "8 of 19 alphabets write the loop as one repeated class.",
        fontsize=5.2, pad=2,
    )
    pubfig.save(fig, FIGDIR / "276g_loop_class_strings_all_alphabets")
    return {
        "residues": residues,
        "n_alphabets": per.height,
        "n_uniform": int((per["n_distinct_classes_in_loop"] == 1).sum()),
        "uniform_alphabets": per.filter(
            pl.col("n_distinct_classes_in_loop") == 1
        )["alphabet"].to_list(),
    }


def figure_loop_reach() -> dict:
    """Whether any k-mer covering the loop occurs in CD47, per alphabet and k.

    This is the question a class string cannot answer on its own: in a larger
    alphabet the loop is a varied string, and only encoding both proteins shows
    whether a k-mer there could ever be shared.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Rectangle

    reach = pl.read_csv(DATA / "09_loop_kmer_reach_all_alphabets.csv")
    order = (
        reach.group_by("alphabet", "n_classes").len().sort("n_classes", "alphabet")
    )
    alphabets = order["alphabet"].to_list()
    ksizes = sorted(reach["ksize"].unique().to_list())
    grid = np.zeros((len(alphabets), len(ksizes)))
    lookup = {
        (r["alphabet"], r["ksize"]): r["n_covering_kmers_found_in_cd47"]
        for r in reach.iter_rows(named=True)
    }
    for i, a in enumerate(alphabets):
        for j, k in enumerate(ksizes):
            grid[i, j] = 1 if lookup.get((a, k), 0) > 0 else 0

    fig, ax = pubfig.figure(width_mm=pubfig.TWO_COLUMN_MM, height_mm=86)
    ax.imshow(grid, cmap=ListedColormap(["#F2F2F2", C_FACE]), aspect="auto",
              vmin=0, vmax=1, interpolation="nearest")
    swept = 0
    for a, k in PR96_SETTINGS:
        if a in alphabets and k in ksizes:
            ax.add_patch(Rectangle((ksizes.index(k) - 0.5, alphabets.index(a) - 0.5),
                                   1, 1, fill=False, edgecolor=C_LOOP, lw=1.1,
                                   zorder=3))
            swept += 1
    ax.set_yticks(range(len(alphabets)))
    ax.set_yticklabels(
        [f"{a} ({order['n_classes'][i]})" for i, a in enumerate(alphabets)],
        fontsize=5,
    )
    ax.set_xticks(range(len(ksizes)))
    ax.set_xticklabels([str(k) for k in ksizes], fontsize=5)
    ax.set_xlabel("k-mer size")
    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=5, color=C_FACE,
                   label="at least one k-mer covering the loop occurs in CD47"),
        plt.Line2D([], [], marker="s", ls="", ms=5, color="#F2F2F2",
                   markeredgecolor=C_BG, markeredgewidth=0.4,
                   label="none does"),
        plt.Line2D([], [], marker="s", ls="", ms=5, markerfacecolor="none",
                   markeredgecolor=C_LOOP, markeredgewidth=1.1,
                   label=f"one of the {swept} settings PR 96 searched"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=1, fontsize=5)
    pubfig.save(fig, FIGDIR / "276i_loop_kmer_reach_by_alphabet_and_k")
    at_swept = sum(lookup.get((a, k), 0) for a, k in PR96_SETTINGS)
    reachable = reach.filter(pl.col("n_covering_kmers_found_in_cd47") > 0)
    return {
        "n_alphabets": len(alphabets),
        "k_range": f"{min(ksizes)}-{max(ksizes)}",
        "n_settings_with_any_reach": int(grid.sum()),
        "highest_k_with_any_reach": int(reachable["ksize"].max()),
        "loop_kmers_in_cd47_at_pr96_settings": int(at_swept),
        "n_pr96_settings_marked": swept,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-pymol", action="store_true")
    args = parser.parse_args()
    pubfig.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)

    print("276a", figure_plddt())
    if not args.skip_pymol:
        print("276b", figure_superposition())
    print("276c", figure_wholefold_background())
    print("276d", figure_loop_background())
    print("276h", figure_claim_diagram())
    print("276e", figure_binding_face_map())
    print("276f", figure_foldseek())
    print("276g", figure_loop_class_strings())
    print("276i", figure_loop_reach())
    print(f"\nFigures in {FIGDIR}")


if __name__ == "__main__":
    main()
