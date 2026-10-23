#!/usr/bin/env python3
"""Where the binding face and the aligned segment sit on CD47's UniProt features.

The face and the segment have been carried through this notebook as residue
ranges with no reference to what CD47 actually has there. This fetches CD47's
whole feature set from UniProt and draws both on top of it, which settles three
things prose had got loosely or not at all:

  * the segment was described as "92% inside the ectodomain". That is right for
    the extracellular topological domain, 19-141, but the immunoglobulin domain
    itself ends at 127, so only about half the segment is in the domain and three
    columns are inside the first transmembrane helix.
  * the CD47 stretch opposite the P66 loop begins with a cysteine that is locked
    in a disulfide bond to residue 41.
  * an N-linked glycosylation site sits two residues inside the segment.

Writes figures/276u_cd47_feature_overlay and a table of what each feature
overlaps.

Run with the project interpreter:
  /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 \
      scripts/nb276_cd47_feature_overlay.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "notebooks"))
import pubfig as pf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb276_p66_cd47_structure import (  # noqa: E402
    LOOP_MATURE_END,
    LOOP_MATURE_START,
    OUTDIR,
    resolve_uniprot,
)
from nb276_why_combining_hurts import CD47_SEGMENT, P66_SEGMENT  # noqa: E402

FIGDIR = Path(__file__).resolve().parent.parent / "figures"
C_FACE = pf.OKABE_ITO["blue"]
C_SEG = pf.OKABE_ITO["vermillion"]
C_DOM = pf.OKABE_ITO["bluish_green"]
C_TM = pf.OKABE_ITO["reddish_purple"]
C_GREY = pf.GREY
#: Zoom panel range, chosen to hold the segment and the domain boundary.
ZOOM = (100, 150)


def features(accession: str) -> list[dict]:
    """CD47's features, cached, straight from the UniProt entry."""
    cache = OUTDIR / "cache" / f"uniprot_features_{accession}.json"
    if not cache.exists():
        with urllib.request.urlopen(
            f"https://rest.uniprot.org/uniprotkb/{accession}?format=json"
        ) as fh:
            cache.write_bytes(fh.read())
    entry = json.loads(cache.read_text())
    out = []
    for feature in entry.get("features", []):
        location = feature["location"]
        out.append(
            {
                "type": feature["type"],
                "start": location["start"]["value"],
                "end": location["end"]["value"],
                "description": feature.get("description", ""),
            }
        )
    return out


def main() -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    pf.use_style()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    cache = OUTDIR / "cache"
    p66 = resolve_uniprot("p66", 139, cache / "uniprot_p66.json")
    cd47 = resolve_uniprot("CD47", 9606, cache / "uniprot_cd47.json")
    offset = p66["signal_peptide_end"]
    loop = (LOOP_MATURE_START + offset, LOOP_MATURE_END + offset)
    face = sorted(
        pl.read_csv(OUTDIR / "04_cd47_sirpalpha_face.csv")[
            "cd47_uniprot_position"
        ].to_list()
    )
    feats = features(cd47["accession"])
    length = cd47["length"]

    def pick(kind, description=None):
        return [
            f
            for f in feats
            if f["type"] == kind
            and (description is None or description.lower() in f["description"].lower())
        ]

    ig = pick("Domain")[0]
    extracellular = [f for f in pick("Topological domain") if "Extracel" in f["description"]]
    cytoplasmic = [f for f in pick("Topological domain") if "Cytopl" in f["description"]]
    tms = pick("Transmembrane")
    strands = pick("Beta strand")
    helices = pick("Helix")
    glycans = pick("Glycosylation")
    disulfides = pick("Disulfide bond")
    signal = pick("Signal")[0]

    # The CD47 stretch opposite the P66 loop.
    opp_start = CD47_SEGMENT[0] + (loop[0] - P66_SEGMENT[0])
    opp_end = CD47_SEGMENT[0] + (loop[1] - P66_SEGMENT[0])
    seg = CD47_SEGMENT

    def overlap(a, b, c, d):
        return max(0, min(b, d) - max(a, c) + 1)

    rows = [
        {
            "feature": f"{f['type']} {f['start']}-{f['end']}"
            + (f" ({f['description']})" if f["description"] else ""),
            "residues_in_segment_109_144": overlap(seg[0], seg[1], f["start"], f["end"]),
            "residues_in_stretch_opposite_the_loop": overlap(
                opp_start, opp_end, f["start"], f["end"]
            ),
            "face_residues_inside": sum(1 for p in face if f["start"] <= p <= f["end"]),
        }
        for f in feats
        if f["type"]
        in {
            "Domain", "Topological domain", "Transmembrane", "Signal", "Chain",
            "Glycosylation", "Disulfide bond", "Beta strand", "Helix",
            "Modified residue",
        }
    ]
    table = pl.DataFrame(rows).filter(
        (pl.col("residues_in_segment_109_144") > 0) | (pl.col("face_residues_inside") > 0)
    ).sort("residues_in_segment_109_144", descending=True)
    table.write_csv(OUTDIR / "41_cd47_feature_overlap.csv")

    fig, (ax, az) = pf.figure(pf.TWO_COLUMN_MM, 104, nrows=2)

    def draw(axis, lo, hi):
        y = {"topo": 5, "dom": 4, "ss": 3, "ptm": 2, "face": 1, "seg": 0}
        axis.hlines(y["topo"], max(1, lo), min(length, hi), color=C_GREY, lw=0.8,
                    zorder=1)
        for f in extracellular:
            axis.add_patch(Rectangle((f["start"], y["topo"] - 0.17),
                                     f["end"] - f["start"] + 1, 0.34,
                                     facecolor="white", edgecolor=C_GREY, lw=0.5))
        for f in cytoplasmic:
            axis.add_patch(Rectangle((f["start"], y["topo"] - 0.17),
                                     f["end"] - f["start"] + 1, 0.34,
                                     facecolor=C_GREY, edgecolor="none", alpha=0.35))
        for f in tms:
            axis.add_patch(Rectangle((f["start"], y["topo"] - 0.26),
                                     f["end"] - f["start"] + 1, 0.52,
                                     facecolor=C_TM, edgecolor="none"))
        axis.add_patch(Rectangle((signal["start"], y["topo"] - 0.26),
                                 signal["end"] - signal["start"] + 1, 0.52,
                                 facecolor="none", edgecolor=C_GREY, lw=0.6))
        axis.add_patch(Rectangle((ig["start"], y["dom"] - 0.22),
                                 ig["end"] - ig["start"] + 1, 0.44,
                                 facecolor=C_DOM, edgecolor="none", alpha=0.55))
        for f in strands:
            axis.add_patch(Rectangle((f["start"], y["ss"] - 0.16),
                                     f["end"] - f["start"] + 1, 0.32,
                                     facecolor=C_GREY, edgecolor="none"))
        for f in helices:
            axis.add_patch(Rectangle((f["start"], y["ss"] - 0.09),
                                     f["end"] - f["start"] + 1, 0.18,
                                     facecolor=C_GREY, edgecolor="none", alpha=0.5))
        for f in glycans:
            axis.plot([f["start"]], [y["ptm"]], marker="o", ms=3.2, mfc="white",
                      mec=C_GREY, mew=0.7, zorder=3)
        for f in disulfides:
            axis.plot([f["start"], f["end"]], [y["ptm"] - 0.28, y["ptm"] - 0.28],
                      color=C_GREY, lw=0.6, zorder=2)
            axis.plot([f["start"], f["end"]], [y["ptm"] - 0.28] * 2, marker="|",
                      ms=3, color=C_GREY, lw=0, zorder=3)
        axis.vlines(face, y["face"] - 0.22, y["face"] + 0.22, color=C_FACE, lw=1.0,
                    zorder=3)
        axis.add_patch(Rectangle((seg[0], y["seg"] - 0.22), seg[1] - seg[0] + 1, 0.44,
                                 facecolor=C_SEG, edgecolor="none", alpha=0.75))
        axis.add_patch(Rectangle((opp_start, y["seg"] - 0.3), opp_end - opp_start + 1,
                                 0.6, facecolor="none", edgecolor=C_SEG, lw=0.9))
        axis.set_yticks(list(y.values()))
        axis.set_yticklabels([
            "topology", "Ig-like V-type", "secondary structure",
            "glycans, disulfides", "touches SIRPα", "aligned to P66",
        ])
        axis.set_ylim(-0.75, 5.6)
        axis.set_xlim(lo, hi)
        for spine in ("left", "right", "top"):
            axis.spines[spine].set_visible(False)
        axis.tick_params(axis="y", length=0)

    draw(ax, 1, length)
    ax.set_xlabel("CD47 residue (UniProt Q08722)")
    pf.panel_label(ax, "a")
    draw(az, *ZOOM)
    az.set_xlabel(f"CD47 residue, {ZOOM[0]}–{ZOOM[1]}")
    az.axvline(ig["end"] + 0.5, color=C_DOM, lw=0.8, ls="--", zorder=4)
    az.annotate(f"Ig domain ends {ig['end']}", (ig["end"] + 0.5, 4.9),
                xytext=(3, 0), textcoords="offset points", fontsize=5, color=C_DOM)
    az.axvline(tms[0]["start"] - 0.5, color=C_TM, lw=0.8, ls="--", zorder=4)
    az.annotate(f"membrane starts {tms[0]['start']}", (tms[0]["start"] - 0.5, 4.9),
                xytext=(-3, 0), textcoords="offset points", fontsize=5, color=C_TM,
                ha="right")
    pf.panel_label(az, "b")

    handles = [
        plt.Line2D([], [], marker="s", ls="", ms=5, color=C_SEG,
                   label="aligned to P66 197–232 (outline: opposite the loop)"),
        plt.Line2D([], [], marker="|", ls="", ms=7, color=C_FACE, mew=1.5,
                   label="within 4.5 Å of SIRPα"),
        plt.Line2D([], [], marker="s", ls="", ms=5, color=C_DOM, alpha=0.55,
                   label=f"Ig-like V-type domain {ig['start']}–{ig['end']}"),
        plt.Line2D([], [], marker="s", ls="", ms=5, color=C_TM,
                   label="transmembrane helix"),
        plt.Line2D([], [], marker="o", ls="", ms=4, mfc="white", mec=C_GREY,
                   label="N-glycosylation site"),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3)
    pf.save(fig, FIGDIR / "276u_cd47_feature_overlay")

    pl.Config.set_tbl_rows(30)
    pl.Config.set_tbl_width_chars(200)
    print(f"CD47 {cd47['accession']}, {length} aa. Segment aligned to P66 "
          f"{P66_SEGMENT[0]}-{P66_SEGMENT[1]} is CD47 {seg[0]}-{seg[1]}; the stretch "
          f"opposite the loop is {opp_start}-{opp_end}.\n")
    print(table)
    in_ig = overlap(seg[0], seg[1], ig["start"], ig["end"])
    in_extra = sum(
        overlap(seg[0], seg[1], f["start"], f["end"]) for f in extracellular
    )
    in_tm = sum(overlap(seg[0], seg[1], f["start"], f["end"]) for f in tms)
    print(f"\nOf the segment's {seg[1] - seg[0] + 1} CD47 columns:")
    print(f"  {in_ig} are in the Ig-like V-type domain ({ig['start']}-{ig['end']}), "
          f"{100 * in_ig / 36:.0f}%")
    print(f"  {in_extra} are in an extracellular topological domain, "
          f"{100 * in_extra / 36:.0f}%")
    print(f"  {in_tm} are in the first transmembrane helix "
          f"({tms[0]['start']}-{tms[0]['end']})")
    print(f"\nFace residues inside the Ig domain: "
          f"{sum(1 for p in face if ig['start'] <= p <= ig['end'])} of {len(face)}; "
          f"the rest are {[p for p in face if p > ig['end']]}.")
    for f in disulfides:
        if seg[0] <= f["end"] <= seg[1] or seg[0] <= f["start"] <= seg[1]:
            print(f"Disulfide {f['start']}-{f['end']} has one cysteine inside the "
                  f"segment.")
    for f in glycans:
        if seg[0] <= f["start"] <= seg[1]:
            print(f"N-glycosylation site at {f['start']} is inside the segment.")
    print(f"\nWrote {OUTDIR / '41_cd47_feature_overlap.csv'}")


if __name__ == "__main__":
    main()
