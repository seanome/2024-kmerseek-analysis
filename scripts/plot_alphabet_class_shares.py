"""Class shares of Swiss-Prot residues for all 19 kmerseek alphabets.

One row per alphabet. Each bar is every Swiss-Prot residue, split by the class of that
alphabet it falls in, with the residues of each class printed inside its segment. The
number at the right is the sum of squared class shares, the chance that two unrelated
residues fall in the same class.

Class definitions are copied from kmerseek's ``src/rust/alphabets.rs`` (dayhoff6 from
sourmash). Composition is UniProtKB/Swiss-Prot release 2026_03 (575,748 entries,
209,017,843 residues), https://web.expasy.org/docs/relnotes/relstat.html.

Usage:
    python scripts/plot_alphabet_class_shares.py [--out figures/alphabet_class_shares]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Swiss-Prot 2026_03 amino-acid composition, as a share of all residues.
SWISSPROT_COMPOSITION: dict[str, float] = {
    "A": 0.0826, "C": 0.0138, "D": 0.0547, "E": 0.0672, "F": 0.0386,
    "G": 0.0708, "H": 0.0227, "I": 0.0591, "K": 0.0580, "L": 0.0965,
    "M": 0.0241, "N": 0.0406, "P": 0.0476, "Q": 0.0393, "R": 0.0553,
    "S": 0.0667, "T": 0.0537, "V": 0.0686, "W": 0.0110, "Y": 0.0292,
}  # fmt: skip

# Classes in the order alphabets.rs lists them, so neighbouring segments match the source.
ALPHABETS: dict[str, list[str]] = {
    "protein20": list("ACDEFGHIKLMNPQRSTVWY"),
    "uniprot18": ["A", "R", "N", "D", "C", "Q", "EP", "G", "HL", "I", "K", "M", "F",
                  "S", "T", "W", "Y", "V"],
    "hsdm17": ["A", "D", "KE", "R", "N", "T", "S", "Q", "Y", "F", "LIV", "M", "C", "W",
               "H", "G", "P"],
    "wass14": ["WM", "DI", "P", "C", "AV", "K", "T", "RE", "G", "L", "Y", "SH", "F", "NQ"],
    "mmseqs12": ["AST", "LM", "IV", "KR", "EQ", "ND", "FY", "C", "G", "H", "P", "W"],
    "sdm12": ["A", "D", "KER", "N", "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"],
    "funcgroups8": ["GVALI", "ST", "CM", "FY", "WHP", "NQ", "DE", "KR"],
    "gbmr7": ["DN", "AEFIKLMQRVWY", "CH", "T", "S", "G", "P"],
    "dayhoff6": ["C", "AGPST", "DENQ", "FWY", "HKR", "ILMV"],
    "wwmj5": ["CMFILVWY", "ATH", "GP", "DE", "SNQRK"],
    "gbmr4": ["ADKERNTSQ", "YFLIVMCWH", "G", "P"],
    "polarity4": ["GAVLIFWMP", "STCYNQ", "DE", "HKR"],
    "hp_lehninger_hpc3": ["AFGILMPVWY", "DEHKNQRST", "C"],
    "hp_lehninger2": ["AFGILMPVWY", "CDEHKNQRST"],
    "hp_lehninger_c_nonpolar2": ["ACFGILMPVWY", "DEHKNQRST"],
    "hp_pbotc_1st_ed2": ["ACFILMPVWY", "DEGHKNQRST"],
    "hp_thomas_dill2": ["ACFILMVWY", "DEGHKNPQRST"],
    "hp_thomas_dill_no_c2": ["AFILMVWY", "CDEGHKNPQRST"],
    "hp_kyte_doolittle2": ["ACFILMV", "DEGHKNPQRSTWY"],
}  # fmt: skip

BAR = "#a9c1ec"
INK = "#1b1f24"
MUTED = "#5b6673"
MONO = "DejaVu Sans Mono"
LABEL_PT = 7.0
MIN_LABEL_PT = 5.0
# DejaVu Sans Mono advance width, as a fraction of the font size.
MONO_EM = 0.602


def class_shares(classes: list[str]) -> list[float]:
    for residue in SWISSPROT_COMPOSITION:
        assert sum(residue in c for c in classes) == 1, residue
    total = sum(SWISSPROT_COMPOSITION.values())
    return [sum(SWISSPROT_COMPOSITION[r] for r in c) / total for c in classes]


def main(out: Path) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "xtick.major.size": 2.5,
    })  # fmt: skip
    # Same order as the artifact: most letters first, ties by name.
    order = sorted(ALPHABETS, key=lambda a: (-len(ALPHABETS[a]), a))

    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_axes([0.235, 0.075, 0.625, 0.82])
    bar_h = 0.74
    bar_width_pt = ax.get_position().width * fig.get_figwidth() * 72

    shrunk = []
    for row, name in enumerate(order):
        classes = ALPHABETS[name]
        shares = class_shares(classes)
        left = 0.0
        for c, q in zip(classes, shares):
            ax.barh(row, q, left=left, height=bar_h, color=BAR,
                    edgecolor="white", linewidth=0.8)  # fmt: skip
            seg_pt = q * bar_width_pt - 1.2
            size = min(LABEL_PT, seg_pt / (len(c) * MONO_EM))
            if size < LABEL_PT:
                shrunk.append((name, c, round(size, 1)))
            ax.text(left + q / 2, row, c, ha="center", va="center", color=INK,
                    fontsize=max(size, MIN_LABEL_PT), family=MONO)  # fmt: skip
            left += q
        s2 = sum(q * q for q in shares)
        ax.text(1.02, row, f"{s2:.3f}", ha="left", va="center", family=MONO,
                fontsize=7.5, transform=ax.get_yaxis_transform())  # fmt: skip

    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_yticks(range(len(order)), order, family=MONO, fontsize=7.5)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100"])
    ax.set_xlabel("Share of Swiss-Prot residues (%)")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    head_y = 0.905
    pos = ax.get_position()
    fig.text(pos.x0 + pos.width / 2, head_y,
             "Classes of each alphabet, with the residues in each class",
             ha="center", va="bottom", fontsize=7.5, color=INK)  # fmt: skip
    fig.text(pos.x1 + 0.02 * pos.width, head_y,
             "Same class\nby chance,\n$\\Sigma_c\\, q_c^2$",
             ha="left", va="bottom", fontsize=7, color=INK, linespacing=1.2)  # fmt: skip
    fig.text(pos.x0 - 0.01, head_y, "Alphabet", ha="right", va="bottom",
             fontsize=7.5, color=INK)  # fmt: skip

    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=300)
    print(f"wrote {out}.{{pdf,svg,png}}")
    print("labels set below 7 pt to fit their segment:")
    for name, c, size in shrunk:
        print(f"  {name:26s} {c:4s} {size} pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("figures/alphabet_class_shares"))
    main(p.parse_args().out)
