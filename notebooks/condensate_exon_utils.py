"""Per-exon sequence scores for notebook 341 (condensate-like exons).

One function per score. Each takes residues and returns one number, so the notebook can
build one row per exon with one named column per score. Nothing here is estimated or
fitted: every value is computed from the residues.

Terms used below:

* sticker / spacer: in a disordered region that phase-separates, the aromatic residues
  (F, W, Y) are "stickers" that bind each other, and the stretches between them are
  "spacers" (Martin et al. 2020).
* kappa: how evenly the positive and negative residues are mixed along a sequence, from
  0 (perfectly mixed) to 1 (all of one charge on one side, all of the other on the
  other) (Das and Pappu 2013).
* IUPred: a per-residue disorder score from 0 to 1, where above 0.5 is read as
  disordered (Dosztanyi et al. 2005). The "long" mode is used here.
* pI: the pH at which the sequence carries no net charge.
"""

from __future__ import annotations

import os
import subprocess

from Bio.SeqUtils.ProtParam import ProteinAnalysis

# aro_hyd_chg_pol4, the alphabet added to kmerseek on branch olgabot/aro-hyd-chg-pol4.
# The letter of each class is the first residue of its cluster, lowercased, as kmerseek
# does (`build_reduced` in src/rust/alphabets.rs).
ARO_HYD_CHG_POL4_CLUSTERS = {
    "aromatic": "FWY",
    "other_hydrophobic": "AILMV",
    "charged": "DEKR",
    "polar": "SNQTGCPH",
}
ARO_HYD_CHG_POL4 = {
    aa: cl[0].lower() for cl in ARO_HYD_CHG_POL4_CLUSTERS.values() for aa in cl
}

# hp_pbotc_1st_ed2: Phillips, Kondev and Theriot 2008, Physical Biology of the Cell, 1st
# edition, Figure 8.30; the same table as kmerseek's PBOTC_1ST_ED_HP.
HP_PBOTC_1ST_ED2 = {
    **{aa: "h" for aa in "ACFILMPVWY"},
    **{aa: "p" for aa in "DEGHKNQRST"},
}

AROMATIC = set("FWY")
BASIC = set("KR")
ACIDIC = set("DE")

# The IUPred build used here: compiled with Apple clang from the iupred_string.c that
# InterProScan ships under bin/mobidb/idrpred/bin/IUPred/src.zip.
IUPRED_DIR = os.environ.get(
    "IUPRED_DIR", "/Users/olga/data/condensate_exons_341/tools/iupred"
)


def encode(residues: str, table: dict[str, str]) -> str:
    """Write `residues` in a reduced alphabet; raises KeyError on a non-canonical residue."""
    return "".join(table[aa] for aa in residues)


def class_fractions(residues: str) -> dict[str, float]:
    """Fraction of residues in each aro_hyd_chg_pol4 class, keyed `frac_<class>`."""
    n = len(residues)
    return {
        f"frac_{name}": sum(aa in cluster for aa in residues) / n
        for name, cluster in ARO_HYD_CHG_POL4_CLUSTERS.items()
    }


def aromatic_count(residues: str) -> int:
    """Number of F, W and Y residues."""
    return sum(aa in AROMATIC for aa in residues)


def aromatic_fraction(residues: str) -> float:
    """Aromatic residues (F, W, Y) as a fraction of all residues."""
    return aromatic_count(residues) / len(residues)


def net_charge_per_residue(residues: str) -> float:
    """(K + R - D - E) / length, at neutral pH with H counted as uncharged."""
    pos = sum(aa in BASIC for aa in residues)
    neg = sum(aa in ACIDIC for aa in residues)
    return (pos - neg) / len(residues)


def fraction_basic(residues: str) -> float:
    """(K + R) / length."""
    return sum(aa in BASIC for aa in residues) / len(residues)


def kappa(residues: str) -> float:
    """Das-Pappu kappa from localCIDER (the Pappu lab's package).

    localCIDER returns -1 when a sequence has no charged residue; that is passed through
    as NaN, since kappa is undefined there. With only a few charged residues kappa is
    defined but carries little information (two charges of opposite sign can give 1.0),
    so read it next to `n_charged`.
    """
    from localcider.sequenceParameters import SequenceParameters

    value = SequenceParameters(residues).get_kappa()
    return float("nan") if value < 0 else float(value)


def isoelectric_point(residues: str) -> float:
    """pI by Biopython's ProteinAnalysis (Bjellqvist pK values)."""
    return ProteinAnalysis(residues).isoelectric_point()


def iupred_long(protein: str) -> list[float]:
    """Per-residue IUPred long-mode scores for a whole protein."""
    out = subprocess.run(
        [os.path.join(IUPRED_DIR, "iupred_string"), protein, "long"],
        env={"IUPred_PATH": IUPRED_DIR},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    scores = [
        float(line.split()[2])
        for line in out.splitlines()
        if line.strip()[:1].isdigit()
    ]
    if len(scores) != len(protein):
        raise ValueError(
            f"IUPred returned {len(scores)} scores for {len(protein)} residues"
        )
    return scores


def iupred_mean_in_context(protein: str, start: int, end: int) -> float:
    """Mean IUPred long score over residues start..end (1-based, inclusive).

    IUPred is run on the whole isoform and then averaged over the exon, because its score
    for a residue depends on the residues around it; running it on the exon alone would
    give the exon's ends a context they do not have in the protein.
    """
    scores = iupred_long(protein)
    window = scores[start - 1 : end]
    return sum(window) / len(window)


def bulk_scores(residues: str) -> dict[str, float]:
    """Every residue-only score in one row, for an exon or a whole isoform."""
    return {
        "length_aa": len(residues),
        "kappa": kappa(residues),
        "n_charged": sum(aa in BASIC or aa in ACIDIC for aa in residues),
        "aromatic_count": aromatic_count(residues),
        "aromatic_fraction": aromatic_fraction(residues),
        **class_fractions(residues),
        "net_charge_per_residue": net_charge_per_residue(residues),
        "fraction_basic": fraction_basic(residues),
        "pI": isoelectric_point(residues),
    }
