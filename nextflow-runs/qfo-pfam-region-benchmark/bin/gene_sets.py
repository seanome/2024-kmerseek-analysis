#!/usr/bin/env python3
"""Curated human gene sets the 200-series notebooks stratify on.

Copied deliberately rather than imported. notebooks/ortholog_analysis_utils.py is the
origin of every list here, but the pipeline runs inside a container that has no reason to
carry the notebook tree, and importing across that boundary would make the container
depend on the analysis repo's layout. `bin/check_gene_sets.py` diffs these against `ou`
whenever it is run somewhere both are importable, so the copy cannot drift silently.

Provenance for each set is recorded next to it, because "which genes count as immune" is
the kind of judgement that becomes uncheckable once it is three files deep.
"""

# ---------------------------------------------------------------------------
# MHC. ou.MHC_CLASSES -- the 25 genes notebooks 210-216 score, grouped into the 7 classes
# those notebooks report by. The class split is not cosmetic: 211 found class I and class II
# answer the k-size question in opposite directions, so collapsing them hides the result.
# ---------------------------------------------------------------------------
MHC_CLASSES: dict[str, str] = {
    "HLA-A": "I classical", "HLA-B": "I classical", "HLA-C": "I classical",
    "HLA-E": "I non-classical", "HLA-F": "I non-classical", "HLA-G": "I non-classical",
    "MICA": "I related (MIC)", "MICB": "I related (MIC)",
    "HLA-DRA": "II alpha", "HLA-DQA1": "II alpha", "HLA-DQA2": "II alpha",
    "HLA-DPA1": "II alpha", "HLA-DMA": "II alpha", "HLA-DOA": "II alpha",
    "HLA-DRB1": "II beta", "HLA-DRB5": "II beta", "HLA-DQB1": "II beta",
    "HLA-DQB2": "II beta", "HLA-DPB1": "II beta", "HLA-DMB": "II beta", "HLA-DOB": "II beta",
    "B2M": "light chain", "TAP1": "processing (ctrl)", "TAP2": "processing (ctrl)",
    "TAPBP": "processing (ctrl)",
}

#: The six class I heavy chains every class I result in the project rests on. Notebook 215
#: is explicit that this is six genes and only three independent lineages (the A/B/C
#: expansion is one), so treat it as n=3, not n=6, when it carries a claim.
MHC_CLASS_I_GENES: list[str] = ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G"]

# ---------------------------------------------------------------------------
# Immunity. Notebook 206's antiviral restriction factor anchor set, verified there against
# current HGNC symbols. A named list rather than an HGNC group because HGNC has no single
# group for "restriction factor" -- it spans TRIM, APOBEC, IFIT and IFITM families.
# ---------------------------------------------------------------------------
ANTIVIRAL_RESTRICTION_FACTORS: list[str] = [
    "TRIM5", "SAMHD1", "BST2", "MX1", "MX2", "ZC3HAV1",
    "APOBEC3A", "APOBEC3B", "APOBEC3C", "APOBEC3D", "APOBEC3F", "APOBEC3G", "APOBEC3H",
    "IFIT1", "IFIT2", "IFIT3", "IFIT5",
    "IFITM1", "IFITM2", "IFITM3", "IFITM5",
]

#: Notebook 214's decoys: immunoglobulin-superfamily genes OUTSIDE the MHC. They exist to
#: answer whether an MHC signal is MHC-specific or generic IgSF, which 214 found flips with
#: the label policy -- so they are a control group, not a positive set.
IGSF_DECOYS: list[str] = ["TTN", "ICAM1", "CD8A", "NCAM1", "VCAM1", "PECAM1"]

# ---------------------------------------------------------------------------
# Fast-evolving families. Notebook 206 used these as its high-divergence anchors, matched
# by HGNC gene_group substring (case-insensitive), restricted to protein-coding genes.
# ---------------------------------------------------------------------------
FAST_EVOLVING_GROUP_PATTERNS: dict[str, str] = {
    "olfactory_receptor": "Olfactory receptor",
    "cytochrome_p450_2_3": "Cytochrome P450 family (2|3)",
}

# ---------------------------------------------------------------------------
# The exclusion. ou.HGNC_EXCLUDE_FAMILY_PATTERN, from notebook 206 section 6 "Traps":
# tandem-repeat C2H2 zinc-finger arrays inflate k-mer sharing through REPEAT CONTENT rather
# than homology. 206's hand-picked family list never included them for that reason, and the
# same exclusion has to apply to any all-groups sweep or the confound reappears at scale --
# it is the largest single HGNC group in the human query set (325 proteins), so left in it
# would dominate the very stratification it corrupts.
# ---------------------------------------------------------------------------
HGNC_EXCLUDE_FAMILY_PATTERN: str = "Zinc finger"

#: Zinc fingers are excluded from the per-group HGNC sweep but NOT from the benchmark. They
#: are cut as their own stratum on the geneset axis, which is the useful middle: the family
#: stays visible and measurable, without leading a sweep whose premise it violates. If HP
#: k-mers behave differently on repeat-driven families, that is a result worth having, and
#: it is only obtainable by keeping them labelled rather than deleted.
ZINC_FINGER_GROUP_PATTERNS: dict[str, str] = {
    "zinc_finger_c2h2": "Zinc fingers C2H2-type",
    "zinc_finger_other": "Zinc finger",
}

#: Below this many query proteins a per-group metric is too noisy to read. Matches notebook
#: 206's own floor of 15 scored pairs.
MIN_FAMILY_SIZE: int = 15
