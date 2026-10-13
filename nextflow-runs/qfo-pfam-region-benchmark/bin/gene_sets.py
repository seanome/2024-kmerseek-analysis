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

# ---------------------------------------------------------------------------
# The MHC's off-chromosome partners. The chr6 midi query set is defined by cytogenetic
# location, so it captures the MHC itself and 736 non-MHC chr6 genes that serve as the
# within-chromosome control -- but it cannot capture the parts of the antigen-presentation
# system that live somewhere else. These four buckets are ADDITIVE to chr6, never a
# replacement: notebook 221 section 2 measures every "the MHC behaves differently" claim
# against those 736 controls, and dropping or diluting them would destroy the comparison.
#
# Every symbol below was resolved against data/covariates/hgnc_complete_set.txt rather than
# recalled, and every accession was checked to be present in BOTH the QfO human proteome
# (UP000005640_9606.fasta) and the Pfam answer key (human_pfam_domains.parquet). A query
# with no truth instance cannot produce a true positive, so an unchecked symbol is a query
# that silently costs compute and returns nothing.
# ---------------------------------------------------------------------------

#: The class I light chain. Beta-2 microglobulin pairs with every classical and
#: non-classical class I heavy chain, but it sits on chr15, so a chr6-only query set scores
#: the heavy chains without the subunit they fold against. One C1-set domain (PF07654).
MHC_LIGHT_CHAIN: list[str] = ["B2M"]

#: Class-I-like molecules that present lipids (CD1) and microbial metabolites (MR1) instead
#: of peptides. CD1A-E are a chr1q23.1 cluster; MR1 is chr1q25.3. They carry the same
#: C1-set fold as the class I heavy chains without the class I sequence identity, which is
#: the case where a fold-aware or HP-patterning method should separate from a
#: sequence-identity one. CD1A-E each carry PF07654 + PF16497; MR1 carries PF00129 + PF07654.
CD1_MR1: list[str] = ["CD1A", "CD1B", "CD1C", "CD1D", "CD1E", "MR1"]

#: The receptors that READ class I, as HGNC gene groups rather than a typed-out list.
#: Membership of the KIR cluster in particular is the kind of thing that is wrong when
#: recalled: the locus is copy-number variable, several members sit on 19q13.4 alternate
#: reference loci, and KIR2DP1/KIR3DP1 are pseudogenes with no UniProt entry at all. Group
#: matching against HGNC's own curation gets that right by construction; a remembered list
#: does not. Resolves to 16 KIR + 11 LILR = 27 accessions, all on chr19q13.4x.
KIR_LILR_GROUPS: list[str] = [
    "Killer cell immunoglobulin like receptors",
    "Activating leukocyte immunoglobulin like receptors",
    "Inhibitory leukocyte immunoglobulin like receptors",
]

#: Bucket name -> how it is resolved. `make_mini_testset.py --gene-set chr6_plus` unions
#: these with the chr6 set and writes one row per query to query_sets.tsv, so every
#: downstream cut can reconstruct the original 964 chr6 queries and the 736-gene control
#: exactly. Ordered: the first bucket a query matches wins, and chr6 is applied first, so a
#: gene that is both on chr6 and in one of these lists stays labelled chr6. (Measured: the
#: overlap is currently empty -- none of the 34 accessions here is on chr6.)
MHC_PARTNER_BUCKETS: dict[str, str] = {
    "b2m": "symbols",
    "cd1_mr1": "symbols",
    "kir_lilr": "gene_groups",
}

#: HGNC gene_group values arrive with stray double quotes when the table is read with
#: quote_char=None, which is how every reader in this pipeline reads it (the file has
#: unbalanced quotes in free-text columns, so real quote parsing breaks it). Strip them
#: before matching or "Killer cell immunoglobulin like receptors" matches 5 of its 18 rows.
GENE_GROUP_STRIP_CHARS: str = '"'
