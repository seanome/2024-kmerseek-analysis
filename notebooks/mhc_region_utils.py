#!/usr/bin/env python3
"""Shared definitions for the MHC-region notebooks (220-223) built on the QfO midi run.

The midi run searches every human chromosome 6 protein against nine whole proteomes with
~3,700 scoring arms per truth set. Notebooks 210-216 asked a different question of a
different run -- human vs mouse, ortholog pairs, one alphabet -- so their helpers in
`ortholog_analysis_utils` (`ou`) stay the source of truth for anything about MHC gene
taxonomy or domain architecture. This module adds only what the region run needs on top:
where a gene sits on chr6, which Pfam family plays which structural role, and how to load
the extracted tables.

Nothing here re-defines a constant `ou` already carries. `ou.MHC_CLASSES`,
`ou.MHC_CLASS_COLORS`, `ou.ARD_I`, `ou.IG_I`, `ou.CONTACT_PROT` and `ou.mhc_gene_arch` are
imported and re-exported so a notebook needs one import, not two.
"""
from __future__ import annotations

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
# Where the extracted midi-run tables live.
#
# The run itself stays on Sherlock: `results/kmerseek` alone is 403 GB and the per-arm call
# files are 5.4 GB across 22,411 files. `scripts/extract_mhc.py` collapses that on the
# cluster to the five tables below, which together are small enough to sit on a laptop.
# ---------------------------------------------------------------------------
MIDI_DIR = Path("/Users/olga/data/qfo-pfam-region-midi")

TRUTH_DIR = MIDI_DIR / "truth"
IDENTITY_DIR = MIDI_DIR / "identity"
EXTRACT_DIR = MIDI_DIR / "extract"

HUMAN_TRUTH = TRUTH_DIR / "human_domain_truth.parquet"
HUMAN_COVARIATES = TRUTH_DIR / "human_query_covariates.parquet"
CHR6_GENE_MAP = MIDI_DIR / "chr6_query_gene_map.parquet"
CHR6_GENCODE = MIDI_DIR / "gencode_v50_chr6_genes.parquet"

GENE_LEVEL_ALL_ARMS = EXTRACT_DIR / "mhc_gene_level_all_arms.parquet"
DOMAIN_LEVEL_ALL_ARMS = EXTRACT_DIR / "mhc_domain_level_all_arms.parquet"
CHR6_CALLS_FOCUS = EXTRACT_DIR / "chr6_calls_focus_arms.parquet"
KMERSEEK_REGIONS = EXTRACT_DIR / "mhc_kmerseek_regions.parquet"
BASELINE_REGIONS = EXTRACT_DIR / "mhc_baseline_regions.parquet"
ALL_DOMAIN_METRICS = EXTRACT_DIR / "all_domain_metrics.parquet"

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
    "HFE": "extended class I", "BTN3A1": "extended class I", "TRIM26": "extended class I",
    "HLA-F": "class I", "HLA-G": "class I", "HLA-A": "class I", "MICA": "class I",
    "HLA-C": "class I", "HLA-B": "class I", "MICB": "class I",
    "TNF": "class III", "LTA": "class III", "C4A": "class III", "CFB": "class III",
    "C2": "class III", "NOTCH4": "class III",
    "HLA-DRA": "class II", "HLA-DRB1": "class II", "HLA-DQB1": "class II",
    "TAP1": "class II", "PSMB9": "class II", "HLA-DPB1": "class II",
    "COL11A2": "extended class II", "RXRB": "extended class II",
}


def assign_subregion(midpoint: pl.Expr) -> pl.Expr:
    """chr6 midpoint -> xMHC sub-region name, null outside the extended MHC.

    Built by folding the interval list in reverse so the first-listed (most telomeric)
    interval ends up as the outermost `when`, which keeps the generated expression's branch
    order the same as `MHC_SUBREGIONS` reads on the page.
    """
    expr = pl.lit(None, dtype=pl.String)
    for name, lo, hi in reversed(MHC_SUBREGIONS):
        expr = pl.when((midpoint >= lo) & (midpoint <= hi)).then(pl.lit(name)).otherwise(expr)
    return expr


# ---------------------------------------------------------------------------
# Target species.
#
# Divergence times are the pipeline's own (`main.nf` SPECIES), repeated rather than parsed
# so a notebook can order an axis without the Nextflow file being on the path. They are
# round numbers for axis ordering, not literature point estimates.
# ---------------------------------------------------------------------------
SPECIES_MYA: dict[str, int] = {
    "mouse": 100, "chicken": 300, "zebrafish": 430, "ciona": 550, "fly": 600,
    "worm": 650, "yeast": 900, "arabidopsis": 1500, "ecoli": 2000,
}

SPECIES_ORDER: list[str] = list(SPECIES_MYA)

SPECIES_LABELS: dict[str, str] = {
    "mouse": "mouse\n100", "chicken": "chicken\n300", "zebrafish": "zebrafish\n430",
    "ciona": "ciona\n550", "fly": "fly\n600", "worm": "worm\n650",
    "yeast": "yeast\n900", "arabidopsis": "arabidopsis\n1500", "ecoli": "E. coli\n2000",
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
}


def domain_role(pfam_id: pl.Expr) -> pl.Expr:
    """Pfam accession -> the role it plays in an MHC molecule, null for anything else."""
    return (
        pl.when(pfam_id.is_in(list(PFAM_PLATFORM))).then(pl.lit("peptide-binding platform"))
        .when(pfam_id == PFAM_IG_C1).then(pl.lit("Ig C1-set"))
        .when(pfam_id == PFAM_CLASS_I_TAIL).then(pl.lit("class I tail"))
        .when(pfam_id.is_in(list(PFAM_TAP))).then(pl.lit("ABC transporter"))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


# ---------------------------------------------------------------------------
# Tool display names and grouping.
# ---------------------------------------------------------------------------
TOOL_FAMILY: dict[str, str] = {
    "hmmer3_phmmer": "sequence", "hmmer3_jackhmmer": "sequence",
    "mmseqs2_seqseq": "sequence", "mmseqs2_iterative": "sequence",
    "hhblits": "profile", "prostt5": "predicted structure",
    "foldseek": "structure", "reseek": "structure", "folddisco": "structure",
    "kmerseek": "kmerseek",
}

TOOL_LABELS: dict[str, str] = {
    "hmmer3_phmmer": "phmmer", "hmmer3_jackhmmer": "jackhmmer",
    "mmseqs2_seqseq": "MMseqs2", "mmseqs2_iterative": "MMseqs2 iterative",
    "hhblits": "HHblits", "prostt5": "ProstT5", "foldseek": "Foldseek",
    "reseek": "Reseek", "folddisco": "Folddisco", "kmerseek": "kmerseek",
}

TOOL_FAMILY_COLORS: dict[str, str] = {
    "sequence": "#999999", "profile": "#6ACC65", "predicted structure": "#B47CC7",
    "structure": "#4878CF", "kmerseek": "#D65F5F",
}


def arm_label(tool: pl.Expr, variant: pl.Expr) -> pl.Expr:
    """`tool`/`variant` -> the short string used on axes.

    kmerseek keeps its variant because the alphabet and k-size *are* the arm; every other
    tool has one variant in this run, so repeating it on an axis is noise.
    """
    return (
        pl.when(tool == "kmerseek")
        .then(pl.lit("kmerseek ") + variant.str.replace("_lcTrue", "").str.replace("_lcFalse", " (no LC mask)"))
        .otherwise(tool.replace_strict(TOOL_LABELS, default=tool))
    )


# ---------------------------------------------------------------------------
# Loaders.
# ---------------------------------------------------------------------------
def load_chr6_gene_map() -> pl.DataFrame:
    """The 964 chr6 query proteins with GENCODE coordinates and xMHC sub-region.

    Three of the 964 have no GENCODE v50 chr6 gene record under their HGNC symbol and carry
    null coordinates; they are kept rather than dropped so per-species counts still add up
    to the query-set size the run reports.
    """
    return pl.read_parquet(CHR6_GENE_MAP)


def load_mhc_window(gene_map: pl.DataFrame | None = None) -> pl.DataFrame:
    """The chr6 queries inside the extended MHC."""
    gm = load_chr6_gene_map() if gene_map is None else gene_map
    return gm.filter(pl.col("mhc_subregion").is_not_null())


def load_human_truth() -> pl.DataFrame:
    return pl.read_parquet(HUMAN_TRUTH)


def load_gene_level() -> pl.DataFrame:
    """One row per (arm, species, MHC-window gene): how much of that gene the arm found."""
    return pl.read_parquet(GENE_LEVEL_ALL_ARMS)


def load_domain_level() -> pl.DataFrame:
    """One row per (arm, species, core MHC gene, Pfam family)."""
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
    return set(dm.filter(pl.col("pfam_id").is_in(pfam_ids))["pfam_id"].unique().to_list())


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
    "human": "UP000005640_9606", "mouse": "UP000000589_10090",
    "chicken": "UP000000539_9031", "zebrafish": "UP000000437_7955",
    "ciona": "UP000008144_7719", "fly": "UP000000803_7227",
    "worm": "UP000001940_6239", "yeast": "UP000002311_559292",
    "arabidopsis": "UP000006548_3702", "ecoli": "UP000000625_83333",
}

#: Species with an Ensembl 116 gene-coordinate table pulled locally.
ENSEMBL_SPECIES: dict[str, str] = {
    "mouse": "mus_musculus", "chicken": "gallus_gallus", "zebrafish": "danio_rerio",
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
    return (pl.read_csv(path, separator="\t", has_header=False,
                        new_columns=["target_acc", "gene_symbol"], infer_schema_length=0)
              .filter(pl.col("gene_symbol").is_not_null() & (pl.col("gene_symbol") != ""))
              .unique(subset="target_acc"))


def load_target_gene_coordinates(species: str) -> pl.DataFrame:
    """Ensembl 116 gene records for a target species.

    Read entirely as text and cast afterwards: seqnames include X, Y, MT and scaffold
    names, so inferring the column type from the first rows commits to i64 and then dies
    on chromosome X.
    """
    path = TARGET_DIR / "gtf" / f"{ENSEMBL_SPECIES[species]}.genes.tsv"
    cols = ["chrom", "start", "end", "strand", "gene_id", "gene_name", "biotype"]
    return (pl.read_csv(path, separator="\t", has_header=False, new_columns=cols,
                        infer_schema_length=0)
              .with_columns(pl.col("start").cast(pl.Int64), pl.col("end").cast(pl.Int64))
              .filter(pl.col("gene_name").is_not_null())
              .with_columns(midpoint=(pl.col("start") + pl.col("end")) // 2))


def target_locus_table(species: str) -> pl.DataFrame:
    """target accession -> (gene symbol, chrom, start, end, midpoint) for one species.

    Symbol matching is case-insensitive: UniProt writes mouse symbols as `H2-D1` and
    zebrafish as `mhc1uba`, and Ensembl's capitalisation does not always agree.
    """
    genes = load_target_gene_coordinates(species).with_columns(
        _key=pl.col("gene_name").str.to_lowercase())
    acc = load_accession_to_gene(species).with_columns(
        _key=pl.col("gene_symbol").str.to_lowercase())
    return (acc.join(genes.unique(subset="_key"), on="_key", how="inner")
               .select("target_acc", "gene_symbol", "chrom", "start", "end", "midpoint",
                       "strand", "biotype"))
