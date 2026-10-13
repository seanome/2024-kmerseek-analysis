"""Gene families and duplicate-copy fates for notebook 282 (Botryllus as the stand-in for
the gene before the two vertebrate whole-genome duplications).

Three parts:

1. Marletaz et al. 2018 (Nature 564: 64-70, doi:10.1038/s41586-018-0734-6): read their
   gene families (Supplementary Datasets 7 and 9) and apply their family classification,
   ported line by line from `main-figures/Analyze_Red_Subf_Spec.pl` in
   gitlab.com/FunctionalAmphioxus/overview (commit 26dcf27b).
2. OrthoFinder families for Botryllus, Ciona, amphioxus and human, with the human copies
   that OHNOLOGS v2 (Singh and Isambert 2020, Nucleic Acids Res 48: D724-D730,
   doi:10.1093/nar/gkz909) lists as kept from the two whole-genome duplications.
3. The Stage 2 classification of cell-type expression, written before any Botryllus
   expression data exists.

The words used throughout: a "family" is every gene descended from one gene in the
chordate ancestor (an OrthoFinder orthogroup rooted at that ancestor). An "ohnolog" is a
human gene copy kept from the two whole-genome duplications. A "domain" (Marletaz et al.'s
word) is one tissue or one cell type in which a gene is either expressed or not.
"""

from collections.abc import Sequence
from pathlib import Path

import polars as pl

SPECIES = ["botryllus", "ciona", "amphioxus", "human"]

# --------------------------------------------------------------------------------------
# 1. Marletaz et al. 2018
# --------------------------------------------------------------------------------------

MARLETAZ_DIR = Path.home() / "data/marletaz2018"
MARLETAZ_DATASETS = MARLETAZ_DIR / "2017-11-15409B-s4"

# The nine homologous samples (their Supplementary Table 7), in the column order the Perl
# script requires.
MARLETAZ_DOMAINS = [
    "Egg/Oocyte",
    "Mid-stage embryo",
    "Male gonads",
    "Female gonads",
    "Epidermis/Skin",
    "Digestive tract",
    "Liver",
    "Muscle",
    "Brain",
]

# "a gene was considered expressed, if it had a normalized cRPKM value higher than 5"
# (Supplementary Information section 11.1; $min_expr_default=5 in the Perl script).
MARLETAZ_MIN_EXPR = 5.0
# "Spec. strong": the copy keeps 2 or fewer domains (Fig. 6 legend; `<= 2` in the script).
MARLETAZ_STRONG_MAX_DOMAINS = 2

# The script's own printed output for the three vertebrates, copied from
# main-figures/F6ae.md in the same GitLab repository. "valid" = families where the single
# amphioxus gene and the union of the vertebrate copies are both expressed in all nine
# domains. These are the only family-level fate counts the authors published; the
# expression tables that produce them are not public (see notebook 282, Stage 0).
MARLETAZ_PUBLISHED_FATES = pl.DataFrame(
    {
        "species": ["mouse", "zebrafish", "frog"],
        "code": ["Mmu", "Dre", "Xtr"],
        "ohnolog_families_start": [1509, 1509, 1509],
        "after_size_filter": [1215, 1142, 1097],
        "valid_all_nine_domains": [370, 366, 360],
        "subfunctionalized": [80, 68, 66],
        "specialized_strong": [77, 99, 66],
        "specialized_mild": [167, 154, 157],
        "redundant": [46, 45, 71],
    }
)


def read_marletaz_families(dataset: int = 7, sheet: str | None = None) -> pl.DataFrame:
    """One row per gene: family_id, species code (Bla, Hsa, ...), gene_id.

    dataset=7 is every gene family (21,576); dataset=9 with sheet="Ohnologs" is their 1,509
    ohnolog families and sheet="1-to-1" their 2,511 single-copy families.
    """
    path = MARLETAZ_DATASETS / f"Supplementary Dataset {dataset}.xlsx"
    sheet = sheet or f"Supplementary Dataset {dataset}"
    df = pl.read_excel(path, sheet_name=sheet, engine="openpyxl")
    return df.select(
        pl.col("Fam_ID_Final").alias("family_id"),
        pl.col("Species").alias("species"),
        pl.col("GeneID").cast(pl.Utf8).alias("gene_id"),
    )


def classify_family_marletaz(
    ancestor_on: Sequence[bool],
    copies_on: Sequence[Sequence[bool]],
    strong_max_domains: int = MARLETAZ_STRONG_MAX_DOMAINS,
) -> str:
    """Fate of one family of vertebrate copies, by the rule of Marletaz et al. 2018.

    ancestor_on: one bool per domain for the single outgroup gene (their amphioxus gene).
    copies_on: one such list per vertebrate copy, domains in the same order.

    Returns one of:
      "not_tested": the outgroup gene is off in some domain, or the copies together
          (their union) are not on in exactly the outgroup's domains. Only families
          expressed everywhere in both enter their Fig. 6d.
      "redundant": every copy is on in every domain.
      "subfunctionalized": no copy is on in every domain (the copies split the domains).
      "specialized_strong": at least one copy is on in every domain, and at least one
          copy that lost domains keeps `strong_max_domains` or fewer.
      "specialized_mild": at least one copy is on in every domain, and every copy that
          lost domains keeps more than `strong_max_domains`.

    Port of the family loop in Analyze_Red_Subf_Spec.pl: `$pos_Sp1_fam >= $min_tis` with
    `$min_tis = 9` (all domains), `$string_fam{Sp1} eq $string_fam{Sp2}`, then
    `$equal_gene` (copy on in as many domains as the outgroup) and `$lower_gene`.
    """
    n = len(ancestor_on)
    if any(len(c) != n for c in copies_on) or len(copies_on) < 2:
        raise ValueError("need two or more copies, each with one value per domain")
    union = [any(c[i] for c in copies_on) for i in range(n)]
    if not all(ancestor_on) or union != list(ancestor_on):
        return "not_tested"
    n_on = [sum(c) for c in copies_on]
    full = [k == n for k in n_on]
    if all(full):
        return "redundant"
    if not any(full):
        return "subfunctionalized"
    lost = [k for k in n_on if k < n]
    return (
        "specialized_strong" if min(lost) <= strong_max_domains else "specialized_mild"
    )


# --------------------------------------------------------------------------------------
# 2. OrthoFinder families and OHNOLOGS v2
# --------------------------------------------------------------------------------------

DATA_282 = Path.home() / "data/282-ohnolog-families"
OHNOLOGS_DIR = Path.home() / "data/ohnologs_v2"
OHNOLOG_CRITERIA = ["strict", "intermediate", "relaxed"]


def read_orthogroups(path: Path) -> pl.DataFrame:
    """OrthoFinder Orthogroups.tsv -> one row per gene: family_id, species, gene_id.

    Genes OrthoFinder could not place with any other gene are not in this file
    (Orthogroups_UnassignedGenes.tsv); they never share a family with a human gene.
    """
    wide = pl.read_csv(path, separator="\t", infer_schema_length=0)
    long = wide.rename({"Orthogroup": "family_id"}).unpivot(
        index="family_id", variable_name="species", value_name="genes"
    )
    return (
        long.drop_nulls("genes")
        .with_columns(pl.col("genes").str.split(", "))
        .explode("genes")
        .rename({"genes": "gene_id"})
        .with_columns(pl.col("gene_id").str.strip_chars())
        .filter(pl.col("gene_id") != "")
    )


def read_ohnolog_pairs(criterion: str = "strict") -> pl.DataFrame:
    """OHNOLOGS v2 human 2R pairs: ohno1, ohno2 (Ensembl gene IDs, release 84), symbols."""
    path = OHNOLOGS_DIR / f"hsapiens.2R.{criterion}.pairs.txt"
    return pl.read_csv(path, separator="\t").select(
        pl.col("Ohno1").alias("ohno1"),
        pl.col("Ohno2").alias("ohno2"),
        pl.col("Symbol1").alias("symbol1"),
        pl.col("Symbol2").alias("symbol2"),
    )


def human_ohnologs_per_family(genes: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Human genes in each family that have an OHNOLOGS partner in the SAME family.

    A human gene listed in OHNOLOGS whose partners all landed in other families is not
    counted here: within this family it has no sister copy from the duplications.
    Returns family_id, gene_id for each such gene.
    """
    human = genes.filter(pl.col("species") == "human").select("family_id", "gene_id")
    both_ways = pl.concat(
        [
            pairs.select(pl.col("ohno1").alias("a"), pl.col("ohno2").alias("b")),
            pairs.select(pl.col("ohno2").alias("a"), pl.col("ohno1").alias("b")),
        ]
    ).unique()
    fam_a = human.rename({"gene_id": "a"})
    fam_b = human.rename({"gene_id": "b", "family_id": "family_b"})
    same = (
        both_ways.join(fam_a, on="a")
        .join(fam_b, on="b")
        .filter(pl.col("family_id") == pl.col("family_b"))
    )
    return (
        same.select("family_id", pl.col("a").alias("gene_id"))
        .unique()
        .sort("family_id", "gene_id")
    )


def family_table(genes: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """One row per family: the genes of each species, the human ohnologs, and their counts.

    Columns: family_id, {species}_genes (sorted list), n_{species}, human_ohnologs,
    n_human_ohnologs, n_human_listed_in_ohnologs (human genes in this family that appear
    anywhere in OHNOLOGS, partner here or not).
    """
    lists = (
        genes.group_by("family_id", "species")
        .agg(pl.col("gene_id").sort())
        .pivot(on="species", index="family_id", values="gene_id")
    )
    for s in SPECIES:
        if s not in lists.columns:
            lists = lists.with_columns(pl.lit(None, dtype=pl.List(pl.Utf8)).alias(s))
    lists = lists.select(
        "family_id",
        *[
            pl.col(s).fill_null(pl.lit([], dtype=pl.List(pl.Utf8))).alias(f"{s}_genes")
            for s in SPECIES
        ],
    )
    ohno = (
        human_ohnologs_per_family(genes, pairs)
        .group_by("family_id")
        .agg(pl.col("gene_id").sort().alias("human_ohnologs"))
    )
    listed = set(pairs["ohno1"]) | set(pairs["ohno2"])
    out = lists.join(ohno, on="family_id", how="left").with_columns(
        pl.col("human_ohnologs").fill_null(pl.lit([], dtype=pl.List(pl.Utf8)))
    )
    return out.with_columns(
        *[pl.col(f"{s}_genes").list.len().alias(f"n_{s}") for s in SPECIES],
        pl.col("human_ohnologs").list.len().alias("n_human_ohnologs"),
        pl.col("human_genes")
        .list.eval(pl.element().is_in(list(listed)))
        .list.sum()
        .alias("n_human_listed_in_ohnologs"),
    ).sort("family_id")


def copy_number_grid(table: pl.DataFrame, cap: int = 5) -> pl.DataFrame:
    """Families counted by (Botryllus copies, human ohnologs), each capped at `cap`
    ("5" means 5 or more). Only families with at least one Botryllus gene or at least one
    human gene are counted."""
    t = table.filter((pl.col("n_botryllus") > 0) | (pl.col("n_human") > 0))
    return (
        t.with_columns(
            pl.col("n_botryllus").clip(upper_bound=cap).alias("botryllus_copies"),
            pl.col("n_human_ohnologs")
            .clip(upper_bound=cap)
            .alias("human_ohnolog_copies"),
        )
        .group_by("botryllus_copies", "human_ohnolog_copies")
        .len("n_families")
        .sort("botryllus_copies", "human_ohnolog_copies")
    )


def compare_with_marletaz(
    genes: pl.DataFrame, marletaz_ohnologs: pl.DataFrame
) -> pl.DataFrame:
    """Do our families keep Marletaz et al.'s amphioxus gene with its human copies?

    For every Marletaz ohnolog family with exactly one amphioxus (Bla) gene and two or more
    human (Hsa) genes, find the family our run put that amphioxus gene in, and count how
    many of their human genes are in it. `agreement` is "all human copies together",
    "some", "none", or "amphioxus gene not in a family" (OrthoFinder left it unassigned,
    or it is not in our amphioxus proteome).
    """
    m = marletaz_ohnologs
    per_fam = m.group_by("family_id").agg(
        pl.col("gene_id").filter(pl.col("species") == "Bla").alias("bla"),
        pl.col("gene_id").filter(pl.col("species") == "Hsa").alias("hsa"),
    )
    per_fam = per_fam.filter(
        (pl.col("bla").list.len() == 1) & (pl.col("hsa").list.len() >= 2)
    )
    ours = genes.select(pl.col("family_id").alias("our_family"), "gene_id")
    bla = per_fam.select(
        pl.col("family_id").alias("marletaz_family"),
        pl.col("bla").list.first().alias("gene_id"),
        "hsa",
    ).join(ours, on="gene_id", how="left")
    hsa = (
        bla.select("marletaz_family", "our_family", "hsa")
        .explode("hsa")
        .join(
            ours.rename({"gene_id": "hsa", "our_family": "hsa_family"}),
            on="hsa",
            how="left",
        )
        .group_by("marletaz_family")
        .agg(
            pl.len().alias("n_human_marletaz"),
            (pl.col("hsa_family") == pl.col("our_family"))
            .fill_null(False)
            .sum()
            .alias("n_human_with_amphioxus"),
        )
    )
    out = bla.drop("hsa").join(hsa, on="marletaz_family")
    return out.with_columns(
        pl.when(pl.col("our_family").is_null())
        .then(pl.lit("amphioxus gene not in a family"))
        .when(pl.col("n_human_with_amphioxus") == pl.col("n_human_marletaz"))
        .then(pl.lit("all human copies together"))
        .when(pl.col("n_human_with_amphioxus") > 0)
        .then(pl.lit("some"))
        .otherwise(pl.lit("none"))
        .alias("agreement")
    ).sort("marletaz_family")


# --------------------------------------------------------------------------------------
# 3. Stage 2 rules (written 2026-10-12, before any Botryllus cell-type data)
# --------------------------------------------------------------------------------------


def classify_copies_against_ancestor(
    ancestor_on: Sequence[bool], copies_on: Sequence[Sequence[bool]]
) -> dict:
    """Stage 2 fate of the human copies of one family, with Botryllus as the ancestor.

    Each list holds one bool per matched cell type (the same order for Botryllus and
    human): is the gene detected there. Returns a dict with:

      "fate": one of
        "ancestor_not_broad": the Botryllus gene is off in some matched cell type.
            Following Marletaz et al., only genes on in every matched cell type are tested,
            because only then can a copy be seen to lose a cell type.
        "both_keep_all": every human copy is on in every cell type (their "redundant").
        "one_narrows": at least one copy is on everywhere, at least one is not (their
            "specialization").
        "split": no copy is on everywhere, but together they cover every cell type (their
            "subfunctionalization", the duplication-degeneration-complementation model of
            Force et al. 1999).
        "cell_type_lost": the copies together miss a cell type the ancestor had.
      "new_cell_type": True if some copy is on in a matched cell type where the Botryllus
          gene is off. Reported beside the fate, not instead of it. With an ancestor that
          is on everywhere it is always False; it matters for the families that fail the
          "broad" filter.
    """
    n = len(ancestor_on)
    if any(len(c) != n for c in copies_on) or len(copies_on) < 2:
        raise ValueError("need two or more copies, each with one value per cell type")
    union = [any(c[i] for c in copies_on) for i in range(n)]
    new = any(u and not a for u, a in zip(union, ancestor_on))
    if not all(ancestor_on):
        return {"fate": "ancestor_not_broad", "new_cell_type": new}
    full = [all(c) for c in copies_on]
    if all(full):
        fate = "both_keep_all"
    elif any(full):
        fate = "one_narrows"
    elif all(union):
        fate = "split"
    else:
        fate = "cell_type_lost"
    return {"fate": fate, "new_cell_type": new}
