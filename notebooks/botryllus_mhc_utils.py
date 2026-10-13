"""Loading and link logic shared by notebook 281 and its interactive page
(scripts/build_281_botryllus_links_page.py).

Inputs live in one folder (on Olga's laptop, /Users/olga/data/botryllus-mhc/):

  mhc_targets.tsv, control_targets.tsv   scripts/mhc_target_sets.py
  botryllus_mhc_regions.csv.gz           tools/dark-regions extract-targets, then annotate (PR 57)
  tool_hits_on_targets_E10.tsv           scripts/botryllus_tool_hits_on_targets.py

A *link* is a (Botryllus protein, Swiss-Prot target) pair with at least one kept kmerseek
region below an E-value cutoff; it is *kmerseek-only* when no phmmer, jackhmmer or mmseqs2
hit on the same pair has E <= the tool cutoff.
"""

from pathlib import Path

import numpy as np
import polars as pl

CATEGORIES = [
    "mhc",
    "mhc_like_nkg2d_ligand",
    "antigen_presentation",
    "nk_receptor",
    "tcr_cd3",
    "c1_set",
    "self_nonself_nonvertebrate",
    "minor_histocompatibility",
]
ORDER = CATEGORIES + ["control"]
CATEGORY_NAME = {
    "mhc": "MHC class I and II",
    "mhc_like_nkg2d_ligand": "MIC / ULBP (NKG2D ligands)",
    "antigen_presentation": "Antigen presentation (TAP, B2M, CD74, ERAP)",
    "nk_receptor": "NK receptors (KIR, LILR, KLR)",
    "tcr_cd3": "T cell receptor and CD3",
    "c1_set": "C1-set domain proteins",
    "self_nonself_nonvertebrate": "Non-vertebrate self/non-self",
    "minor_histocompatibility": "Minor histocompatibility antigens",
    "control": "Control (length-matched random)",
}
TOOLS = ["phmmer", "jackhmmer", "mmseqs2"]
_FLOATS = [
    "region_evalue",
    "query_lc",
    "query_tm",
    "target_lc",
    "target_tm",
    "region_ka_bits",
]
_INTS = [
    "ksize",
    "region_start",
    "region_end",
    "target_start",
    "target_end",
    "region_length",
    "query_length",
    "target_length",
]


def load_targets(data: Path) -> pl.DataFrame:
    """Target and control proteins, with the control's length decile of each (0-9)."""
    targets = pl.concat(
        [
            pl.read_csv(data / "mhc_targets.tsv", separator="\t"),
            pl.read_csv(data / "control_targets.tsv", separator="\t"),
        ]
    )
    assert targets["accession"].n_unique() == targets.height
    edges = np.quantile(
        targets.filter(pl.col("category") == "control")["length"], np.linspace(0, 1, 11)
    )[1:-1]
    return targets.with_columns(
        length_decile=pl.col("length").map_elements(
            lambda x: int(np.searchsorted(edges, x, side="right")),
            return_dtype=pl.Int64,
        )
    )


def load_regions(
    data: Path, targets: pl.DataFrame, max_biased_share: float
) -> pl.DataFrame:
    """Every extracted kmerseek region (E < 10), with its category and why it would be dropped.

    low_complexity / helix: at least max_biased_share of the region, on either protein, is low
    complexity / transmembrane helix. wrong_target: the target residues are not in the named
    target (kmerseek PR 141). kept: none of the three.
    """
    regions = (
        pl.read_csv(data / "botryllus_mhc_regions.csv.gz", infer_schema_length=0)
        .with_columns(
            [pl.col(c).cast(pl.Float64) for c in _FLOATS]
            + [pl.col(c).cast(pl.Int64) for c in _INTS]
        )
        .rename({"moltype": "alphabet"})
        .join(
            targets.select("accession", "category"),
            left_on="target_name",
            right_on="accession",
            how="left",
        )
    )
    assert regions["category"].null_count() == 0
    return regions.with_columns(
        low_complexity=pl.max_horizontal("query_lc", "target_lc") >= max_biased_share,
        helix=pl.max_horizontal("query_tm", "target_tm") >= max_biased_share,
        wrong_target=pl.col("target_coords") == "not_found",
    ).with_columns(
        kept=~(pl.col("low_complexity") | pl.col("helix") | pl.col("wrong_target"))
    )


def load_tool_hits(data: Path, targets: pl.DataFrame) -> pl.DataFrame:
    """phmmer, jackhmmer and mmseqs2 hits of Botryllus proteins on target and control proteins, E <= 10."""
    return pl.read_csv(
        data / "tool_hits_on_targets_E10.tsv",
        separator="\t",
        has_header=False,
        new_columns=[
            "tool",
            "query_name",
            "target_name",
            "tstart",
            "tend",
            "bits",
            "evalue",
        ],
    ).join(
        targets.select("accession", "category"),
        left_on="target_name",
        right_on="accession",
    )


def links_at(
    regions: pl.DataFrame, tool_pairs: pl.DataFrame, cutoff: float
) -> pl.DataFrame:
    """One row per (Botryllus protein, target) link at E < cutoff, with its best region and
    whether any tool found the same pair (tool_pairs: query_name, target_name)."""
    r = regions.filter(pl.col("region_evalue") < cutoff)
    best = (
        r.sort("region_evalue", "alphabet", "ksize", "region_start")
        .group_by("query_name", "target_name", maintain_order=True)
        .agg(
            pl.all().first(),
            n_regions=pl.len(),
            n_alphabets=pl.col("alphabet").n_unique(),
            n_settings=pl.struct("alphabet", "ksize").n_unique(),
        )
    )
    return best.join(
        tool_pairs.with_columns(tool_found=pl.lit(True)),
        on=["query_name", "target_name"],
        how="left",
    ).with_columns(pl.col("tool_found").fill_null(False))


def alphabet_letter_maps(regions: pl.DataFrame) -> dict[str, dict[str, str]]:
    """alphabet -> {amino acid: the alphabet's letter for it}, read off the run's own output.

    kmerseek writes moltype_seq as the target residues in the alphabet's letters, so each
    target residue and its letter give the map. Stops if a residue maps to two letters.
    """
    maps = {}
    same = regions.filter(
        pl.col("target_subseq").str.len_chars() == pl.col("moltype_seq").str.len_chars()
    )
    for (alphabet,), g in same.group_by("alphabet"):
        pairs = (
            g.select(
                aa=pl.col("target_subseq").str.split(""),
                letter=pl.col("moltype_seq").str.split(""),
            )
            .explode("aa", "letter")
            .unique()
        )
        dup = pairs.group_by("aa").len().filter(pl.col("len") > 1)
        if dup.height:
            raise ValueError(
                f"{alphabet}: residues with more than one letter: {dup['aa'].to_list()}"
            )
        maps[alphabet] = dict(pairs.iter_rows())
    return maps


def encode(seq: str, letter_map: dict[str, str]) -> str:
    """seq in an alphabet's letters; a residue the map has never seen is written '?'."""
    return "".join(letter_map.get(x, "?") for x in seq)


def identity_and_chance(a: str, b: str) -> tuple[float, float]:
    """Identity of two equal-length gapless strings, and the identity expected from their compositions."""
    n = len(a)
    ident = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = {}, {}
    for x in a:
        ca[x] = ca.get(x, 0) + 1
    for x in b:
        cb[x] = cb.get(x, 0) + 1
    chance = sum(ca[x] * cb.get(x, 0) for x in ca) / (n * n)
    return ident, chance


def with_identity(
    links: pl.DataFrame, letter_maps: dict[str, dict[str, str]]
) -> pl.DataFrame:
    """Add both proteins' residues in the region's alphabet, and identity and chance identity
    in amino acids and in alphabet letters. Regions whose two strings differ in length (chained
    across an insertion or deletion) get no identity."""
    q_enc = [
        encode(q, letter_maps[a])
        for q, a in zip(links["region_subseq"], links["alphabet"])
    ]
    t_enc = [
        encode(t, letter_maps[a])
        for t, a in zip(links["target_subseq"], links["alphabet"])
    ]
    aa = [
        identity_and_chance(q, t) if len(q) == len(t) else (None, None)
        for q, t in zip(links["region_subseq"], links["target_subseq"])
    ]
    cls = [
        identity_and_chance(q, t) if len(q) == len(t) else (None, None)
        for q, t in zip(q_enc, t_enc)
    ]
    col = lambda vals, i: pl.Series([v[i] for v in vals], dtype=pl.Float64)
    return links.with_columns(
        query_letters=pl.Series(q_enc),
        target_letters=pl.Series(t_enc),
        identity=col(aa, 0),
        chance_identity=col(aa, 1),
        letter_identity=col(cls, 0),
        chance_letter_identity=col(cls, 1),
    ).with_columns(
        identity_above_chance=(pl.col("identity") - pl.col("chance_identity")) * 100,
        letter_identity_above_chance=(
            pl.col("letter_identity") - pl.col("chance_letter_identity")
        )
        * 100,
        group=pl.when(pl.col("category") == "control")
        .then(pl.lit("control"))
        .otherwise(pl.lit("categories")),
    )


def residue_lines(
    query: str, target: str, query_letters: str, target_letters: str, width: int = 80
) -> list[str]:
    """Five lines per block: the Botryllus residues, them in the alphabet's letters, a match
    line ('|' same amino acid, ':' same letter only), the target in letters, the target.
    """
    lines = []
    same = len(query) == len(target)
    for i in range(0, max(len(query), len(target)), width):
        q, t = query[i : i + width], target[i : i + width]
        qe, te = query_letters[i : i + width], target_letters[i : i + width]
        mid = (
            "".join(
                "|" if x == y else ":" if u == v else " "
                for x, y, u, v in zip(q, t, qe, te)
            )
            if same
            else ""
        )
        lines += (
            ["  Q   " + q, "  Q a " + qe]
            + (["      " + mid] if same else [])
            + ["  T a " + te, "  T   " + t]
        )
    return lines
