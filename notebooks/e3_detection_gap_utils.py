"""
E3 — detection-gap benchmark harness (domain-level).

The unit of analysis is the **annotated domain instance** in a non-human proteome,
not the protein pair.

Question
--------
For each Pfam-annotated domain instance in a held-out (invertebrate) proteome, which
homology-search methods recover it by transfer from a human protein carrying the same
Pfam family?  The headline readout is the **set difference**: domains recovered by the
HP (2-letter) arm and by no other arm.

Design
------
* **Ground truth** = Pfam domain instances per species, built by
  ``nextflow-runs/qfo-pfam-benchmark`` into
  ``results/pfam_benchmark/annotations/{species}_pfam_domains.parquet``
  (columns: accession, pfam_id, domain_start, domain_end, protein_length, ...).
* **Evaluated pair set** = ``results/pfam_benchmark/pairs/human_vs_{species}_ground_truth.parquet``
  — 20_000 positive (share >=1 Pfam family) and 40_000 negative human/species pairs.
* **Arms** each produce a score per (human_accession, species_accession) pair.
  Missing pair => no detection.
* **Matched specificity.**  Every arm is thresholded at the *same* false-positive rate
  measured on the 40_000 negative pairs, so no arm gets credit for being looser.
  Without this the comparison is meaningless: bitscores, -log10 E-values and kmerseek
  containment are not on a common scale.
* A domain instance is **recoverable** if at least one positive benchmark pair carries
  its Pfam family; only recoverable domains enter the denominator.  A domain is
  **recovered by arm M** if at least one such pair is called by M.

Circularity note (read before quoting any number)
-------------------------------------------------
The ground truth is itself Pfam-HMM derived.  A fresh ``hmmscan``/InterProScan Pfam arm
therefore recovers ~100% *by construction* and is reported as the annotation ceiling,
not as a competing arm.  What E3 measures is **homology transfer**: given an
unannotated protein, which method links it to an annotated human relative.  That is the
question BHF panel C asks at n=1.  Genuinely *cryptic* (un-annotated) domains cannot be
scored here at all — that is experiment E5's job, and this module deliberately does not
pretend otherwise.

Conventions
-----------
* Counts are reported as ``n_domains_all`` (every recoverable domain) and
  ``n_domains_ref`` (the subset on which a structure arm is even applicable, i.e. both
  proteins have an AlphaFold model) **separately**.  A domain with no AFDB model is a
  FoldSeek miss, not an out-of-benchmark domain.
* Any arm whose results file is absent or an empty gzip is reported as
  ``status='no data'`` and is never silently dropped from the table.
"""

from __future__ import annotations

import gzip
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

HOME = Path(os.path.expanduser("~"))
REPO = HOME / "code" / "2024-kmerseek-analysis"
PFAM_BENCH = REPO / "results" / "pfam_benchmark"

ANNOTATIONS = PFAM_BENCH / "annotations"
PAIRS = PFAM_BENCH / "pairs"
PAIRS_STRATIFIED = PFAM_BENCH / "pairs_stratified"
MERGED = PFAM_BENCH / "merged"
DIAMOND_FRESH = PFAM_BENCH / "diamond_fresh"
AFDB_STRUCTURES = PFAM_BENCH / "alphafold_structures" / "all_species"

TOOLS_RESULTS = HOME / "data" / "pfam-benchmark-tools" / "results"

# E3-specific outputs (produced by nextflow-runs/e3-detection-gap)
E3_RESULTS = HOME / "data" / "e3-detection-gap"
E3_FOLDSEEK = E3_RESULTS / "foldseek"
E3_JACKHMMER = E3_RESULTS / "jackhmmer"
E3_HHBLITS = E3_RESULTS / "hhblits"
E3_INTERPROSCAN = E3_RESULTS / "interproscan"
E3_PLM = E3_RESULTS / "plm"

# An empty gzip stream is 20 bytes.  Several arms in the existing benchmark wrote these
# (hhblits with no profile DB, the first diamond run) and they must not read as "0 hits,
# arm ran fine".
EMPTY_GZIP_BYTES = 32

# --------------------------------------------------------------------------------------
# Species
# --------------------------------------------------------------------------------------

SPECIES_MYA = {
    "mouse": 100,
    "chicken": 320,
    "zebrafish": 450,
    "ciona": 550,
    "fly": 800,
    "worm": 800,
    "yeast": 1100,
    "arabidopsis": 1500,
    "ecoli": 2000,
}

INVERTEBRATES = ("ciona", "fly", "worm")
VERTEBRATES = ("mouse", "chicken", "zebrafish")
NON_METAZOA = ("yeast", "arabidopsis", "ecoli")

ALL_SPECIES = tuple(SPECIES_MYA)

# The held-out invertebrate proteome for the headline number.  Ciona intestinalis is the
# closest available relative of Botryllus schlosseri (both tunicates), which is the
# proteome the BHF figure is built on, and it is the one with the deepest Pfam
# annotation of the three invertebrates.
HEADLINE_SPECIES = "ciona"


# --------------------------------------------------------------------------------------
# Arm registry
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One method in the detection-gap comparison."""

    name: str
    label: str
    kind: str  # 'tsv_pipe' | 'tsv_bare' | 'kmerseek' | 'foldseek' | 'ceiling'
    path_template: str
    family: str  # 'sequence-sequence' | 'profile' | 'structure' | 'plm' | 'hp' | 'ceiling'
    score_is_bitscore: bool = True
    higher_is_better: bool = True
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def path(self, species: str, **kwargs) -> Path:
        return Path(self.path_template.format(species=species, **kwargs))


def _kmerseek_arm(ksize: int, score_col: str = "score_tfidf_cont") -> Arm:
    return Arm(
        name=f"kmerseek_hp_k{ksize}",
        label=f"kmerseek HP k={ksize}",
        kind="kmerseek",
        path_template=str(MERGED / f"human_vs_{{species}}.hp.k{ksize}.merged.parquet"),
        family="hp",
        score_is_bitscore=False,
        notes="2-letter hydrophobic-polar alphabet, sequence only, no structure",
        extra={"ksize": ksize, "score_col": score_col},
    )


#: Arms that do not need the heavy pipeline — data already on disk (or provably absent).
BASE_ARMS: dict[str, Arm] = {
    "diamond": Arm(
        name="diamond",
        label="DIAMOND (BLASTp-class)",
        kind="tsv_pipe",
        path_template=str(DIAMOND_FRESH / "human_vs_{species}.diamond.tsv.gz"),
        family="sequence-sequence",
        notes="single-sequence pairwise; the baseline BHF panel C already uses",
    ),
    "mmseqs2_seqseq": Arm(
        name="mmseqs2_seqseq",
        label="MMseqs2 (sequence-sequence)",
        kind="tsv_bare",
        path_template=str(
            TOOLS_RESULTS / "mmseqs2_seqseq" / "human_vs_{species}.mmseqs2_seqseq.tsv.gz"
        ),
        family="sequence-sequence",
    ),
    "phmmer": Arm(
        name="phmmer",
        label="phmmer (HMMER3, 1 iteration)",
        kind="tsv_pipe",
        path_template=str(
            TOOLS_RESULTS / "hmmer3_phmmer" / "human_vs_{species}.hmmer3_phmmer.tsv.gz"
        ),
        family="profile",
        notes="single-query profile; the second baseline in BHF panel C",
    ),
    "mmseqs2_iterative": Arm(
        name="mmseqs2_iterative",
        label="MMseqs2 iterative profile search",
        kind="tsv_bare",
        path_template=str(
            TOOLS_RESULTS
            / "mmseqs2_iterative"
            / "human_vs_{species}.mmseqs2_iterative.tsv.gz"
        ),
        family="profile",
        notes=(
            "iterative profile search — the strongest sequence-only baseline with data "
            "already on disk, and the closest stand-in for jackhmmer/HHblits until "
            "those arms land"
        ),
    ),
    "hhblits": Arm(
        name="hhblits",
        label="HHblits (3 iter vs UniRef30)",
        kind="tsv_bare",
        path_template=str(
            TOOLS_RESULTS / "hhblits" / "human_vs_{species}.hhblits.tsv.gz"
        ),
        family="profile",
        notes=(
            "PRIORITY-1 arm. The existing files are 20-byte empty gzips: the pipeline "
            "was run with params.hhblits_db=null, so hhblits built single-sequence a3m "
            "profiles with 0 iterations and emitted nothing. Needs UniRef30."
        ),
    ),
}

#: Arms produced by nextflow-runs/e3-detection-gap.
E3_ARMS: dict[str, Arm] = {
    "jackhmmer": Arm(
        name="jackhmmer",
        label="jackhmmer (3 iter vs UniRef30)",
        kind="tsv_bare",
        path_template=str(E3_JACKHMMER / "human_vs_{species}.jackhmmer.tsv.gz"),
        family="profile",
        notes="PRIORITY-1 arm; sequence-only state of the art for remote domain detection",
    ),
    "hhblits_uniref30": Arm(
        name="hhblits_uniref30",
        label="HHblits (3 iter vs UniRef30)",
        kind="tsv_bare",
        path_template=str(E3_HHBLITS / "human_vs_{species}.hhblits.tsv.gz"),
        family="profile",
        notes="PRIORITY-1 arm; rerun of the hhblits arm with a real profile database",
    ),
    "foldseek_afdb": Arm(
        name="foldseek_afdb",
        label="FoldSeek (AlphaFold models)",
        kind="tsv_bare",
        path_template=str(E3_FOLDSEEK / "human_vs_{species}.foldseek.tsv.gz"),
        family="structure",
        notes="structure arm; AFDB models already on disk for 54_339 accessions",
    ),
    "esm2_windowed": Arm(
        name="esm2_windowed",
        label="ESM-2 windowed embedding search",
        kind="tsv_bare",
        path_template=str(E3_PLM / "human_vs_{species}.esm2.tsv.gz"),
        family="plm",
        notes="PLM arm; needs a GPU — scaffolded and handed off, not run locally",
    ),
    "interproscan_pfam": Arm(
        name="interproscan_pfam",
        label="InterProScan Pfam scan (annotation ceiling)",
        kind="ceiling",
        path_template=str(E3_INTERPROSCAN / "{species}.pfam.tsv.gz"),
        family="ceiling",
        notes=(
            "CIRCULAR by construction: the ground truth is Pfam-derived, so this arm "
            "recovers ~100%. Reported as the annotation ceiling, excluded from the "
            "HP-only set difference."
        ),
    ),
}

ARMS: dict[str, Arm] = {**BASE_ARMS, **E3_ARMS}


def register_kmerseek_arms(
    ksizes=(10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40),
    score_col: str = "score_tfidf_cont",
) -> dict[str, Arm]:
    """Add one HP arm per k-size to the registry and return the added arms."""
    added = {}
    for k in ksizes:
        arm = _kmerseek_arm(k, score_col=score_col)
        ARMS[arm.name] = arm
        added[arm.name] = arm
    return added


# --------------------------------------------------------------------------------------
# Availability
# --------------------------------------------------------------------------------------


def _is_empty_file(path: Path) -> bool:
    if not path.exists():
        return True
    if path.stat().st_size <= EMPTY_GZIP_BYTES:
        return True
    return False


def arm_status(arm_name: str, species: str) -> dict:
    """Report whether an arm has usable data for a species, and why not if it does not."""
    arm = ARMS[arm_name]
    if arm.kind == "ceiling":
        p = arm.path(species)
        return {
            "arm": arm_name,
            "label": arm.label,
            "family": arm.family,
            "species": species,
            "path": str(p),
            "exists": p.exists(),
            "status": "ceiling (circular)" if p.exists() else "no data",
            "n_bytes": p.stat().st_size if p.exists() else 0,
            "notes": arm.notes,
        }
    p = arm.path(species)
    if not p.exists():
        status = "no data (file missing)"
    elif p.stat().st_size <= EMPTY_GZIP_BYTES:
        status = "no data (empty output)"
    else:
        status = "ok"
    return {
        "arm": arm_name,
        "label": arm.label,
        "family": arm.family,
        "species": species,
        "path": str(p),
        "exists": p.exists(),
        "status": status,
        "n_bytes": p.stat().st_size if p.exists() else 0,
        "notes": arm.notes,
    }


def arm_status_table(arm_names, species_list) -> pl.DataFrame:
    rows = [arm_status(a, s) for a in arm_names for s in species_list]
    return pl.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------

_PIPE_ACC = r"^(?:sp|tr)\|([A-Z0-9]+)\|"


def _strip_pipe(col: str) -> pl.Expr:
    """`sp|Q9NR96|TLR9_HUMAN` -> `Q9NR96`; passes bare accessions through unchanged."""
    return (
        pl.when(pl.col(col).str.contains(r"\|"))
        .then(pl.col(col).str.extract(_PIPE_ACC, 1))
        .otherwise(pl.col(col))
        .alias(col)
    )


def load_pfam_domains(species: str) -> pl.DataFrame:
    """Ground-truth Pfam domain instances for one proteome."""
    return pl.read_parquet(ANNOTATIONS / f"{species}_pfam_domains.parquet")


def load_pfam_summary(species: str) -> dict:
    with open(ANNOTATIONS / f"{species}_pfam_summary.json") as fh:
        return json.load(fh)


def load_benchmark_pairs(species: str, with_strata: bool = True) -> pl.DataFrame:
    """The 60_000 labelled human/species pairs, optionally joined to hard/easy strata."""
    pairs = pl.read_parquet(PAIRS / f"human_vs_{species}_ground_truth.parquet")
    if with_strata:
        strat_path = PAIRS_STRATIFIED / f"human_vs_{species}_hard_easy.parquet"
        if strat_path.exists():
            strata = pl.read_parquet(strat_path).select(
                "human_accession",
                "species_accession",
                "stratum",
                "nw_identity",
                "is_holdout",
                "diamond_best_bitscore",
            )
            pairs = pairs.join(
                strata, on=["human_accession", "species_accession"], how="left"
            )
    return pairs


def load_arm_scores(arm_name: str, species: str) -> pl.DataFrame | None:
    """
    One row per (human_accession, species_accession) with a single ``score`` column.

    Returns ``None`` when the arm has no usable data, so callers can report the arm as
    missing rather than as zero-recall.
    """
    arm = ARMS[arm_name]
    path = arm.path(species)

    if arm.kind == "kmerseek":
        if not path.exists():
            return None
        score_col = arm.extra["score_col"]
        df = pl.read_parquet(path)
        if score_col not in df.columns:
            raise KeyError(f"{score_col} not in {path.name}: {df.columns}")
        return (
            df.select(
                "human_accession",
                "species_accession",
                pl.col(score_col).alias("score"),
            )
            .drop_nulls("score")
            .group_by("human_accession", "species_accession")
            .agg(pl.col("score").max())
        )

    if arm.kind == "ceiling":
        return None

    if _is_empty_file(path):
        return None

    df = pl.read_csv(
        path,
        separator="\t",
        has_header=False,
        new_columns=["query", "target", "bitscore", "evalue"],
        infer_schema_length=10_000,
    )
    df = df.with_columns(_strip_pipe("query"), _strip_pipe("target"))
    return (
        df.select(
            pl.col("query").alias("human_accession"),
            pl.col("target").alias("species_accession"),
            pl.col("bitscore").cast(pl.Float64).alias("score"),
        )
        .drop_nulls(["human_accession", "species_accession", "score"])
        .group_by("human_accession", "species_accession")
        .agg(pl.col("score").max())
    )


def load_afdb_accessions() -> set[str]:
    """Accessions with an AlphaFold model already downloaded."""
    if not AFDB_STRUCTURES.exists():
        return set()
    return {p.name.split("-")[1] for p in AFDB_STRUCTURES.glob("*.cif")}


# --------------------------------------------------------------------------------------
# Matched-specificity calibration
# --------------------------------------------------------------------------------------


def attach_arm_scores(
    pairs: pl.DataFrame, arm_names, species: str
) -> tuple[pl.DataFrame, list[str]]:
    """Left-join each available arm's score onto the pair table.

    Returns (pair table, list of arms that actually had data).
    """
    out = pairs
    available = []
    for name in arm_names:
        scores = load_arm_scores(name, species)
        if scores is None:
            continue
        out = out.join(
            scores.rename({"score": f"score_{name}"}),
            on=["human_accession", "species_accession"],
            how="left",
        )
        available.append(name)
    return out, available


def threshold_at_fpr(pairs: pl.DataFrame, arm_name: str, fpr: float) -> float:
    """
    Score threshold placing exactly ``fpr`` of the *negative* pairs above it.

    Non-detections count as negatives that were correctly rejected, which is what makes
    this comparable across arms with wildly different hit counts.  If an arm produces
    fewer above-threshold negatives than the budget allows, the threshold falls to the
    lowest score it reported (its "any hit" operating point).
    """
    col = f"score_{arm_name}"
    neg = (
        pairs.filter(~pl.col("label"))
        .select(col)
        .drop_nulls()
        .get_column(col)
        .to_numpy()
    )
    n_neg = pairs.filter(~pl.col("label")).height
    budget = int(np.floor(fpr * n_neg))
    if neg.size == 0:
        return np.inf
    if neg.size <= budget:
        return float(neg.min())
    neg_sorted = np.sort(neg)[::-1]
    return float(neg_sorted[budget])


def add_calls(
    pairs: pl.DataFrame, arm_names, fpr: float = 0.01
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Add a boolean ``call_{arm}`` column per arm at matched FPR.

    Returns (pair table with calls, threshold table).
    """
    thresholds = {}
    exprs = []
    for name in arm_names:
        col = f"score_{name}"
        if col not in pairs.columns:
            continue
        thr = threshold_at_fpr(pairs, name, fpr)
        thresholds[name] = thr
        exprs.append(
            (pl.col(col).is_not_null() & (pl.col(col) > thr)).alias(f"call_{name}")
        )
    out = pairs.with_columns(exprs)

    n_neg = pairs.filter(~pl.col("label")).height
    n_pos = pairs.filter(pl.col("label")).height
    rows = []
    for name, thr in thresholds.items():
        called = out.filter(pl.col(f"call_{name}"))
        rows.append(
            {
                "arm": name,
                "label": ARMS[name].label,
                "family": ARMS[name].family,
                "threshold": thr,
                "target_fpr": fpr,
                "realised_fpr": called.filter(~pl.col("label")).height / n_neg,
                "pair_recall": called.filter(pl.col("label")).height / n_pos,
                "n_pairs_called": called.height,
            }
        )
    return out, pl.DataFrame(rows).sort("pair_recall", descending=True)


# --------------------------------------------------------------------------------------
# Domain-level readout
# --------------------------------------------------------------------------------------


def recoverable_domains(species: str, pairs: pl.DataFrame) -> pl.DataFrame:
    """
    Domain instances that *any* homology-transfer method could in principle recover:
    the species protein carries the domain, and at least one positive benchmark pair
    links it to a human protein carrying the same Pfam family.

    One row per (species_accession, pfam_id, domain_start, domain_end).
    """
    domains = load_pfam_domains(species).select(
        pl.col("accession").alias("species_accession"),
        "pfam_id",
        "domain_start",
        "domain_end",
        "domain_length",
        "protein_length",
    )
    positives = (
        pairs.filter(pl.col("label"))
        .select("human_accession", "species_accession", "shared_pfam_ids")
        .with_columns(pl.col("shared_pfam_ids").str.split(";").alias("pfam_id"))
        .explode("pfam_id")
        .filter(pl.col("pfam_id").str.len_chars() > 0)
        .unique(["human_accession", "species_accession", "pfam_id"])
    )
    linked = positives.select("species_accession", "pfam_id").unique()
    return domains.join(linked, on=["species_accession", "pfam_id"], how="inner")


def domain_recovery(
    species: str, pairs_with_calls: pl.DataFrame, arm_names
) -> pl.DataFrame:
    """
    One row per recoverable domain instance with a ``found_{arm}`` boolean per arm.

    A domain is found by arm M if M calls at least one positive pair that links the
    species protein to a human protein sharing that Pfam family.
    """
    call_cols = [f"call_{a}" for a in arm_names if f"call_{a}" in pairs_with_calls.columns]
    used = [c.removeprefix("call_") for c in call_cols]

    positives = (
        pairs_with_calls.filter(pl.col("label"))
        .select("human_accession", "species_accession", "shared_pfam_ids", *call_cols)
        .with_columns(pl.col("shared_pfam_ids").str.split(";").alias("pfam_id"))
        .explode("pfam_id")
        .filter(pl.col("pfam_id").str.len_chars() > 0)
    )
    per_protein_family = positives.group_by("species_accession", "pfam_id").agg(
        [pl.col(c).any().alias(f"found_{a}") for c, a in zip(call_cols, used)]
        + [pl.len().alias("n_human_partners")]
    )

    domains = recoverable_domains(species, pairs_with_calls)
    out = domains.join(
        per_protein_family, on=["species_accession", "pfam_id"], how="left"
    ).with_columns(
        [pl.col(f"found_{a}").fill_null(False) for a in used]
    )
    return out.with_columns(pl.lit(species).alias("species"))


def arm_coverage(domains: pl.DataFrame, arm_names) -> pl.DataFrame:
    """Per-arm domain-level recall, with the honest denominator printed alongside."""
    used = [a for a in arm_names if f"found_{a}" in domains.columns]
    n = domains.height
    rows = []
    for a in used:
        n_found = domains.filter(pl.col(f"found_{a}")).height
        rows.append(
            {
                "arm": a,
                "label": ARMS[a].label,
                "family": ARMS[a].family,
                "n_domains_all": n,
                "n_domains_found": n_found,
                "domain_recall": n_found / n if n else float("nan"),
            }
        )
    return pl.DataFrame(rows).sort("domain_recall", descending=True)


def set_difference(
    domains: pl.DataFrame, hp_arm: str, other_arms
) -> dict:
    """
    The headline number: domains recovered by the HP arm and by no other arm.

    Also returns the reverse (recovered by others, missed by HP), because a claim that
    only reports its own wins does not survive review.
    """
    others = [a for a in other_arms if f"found_{a}" in domains.columns and a != hp_arm]
    if f"found_{hp_arm}" not in domains.columns:
        raise KeyError(f"HP arm {hp_arm} has no found_ column")
    any_other = pl.any_horizontal([pl.col(f"found_{a}") for a in others])
    hp = pl.col(f"found_{hp_arm}")

    d = domains.with_columns(any_other.alias("found_any_other"))
    n = d.height
    n_hp = d.filter(hp).height
    n_other = d.filter(pl.col("found_any_other")).height
    n_hp_only = d.filter(hp & ~pl.col("found_any_other")).height
    n_other_only = d.filter(~hp & pl.col("found_any_other")).height
    n_both = d.filter(hp & pl.col("found_any_other")).height
    n_neither = d.filter(~hp & ~pl.col("found_any_other")).height

    return {
        "hp_arm": hp_arm,
        "other_arms": others,
        "n_domains_all": n,
        "n_found_hp": n_hp,
        "n_found_any_other": n_other,
        "n_hp_only": n_hp_only,
        "n_other_only": n_other_only,
        "n_both": n_both,
        "n_neither": n_neither,
        "hp_only_rate": n_hp_only / n if n else float("nan"),
        "other_only_rate": n_other_only / n if n else float("nan"),
        # The falsifier statistic: of the domains HP finds, what fraction does the best
        # sequence-only profile method also find?  ->1 means the claim narrows to cost.
        "frac_of_hp_also_found_by_others": (n_both / n_hp) if n_hp else float("nan"),
    }


def per_arm_hp_overlap(domains: pl.DataFrame, hp_arm: str, other_arms) -> pl.DataFrame:
    """Pairwise HP-vs-one-arm breakdown; this is where the falsifier is read off."""
    hp = pl.col(f"found_{hp_arm}")
    rows = []
    n_hp = domains.filter(hp).height
    for a in other_arms:
        c = f"found_{a}"
        if c not in domains.columns or a == hp_arm:
            continue
        other = pl.col(c)
        n_both = domains.filter(hp & other).height
        rows.append(
            {
                "other_arm": a,
                "label": ARMS[a].label,
                "family": ARMS[a].family,
                "n_domains_all": domains.height,
                "n_found_hp": n_hp,
                "n_found_other": domains.filter(other).height,
                "n_both": n_both,
                "n_hp_not_other": domains.filter(hp & ~other).height,
                "n_other_not_hp": domains.filter(~hp & other).height,
                "frac_of_hp_covered_by_other": (n_both / n_hp) if n_hp else float("nan"),
            }
        )
    return pl.DataFrame(rows).sort("frac_of_hp_covered_by_other", descending=True)


def structure_applicable_subset(
    domains: pl.DataFrame, pairs: pl.DataFrame, afdb: set[str]
) -> pl.DataFrame:
    """
    Restrict to domains where a structure arm is even applicable — both the species
    protein and at least one linked human partner have an AlphaFold model.

    This yields ``n_domains_ref``.  Domains excluded here are FoldSeek misses caused by
    missing models, and are still counted in ``n_domains_all``.
    """
    if not afdb:
        return domains.head(0)
    human_ok = (
        pairs.filter(pl.col("label"))
        .filter(pl.col("human_accession").is_in(list(afdb)))
        .select("species_accession", "shared_pfam_ids")
        .with_columns(pl.col("shared_pfam_ids").str.split(";").alias("pfam_id"))
        .explode("pfam_id")
        .select("species_accession", "pfam_id")
        .unique()
    )
    return domains.filter(pl.col("species_accession").is_in(list(afdb))).join(
        human_ok, on=["species_accession", "pfam_id"], how="inner"
    )


# --------------------------------------------------------------------------------------
# Cost / throughput
# --------------------------------------------------------------------------------------


TRACE_SOURCES = {
    # arm -> (trace glob, process name(s) that do the search, tag->species parser)
    "tools": (
        REPO / "nextflow-runs" / "pfam-benchmark-tools",
        "pfam_benchmark_tools.*.trace.txt",
    ),
    "kmerseek": (
        REPO / "nextflow-runs" / "qfo-pfam-benchmark",
        "qfo_pfam_benchmark.*.trace.txt",
    ),
}

_DUR = {"d": 86_400.0, "h": 3_600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_duration(text: str) -> float:
    """Nextflow trace durations: '2h 3m 43s', '11.3s', '0ms', '-'.  Returns seconds."""
    import re

    if text is None:
        return float("nan")
    text = text.strip()
    if not text or text == "-":
        return float("nan")
    total = 0.0
    found = False
    for value, unit in re.findall(r"([0-9]*\.?[0-9]+)\s*(ms|[dhms])", text):
        total += float(value) * _DUR[unit]
        found = True
    return total if found else float("nan")


def parse_size(text: str) -> float:
    """Nextflow trace memory strings: '1.1 GB', '999.5 MB', '-'.  Returns GB."""
    import re

    if not text or text.strip() in {"", "-"}:
        return float("nan")
    m = re.match(r"([0-9]*\.?[0-9]+)\s*([KMGT]?B)", text.strip())
    if not m:
        return float("nan")
    scale = {"B": 1 / 1024**3, "KB": 1 / 1024**2, "MB": 1 / 1024, "GB": 1.0, "TB": 1024.0}
    return float(m.group(1)) * scale[m.group(2)]


def load_trace_runtimes() -> pl.DataFrame:
    """
    Real wall-clock and peak-RSS per search task, scraped from the Nextflow trace files
    the existing pipelines already wrote.  Nothing here is estimated.

    Caveat that must travel with these numbers: they come from macOS runs.  Resource
    accounting is only reliable because those runs used Docker; a macOS run without
    containers records no peak_rss at all.  Anything quoted in the paper should be
    re-measured on Linux.
    """
    frames = []
    for source, (directory, pattern) in TRACE_SOURCES.items():
        for path in sorted(directory.glob(pattern)):
            df = pl.read_csv(path, separator="\t", infer_schema_length=0)
            keep = [c for c in ("process", "tag", "status", "realtime", "%cpu", "peak_rss", "cpus") if c in df.columns]
            frames.append(
                df.select(keep).with_columns(
                    pl.lit(source).alias("source"),
                    pl.lit(path.name).alias("trace_file"),
                )
            )
    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal")
    return out.with_columns(
        pl.col("realtime")
        .map_elements(parse_duration, return_dtype=pl.Float64)
        .alias("realtime_s"),
        pl.col("peak_rss")
        .map_elements(parse_size, return_dtype=pl.Float64)
        .alias("peak_rss_gb"),
        pl.col("cpus").cast(pl.Float64, strict=False).alias("n_cpus"),
    )


def cost_table(rows: list[dict]) -> pl.DataFrame:
    """
    Cost/throughput comparison.  Kept as an explicit hand-entered table rather than
    scraped, so that every number carries its provenance string and nothing is invented.
    """
    return pl.DataFrame(rows)


__all__ = [name for name in dir() if not name.startswith("_")]
