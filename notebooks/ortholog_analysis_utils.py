"""
ortholog_analysis_utils.py
Shared utilities for human–mouse ortholog analysis notebooks (118–126, 200–215).

Covers:
- JSON cache I/O
- MGI ground-truth loading
- Kmerseek data loading with Poisson p-values and conservative multiple-testing corrections
- OrthoFinder prediction loading
- GENCODE protein-ID extraction
- UniProt accession extraction (Pfam / QfO benchmark)
- KS_COLS, N_PROTEINS: benchmark column lists and species protein counts
- SCORE_COLS, SCORE_LABELS, add_composite_scores: scoring for all benchmarks
- compute_aucs: ROC-AUC and PR-AUC over all score columns
- AA_FREQUENCIES, ALPHABET_AA_GROUPS, entropy_per_symbol, bits_per_kmer: real (not naive
  log2(n_symbols)) information content per alphabet, for comparing protein/dayhoff/HP on a
  common bits-of-sequence-information axis instead of raw, alphabet-incomparable k-size
- UniProt ID mapping (Ensembl Protein → UniProtKB)
- UniProt annotation fetching and parsing
- MobiDB intrinsic-disorder fetching
- DataFrame annotation helpers
- Summary-statistics utilities
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import requests

# ---------------------------------------------------------------------------
# Paths (defaults – callers can override)
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Users/olga/data/gencode/results-human-mouse-orthologs")

# The kmerseek 0.4.0 HP-only pipeline (nextflow-runs/human-mouse-gencode-orthologs-hp-v040)
# keeps a fully separate outdir so it can never invalidate the main pipeline's resume cache.
# Its k=18-19 HP results live only here, so `genome_wide_results_file` falls back to these
# dirs before giving up.
EXTRA_DATA_DIRS = [Path("/Users/olga/data/gencode/results-human-mouse-orthologs-hp-v040")]
OF_DIR = Path("/Users/olga/data/gencode/data-for-orthofinder/OrthoFinder/Results_Mar03")

# ---------------------------------------------------------------------------
# Alphabet information content (bits per encoded symbol / per k-mer)
# ---------------------------------------------------------------------------
# Real amino-acid background frequencies -- NOT a textbook or uniform assumption -- computed by
# counting every residue in the actual human+mouse proteomes this project searches against
# (gencode.v49.pc_translations.fa: 245,535 sequences + gencode.vM38.pc_translations.fa: 66,668
# sequences; 145,111,545 standard-amino-acid residues after excluding X/U/stop-codon-fragment
# characters, which together account for <0.04% of residues). Computed 2026-08-05.
AA_FREQUENCIES: dict[str, float] = {
    'L': 0.099040, 'S': 0.082679, 'E': 0.072152, 'A': 0.070069, 'G': 0.064989,
    'P': 0.061661, 'V': 0.060746, 'K': 0.058080, 'R': 0.056316, 'T': 0.053380,
    'D': 0.048762, 'Q': 0.048394, 'I': 0.043419, 'F': 0.036325, 'N': 0.035909,
    'Y': 0.026507, 'H': 0.025944, 'M': 0.022029, 'C': 0.021521, 'W': 0.011940,
}

# Amino-acid -> symbol groupings for every alphabet kmerseek's pipeline supports, transcribed
# directly from kmerseek's own encoding tables (kmerseek/src/rust/hp_alphabets.rs for the 6
# named hp_* variants; REFERENCE_sourmash/src/core/src/encodings.rs for protein/dayhoff/hp) --
# not re-derived, so entropy computed on these reflects the real per-symbol information content
# of what kmerseek actually searches with, not a guess at the grouping.
ALPHABET_AA_GROUPS: dict[str, dict[str, str]] = {
    "protein": {aa: aa for aa in AA_FREQUENCIES},
    "dayhoff": {
        "a": "C", "b": "AGPST", "c": "DENQ", "d": "HKR", "e": "ILMV", "f": "FWY",
    },
    "hp": {"h": "AFGILMPVWY", "p": "CDEHKNQRST"},                  # Lehninger; alias of hp_lehninger
    "hp_lehninger": {"h": "AFGILMPVWY", "p": "CDEHKNQRST"},
    "hp_lehninger_plus_c": {"h": "ACFGILMPVWY", "p": "DEHKNQRST"},
    "hp_kyte_doolittle": {"h": "ACFILMV", "p": "DEGHKNPQRSTWY"},
    "hp_pbotc_1st_ed": {"h": "ACFILMPVWY", "p": "DEGHKNQRST"},
    "hp_thomas_dill": {"h": "ACFILMVWY", "p": "DEGHKNPQRST"},
    "hp_thomas_dill_no_c": {"h": "AFILMVWY", "p": "CDEGHKNPQRST"},
}


def entropy_per_symbol(encoding: str) -> float:
    """Shannon entropy (bits) of one encoded symbol under *encoding*, from the REAL amino-acid
    background frequency (:data:`AA_FREQUENCIES`) grouped the way kmerseek actually groups it
    (:data:`ALPHABET_AA_GROUPS`) -- not `log2(n_symbols)`, which assumes every symbol is equally
    likely and overstates every alphabet coarser than `protein` (e.g. every 2-letter HP alphabet
    has naive entropy exactly 1.0 bit, but real entropy ranges 0.94-1.00 bits depending on how
    balanced that specific H/P split happens to fall against real amino-acid usage).
    """
    groups = ALPHABET_AA_GROUPS[encoding]
    ent = 0.0
    for aas in groups.values():
        p = sum(AA_FREQUENCIES[aa] for aa in aas)
        if p > 0:
            ent -= p * math.log2(p)
    return ent


def bits_per_kmer(encoding: str, ksize: int) -> float:
    """Bits of sequence information in one k-mer under *encoding* (`ksize * entropy_per_symbol`,
    the standard iid-positions approximation). The fair x-axis for comparing alphabets of
    different symbol-cardinality: e.g. HP k=30 (30 x ~0.97 bit =~29 bits) and protein k=7
    (7 x ~4.18 bit =~29 bits) carry roughly the same raw information despite a 4x difference in
    raw k, which is invisible if you only plot against k-size.
    """
    return ksize * entropy_per_symbol(encoding)


KSIZE = 24
ALPHA = 0.05

KMERSEEK_TSV = DATA_DIR / f"ortholog_evaluation.hp.k{KSIZE}.tsv.gz"
MGI_FILE = DATA_DIR / "HOM_MouseHumanSequence.rpt.gz"
OF_ORTHOLOGS_TSV = (
    OF_DIR
    / "Orthologues/Orthologues_gencode.v49.pc_translations"
    / "gencode.v49.pc_translations__v__gencode.vM38.pc_translations.tsv"
)

KMERSEEK_USECOLS = [
    "query_name", "target_name",
    "jaccard", "containment", "n_intersecting_hashes", "expected_shared_kmers",
    "poisson_pvalue", "enrichment", "query_tfidf", "mean_matched_kmer_freq",
    "human_gene", "mouse_gene", "is_ortholog",
]

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_cache(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# MGI ground truth
# ---------------------------------------------------------------------------

def load_mgi_orthologs(mgi_file: Path = MGI_FILE) -> tuple[pl.DataFrame, set[tuple[str, str]]]:
    """Return (mgi_pairs DataFrame, mgi_ortholog_set of (HUMAN_UPPER, MOUSE_UPPER) tuples)."""
    mgi_raw = pl.read_csv(
        str(mgi_file),
        separator="\t",
        infer_schema_length=10000,
        null_values=["", "NA"],
    )

    human_rows = (
        mgi_raw
        .filter(pl.col("NCBI Taxon ID") == 9606)
        .select(["DB Class Key", "Symbol"])
        .rename({"Symbol": "human_symbol"})
    )
    mouse_rows = (
        mgi_raw
        .filter(pl.col("NCBI Taxon ID") == 10090)
        .select(["DB Class Key", "Symbol"])
        .rename({"Symbol": "mouse_symbol"})
    )

    mgi_pairs = (
        human_rows
        .join(mouse_rows, on="DB Class Key", how="inner")
        .with_columns([
            pl.col("human_symbol").str.to_uppercase().alias("human_upper"),
            pl.col("mouse_symbol").str.to_uppercase().alias("mouse_upper"),
        ])
    )

    mgi_ortholog_set = set(
        zip(mgi_pairs["human_upper"].to_list(), mgi_pairs["mouse_upper"].to_list())
    )
    return mgi_pairs, mgi_ortholog_set


# ---------------------------------------------------------------------------
# Conservative multiple-testing corrections
# ---------------------------------------------------------------------------

def bh_conservative(p_values: np.ndarray, m_total: int) -> np.ndarray:
    """BH FDR adjustment using *m_total* as the full denominator (including untested pairs)."""
    n = len(p_values)
    order = np.argsort(p_values)
    adj = p_values[order] * m_total / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(adj, 1.0)
    return result


def by_conservative(p_values: np.ndarray, m_total: int) -> np.ndarray:
    """BY FDR (valid under arbitrary dependence) using *m_total* total comparisons."""
    c_m = np.log(m_total) + 0.5772156649015329  # Euler-Mascheroni
    n = len(p_values)
    order = np.argsort(p_values)
    adj = p_values[order] * m_total * c_m / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(adj, 1.0)
    return result


# ---------------------------------------------------------------------------
# Kmerseek data loading
# ---------------------------------------------------------------------------

def load_kmerseek_data(
    kmerseek_tsv: Path = KMERSEEK_TSV,
    mgi_ortholog_set: set[tuple[str, str]] | None = None,
    alpha: float = ALPHA,
    usecols: list[str] = KMERSEEK_USECOLS,
) -> tuple[pl.DataFrame, int, int, float]:
    """Load Kmerseek TSV, compute Poisson p-values, and add conservative corrections.

    Returns:
        df              – Polars DataFrame with p-value columns added
        N_HUMAN         – unique human proteins
        N_MOUSE         – unique mouse proteins
        BONF_THRESHOLD  – raw-p threshold for conservative Bonferroni p < alpha

    Note: older runs (e.g. the original hp k=24 evaluation) don't ship a
    ``poisson_pvalue`` column directly — only ``n_intersecting_hashes`` and
    ``expected_shared_kmers``, from which it's computed here via
    ``poisson.sf(k - 1, lambda)``. Newer (v0.4.0+) runs ship ``poisson_pvalue``
    precomputed, which is read as-is (NOT the same quantity as ``prob_overlap``,
    also present in some files — the two are unrelated scores).
    """
    available = set(pl.scan_csv(str(kmerseek_tsv), separator="\t").collect_schema().names())
    has_poisson_col = "poisson_pvalue" in available
    read_cols = [c for c in usecols if c in available]
    if not has_poisson_col:
        for extra in ("n_intersecting_hashes", "expected_shared_kmers"):
            if extra not in read_cols:
                read_cols.append(extra)

    df = pl.read_csv(
        str(kmerseek_tsv),
        separator="\t",
        columns=read_cols,
        ignore_errors=True,
        null_values=["", "NA", "NaN"],
    )

    if not has_poisson_col:
        from scipy.stats import poisson as _poisson
        k_arr = df["n_intersecting_hashes"].to_numpy()
        lam_arr = df["expected_shared_kmers"].cast(pl.Float64).fill_null(0.0).to_numpy()
        poisson_pvalue = _poisson.sf(k_arr - 1, np.maximum(lam_arr, 1e-300))
        df = df.with_columns(pl.Series("poisson_pvalue", poisson_pvalue))

    N_HUMAN = df["query_name"].n_unique()
    N_MOUSE = df["target_name"].n_unique()
    N_TOTAL = N_HUMAN * N_MOUSE
    BONF_THRESHOLD = alpha / N_TOTAL

    poisson_p = df["poisson_pvalue"].fill_null(1.0).to_numpy()
    bonf = np.clip(poisson_p * N_TOTAL, 0, 1)
    bh = bh_conservative(poisson_p, N_TOTAL)
    by = by_conservative(poisson_p, N_TOTAL)

    df = df.with_columns([
        pl.Series("poisson_p_bonf_conservative", bonf),
        pl.Series("poisson_p_bh_conservative", bh),
        pl.Series("poisson_p_by_conservative", by),
    ])

    df = df.with_columns([
        pl.col("human_gene").str.to_uppercase().alias("human_upper"),
        pl.col("mouse_gene").str.to_uppercase().alias("mouse_upper"),
    ])

    if mgi_ortholog_set is not None:
        is_mgi = [
            (h, m) in mgi_ortholog_set
            for h, m in zip(df["human_upper"].to_list(), df["mouse_upper"].to_list())
        ]
        df = df.with_columns(pl.Series("is_mgi_ortholog", is_mgi))

    return df, N_HUMAN, N_MOUSE, BONF_THRESHOLD


# ---------------------------------------------------------------------------
# OrthoFinder predictions
# ---------------------------------------------------------------------------

def parse_gene_from_id(protein_id: str) -> str:
    parts = protein_id.split("|")
    return parts[-2] if len(parts) >= 2 else protein_id


def load_orthofinder_predictions(
    of_tsv: Path = OF_ORTHOLOGS_TSV,
    human_col: str = "gencode.v49.pc_translations",
    mouse_col: str = "gencode.vM38.pc_translations",
) -> set[tuple[str, str]]:
    """Return set of (HUMAN_GENE_UPPER, MOUSE_GENE_UPPER) OrthoFinder predictions."""
    of_raw = pl.read_csv(str(of_tsv), separator="\t", infer_schema_length=10000)
    of_pairs: set[tuple[str, str]] = set()
    for h_field, m_field in zip(of_raw[human_col].to_list(), of_raw[mouse_col].to_list()):
        humans = [x.strip() for x in str(h_field).split(",") if x.strip() and x.strip() != "nan"]
        mice = [x.strip() for x in str(m_field).split(",") if x.strip() and x.strip() != "nan"]
        for h in humans:
            for m in mice:
                of_pairs.add((parse_gene_from_id(h).upper(), parse_gene_from_id(m).upper()))
    return of_pairs


# ---------------------------------------------------------------------------
# Gene family membership (HGNC gene groups) — real, non-hand-picked family
# definitions, as opposed to a hardcoded gene-symbol dict (cf. the MHC block's (210-215)
# MHC_CLASSES, which predates this and is kept as-is for the MHC recap).
# ---------------------------------------------------------------------------

HGNC_FILE = DATA_DIR / "hgnc_complete_set.txt"


def load_hgnc_gene_groups(hgnc_file: Path = HGNC_FILE) -> pl.DataFrame:
    """Load HGNC's complete gene set, keeping columns needed for family lookups.

    Returns a DataFrame with columns: symbol, locus_group, locus_type, gene_group,
    gene_group_id, ensembl_gene_id. `gene_group` / `gene_group_id` are pipe-delimited
    for genes in multiple groups.
    """
    return pl.read_csv(
        str(hgnc_file),
        separator="\t",
        columns=["symbol", "locus_group", "locus_type", "gene_group", "gene_group_id", "ensembl_gene_id"],
        infer_schema_length=50000,
        null_values=["", "NA"],
    )


def genes_in_group(hgnc_df: pl.DataFrame, pattern: str) -> pl.DataFrame:
    """Filter `hgnc_df` to genes whose `gene_group` matches `pattern` (case-insensitive substring/regex).

    E.g. `genes_in_group(hgnc, "Olfactory receptor")` or `genes_in_group(hgnc, "Cytochrome P450 family 2")`.
    """
    return hgnc_df.filter(
        pl.col("gene_group").is_not_null() & pl.col("gene_group").str.contains(f"(?i){pattern}")
    )


# ---------------------------------------------------------------------------
# RBH (reciprocal-best-hit) scoring — the fair way to compare kmerseek to
# OrthoFinder's own 1:1 orthogroup clustering (see notebook 200, lesson 1: a
# raw p-value significance threshold badly undercounts kmerseek).
# ---------------------------------------------------------------------------

# A kmerseek run that indexed an empty database still writes a well-formed but rowless
# results file: a 13-byte empty zstd frame, or a header-only parquet. Those are the
# dash/underscore empty-index bug's fallout. They must not shadow a real result for the
# same combo in another dir, so treat them as absent rather than as data.
_EMPTY_CSV_ZST_MAX_BYTES = 1024


def _is_empty_results_file(f: Path) -> bool:
    """Cheap enough to call per lookup: reads the parquet footer only, never the data.

    (`pl.scan_parquet(...).select(pl.len()).collect()` would also avoid the row groups but
    spins up a query engine each time, and this runs once per combo per sweep.) Not cached,
    so a conversion finishing mid-session is picked up immediately.
    """
    if f.suffix == ".parquet":
        try:
            return pq.ParquetFile(str(f)).metadata.num_rows == 0
        except Exception:
            return True  # unreadable/truncated parquet is not usable data either
    return f.stat().st_size <= _EMPTY_CSV_ZST_MAX_BYTES


def genome_wide_results_file(encoding: str, ksize: int, data_dir: Path = DATA_DIR) -> Path:
    """Locate a `human_vs_mouse.{encoding}.k{ksize}.results.*` file.

    Prefers `.parquet` (produced by `scripts/convert_results_to_parquet.py` — same rows,
    minus the two unused md5 columns) over the original `.csv.{zst,gz}`, so callers get the
    smaller/faster format automatically once a combo has been converted.

    Searches `data_dir` first, then `EXTRA_DATA_DIRS`. The kmerseek 0.4.0 HP-only pipeline
    writes k=18-19 to its own outdir (see EXTRA_DATA_DIRS), so combos absent from the main
    results dir are picked up there instead of being reported MISSING.
    """
    search_dirs = [data_dir] + [d for d in EXTRA_DATA_DIRS if d != data_dir]
    for d in search_dirs:
        base = d / f"human_vs_mouse.{encoding}.k{ksize}.results"
        for suffix in (".parquet", ".csv.zst", ".csv.gz"):
            f = Path(f"{base}{suffix}")
            if f.exists() and not _is_empty_results_file(f):
                return f
    searched = ", ".join(str(d) for d in search_dirs)
    raise FileNotFoundError(
        f"No human_vs_mouse.{encoding}.k{ksize}.results.{{parquet,csv.zst,csv.gz}} "
        f"in any of: {searched}"
    )


def scan_genome_wide_results(
    encoding: str, ksize: int, data_dir: Path = DATA_DIR, columns: list[str] | None = None,
) -> pl.LazyFrame:
    """`genome_wide_results_file` + format-aware lazy scan (parquet vs csv), optionally
    projected to `columns` right away (cheap for parquet's columnar layout)."""
    f = genome_wide_results_file(encoding, ksize, data_dir)
    lf = pl.scan_parquet(str(f)) if f.suffix == ".parquet" else pl.scan_csv(str(f), ignore_errors=True)
    return lf.select(columns) if columns is not None else lf


def scan_available_columns(
    encoding: str, ksize: int, data_dir: Path, wanted_columns: list[str], quiet: bool = False,
) -> pl.LazyFrame:
    """`scan_genome_wide_results`, restricted to whichever of *wanted_columns* this specific
    combo's file actually has, instead of erroring on the first missing one.

    Needed because the pipeline's output schema isn't identical across every genome-wide file:
    a handful of hp-lehninger re-runs (k24, k26-30) shipped `prob_overlap` instead of
    `poisson_pvalue` -- an older/newer kmerseek version than every other file in this project --
    while everything else has `poisson_pvalue`. Pair with `ensure_poisson_pvalue` below to
    recompute it from `n_intersecting_hashes`/`expected_shared_kmers` when it's missing, the same
    way `load_kmerseek_data` already does for pre-poisson_pvalue TSVs.
    """
    lf = scan_genome_wide_results(encoding, ksize, data_dir)
    available = set(lf.collect_schema().names())
    cols = [c for c in wanted_columns if c in available]
    missing = [c for c in wanted_columns if c not in available]
    if missing and not quiet:
        print(f"  note: {encoding} k={ksize} genome-wide file is missing columns {missing} -- proceeding without them")
    return lf.select(cols)


def ensure_poisson_pvalue(df: pl.DataFrame) -> pl.DataFrame:
    """If `poisson_pvalue` wasn't in the source file (see `scan_available_columns`), compute it
    from `n_intersecting_hashes` and `expected_shared_kmers` -- the same math `load_kmerseek_data`
    already applies for older TSVs that predate the precomputed column. No-op if the column is
    already present."""
    if "poisson_pvalue" in df.columns:
        return df
    from scipy.stats import poisson as _poisson
    k_arr = df["n_intersecting_hashes"].fill_null(0).to_numpy()
    lam_arr = df["expected_shared_kmers"].cast(pl.Float64).fill_null(0.0).to_numpy()
    p = _poisson.sf(k_arr - 1, np.maximum(lam_arr, 1e-300))
    return df.with_columns(pl.Series("poisson_pvalue", p))


def load_all_alphabet_ksize_combos(data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """Every (encoding, ksize) combo notebook 200 confirmed has real genome-wide results for --
    the union of its base protein/dayhoff/hp sweep (`200_alphabet_ksize_matched_scope_comparison.csv`,
    the CSV notebook 200's own `SWEEP_CSV` cell currently writes -- NOT the older, now-stale
    `200_alphabet_ksize_rbh_sweep.csv` name notebook 206 still references, which predates 200's
    hp k=21 result and is missing that one combo) and its 6-variant HP sweep
    (`200_hp_variants_full_sweep.csv`: hp-lehninger, hp-lehninger-plus-c, hp-kyte-doolittle,
    hp-pbotc-1st-ed, hp-thomas-dill, hp-thomas-dill-no-c). Single source of truth for "all
    alphabets and ksizes" so downstream notebooks (203, 206, 211, ...) can't drift from what 200
    actually validated -- add a new alphabet/ksize there first, then it shows up here automatically.

    Returns one row per combo with columns:
    - `dash_encoding`: the --encoding argument scan_genome_wide_results/kmerseek expect
    - `display_encoding`: the underscore label used in this project's figures/labels
    - `ksize`
    """
    base = pl.read_csv(data_dir / "200_alphabet_ksize_matched_scope_comparison.csv").select(
        pl.col("encoding").alias("dash_encoding"),
        pl.col("encoding").alias("display_encoding"),
        "ksize",
    )
    variants = pl.read_csv(data_dir / "200_hp_variants_full_sweep.csv").select(
        "dash_encoding",
        pl.col("encoding").alias("display_encoding"),
        "ksize",
    )
    return (
        pl.concat([base, variants], how="vertical_relaxed")
        .unique(["dash_encoding", "ksize"])
        .sort(["display_encoding", "ksize"])
    )


def alphabet_combo_colors(
    combos: list[tuple[str, int]], display_names: dict[str, str],
) -> tuple[dict[str, str], dict[str, tuple]]:
    """Per-alphabet color families for combo x k-size figures: hue = alphabet (cycled from a
    fixed colormap palette, sorted by display name so assignment is stable run-to-run), shade =
    k-size rank within that alphabet. Shared by 203/206/211 so they can't each hand-roll a
    different color scheme for the same ~100-combo "all alphabets and ksizes" sweep.

    Returns (alphabet_cmaps, combo_colors):
    - `alphabet_cmaps`: dash_encoding -> matplotlib colormap name
    - `combo_colors`: "{display_encoding}_k{ksize}" -> RGBA color
    """
    import matplotlib.pyplot as plt

    palette = ["cividis", "viridis", "plasma_r", "magma", "winter", "copper", "spring", "summer", "Purples"]
    alphabets_sorted = sorted({e for e, _ in combos}, key=lambda e: display_names[e])
    alphabet_cmaps = {enc: palette[i % len(palette)] for i, enc in enumerate(alphabets_sorted)}
    combo_colors = {}
    for enc in {e for e, _ in combos}:
        enc_ks = sorted(k for e, k in combos if e == enc)
        cmap = plt.get_cmap(alphabet_cmaps[enc])
        for i, k in enumerate(enc_ks):
            frac = 0.25 + 0.65 * (i / max(len(enc_ks) - 1, 1))
            combo_colors[f"{display_names.get(enc, enc)}_k{k}"] = cmap(frac)
    return alphabet_cmaps, combo_colors


PAIR_CACHE_DIR = DATA_DIR / "pair_cache"


def load_pair_table(encoding: str, ksize: int, data_dir: Path = DATA_DIR) -> pl.DataFrame:
    """Per-gene-pair max-jaccard table for one alphabet/ksize combo, aggregated from the
    raw genome-wide `human_vs_mouse.{encoding}.k{ksize}.results.*` file -- the single expensive
    step (multi-GB-to-100+GB raw scan) that every downstream per-combo analysis in this project
    needs (notebook 200's F1/AUC sweep, 203/206/211's per-combo comparisons, 202's per-gene pLDDT
    correctness table). Cached to `{data_dir}/pair_cache/{encoding}_k{ksize}_jaccard.parquet` on
    first call so notebooks 200/202/203/206/210 stop each re-scanning the same raw file
    independently -- everything past this point (RBH calling, F1, per-gene tables) is cheap
    in-memory work on a table with one row per observed (human_gene, mouse_gene) candidate, not
    per matched region.

    Metric is jaccard, not containment: the notebook 200 section 2b full 97-combo sweep found
    jaccard wins the ranking-metric comparison for 65/97 combos, almost entirely HP alphabets
    (65/78) -- and HP is this project's priority alphabet family. The cache filename is
    metric-qualified so it doesn't collide with (and silently misread) any pre-existing
    containment-based `{encoding}_k{ksize}.parquet` cache from before this switch.

    Columns: human_gene, mouse_gene, jaccard (max over that gene pair's matched regions).
    """
    cache_path = PAIR_CACHE_DIR / f"{encoding}_k{ksize}_jaccard.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    lf = (
        scan_genome_wide_results(encoding, ksize, data_dir, columns=["query_name", "target_name", "jaccard"])
        .with_columns([
            pl.col("query_name").str.split("|").list.get(6).str.to_uppercase().alias("human_gene"),
            pl.col("target_name").str.split("|").list.get(6).str.to_uppercase().alias("mouse_gene"),
        ])
        .group_by(["human_gene", "mouse_gene"])
        .agg(pl.col("jaccard").max().alias("jaccard"))
    )
    # streaming: low-k HP-variant files can be 100+ GB uncompressed (e.g. hp-thomas-dill
    # k25/26) -- a plain .collect() tries to materialize the whole thing at once and can
    # exhaust available RAM/crash the kernel; streaming processes it in batches instead.
    pair = lf.collect(engine="streaming")

    PAIR_CACHE_DIR.mkdir(exist_ok=True)
    pair.write_parquet(cache_path)
    return pair


def rbh_pairs_and_scope(
    encoding: str, ksize: int, data_dir: Path = DATA_DIR,
) -> tuple[set[tuple[str, str]], set[str]]:
    """Aggregate to per-gene-pair max jaccard (via the shared `load_pair_table` cache) and
    call reciprocal-best-hit (RBH) pairs: each human gene's top mouse hit must agree with that
    mouse gene's top human hit.

    Returns (rbh_set, human_genes_in_scope) — scope is every human gene the file covers, for
    re-scoring OrthoFinder against the same matched gene universe (notebook 200, lesson 2).
    """
    pair = load_pair_table(encoding, ksize, data_dir)
    scope = set(pair["human_gene"].unique().to_list())

    # secondary sort key makes tie-breaking (equal max jaccard) deterministic
    best_h2m = (pair.sort(["jaccard", "mouse_gene"], descending=[True, False])
                .group_by("human_gene").agg(pl.col("mouse_gene").first().alias("best_mouse")))
    best_m2h = (pair.sort(["jaccard", "human_gene"], descending=[True, False])
                .group_by("mouse_gene").agg(pl.col("human_gene").first().alias("best_human")))
    rbh = (best_h2m.join(best_m2h, left_on=["human_gene", "best_mouse"], right_on=["best_human", "mouse_gene"])
           .select(["human_gene", "best_mouse"]).rename({"best_mouse": "mouse_gene"}))
    rbh_set = set(zip(rbh["human_gene"].to_list(), rbh["mouse_gene"].to_list()))
    return rbh_set, scope


def per_gene_rbh_table(
    encoding: str, ksize: int, truth_lists: pl.DataFrame, data_dir: Path = DATA_DIR,
) -> pl.DataFrame:
    """Per-human-gene RBH call + MGI correctness (notebook 202's `per_gene_rbh`, generalized to
    read from the shared `load_pair_table` cache instead of re-scanning the raw file).

    `truth_lists` must have columns `human_gene`, `true_mouse_genes` (list[str]) -- e.g.
    `mgi_pairs.group_by("human_gene").agg(pl.col("mouse_gene").alias("true_mouse_genes"))`.

    Returns one row per human gene in scope: human_gene, mouse_gene (RBH pick), jaccard,
    is_rbh, has_mgi_truth, is_correct.
    """
    pair = load_pair_table(encoding, ksize, data_dir)

    best_h2m = (pair.sort(["jaccard", "mouse_gene"], descending=[True, False])
                .group_by("human_gene")
                .agg([pl.col("mouse_gene").first().alias("rbh_mouse_gene"),
                      pl.col("jaccard").first().alias("jaccard")]))
    best_m2h = (pair.sort(["jaccard", "human_gene"], descending=[True, False])
                .group_by("mouse_gene").agg(pl.col("human_gene").first().alias("best_human")))

    recip = (best_h2m.select(["human_gene", "rbh_mouse_gene"])
              .join(best_m2h, left_on="rbh_mouse_gene", right_on="mouse_gene", how="left")
              .with_columns((pl.col("best_human") == pl.col("human_gene")).fill_null(False).alias("is_rbh")))

    final = (
        best_h2m
        .join(recip.select(["human_gene", "is_rbh"]), on="human_gene", how="left")
        .join(truth_lists, on="human_gene", how="left")
        .with_columns([
            pl.col("true_mouse_genes").is_not_null().alias("has_mgi_truth"),
            pl.struct(["rbh_mouse_gene", "true_mouse_genes"]).map_elements(
                lambda s: s["true_mouse_genes"] is not None and s["rbh_mouse_gene"] in s["true_mouse_genes"],
                return_dtype=pl.Boolean,
            ).alias("_in_truth_set"),
        ])
        .with_columns((pl.col("is_rbh") & pl.col("_in_truth_set")).alias("is_correct"))
        .select(["human_gene", "rbh_mouse_gene", "jaccard", "is_rbh", "has_mgi_truth", "is_correct"])
        .rename({"rbh_mouse_gene": "mouse_gene"})
        .with_columns([pl.lit(encoding).alias("encoding"), pl.lit(ksize).alias("ksize")])
    )
    return final


def prf1(called: set, truth: set) -> tuple[float, float, float, int, int, int]:
    """Precision, recall, F1, TP, FP, FN for a called-pair set vs. a ground-truth pair set."""
    tp, fp, fn = len(called & truth), len(called - truth), len(truth - called)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1, tp, fp, fn


def load_family_kmerseek_scores(
    encoding: str,
    ksize: int,
    human_genes: set[str] | None = None,
    mouse_genes: set[str] | None = None,
    mgi_ortholog_set: set[tuple[str, str]] | None = None,
    data_dir: Path = DATA_DIR,
    n_total: int | None = None,
) -> pl.DataFrame:
    """Load one alphabet/ksize genome-wide results file, restricted to rows touching a gene
    family, with composite scores (`add_composite_scores`) and an MGI-truth `label` column.

    At least one of `human_genes` / `mouse_genes` must be given (rows are kept if the human
    gene is in `human_genes` OR the mouse gene is in `mouse_genes` — usually only one side is
    a real family and the other is left as every partner it was compared against).

    `n_total` (Bonferroni/BH denominator) defaults to `N_PROTEINS['human'] * N_PROTEINS['mouse']`
    — the full proteome search space — rather than this file's own row-derived unique-gene count,
    so the correction stays consistent and comparable across alphabets/k without an extra
    full-file scan per family lookup. This is a deliberate simplification, stated here rather
    than silently: it's the conventional "full search space" Bonferroni denominator, not a
    per-alphabet realized-hit count.
    """
    if human_genes is None and mouse_genes is None:
        raise ValueError("must supply human_genes and/or mouse_genes")
    if n_total is None:
        n_total = N_PROTEINS["human"] * N_PROTEINS["mouse"]

    # Older genome-wide runs (e.g. plain "hp") don't ship `poisson_pvalue` — only
    # `n_intersecting_hashes`/`expected_shared_kmers`, from which it's derived (same
    # fallback as `load_kmerseek_data`). NOT the same quantity as `prob_overlap`, also
    # present in some of these older files — the two are unrelated scores.
    f = genome_wide_results_file(encoding, ksize, data_dir)
    schema_names = set(
        pl.scan_parquet(str(f)).collect_schema().names() if f.suffix == ".parquet"
        else pl.scan_csv(str(f), n_rows=1).collect_schema().names()
    )
    has_poisson_col = "poisson_pvalue" in schema_names
    base_cols = ["human_gene", "mouse_gene", "enrichment", "containment", "jaccard", "mean_matched_kmer_freq", "query_tfidf"]
    extra_cols = ["poisson_pvalue"] if has_poisson_col else ["n_intersecting_hashes", "expected_shared_kmers"]
    raw_cols = ["query_name", "target_name"] + base_cols[2:] + extra_cols

    lf = (
        scan_genome_wide_results(encoding, ksize, data_dir, columns=raw_cols)
        .with_columns([
            pl.col("query_name").str.split("|").list.get(6).str.to_uppercase().alias("human_gene"),
            pl.col("target_name").str.split("|").list.get(6).str.to_uppercase().alias("mouse_gene"),
        ])
    )
    cond = pl.lit(False)
    if human_genes is not None:
        cond = cond | pl.col("human_gene").is_in(list(human_genes))
    if mouse_genes is not None:
        cond = cond | pl.col("mouse_gene").is_in(list(mouse_genes))
    df = lf.filter(cond).select(base_cols + extra_cols).collect(engine="streaming")
    if df.height == 0:
        return df

    if not has_poisson_col:
        from scipy.stats import poisson as _poisson
        k_arr = df["n_intersecting_hashes"].to_numpy()
        lam_arr = df["expected_shared_kmers"].cast(pl.Float64).fill_null(0.0).to_numpy()
        df = df.with_columns(pl.Series("poisson_pvalue", _poisson.sf(k_arr - 1, np.maximum(lam_arr, 1e-300))))

    poisson_p = df["poisson_pvalue"].fill_null(1.0).to_numpy()
    df = df.with_columns([
        pl.Series("poisson_p_bonf_conservative", np.clip(poisson_p * n_total, 0, 1)),
        pl.Series("poisson_p_bh_conservative", bh_conservative(poisson_p, n_total)),
    ])
    df = add_composite_scores(df)

    if mgi_ortholog_set is not None:
        label = [
            int((h, m) in mgi_ortholog_set)
            for h, m in zip(df["human_gene"].to_list(), df["mouse_gene"].to_list())
        ]
        df = df.with_columns(pl.Series("label", label, dtype=pl.Int8))
    return df


def load_families_kmerseek_scores(
    encoding: str,
    ksize: int,
    family_gene_sets: dict[str, set[str]],
    mgi_ortholog_set: set[tuple[str, str]] | None = None,
    data_dir: Path = DATA_DIR,
    n_total: int | None = None,
) -> dict[str, pl.DataFrame]:
    """Batched version of `load_family_kmerseek_scores` for MULTIPLE human-gene families against
    the same (encoding, ksize) file, in a single raw-file scan.

    Motivation: scoring N families one at a time (N calls to `load_family_kmerseek_scores`) rescans
    the same multi-GB genome-wide file once per family that happens to share a combo -- for
    notebook 206's 5-family x ~9-combo deep dive that's 45 raw scans when only ~9 are actually
    needed. This scans each (encoding, ksize) file exactly ONCE, filtered to the union of every
    family's genes, computes composite scores once on the combined result, then splits it back
    out per family in memory (cheap) -- the raw scan is the expensive part this avoids repeating.

    Human-genes-only (unlike `load_family_kmerseek_scores`, which also accepts `mouse_genes`) --
    every current caller only queries by human gene family membership.

    Returns {family_name: DataFrame}, same columns/labeling as `load_family_kmerseek_scores`.
    A family with zero matching rows gets an empty (but column-complete after collect) DataFrame,
    not omitted, so callers can distinguish "family present but empty" from a missing combo.
    """
    if n_total is None:
        n_total = N_PROTEINS["human"] * N_PROTEINS["mouse"]
    all_genes = set().union(*family_gene_sets.values()) if family_gene_sets else set()

    f = genome_wide_results_file(encoding, ksize, data_dir)
    schema_names = set(
        pl.scan_parquet(str(f)).collect_schema().names() if f.suffix == ".parquet"
        else pl.scan_csv(str(f), n_rows=1).collect_schema().names()
    )
    has_poisson_col = "poisson_pvalue" in schema_names
    base_cols = ["human_gene", "mouse_gene", "enrichment", "containment", "jaccard", "mean_matched_kmer_freq", "query_tfidf"]
    extra_cols = ["poisson_pvalue"] if has_poisson_col else ["n_intersecting_hashes", "expected_shared_kmers"]
    raw_cols = ["query_name", "target_name"] + base_cols[2:] + extra_cols

    lf = (
        scan_genome_wide_results(encoding, ksize, data_dir, columns=raw_cols)
        .with_columns([
            pl.col("query_name").str.split("|").list.get(6).str.to_uppercase().alias("human_gene"),
            pl.col("target_name").str.split("|").list.get(6).str.to_uppercase().alias("mouse_gene"),
        ])
    )
    df = (
        lf.filter(pl.col("human_gene").is_in(list(all_genes)))
        .select(base_cols + extra_cols)
        .collect(engine="streaming")
    )

    if df.height > 0:
        if not has_poisson_col:
            from scipy.stats import poisson as _poisson
            k_arr = df["n_intersecting_hashes"].to_numpy()
            lam_arr = df["expected_shared_kmers"].cast(pl.Float64).fill_null(0.0).to_numpy()
            df = df.with_columns(pl.Series("poisson_pvalue", _poisson.sf(k_arr - 1, np.maximum(lam_arr, 1e-300))))

        poisson_p = df["poisson_pvalue"].fill_null(1.0).to_numpy()
        df = df.with_columns([
            pl.Series("poisson_p_bonf_conservative", np.clip(poisson_p * n_total, 0, 1)),
            pl.Series("poisson_p_bh_conservative", bh_conservative(poisson_p, n_total)),
        ])
        df = add_composite_scores(df)
        if mgi_ortholog_set is not None:
            label = [
                int((h, m) in mgi_ortholog_set)
                for h, m in zip(df["human_gene"].to_list(), df["mouse_gene"].to_list())
            ]
            df = df.with_columns(pl.Series("label", label, dtype=pl.Int8))

    return {
        fam_name: (df.filter(pl.col("human_gene").is_in(list(genes))) if df.height > 0 else df)
        for fam_name, genes in family_gene_sets.items()
    }


def length_floor_mask(
    seq_lengths: np.ndarray, ksize: int, scaled: int = 1, min_expected_kmers: float = 2.0,
) -> np.ndarray:
    """Flag sequences too short, at a given k/scaled, to produce a meaningful sketch.

    A sequence of length L has (L - k + 1) k-mers; at FracMinHash `scaled`, the expected
    sketch size is (L - k + 1) / scaled. Below `min_expected_kmers` the sketch is dominated
    by sampling noise (see notebook 085's hash-floor analysis). Returns a boolean array,
    True = sequence clears the floor (keep).
    """
    seq_lengths = np.asarray(seq_lengths, dtype=float)
    n_kmers = np.maximum(seq_lengths - ksize + 1, 0)
    expected_sketch = n_kmers / scaled
    return expected_sketch >= min_expected_kmers


# ---------------------------------------------------------------------------
# Gene coordinates (GENCODE GTF) — notebook 215's synteny work.
#
# Human uses the `chr_patch_hapl_scaff` flavor (not `basic.annotation`, which GENCODE ships
# primary-assembly-only) specifically so the 8 GRCh38 MHC alt haplotypes are present and the
# ALT-contig filter this notebook needs (see 215's ALT-contig trap check) has something real to remove instead
# of trivially reporting zero. GENCODE names alt/patch/scaffold sequences by their GenBank
# accession (e.g. `GL000251.2`), not UCSC's `chr6_GL000251v2_alt` — HUMAN_MHC_ALT_CONTIGS below
# is the accession form, confirmed present with real gene annotations (184-341 genes each).
# Mouse GRCm39 carries no alt MHC haplotype, so `basic.annotation` (primary assembly only) is
# sufficient there.
# ---------------------------------------------------------------------------

GENCODE_DIR = Path("/Users/olga/data/gencode")
HUMAN_GTF_CHR_PATCH_HAPL_SCAFF = GENCODE_DIR / "human/v49/gencode.v49.chr_patch_hapl_scaff.annotation.gtf.gz"
MOUSE_GTF_BASIC = GENCODE_DIR / "mouse/m38/gencode.vM38.basic.annotation.gtf.gz"

HUMAN_PRIMARY_CHROMS: set[str] = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY", "chrM"}
MOUSE_PRIMARY_CHROMS: set[str] = {f"chr{i}" for i in range(1, 20)} | {"chrX", "chrY", "chrM"}

# GRCh38's 8 MHC alt haplotypes (GENCODE/GenBank accession naming; see module note above).
HUMAN_MHC_ALT_CONTIGS: set[str] = {
    "GL000250.2", "GL000251.2", "GL000252.2", "GL000253.2",
    "GL000254.2", "GL000255.2", "GL000256.2", "GL000257.2",
}


def parse_gencode_gtf_genes(gtf_path: Path, chroms: set[str] | None = None) -> pl.DataFrame:
    """Parse `gene`-feature rows out of a GENCODE GTF (gzip-transparent via polars' native reader).

    `chroms`, if given, restricts the scan to those seqnames before the attribute-column regex
    extraction (the expensive part) runs -- e.g. {"chr17"} for a single-chromosome notebook
    instead of the whole ~50-60k-gene genome. Returns chrom, start, end, strand, gene_id
    (versionless), gene_type, gene_name -- one row per gene model (a gene present on both a
    primary chromosome and an alt/patch contig gets one row each, by design: that duplication is
    exactly what the ALT-contig filter downstream needs to see and remove).
    """
    lf = pl.scan_csv(
        str(gtf_path), separator="\t", has_header=False, comment_prefix="#", quote_char=None,
        new_columns=["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attr"],
    ).filter(pl.col("feature") == "gene")
    if chroms is not None:
        lf = lf.filter(pl.col("chrom").is_in(list(chroms)))
    lf = lf.select([
        "chrom", "start", "end", "strand",
        pl.col("attr").str.extract(r'gene_id "([^"]+)"', 1).str.split(".").list.get(0).alias("gene_id"),
        pl.col("attr").str.extract(r'gene_type "([^"]+)"', 1).alias("gene_type"),
        pl.col("attr").str.extract(r'gene_name "([^"]+)"', 1).alias("gene_name"),
    ])
    return lf.collect()


def region_by_anchor_genes(gene_table: pl.DataFrame, anchor_names: tuple[str, str]) -> tuple[int, int]:
    """Define a region as "anchor gene 1 through anchor gene 2 inclusive" -- the gene-model
    extents of two named boundary genes -- rather than a hardcoded bp interval.

    Motivation (notebook 215): a literature-cited region boundary (e.g. Shiina/Kulski's classical
    HLA span) is only as stable as whichever single transcript its source used to define GABBR1's
    start; GENCODE's own `gene` feature start is the 5'-most extent of *any* annotated transcript
    and drifts as isoforms get added (confirmed here: GABBR1 differs from the literature value by
    ~46kb, entirely attributable to alternative-promoter isoforms, not an assembly-build shift --
    KIFC1 in the same pair matches within 215bp). Re-deriving the boundary from this project's own
    anchor gene models, every time, is robust to that drift in a way a pasted-in bp pair is not.
    """
    rows = gene_table.filter(pl.col("gene_name").is_in(list(anchor_names)))
    return int(rows["start"].min()), int(rows["end"].max())


def load_gene_coordinates(
    human_chroms: set[str] = {"chr6"} | HUMAN_MHC_ALT_CONTIGS,
    mouse_chroms: set[str] = frozenset({"chr17"}),
    human_gtf: Path = HUMAN_GTF_CHR_PATCH_HAPL_SCAFF,
    mouse_gtf: Path = MOUSE_GTF_BASIC,
    cache_path: Path = DATA_DIR / "215_gene_coordinates.parquet",
    force: bool = False,
) -> pl.DataFrame:
    """Gene -> (assembly, chrom, start, end, strand, gene_type, is_primary_assembly) table for
    notebook 215, restricted by default to human chr6 (+ its 8 MHC alt contigs) and mouse chr17
    -- the synteny notebook's "start with just the MHC window, then add on" scoping decision. Widen
    `human_chroms`/`mouse_chroms` (and pass `force=True` to bypass the cache) if a later section
    needs anchors outside that window.

    Cached to `cache_path` (default `215_gene_coordinates.parquet`) since re-parsing the 93MB
    chr_patch_hapl_scaff GTF on every notebook run is wasted work once the window is fixed.
    """
    if cache_path.exists() and not force:
        return pl.read_parquet(cache_path)

    human = parse_gencode_gtf_genes(human_gtf, human_chroms).with_columns([
        pl.lit("human").alias("assembly"),
        pl.col("chrom").is_in(HUMAN_PRIMARY_CHROMS).alias("is_primary_assembly"),
    ])
    mouse = parse_gencode_gtf_genes(mouse_gtf, mouse_chroms).with_columns([
        pl.lit("mouse").alias("assembly"),
        pl.col("chrom").is_in(MOUSE_PRIMARY_CHROMS).alias("is_primary_assembly"),
    ])
    combined = pl.concat([human, mouse], how="vertical_relaxed")
    cache_path.parent.mkdir(exist_ok=True, parents=True)
    combined.write_parquet(cache_path)
    return combined


# ---------------------------------------------------------------------------
# MHC benchmark: gene taxonomy and domain architecture
#
# Shared by the 210-215 MHC block. Previously redefined inline in the pre-split notebook 201; kept here so a
# correction to a domain boundary or a class assignment lands in every notebook at once instead
# of drifting between them.
#
# Domain boundaries are real UniProt feature-table values in PROTEIN numbering (signal peptide
# included), not mature-chain numbering:
#   class I heavy (HLA-A, P04439): signal 1-24 | ARD a1 25-114 + a2 115-206 | Ig-C1 a3 207-298 | TM/cyto 299-365
#   class II alpha (HLA-DRA, P01903): signal 1-25 | ARD a1 26-109 | Ig a2 110-203 | TM 204-254
#   class II beta  (HLA-DRB1, P01911): signal 1-29 | ARD b1 30-124 | Ig b2 125-227 | TM 228-266
#   light chain    (B2M, P61769): signal 1-20 | no ARD | Ig 25-113
# ---------------------------------------------------------------------------

#: Structural taxonomy of the 25-gene human MHC query set. Hand-curated rather than pulled from an
#: HGNC group: HGNC's own "Histocompatibility complex" group is a single flat bucket, not split by
#: class, so it cannot substitute for this.
MHC_CLASSES: dict[str, str] = {
    "HLA-A": "I classical", "HLA-B": "I classical", "HLA-C": "I classical",
    "HLA-E": "I non-classical", "HLA-F": "I non-classical", "HLA-G": "I non-classical",
    "MICA": "I related (MIC)", "MICB": "I related (MIC)",
    "HLA-DRA": "II alpha", "HLA-DQA1": "II alpha", "HLA-DQA2": "II alpha",
    "HLA-DPA1": "II alpha", "HLA-DMA": "II alpha", "HLA-DOA": "II alpha",
    "HLA-DRB1": "II beta", "HLA-DRB5": "II beta", "HLA-DQB1": "II beta", "HLA-DQB2": "II beta",
    "HLA-DPB1": "II beta", "HLA-DMB": "II beta", "HLA-DOB": "II beta",
    "B2M": "light chain", "TAP1": "processing (ctrl)", "TAP2": "processing (ctrl)",
    "TAPBP": "processing (ctrl)",
}

MHC_CLASS_ORDER: list[str] = [
    "I classical", "I non-classical", "I related (MIC)", "II alpha", "II beta",
    "light chain", "processing (ctrl)",
]

MHC_CLASS_COLORS: dict[str, str] = {
    "I classical": "#D65F5F", "I non-classical": "#E1917F", "I related (MIC)": "#C4AD66",
    "II alpha": "#4878CF", "II beta": "#77BEDB", "light chain": "#6ACC65",
    "processing (ctrl)": "#999999",
}

#: The six class I heavy chains -- the genes every class I AUC in this project rests on.
MHC_CLASS_I_GENES: list[str] = ["HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G"]

#: HLA-A (P04439) domain architecture for the localisation figures: (name, start, end, color).
DOMAINS_CLASS_I: list[tuple[str, int, int, str]] = [
    ("signal", 1, 24, "#cccccc"), ("α1 (ARD)", 25, 114, "#D65F5F"),
    ("α2 (ARD)", 115, 206, "#E1917F"), ("α3 (Ig-C1)", 207, 298, "#4878CF"),
    ("TM/cyto", 299, 365, "#dddddd"),
]
ARD_I: tuple[int, int] = (25, 206)
IG_I: tuple[int, int] = (207, 298)
TM_I: tuple[int, int] = (299, 365)
LEN_HLA_A: int = 365

#: HLA class I peptide-contact residues, Saper/Bjorkman groove set, converted from mature to
#: protein numbering (+24 for HLA-A's 24-residue signal peptide).
CONTACT_MATURE: list[int] = [
    5, 7, 9, 24, 45, 59, 62, 63, 66, 67, 69, 70, 73, 74, 76, 77, 80, 81, 84, 95, 97, 99, 114, 116,
    123, 143, 146, 147, 152, 156, 159, 163, 167, 171,
]
CONTACT_PROT: list[int] = [r + 24 for r in CONTACT_MATURE]

#: Per-gene-class (ARD ranges, Ig range) in protein numbering, for within-molecule density.
MHC_ARCH: dict[str, tuple[list[tuple[int, int]], tuple[int, int]]] = {
    "I": ([(25, 114), (115, 206)], (207, 298)),
    "II alpha": ([(26, 109)], (110, 203)),
    "II beta": ([(30, 124)], (125, 227)),
}

#: Real UniProt boundaries for the two mouse classical class I genes that receive most class-I
#: matched regions. H2-D1's boundaries are numerically identical to HLA-A's; H2-K1's are offset
#: by -3 (a 3-aa-shorter signal peptide). Both verified against the UniProt REST API, not scaled
#: or guessed from HLA-A.
MOUSE_DOMAINS_I: dict[str, dict] = {
    "H2-D1": {"acc": "P01899", "ARD": [(25, 114), (115, 206)], "Ig": (207, 298)},
    "H2-K1": {"acc": "P01901", "ARD": [(22, 111), (112, 203)], "Ig": (204, 295)},
}


def mhc_gene_arch(mhc_class: str) -> tuple[list[tuple[int, int]], tuple[int, int]] | None:
    """(ARD ranges, Ig range) for an :data:`MHC_CLASSES` value, or None if the class has no ARD.

    "I related (MIC)" reuses the class I architecture: MICA/MICB share the a1/a2/a3 fold even
    though they do not bind peptide. "light chain" (B2M) and "processing (ctrl)" return None --
    B2M is a free-standing Ig domain with no ARD of its own, so it is excluded from any ARD-vs-Ig
    ratio by construction rather than by oversight.
    """
    if mhc_class in ("I classical", "I non-classical", "I related (MIC)"):
        return MHC_ARCH["I"]
    return MHC_ARCH.get(mhc_class)


# ---------------------------------------------------------------------------
# One-to-many ortholog label policies
#
# A gene with N > 1 MGI-listed mouse partners has no single "the ortholog", so "is this pair a
# true ortholog?" has no answer until a rule is named. Three rules are defensible and they give
# materially different AUCs, so the rule has to be stated rather than inherited from whatever a
# join happened to do:
#
#   any_member           y=1 for every MGI-listed partner. Standard and assumption-light, but the
#                        positive set grows with the expansion (|P| = 14 for HLA-A vs 1 for a
#                        class II gene), so class I and class II AUCs are not comparable.
#   best_scoring_member  y=1 only for the highest-scoring MGI partner. Gives |P| = 1, but the
#                        label is chosen using the score being evaluated -- circular, and it
#                        inflates AUC by construction. Reported as a bound, never as the headline.
#   synteny_nearest      y=1 only for the partner closest to the position synteny predicts. Gives
#                        |P| = 1 from evidence (genomic position) that is independent of the
#                        protein sequence, so it is the only rule under which a one-to-many class
#                        I number is comparable to a 1:1 class II number.
# ---------------------------------------------------------------------------

LABEL_POLICIES: tuple[str, str, str] = ("any_member", "best_scoring_member", "synteny_nearest")

LABEL_POLICY_LABELS: dict[str, str] = {
    "any_member": "any MGI member",
    "best_scoring_member": "best-scoring member (circular)",
    "synteny_nearest": "syntenically nearest",
}

#: Column name each policy writes into a pair table.
LABEL_POLICY_COLS: dict[str, str] = {p: f"label_{p}" for p in LABEL_POLICIES}


def syntenic_nearest_partner(
    mgi_pairs: pl.DataFrame,
    human_coords: pl.DataFrame,
    mouse_coords: pl.DataFrame,
    *,
    anchor_pairs: pl.DataFrame | None = None,
    human_gene_col: str = "human_upper",
    mouse_gene_col: str = "mouse_upper",
) -> pl.DataFrame:
    """For each human gene, pick the ONE MGI mouse partner sitting closest to the genomic position
    local synteny predicts -- the label rule that gives a one-to-many gene a size-1 positive set.

    The prediction comes from *flanking 1:1 anchors*, not from a whole-region average: human genes
    with exactly one MGI partner on the region's modal mouse chromosome are sorted by human
    coordinate, and a query gene's expected mouse position is linearly interpolated between the
    nearest anchor on each side. Interpolating (rather than assuming a fixed offset) handles the
    mouse MHC's inverted orientation for free -- ascending human coordinate runs
    GABBR1->KIFC1 while ascending mouse runs Kifc1->Gabbr1, so the fitted local slope is simply
    negative and no axis flip has to be hardcoded anywhere.

    `anchor_pairs` supplies the gene set the anchor ladder is built from, and should normally be
    the WHOLE surrounding syntenic region rather than just the genes being labelled. Defaulting it
    to `mgi_pairs` is only safe when `mgi_pairs` is itself region-wide: a query set of two dozen
    genes contains too few 1:1 anchors to interpolate between, so every gene falls outside the
    anchor range, every prediction degrades to `extrapolated`, and whole classes collapse onto a
    single nearest partner. Observed directly while building notebook 210 -- all six class I genes
    were assigned H2-D1 until the ladder was widened to the full MHC region.

    A gene is never an anchor for itself, so a 1:1 gene's own label is still predicted from its
    neighbours rather than trivially from its own position.

    Returns one row per human gene: n_partners, syntenic_partner, expected_mouse_start,
    distance_bp (partner to prediction), and `basis` -- one of `interpolated` (anchors on both
    sides), `extrapolated` (anchors on one side only, local slope from the two nearest),
    `single_anchor` (one anchor total, offset unknown, position copied), `no_anchor`, or
    `off_modal_chrom` (the chosen partner is not on the modal chromosome, so the distance is not
    physically meaningful). `basis` is returned rather than logged so a caller can drop or
    down-weight the weak cases instead of silently trusting all of them equally.
    """
    h_lookup = human_coords.select([
        pl.col("gene_name").str.to_uppercase().alias(human_gene_col),
        pl.col("start").alias("human_start"),
    ]).unique(subset=[human_gene_col])
    m_lookup = mouse_coords.select([
        pl.col("gene_name").str.to_uppercase().alias(mouse_gene_col),
        pl.col("chrom").alias("mouse_chrom"),
        pl.col("start").alias("mouse_start"),
    ]).unique(subset=[mouse_gene_col])

    def _with_coords(df: pl.DataFrame) -> pl.DataFrame:
        return (df.select([human_gene_col, mouse_gene_col]).unique()
                .join(h_lookup, on=human_gene_col, how="inner")
                .join(m_lookup, on=mouse_gene_col, how="inner"))

    pairs = _with_coords(mgi_pairs)
    anchor_src = _with_coords(anchor_pairs) if anchor_pairs is not None else pairs
    if pairs.is_empty() or anchor_src.is_empty():
        return pl.DataFrame(schema={
            human_gene_col: pl.String, "n_partners": pl.UInt32, "syntenic_partner": pl.String,
            "expected_mouse_start": pl.Float64, "distance_bp": pl.Float64, "basis": pl.String,
        })

    # Anchors are indexed BY MOUSE CHROMOSOME, not pooled under one global "modal" chromosome.
    # A contiguous human region does not have to map to a single mouse chromosome: the human xMHC
    # splits across mouse chr17 (the MHC proper) and mouse chr13 (the Btn/Rfp block carried away
    # during the region's evolution), at 251 vs 255 anchor pairs. A single global mode picks chr13
    # by a four-pair margin and then treats every chr17 MHC partner as off-target -- which silently
    # discards the synteny signal for exactly the genes this function exists to label. The target
    # chromosome is therefore chosen per gene, from that gene's own partners.
    n_by_anchor = anchor_src.group_by(human_gene_col).len().rename({"len": "n_anchor_partners"})
    one_to_one = anchor_src.join(n_by_anchor, on=human_gene_col).filter(
        pl.col("n_anchor_partners") == 1)
    anchors_by_chrom: dict[str, tuple[list[str], np.ndarray, np.ndarray]] = {}
    for chrom, grp in one_to_one.group_by("mouse_chrom"):
        chrom = chrom[0] if isinstance(chrom, tuple) else chrom
        grp = grp.sort("human_start")
        anchors_by_chrom[chrom] = (
            grp[human_gene_col].to_list(),
            grp["human_start"].to_numpy().astype(float),
            grp["mouse_start"].to_numpy().astype(float),
        )

    def _predict(gene: str, h: float, chrom: str) -> tuple[float | None, str]:
        if chrom not in anchors_by_chrom:
            return None, "no_anchor"
        a_human, a_h, a_m = anchors_by_chrom[chrom]
        keep = np.array([g != gene for g in a_human])
        hs, ms = a_h[keep], a_m[keep]
        if len(hs) == 0:
            return None, "no_anchor"
        if len(hs) == 1:
            return float(ms[0]), "single_anchor"
        i = int(np.searchsorted(hs, h))
        if 0 < i < len(hs):                      # flanked on both sides
            lo, hi = i - 1, i
            basis = "interpolated"
        elif i == 0:                             # left of every anchor
            lo, hi = 0, 1
            basis = "extrapolated"
        else:                                    # right of every anchor
            lo, hi = len(hs) - 2, len(hs) - 1
            basis = "extrapolated"
        span = hs[hi] - hs[lo]
        if span == 0:
            return float(ms[lo]), basis
        slope = (ms[hi] - ms[lo]) / span
        return float(ms[lo] + (h - hs[lo]) * slope), basis

    rows = []
    for gene, grp in pairs.group_by(human_gene_col):
        gene = gene[0] if isinstance(gene, tuple) else gene
        h = float(grp["human_start"][0])
        # Target chromosome = where this gene's own partners actually are. Ties are broken toward
        # the chromosome carrying more anchors, i.e. the better-supported syntenic block.
        counts = grp.group_by("mouse_chrom").len()
        best_n = counts["len"].max()
        tied = counts.filter(pl.col("len") == best_n)["mouse_chrom"].to_list()
        target_chrom = max(tied, key=lambda c: len(anchors_by_chrom.get(c, ([], [], []))[0]))

        expected, basis = _predict(gene, h, target_chrom)
        pool = grp.filter(pl.col("mouse_chrom") == target_chrom)
        if expected is None:
            # No anchors on this chromosome at all: fall back to the first partner by name so the
            # choice is at least deterministic and reproducible, and say so via `basis`.
            rows.append({
                human_gene_col: gene, "n_partners": grp.height, "mouse_chrom": target_chrom,
                "syntenic_partner": pool.sort(mouse_gene_col)[mouse_gene_col][0],
                "expected_mouse_start": None, "distance_bp": None, "basis": basis,
            })
            continue
        d = np.abs(pool["mouse_start"].to_numpy().astype(float) - expected)
        j = int(np.argmin(d))
        rows.append({
            human_gene_col: gene, "n_partners": grp.height, "mouse_chrom": target_chrom,
            "syntenic_partner": pool[mouse_gene_col][j],
            "expected_mouse_start": expected, "distance_bp": float(d[j]), "basis": basis,
        })
    return pl.DataFrame(rows).sort(human_gene_col)


def add_label_policies(
    pair_df: pl.DataFrame,
    partners_by_human: dict[str, set[str]],
    synteny_partner: dict[str, str],
    *,
    score_col: str = "containment",
    group_cols: tuple[str, ...] = ("combo",),
    human_col: str = "human_gene",
    mouse_col: str = "mouse_gene",
) -> pl.DataFrame:
    """Add `label_any_member`, `label_best_scoring_member`, `label_synteny_nearest` to a candidate
    pair table -- the three rules in :data:`LABEL_POLICIES`, computed side by side so a notebook
    can report all three instead of inheriting whichever one its join implied.

    `best_scoring_member` is resolved *within* `group_cols` (default: per alphabet/k-size combo),
    because the highest-scoring partner is a property of the scoring run, not of the gene.
    """
    out = pair_df.with_columns([
        pl.struct([human_col, mouse_col]).map_elements(
            lambda s: s[mouse_col] in partners_by_human.get(s[human_col], ()),
            return_dtype=pl.Boolean,
        ).alias("label_any_member"),
        pl.struct([human_col, mouse_col]).map_elements(
            lambda s: synteny_partner.get(s[human_col]) == s[mouse_col],
            return_dtype=pl.Boolean,
        ).alias("label_synteny_nearest"),
    ])
    # Best-scoring MGI member, per (group_cols..., human gene): rank only among true members so a
    # gene whose partners were all missed contributes no positive rather than promoting a decoy.
    keys = list(group_cols) + [human_col]
    best = (
        out.filter(pl.col("label_any_member"))
        .sort(score_col, descending=True, nulls_last=True)
        .group_by(keys, maintain_order=True)
        .agg(pl.col(mouse_col).first().alias("_best_member"))
    )
    return (
        out.join(best, on=keys, how="left")
        .with_columns((pl.col(mouse_col) == pl.col("_best_member")).fill_null(False)
                      .alias("label_best_scoring_member"))
        .drop("_best_member")
    )


def prevalence_table(
    pair_df: pl.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("combo", "mhc_class"),
    label_cols: tuple[str, ...] | None = None,
) -> pl.DataFrame:
    """Positive-class size and prevalence per group, for every label policy present.

    AUPRC's no-skill baseline IS the prevalence, so an AUPRC reported without it is unreadable:
    a stratum whose positive class is 15x larger than another's starts from a 15x higher floor.
    Always plot/print this alongside any AUPRC comparison across strata or across k.
    """
    label_cols = label_cols or tuple(
        c for c in LABEL_POLICY_COLS.values() if c in pair_df.columns
    )
    aggs = [pl.len().alias("n_candidates")]
    for c in label_cols:
        aggs += [
            pl.col(c).sum().alias(f"n_pos_{c.removeprefix('label_')}"),
            pl.col(c).mean().alias(f"prevalence_{c.removeprefix('label_')}"),
        ]
    return pair_df.group_by(list(group_cols)).agg(aggs).sort(list(group_cols))


# ---------------------------------------------------------------------------
# GENCODE name → Ensembl IDs
# ---------------------------------------------------------------------------

def extract_ids(name_series: pl.Series) -> pl.DataFrame:
    """Parse ENSP|ENST|ENSG|…|gene_name|length GENCODE IDs.

    Returns a DataFrame with columns: protein_id, gene_id, gene_name.
    """
    rows = []
    for name in name_series.to_list():
        parts = str(name).split("|")
        protein_id = parts[0].split(".")[0] if parts else ""
        gene_id = parts[2].split(".")[0] if len(parts) > 2 else ""
        gene_name = parts[-2] if len(parts) >= 2 else ""
        rows.append({"protein_id": protein_id, "gene_id": gene_id, "gene_name": gene_name})
    return pl.DataFrame(rows)


def prepare_set_df(subset: pl.DataFrame, label: str) -> pl.DataFrame:
    """Add human/mouse Ensembl IDs and a set_label column to a Kmerseek subset."""
    human_ids = extract_ids(subset["query_name"])
    mouse_ids = extract_ids(subset["target_name"])
    return subset.with_columns([
        human_ids["protein_id"].alias("human_protein_id"),
        human_ids["gene_id"].alias("human_gene_id"),
        human_ids["gene_name"].alias("human_gene_name"),
        mouse_ids["protein_id"].alias("mouse_protein_id"),
        mouse_ids["gene_id"].alias("mouse_gene_id"),
        mouse_ids["gene_name"].alias("mouse_gene_name"),
        pl.lit(label).alias("set_label"),
    ])


# ---------------------------------------------------------------------------
# UniProt ID mapping (Ensembl Protein → UniProtKB)
# ---------------------------------------------------------------------------

def submit_uniprot_mapping(ids: list[str], from_db: str = "Ensembl_Protein", to_db: str = "UniProtKB") -> str:
    resp = requests.post(
        "https://rest.uniprot.org/idmapping/run",
        data={"from": from_db, "to": to_db, "ids": ",".join(ids)},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("jobId", "")


def poll_uniprot_job(job_id: str, max_wait: int = 300) -> dict:
    status_url = f"https://rest.uniprot.org/idmapping/status/{job_id}"
    stream_url = f"https://rest.uniprot.org/idmapping/stream/{job_id}"
    waited = 0
    while waited < max_wait:
        resp = requests.get(status_url, timeout=30)
        resp.raise_for_status()
        status = resp.json()
        if "results" in status or status.get("jobStatus") == "FINISHED":
            break
        if status.get("jobStatus") in ("ERROR", "FAILURE"):
            print(f"Job failed: {status}")
            return {}
        time.sleep(5)
        waited += 5
    resp = requests.get(stream_url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def batch_map_ensembl_to_uniprot(
    ensembl_ids: list[str],
    cache: dict,
    batch_size: int = 500,
) -> dict:
    """Query UniProt ID-mapping API for IDs not already in *cache*. Returns updated cache."""
    to_query = [eid for eid in ensembl_ids if eid and eid not in cache]
    print(f"Querying {len(to_query):,} new IDs ({len(ensembl_ids) - len(to_query):,} cached)")
    for i in range(0, len(to_query), batch_size):
        batch = to_query[i : i + batch_size]
        print(f"  Batch {i // batch_size + 1}: {len(batch)} IDs ...", end=" ", flush=True)
        try:
            job_id = submit_uniprot_mapping(batch)
            if not job_id:
                continue
            result = poll_uniprot_job(job_id)
            n_mapped = 0
            for entry in result.get("results", []):
                from_id = entry.get("from", "")
                to_info = entry.get("to", {})
                accession = (
                    to_info.get("primaryAccession")
                    if isinstance(to_info, dict)
                    else str(to_info)
                )
                if accession:
                    cache[from_id] = accession
                    n_mapped += 1
            print(f"OK ({n_mapped} mapped)")
        except Exception as exc:
            print(f"ERROR: {exc}")
            time.sleep(5)
    return cache


# ---------------------------------------------------------------------------
# UniProt annotation fetching
# ---------------------------------------------------------------------------

_ann_session = requests.Session()
_ann_session.headers["User-Agent"] = "kmerseek-ortho-characterization/1.0"


def fetch_uniprot_entry(accession: str) -> dict:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.json"
    try:
        resp = _ann_session.get(url, timeout=30)
        if resp.status_code == 429:
            time.sleep(5)
            resp = _ann_session.get(url, timeout=30)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        print(f"  WARN {accession}: {exc}")
        return {}


def parse_uniprot_entry(entry: dict) -> dict:
    if not entry:
        return {}
    result: dict = {}
    result["seq_length"] = entry.get("sequence", {}).get("length")
    keywords = [kw.get("name", "") for kw in entry.get("keywords", [])]
    result["keywords"] = "; ".join(keywords)
    subloc_texts = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SUBCELLULAR LOCATION":
            for loc in comment.get("subcellularLocations", []):
                val = loc.get("location", {}).get("value", "")
                if val:
                    subloc_texts.append(val)
    result["subcellular_location"] = "; ".join(subloc_texts)
    combined = (result["subcellular_location"] + " " + result["keywords"]).lower()
    result["is_membrane"] = any(x in combined for x in ["membrane", "transmembrane"])
    result["is_secreted"] = any(x in combined for x in ["secreted", "extracellular"])
    result["is_nuclear"] = any(x in combined for x in ["nucleus", "nuclear"])
    tm_count = 0
    disordered = 0
    for feat in entry.get("features", []):
        ftype = feat.get("type", "")
        if ftype == "Transmembrane":
            tm_count += 1
        if ftype == "Region" and "disordered" in feat.get("description", "").lower():
            disordered += 1
    result["n_transmembrane"] = tm_count
    result["n_disordered_regions"] = disordered
    result["has_disordered_regions"] = disordered > 0
    return result


def fetch_all_annotations(
    accessions: list[str],
    cache: dict,
    sleep: float = 0.1,
    checkpoint_every: int = 100,
    cache_path: Path | None = None,
) -> dict:
    """Fetch UniProt entries for *accessions* not in *cache*. Returns updated cache."""
    to_query = [a for a in accessions if a and a not in cache]
    print(f"Total accessions: {len(accessions):,}  |  To query: {len(to_query):,}")
    for i, acc in enumerate(to_query):
        cache[acc] = fetch_uniprot_entry(acc)
        if (i + 1) % checkpoint_every == 0:
            print(f"  {i + 1}/{len(to_query)} ...")
            if cache_path:
                save_cache(cache, cache_path)
        time.sleep(sleep)
    if cache_path:
        save_cache(cache, cache_path)
    print(f"Done. Cache has {len(cache):,} entries.")
    return cache


def add_annotations(df_pl: pl.DataFrame, ann_cache: dict) -> pl.DataFrame:
    """Append UniProt annotation columns to *df_pl* using *ann_cache*."""
    rows = [
        parse_uniprot_entry(ann_cache.get(acc, {}) if acc else {})
        for acc in df_pl["uniprot_acc"].to_list()
    ]
    ann_df = pl.DataFrame({
        "seq_length": [r.get("seq_length") for r in rows],
        "keywords": [r.get("keywords", "") for r in rows],
        "subcellular_location": [r.get("subcellular_location", "") for r in rows],
        "is_membrane": [bool(r.get("is_membrane", False)) for r in rows],
        "is_secreted": [bool(r.get("is_secreted", False)) for r in rows],
        "is_nuclear": [bool(r.get("is_nuclear", False)) for r in rows],
        "n_transmembrane": [int(r.get("n_transmembrane", 0)) for r in rows],
        "n_disordered_regions": [int(r.get("n_disordered_regions", 0)) for r in rows],
        "has_disordered_regions": [bool(r.get("has_disordered_regions", False)) for r in rows],
    })
    return pl.concat([df_pl, ann_df], how="horizontal")


# ---------------------------------------------------------------------------
# MobiDB intrinsic disorder
# ---------------------------------------------------------------------------

_mobi_session = requests.Session()


def fetch_mobidb(uniprot_id: str) -> float | None:
    url = f"https://mobidb.bio.unipd.it/api/download?acc={uniprot_id}&format=json"
    try:
        resp = _mobi_session.get(url, timeout=30)
        if resp.status_code == 404:
            return None
        if resp.status_code == 429:
            time.sleep(10)
            resp = _mobi_session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if entry.get("acc") == uniprot_id:
                for key in ["prediction-disorder-th_90", "curated-disorder-merge"]:
                    val = entry.get(key, {})
                    if val and val.get("content_fraction") is not None:
                        return float(val["content_fraction"])
                consensus = entry.get("disorder", {}).get("consensus", {})
                if consensus.get("content_fraction") is not None:
                    return float(consensus["content_fraction"])
        return None
    except Exception as exc:
        print(f"  WARN MobiDB {uniprot_id}: {exc}")
        return None


def fetch_all_mobidb(
    accessions: list[str],
    cache: dict,
    sleep: float = 0.1,
    checkpoint_every: int = 100,
    cache_path: Path | None = None,
) -> dict:
    """Fetch MobiDB disorder fractions for *accessions* not in *cache*."""
    to_query = [a for a in accessions if a and a not in cache]
    print(f"To query: {len(to_query):,}")
    for i, acc in enumerate(to_query):
        cache[acc] = fetch_mobidb(acc)
        if (i + 1) % checkpoint_every == 0:
            print(f"  {i + 1}/{len(to_query)} ...")
            if cache_path:
                save_cache(cache, cache_path)
        time.sleep(sleep)
    if cache_path:
        save_cache(cache, cache_path)
    print(f"Done. MobiDB cache has {len(cache):,} entries.")
    return cache


def add_disorder_fraction(df_pl: pl.DataFrame, mobi_cache: dict) -> pl.DataFrame:
    fracs = [mobi_cache.get(acc) for acc in df_pl["uniprot_acc"].to_list()]
    return df_pl.with_columns(pl.Series("disorder_fraction", fracs, dtype=pl.Float64))


# ---------------------------------------------------------------------------
# Ensembl BioMart percent-identity (human-mouse divergence covariate; see
# notebook 206 — dN/dS was discontinued genome-wide by Ensembl at Release 100,
# but percent identity is still live on BioMart and is the mechanistically
# correct covariate for a substitution-tolerance argument anyway).
# ---------------------------------------------------------------------------

_BIOMART_URL = "https://www.ensembl.org/biomart/martservice"


#: Rules for collapsing a one-to-many gene's per-homolog percent identities to one number.
#: `max` is an extreme order statistic -- E[max of N] rises with N, so a gene with 14 mouse
#: partners is biased upward relative to a 1:1 gene by the very expansion that makes it hard.
#: `median` is not biased that way and is the default for exactly that reason.
PERC_ID_POLICIES: tuple[str, ...] = ("median", "max", "mean", "one_to_one_only")


def fetch_biomart_perc_id_pairs(
    ensg_ids: list[str], batch_size: int = 100, timeout: int = 180, retries: int = 2,
) -> pl.DataFrame:
    """Batch-query BioMart for per-HOMOLOG-PAIR human-mouse percent identity.

    Returns one row per (human ENSG, mouse homolog): `ensg`, `mouse_symbol`, `perc_id_h2m`,
    `perc_id_m2h`, `perc_id` (symmetric mean of the two directions, which differ slightly because
    of alignment-length asymmetry). Genes with no mouse homolog, and failed batches, are absent.

    This is the pair-level primitive; use :func:`collapse_perc_id` to reduce it to one value per
    gene under a NAMED policy. Keeping the pair level is the whole point: BioMart returns one row
    per homolog, so a gene with 14 mouse partners produces 14 rows, and any per-gene number is an
    order statistic over those rows whose identity has to be chosen deliberately.

    `batch_size` default lowered from the original 300 to 100, and `timeout` raised from 60s to
    180s (notebook 215 hit repeated read timeouts on a 300-gene batch -- a single-gene
    smoke test against this same endpoint took 27s on its own, so 300 genes at 60s was
    underprovisioned, not a transient fluke). `retries` gives each batch a second attempt before
    it's given up on and reported as a WARN, since a single stalled request shouldn't sacrifice an
    otherwise-fine batch.
    """
    rows: list[dict] = []
    for i in range(0, len(ensg_ids), batch_size):
        batch = ensg_ids[i : i + batch_size]
        query = (
            '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
            '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
            'count="" datasetConfigVersion="0.6">'
            '<Dataset name="hsapiens_gene_ensembl" interface="default">'
            f'<Filter name="ensembl_gene_id" value="{",".join(batch)}"/>'
            '<Attribute name="ensembl_gene_id"/>'
            '<Attribute name="mmusculus_homolog_associated_gene_name"/>'
            '<Attribute name="mmusculus_homolog_perc_id"/>'
            '<Attribute name="mmusculus_homolog_perc_id_r1"/>'
            "</Dataset></Query>"
        )
        for attempt in range(retries + 1):
            try:
                resp = requests.get(_BIOMART_URL, params={"query": query}, timeout=timeout)
                resp.raise_for_status()
                text = resp.text.strip()
                # BioMart serves its outage and error pages as HTML with HTTP 200, so
                # raise_for_status() does NOT catch them. Left undetected, the TSV parser below
                # simply matches no lines, the batch looks like "this gene has no mouse homolog",
                # and the empty result gets cached as if it were real -- silently turning an
                # outage into a permanent data gap. Detected explicitly and raised so the retry
                # path runs and a failure is reported rather than absorbed.
                head = text[:200].lstrip().lower()
                if head.startswith(("<html", "<!doctype", "<?xml")) or "service unavailable" in head:
                    raise RuntimeError(
                        f"BioMart returned an HTML error page, not TSV (HTTP {resp.status_code}): "
                        f"{text[:120]!r}")
                for line in text.splitlines():
                    parts = line.split("\t")
                    if len(parts) != 4:
                        continue
                    ensg, mouse_symbol, pid, pid_r1 = parts
                    try:
                        vals = [float(v) for v in (pid, pid_r1) if v not in ("", "NA")]
                    except ValueError:
                        continue
                    if not vals:
                        continue
                    rows.append({
                        "ensg": ensg,
                        "mouse_symbol": mouse_symbol.upper() or None,
                        "perc_id_h2m": float(pid) if pid not in ("", "NA") else None,
                        "perc_id_m2h": float(pid_r1) if pid_r1 not in ("", "NA") else None,
                        "perc_id": float(np.mean(vals)),
                    })
                break
            except Exception as exc:
                if attempt < retries:
                    time.sleep(3)
                    continue
                print(f"  WARN BioMart batch {i // batch_size + 1} (after {retries + 1} attempts): {exc}")
        time.sleep(0.2)
    schema = {"ensg": pl.String, "mouse_symbol": pl.String, "perc_id_h2m": pl.Float64,
              "perc_id_m2h": pl.Float64, "perc_id": pl.Float64}
    return pl.DataFrame(rows, schema=schema) if rows else pl.DataFrame(schema=schema)


def collapse_perc_id(
    pair_df: pl.DataFrame, policy: str = "median", *, key: str = "ensg",
) -> pl.DataFrame:
    """Collapse the pair-level table from :func:`fetch_biomart_perc_id_pairs` to one percent
    identity per gene under an explicit `policy` from :data:`PERC_ID_POLICIES`.

    Returns `key`, `perc_id`, `n_homologs`, `perc_id_policy` -- the policy travels with the data
    so a downstream ledger cannot end up mixing estimators across rows without it being visible.

    `one_to_one_only` nulls out any gene with more than one homolog rather than picking among
    them: the strictest option, and the one that makes a cross-stratum comparison unambiguous at
    the cost of dropping exactly the expanded families that motivated the question.
    """
    if policy not in PERC_ID_POLICIES:
        raise ValueError(f"policy must be one of {PERC_ID_POLICIES}, got {policy!r}")
    agg = {
        "median": pl.col("perc_id").median(),
        "max": pl.col("perc_id").max(),
        "mean": pl.col("perc_id").mean(),
        "one_to_one_only": pl.when(pl.len() == 1).then(pl.col("perc_id").first()).otherwise(None),
    }[policy]
    return (
        pair_df.group_by(key)
        .agg([agg.alias("perc_id"), pl.len().alias("n_homologs")])
        .with_columns(pl.lit(policy).alias("perc_id_policy"))
        .sort(key)
    )


def fetch_biomart_perc_id(
    ensg_ids: list[str], batch_size: int = 100, timeout: int = 180, retries: int = 2,
    policy: str = "median",
) -> dict[str, float]:
    """ENSG -> percent identity, collapsed under a NAMED `policy` (default `median`).

    .. warning::
       Before 2026-08-08 this function had no policy at all. It looped over BioMart's response
       lines assigning ``out[ensg] = ...`` and BioMart returns **one line per homolog**, so for a
       one-to-many gene the retained value was whichever homolog happened to come last in the
       response -- an arbitrary order statistic, neither max nor median, and not stable across
       calls. Any cached value produced by that version should be refetched, not reused; caches
       written by it use the old filenames and are deliberately not read back by the new code.
    """
    pairs = fetch_biomart_perc_id_pairs(ensg_ids, batch_size, timeout, retries)
    if pairs.is_empty():
        return {}
    collapsed = collapse_perc_id(pairs, policy).drop_nulls("perc_id")
    return dict(zip(collapsed["ensg"].to_list(), collapsed["perc_id"].to_list()))


# ---------------------------------------------------------------------------
# Summary statistics helpers
# ---------------------------------------------------------------------------

def fmt_median(s: pl.Series) -> str:
    val = s.median()
    return f"{val:.3f}" if val is not None else "N/A"


def safe_mwu(a, b):
    from scipy.stats import mannwhitneyu
    a = [v for v in a if v is not None and not (isinstance(v, float) and np.isnan(v))]
    b = [v for v in b if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return mannwhitneyu(a, b, alternative="two-sided")[1]


# ---------------------------------------------------------------------------
# Pfam / QfO benchmark helpers
# ---------------------------------------------------------------------------

#: Kmerseek numeric columns to select from a results CSV.
KS_COLS: list[str] = [
    'enrichment', 'containment', 'jaccard', 'query_tfidf',
    'mean_matched_kmer_freq', 'sum_matched_kmer_freq',
    'expected_shared_kmers', 'poisson_pvalue',
]

#: Canonical protein counts per species, used as Bonferroni/BH denominators.
N_PROTEINS: dict[str, int] = {
    'human':       20600,  # UP000005640_9606
    'mouse':       21989,  # UP000000589_10090
    'chicken':     18116,  # UP000000539_9031
    'zebrafish':   25698,  # UP000000437_7955
    'ciona':       16678,  # UP000008144_7719
    'fly':         13811,  # UP000000803_7227
    'worm':        19819,  # UP000001940_6239
    'yeast':        6049,  # UP000002311_559292
    'arabidopsis': 27500,  # UP000006548_3702
    'ecoli':        4391,  # UP000000625_83333
}


def extract_accession_expr(col: str) -> pl.Expr:
    """Return a Polars expression that extracts a UniProt accession from *col*.

    Handles two common formats found in kmerseek query_name / target_name:

    - Swiss-Prot / TrEMBL pipe-delimited:  ``sp|P12345|GENE_HUMAN``
    - Raw accession at start of string:     ``P12345 …``
    """
    return (
        pl.col(col)
        .str.extract(r'(?:sp|tr)\|([A-Z0-9]+)\|', group_index=1)
        .fill_null(
            pl.col(col).str.extract(
                r'^([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})',
                group_index=1
            )
        )
    )


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

#: All score column names produced by :func:`add_composite_scores` plus raw kmerseek metrics.
SCORE_COLS: list[str] = [
    'score_enr_cont_freq', 'score_enr_cont', 'score_neglogp_cont',
    'score_tfidf_cont', 'score_neglogp', 'score_bonf_neglogp',
    'score_bonf_neglogp_cont', 'score_bh_neglogq', 'score_bh_neglogq_cont',
    'enrichment', 'containment', 'jaccard',
]

#: Human-readable display labels for each score column.
SCORE_LABELS: dict[str, str] = {
    'score_enr_cont_freq':     'enr × cont / freq  ★',
    'score_enr_cont':          'enr × cont',
    'score_neglogp_cont':      '-log p × cont',
    'score_tfidf_cont':        'tfidf × cont',
    'score_neglogp':           '-log p (raw)',
    'score_bonf_neglogp':      '-log(p·N_bonf)',
    'score_bonf_neglogp_cont': '-log(p·N_bonf) × cont',
    'score_bh_neglogq':        '-log q_BH',
    'score_bh_neglogq_cont':   '-log q_BH × cont',
    'enrichment':              'enrichment',
    'containment':             'containment',
    'jaccard':                 'jaccard',
}


def add_composite_scores(df: pl.DataFrame) -> pl.DataFrame:
    """Add composite scoring columns for kmerseek benchmarking.

    All scores are oriented so that HIGHER = more likely a true positive.

    Required input columns
    ----------------------
    enrichment, containment, mean_matched_kmer_freq, query_tfidf,
    poisson_pvalue, poisson_p_bonf_conservative, poisson_p_bh_conservative
    (the latter two are produced by :func:`load_kmerseek_data`).
    """
    eps = 1e-9
    return df.with_columns([
        (pl.col('enrichment') * pl.col('containment')
         / (pl.col('mean_matched_kmer_freq') + eps)).alias('score_enr_cont_freq'),
        (pl.col('enrichment') * pl.col('containment')).alias('score_enr_cont'),
        (-pl.col('poisson_pvalue').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
         * pl.col('containment')).alias('score_neglogp_cont'),
        (pl.col('query_tfidf') * pl.col('containment')).alias('score_tfidf_cont'),
        (-pl.col('poisson_pvalue').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
        ).alias('score_neglogp'),
        (-pl.col('poisson_p_bonf_conservative').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
        ).alias('score_bonf_neglogp'),
        (-pl.col('poisson_p_bonf_conservative').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
         * pl.col('containment')).alias('score_bonf_neglogp_cont'),
        (-pl.col('poisson_p_bh_conservative').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
        ).alias('score_bh_neglogq'),
        (-pl.col('poisson_p_bh_conservative').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
         * pl.col('containment')).alias('score_bh_neglogq_cont'),
    ])


def compute_aucs(df: pl.DataFrame, score_cols: list[str] | None = None) -> dict:
    """Compute ROC-AUC and PR-AUC for every score column present in *df*.

    Parameters
    ----------
    df         : DataFrame with a ``label`` column (0/1 ints) and score columns.
    score_cols : columns to evaluate; defaults to :data:`SCORE_COLS`.

    Returns
    -------
    dict mapping col_name → ``{"roc_auc": float, "pr_auc": float}``
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    cols = score_cols if score_cols is not None else SCORE_COLS
    y = df["label"].cast(pl.Int8).to_numpy()
    if y.sum() == 0 or (1 - y).sum() == 0:
        return {}
    out = {}
    for col in cols:
        if col not in df.columns:
            continue
        scores = df[col].fill_null(0.0).fill_nan(0.0).to_numpy()
        try:
            out[col] = {
                "roc_auc": roc_auc_score(y, scores),
                "pr_auc":  average_precision_score(y, scores),
            }
        except Exception:
            pass
    return out


def leaderboard_df(aucs: dict) -> pl.DataFrame:
    """Turn a :func:`compute_aucs` dict into a DataFrame ranked by AUPRC (descending)."""
    rows = [{"metric": k, "label": SCORE_LABELS.get(k, k), **v} for k, v in aucs.items()]
    lb = pl.DataFrame(rows).sort("pr_auc", descending=True)
    return lb.with_columns(pl.Series("rank", range(1, lb.height + 1)))


def plot_metric_leaderboard(aucs: dict, ax, title: str = "") -> pl.DataFrame:
    """Horizontal AUPRC bar chart for all metrics in *aucs*, winner in navy, the
    old score_enr_cont_freq incumbent in red (see notebook 120). Returns the
    underlying :func:`leaderboard_df` (rank 1 = best) for printing alongside.
    """
    lb = leaderboard_df(aucs)
    labels = lb["label"].to_list()
    values = lb["pr_auc"].to_list()
    metrics = lb["metric"].to_list()
    colors = [
        "#e31a1c" if m == "score_enr_cont_freq" else ("#084594" if i == 0 else "#9ecae1")
        for i, m in enumerate(metrics)
    ]
    ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("AUPRC")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    return lb


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 0,
    metric: str = "roc_auc",
) -> dict:
    """Bootstrap-resample (y_true, y_score) pairs to get a CI on ROC-AUC or PR-AUC.

    Needed because family-level n can be as low as ~15-40 pairs (see notebook 206),
    where a point-estimate AUC (as in `compute_aucs`) overstates precision. Resamples
    pairs with replacement `n_boot` times; degenerate resamples (all-one-class) are
    skipped and don't count toward `n_boot`.

    Returns dict: {"point": float, "lo": float, "hi": float, "n_boot_used": int}.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    scorer = roc_auc_score if metric == "roc_auc" else average_precision_score
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    point = scorer(y_true, y_score)

    rng = np.random.default_rng(random_state)
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        boot_vals.append(scorer(yt, ys))

    alpha = (1 - ci) / 2
    lo, hi = (np.quantile(boot_vals, [alpha, 1 - alpha]) if boot_vals else (float("nan"), float("nan")))
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "n_boot_used": len(boot_vals)}


def print_threshold_comparison(df: pl.DataFrame, alpha: float = ALPHA) -> None:
    """Print a table comparing TP / kmerseek-only counts under different thresholds."""
    thresholds = {
        "raw p<0.05":                  pl.col("poisson_pvalue") < alpha,
        "conservative Bonferroni":     pl.col("poisson_p_bonf_conservative") < alpha,
        "conservative BH (FDR)":       pl.col("poisson_p_bh_conservative") < alpha,
        "conservative BY (any dep.)":  pl.col("poisson_p_by_conservative") < alpha,
    }
    of_tps = df.filter(pl.col("is_orthofinder") & pl.col("is_mgi_ortholog"))
    print(f"{'Threshold':<35} {'TPs':>8} {'FPs':>8} {'kmerseek-only TPs':>18}")
    print("-" * 73)
    for label, expr in thresholds.items():
        tps = df.filter(expr & pl.col("is_mgi_ortholog"))
        fps = df.filter(expr & ~pl.col("is_mgi_ortholog"))
        only = df.filter(expr & pl.col("is_mgi_ortholog") & ~pl.col("is_orthofinder"))
        print(f"{label:<35} {len(tps):>8,} {len(fps):>8,} {len(only):>18,}")
    print(f"{'OrthoFinder (reference)':<35} {len(of_tps):>8,} {'—':>8} {'—':>18}")
