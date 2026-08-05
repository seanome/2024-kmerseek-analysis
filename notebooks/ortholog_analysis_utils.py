"""
ortholog_analysis_utils.py
Shared utilities for human–mouse ortholog analysis notebooks (118–126).

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
# definitions, as opposed to a hardcoded gene-symbol dict (cf. notebook 201's
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
    alphabets and ksizes" so downstream notebooks (201, 203, 206, ...) can't drift from what 200
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
    k-size rank within that alphabet. Shared by 201/203/206 so they can't each hand-roll a
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
    """Per-gene-pair max-containment table for one alphabet/ksize combo, aggregated from the
    raw genome-wide `human_vs_mouse.{encoding}.k{ksize}.results.*` file -- the single expensive
    step (multi-GB-to-100+GB raw scan) that every downstream per-combo analysis in this project
    needs (notebook 200's F1/AUC sweep, 201/203/206's per-combo comparisons, 202's per-gene pLDDT
    correctness table). Cached to `{data_dir}/pair_cache/{encoding}_k{ksize}.parquet` on first call
    so notebooks 200/201/202/203/206 stop each re-scanning the same raw file independently --
    everything past this point (RBH calling, F1, per-gene tables) is cheap in-memory work on a
    table with one row per observed (human_gene, mouse_gene) candidate, not per matched region.

    Columns: human_gene, mouse_gene, containment (max over that gene pair's matched regions).
    """
    cache_path = PAIR_CACHE_DIR / f"{encoding}_k{ksize}.parquet"
    if cache_path.exists():
        return pl.read_parquet(cache_path)

    lf = (
        scan_genome_wide_results(encoding, ksize, data_dir, columns=["query_name", "target_name", "containment"])
        .with_columns([
            pl.col("query_name").str.split("|").list.get(6).str.to_uppercase().alias("human_gene"),
            pl.col("target_name").str.split("|").list.get(6).str.to_uppercase().alias("mouse_gene"),
        ])
        .group_by(["human_gene", "mouse_gene"])
        .agg(pl.col("containment").max().alias("containment"))
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
    """Aggregate to per-gene-pair max containment (via the shared `load_pair_table` cache) and
    call reciprocal-best-hit (RBH) pairs: each human gene's top mouse hit must agree with that
    mouse gene's top human hit.

    Returns (rbh_set, human_genes_in_scope) — scope is every human gene the file covers, for
    re-scoring OrthoFinder against the same matched gene universe (notebook 200, lesson 2).
    """
    pair = load_pair_table(encoding, ksize, data_dir)
    scope = set(pair["human_gene"].unique().to_list())

    # secondary sort key makes tie-breaking (equal max containment) deterministic
    best_h2m = (pair.sort(["containment", "mouse_gene"], descending=[True, False])
                .group_by("human_gene").agg(pl.col("mouse_gene").first().alias("best_mouse")))
    best_m2h = (pair.sort(["containment", "human_gene"], descending=[True, False])
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

    Returns one row per human gene in scope: human_gene, mouse_gene (RBH pick), containment,
    is_rbh, has_mgi_truth, is_correct.
    """
    pair = load_pair_table(encoding, ksize, data_dir)

    best_h2m = (pair.sort(["containment", "mouse_gene"], descending=[True, False])
                .group_by("human_gene")
                .agg([pl.col("mouse_gene").first().alias("rbh_mouse_gene"),
                      pl.col("containment").first().alias("containment")]))
    best_m2h = (pair.sort(["containment", "human_gene"], descending=[True, False])
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
        .select(["human_gene", "rbh_mouse_gene", "containment", "is_rbh", "has_mgi_truth", "is_correct"])
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


def fetch_biomart_perc_id(ensg_ids: list[str], batch_size: int = 300) -> dict[str, float]:
    """Batch-query BioMart `mmusculus_homolog_perc_id[_r1]` for human ENSG IDs.

    Returns dict ENSG -> symmetric percent identity (mean of the two BioMart-reported
    directions). IDs with no mouse homolog or a failed batch are simply absent.
    """
    out: dict[str, float] = {}
    for i in range(0, len(ensg_ids), batch_size):
        batch = ensg_ids[i : i + batch_size]
        query = (
            '<?xml version="1.0" encoding="UTF-8"?><!DOCTYPE Query>'
            '<Query virtualSchemaName="default" formatter="TSV" header="0" uniqueRows="1" '
            'count="" datasetConfigVersion="0.6">'
            '<Dataset name="hsapiens_gene_ensembl" interface="default">'
            f'<Filter name="ensembl_gene_id" value="{",".join(batch)}"/>'
            '<Attribute name="ensembl_gene_id"/>'
            '<Attribute name="mmusculus_homolog_perc_id"/>'
            '<Attribute name="mmusculus_homolog_perc_id_r1"/>'
            "</Dataset></Query>"
        )
        try:
            resp = requests.get(_BIOMART_URL, params={"query": query}, timeout=60)
            resp.raise_for_status()
            for line in resp.text.strip().splitlines():
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                ensg, pid, pid_r1 = parts
                try:
                    vals = [float(v) for v in (pid, pid_r1) if v not in ("", "NA")]
                except ValueError:
                    continue
                if vals:
                    out[ensg] = float(np.mean(vals))
        except Exception as exc:
            print(f"  WARN BioMart batch {i // batch_size + 1}: {exc}")
        time.sleep(0.2)
    return out


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
