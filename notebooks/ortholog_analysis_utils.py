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
- UniProt ID mapping (Ensembl Protein → UniProtKB)
- UniProt annotation fetching and parsing
- MobiDB intrinsic-disorder fetching
- DataFrame annotation helpers
- Summary-statistics utilities
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import requests

# ---------------------------------------------------------------------------
# Paths (defaults – callers can override)
# ---------------------------------------------------------------------------

DATA_DIR = Path("/Users/olga/data/gencode/results-human-mouse-orthologs")
OF_DIR = Path("/Users/olga/data/gencode/data-for-orthofinder/OrthoFinder/Results_Mar03")

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
    "poisson_pvalue", "enrichment", "query_tfidf",
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
    """
    df = pl.read_csv(
        str(kmerseek_tsv),
        separator="\t",
        columns=usecols,
        ignore_errors=True,
        null_values=["", "NA", "NaN"],
    )

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
    poisson_pvalue, bonf_pvalue, bh_pvalue
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
        (-pl.col('bonf_pvalue').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
        ).alias('score_bonf_neglogp'),
        (-pl.col('bonf_pvalue').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
         * pl.col('containment')).alias('score_bonf_neglogp_cont'),
        (-pl.col('bh_pvalue').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
        ).alias('score_bh_neglogq'),
        (-pl.col('bh_pvalue').clip(lower_bound=1e-300).log(base=10).fill_nan(0.0)
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
