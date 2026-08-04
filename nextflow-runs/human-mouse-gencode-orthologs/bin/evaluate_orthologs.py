#!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
"""Evaluate kmerseek ortholog detection accuracy for one k-size.

Usage:
    evaluate_orthologs.py <ksize> <results_zst> <ortholog_pairs>

Memory strategy to handle very large result files (k=18 = 23 GB compressed):
  - sink_csv for full eval TSV: streaming write, near-zero RAM
  - Histograms via lazy cut+group_by: O(bins) RAM, not O(rows)
  - Metric stats via lazy group_by aggregation: O(1) RAM
  - MHT with Categorical gene names: ~4 bytes/row vs ~15 chars/row
  - ROC curve: only 2 columns collected
"""

import sys
import os
import subprocess
import json
import math
import gc

import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


METRIC_COLS = [
    'n_intersecting_hashes', 'jaccard', 'containment', 'query_tfidf',
    'mean_matched_kmer_freq', 'sum_matched_kmer_freq',
    'expected_shared_kmers', 'enrichment', 'poisson_pvalue',
]

# Encoding name used in every output filename. Set from argv[5] in main().
# Nextflow's evaluateOrthologs output block expects ortholog_evaluation.<encoding>.k<ksize>.*,
# so this MUST match the encoding the process was launched with (hp, protein, hp-thomas-dill, …).
ENCODING = 'hp'


def zst_compress(path: str) -> None:
    subprocess.run(['zstd', '-T2', '--rm', '-f', '-q', path], check=True)


def write_empty_outputs(ksize: int, reason: str = 'empty file') -> None:
    """Write placeholder outputs when the search file has no results."""
    print(f'k={ksize}: no search results ({reason}) — writing empty outputs')
    for suffix in ['tsv', 'roc_data.tsv', 'mht.csv']:
        p = f'ortholog_evaluation.{ENCODING}.k{ksize}.{suffix}'
        open(p, 'w').close()
        zst_compress(p)
    open(f'ortholog_evaluation.{ENCODING}.k{ksize}.summary.txt', 'w').write(
        f'K-size: {ksize}\nEncoding: {ENCODING}\n\nNo search results ({reason}).\n'
    )
    fig, ax = plt.subplots()
    ax.set_title(f'No results for k={ksize}')
    plt.savefig(f'metrics_no_results.{ENCODING}.k{ksize}.png', dpi=72)
    plt.close()


def harmonic_number(n: int) -> float:
    """H_n = sum(1/k, k=1..n). Uses log approximation for large n."""
    if n <= 500_000:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))
    # Euler-Mascheroni: H_n ≈ ln(n) + γ + 1/(2n)
    return math.log(n) + 0.5772156649015329 + 1.0 / (2 * n)


def scan_results(results_path: str) -> pl.LazyFrame:
    """Format-aware lazy scan — accepts either .results.csv.zst or a
    .results.parquet (post-convert_results_to_parquet.py)."""
    return pl.scan_parquet(results_path) if results_path.endswith('.parquet') else pl.scan_csv(results_path)


def build_lazy_base(results_zst: str, ortho: pl.DataFrame) -> pl.LazyFrame:
    """Lazy query: parse gene names from FASTA headers, join ortholog labels."""
    return (
        scan_results(results_zst)
        .with_columns([
            pl.col('query_name').str.split('|').list.get(6).alias('human_gene'),
            pl.col('target_name').str.split('|').list.get(6).alias('mouse_gene'),
        ])
        .with_columns(pl.col('human_gene').str.to_uppercase())
        .join(
            ortho.lazy().select(['human_gene', 'mouse_gene', 'is_ortholog']),
            on=['human_gene', 'mouse_gene'],
            how='left',
        )
        .with_columns(pl.col('is_ortholog').fill_null(False))
    )


def compute_histograms(
    lazy_base: pl.LazyFrame,
    metrics: list,
    ksize: int,
    n_orth: int,
    n_non: int,
    streaming: bool = True,
) -> None:
    """Histograms via cut+group_by — O(bins) RAM, not O(rows).

    Uses two scans: one for percentile bin edges (all metrics together),
    one per metric for binned counts.
    streaming=False uses polars hash-join collect instead of the streaming engine,
    which is more reliable when lazy_base contains a join (streaming engine +
    join can silently fall back to a full materialization and OOM).
    """
    if not metrics:
        return

    n_bins = 60
    collect_kw = {'engine': 'streaming'} if streaming else {}

    # Single scan for all percentile edges
    percs_df = lazy_base.select([
        expr
        for m in metrics
        for expr in [
            pl.col(m).quantile(0.005, interpolation='linear').alias(f'{m}_lo'),
            pl.col(m).quantile(0.995, interpolation='linear').alias(f'{m}_hi'),
        ]
    ]).collect(**collect_kw)

    bin_labels = [str(i) for i in range(n_bins)]

    for metric in metrics:
        lo = percs_df[f'{metric}_lo'].item()
        hi = percs_df[f'{metric}_hi'].item()
        if lo is None or hi is None or lo >= hi:
            continue

        breaks = list(np.linspace(lo, hi, n_bins + 1)[1:-1])  # n_bins-1 interior breaks

        hist = (
            lazy_base
            .select(['is_ortholog', metric])
            .filter(pl.col(metric).is_not_null())
            .with_columns(
                pl.col(metric)
                .cut(breaks=breaks, labels=bin_labels, left_closed=True)
                .alias('bin_idx')
            )
            .group_by(['is_ortholog', 'bin_idx'])
            .agg(pl.len().alias('count'))
            .collect(**collect_kw)
        )

        def get_counts(is_orth: bool) -> np.ndarray:
            counts = np.zeros(n_bins)
            for row in hist.filter(pl.col('is_ortholog') == is_orth).iter_rows(named=True):
                try:
                    counts[int(row['bin_idx'])] = row['count']
                except (ValueError, IndexError):
                    pass
            return counts

        ov = get_counts(True)
        nv = get_counts(False)
        bin_width   = (hi - lo) / n_bins
        bin_centers = np.linspace(lo, hi, n_bins * 2 + 1)[1::2]

        fig, ax = plt.subplots(figsize=(8, 5))
        if nv.sum() > 0:
            ax.bar(bin_centers, nv / nv.sum() / bin_width, width=bin_width,
                   alpha=0.6, label=f'Non-ortholog (n={n_non:,})', color='steelblue')
        if ov.sum() > 0:
            ax.bar(bin_centers, ov / ov.sum() / bin_width, width=bin_width,
                   alpha=0.6, label=f'Ortholog (n={n_orth:,})', color='tomato')
        ax.set_xlabel(metric)
        ax.set_ylabel('Density')
        ax.set_title(f'{metric}  —  HP k={ksize}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(f'metrics_{metric}.{ENCODING}.k{ksize}.png', dpi=150)
        plt.close()


def compute_mht(
    lazy_base: pl.LazyFrame,
    n: int,
    ksize: int,
    schema,
    alpha: float = 0.05,
) -> tuple:
    """BH, BY, Bonferroni, two-stage BH via polars expressions — no numpy argsort.

    Uses Categorical encoding for gene name columns to cut RAM ~4x vs plain strings.
    Intermediate columns are dropped early to keep peak memory low.
    Returns (summary_dict, mht_path, n_orth_in_data) where n_orth_in_data is the
    count of ortholog rows that passed the alpha filter (used by callers that don't
    have a prior streaming count, e.g. the pre-filtered pathway).
    """
    mht_path = f'ortholog_evaluation.{ENCODING}.k{ksize}.mht.csv'

    if 'poisson_pvalue' not in schema:
        print(f'k={ksize}: poisson_pvalue not in columns — skipping MHT')
        open(mht_path, 'w').close()
        return {}, mht_path, 0

    c_n = harmonic_number(n)
    # Only load what's needed for BH corrections — omit containment/jaccard to
    # cut peak RAM by ~16 bytes/row (two float64 columns) before the sort.
    # Categorical gene names: ~4 bytes/row instead of ~15 chars/row.
    # Pre-filter to p < alpha: no row with p >= alpha can be rejected under any
    # correction (BH adjusted = p*n/rank >= p since rank <= n). Ranks within
    # this filtered set equal ranks in the full ascending sort because all
    # p >= alpha rows sort after them. Use n (n_total) in formulas, not filtered count.
    mht_df = (
        lazy_base
        .select(['human_gene', 'mouse_gene', 'is_ortholog', 'poisson_pvalue'])
        .with_columns([
            pl.col('human_gene').cast(pl.Categorical),
            pl.col('mouse_gene').cast(pl.Categorical),
            pl.col('poisson_pvalue').fill_null(1.0),
        ])
        .filter(pl.col('poisson_pvalue') < alpha)
        .sort('poisson_pvalue')
        .with_row_index('__rank__', offset=1)
        .with_columns([
            (pl.col('poisson_pvalue') * n).clip(upper_bound=1.0).alias('bonferroni'),
            (pl.col('poisson_pvalue') * n / pl.col('__rank__'))
                .clip(upper_bound=1.0).alias('__bh_pre__'),
            (pl.col('poisson_pvalue') * n * c_n / pl.col('__rank__'))
                .clip(upper_bound=1.0).alias('__by_pre__'),
        ])
        .with_columns([
            pl.col('__bh_pre__').reverse().cum_min().reverse().alias('bh'),
            pl.col('__by_pre__').reverse().cum_min().reverse().alias('by'),
        ])
        .drop(['__bh_pre__', '__by_pre__'])  # free intermediate RAM before collect
        .collect()
    )

    # Two-stage BH: re-estimate m0 from BH rejection count, then rerun
    m0 = max(n - int((mht_df['bh'] <= alpha).sum()), 1)
    mht_df = (
        mht_df.lazy()
        .with_columns(
            (pl.col('poisson_pvalue') * m0 / pl.col('__rank__'))
            .clip(upper_bound=1.0)
            .alias('__2sbh_pre__')
        )
        .with_columns(
            pl.col('__2sbh_pre__').reverse().cum_min().reverse().alias('two_stage_bh')
        )
        .drop(['__rank__', '__2sbh_pre__'])
        .collect()
    )

    mht_df.write_csv(mht_path)

    is_orth      = mht_df['is_ortholog'].to_numpy()
    n_orth_alpha = int(is_orth.sum())  # orthologs that passed the alpha pre-filter

    def mht_stats(col: str) -> dict:
        adj = mht_df[col].to_numpy()
        rej = adj <= alpha
        tp  = int((rej & is_orth).sum())
        tot = int(rej.sum())
        return {
            'rejected':  tot,
            'TP':        tp,
            'precision': round(tp / tot         if tot         else 0.0, 4),
            'recall':    round(tp / n_orth_alpha if n_orth_alpha else 0.0, 4),
        }

    summary = {
        'bonferroni':   mht_stats('bonferroni'),
        'bh':           mht_stats('bh'),
        'by':           mht_stats('by'),
        'two_stage_bh': mht_stats('two_stage_bh'),
    }

    # Explicitly free the sorted MHT frame before the ROC sort to avoid
    # holding two large sorted datasets in memory simultaneously.
    del mht_df, is_orth
    gc.collect()

    return summary, mht_path, n_orth_alpha


def compute_metric_stats(
    lazy_base: pl.LazyFrame, metrics: list, n_orth: int, n_non: int,
    streaming: bool = True,
) -> dict:
    """Mean/median per metric × ortholog status via lazy group_by — O(1) RAM."""
    if not metrics:
        return {}

    collect_kw = {'engine': 'streaming'} if streaming else {}
    stats_df = (
        lazy_base
        .select(['is_ortholog'] + metrics)
        .group_by('is_ortholog')
        .agg(
            [pl.col(m).mean().alias(f'{m}_mean')   for m in metrics] +
            [pl.col(m).median().alias(f'{m}_median') for m in metrics]
        )
        .collect(**collect_kw)
    )

    def get_row(is_orth: bool) -> dict:
        rows = stats_df.filter(pl.col('is_ortholog') == is_orth)
        return rows.row(0, named=True) if len(rows) else {}

    orth_row = get_row(True)
    nort_row = get_row(False)

    return {
        m: {
            'ortholog':     {
                'mean':   float(orth_row.get(f'{m}_mean',   0.0) or 0.0),
                'median': float(orth_row.get(f'{m}_median', 0.0) or 0.0),
                'n': n_orth,
            },
            'non_ortholog': {
                'mean':   float(nort_row.get(f'{m}_mean',   0.0) or 0.0),
                'median': float(nort_row.get(f'{m}_median', 0.0) or 0.0),
                'n': n_non,
            },
        }
        for m in metrics
    }


def compute_roc_data(
    lazy_base: pl.LazyFrame,
    n_total: int,
    n_orth: int,
    n_non: int,
    ksize: int,
    schema,
) -> str:
    """ROC curve sampled to ≤10k points. Only collects 2 columns."""
    roc_path = f'ortholog_evaluation.{ENCODING}.k{ksize}.roc_data.tsv'

    if 'poisson_pvalue' in schema:
        roc_sort_col, descending = 'poisson_pvalue', False
    elif 'containment' in schema:
        roc_sort_col, descending = 'containment', True
    else:
        open(roc_path, 'w').close()
        return roc_path

    step  = max(1, n_total // 10_000)
    n_pos = max(n_orth, 1)
    n_neg = max(n_non, 1)

    (
        lazy_base
        .select(['is_ortholog', roc_sort_col])
        .sort(roc_sort_col, descending=descending)
        .with_columns([
            pl.col('is_ortholog').cast(pl.Int32).cum_sum().alias('cum_tp'),
            (~pl.col('is_ortholog')).cast(pl.Int32).cum_sum().alias('cum_fp'),
        ])
        .with_row_index('__row__')
        .filter(pl.col('__row__') % step == 0)
        .select([
            pl.col(roc_sort_col).alias('threshold'),
            (pl.col('cum_tp') / n_pos).alias('TPR'),
            (pl.col('cum_fp') / n_neg).alias('FPR'),
        ])
        .collect()
        .write_csv(roc_path, separator='\t')
    )
    return roc_path


def main() -> None:
    ksize          = int(sys.argv[1])
    results_zst    = sys.argv[2]
    ortholog_pairs = sys.argv[3]
    # Optional: bash pre-filter passes n_total_full so MHT denominators are correct
    # even though results_zst only contains rows with poisson_pvalue < 0.05.
    n_total_full = int(sys.argv[4]) if len(sys.argv) > 4 else None
    prefiltered  = n_total_full is not None
    # argv[5] is the encoding (hp, protein, dayhoff, hp-thomas-dill, …); drives all output
    # filenames so they match what Nextflow's evaluateOrthologs output block expects.
    global ENCODING
    if len(sys.argv) > 5:
        ENCODING = sys.argv[5]

    # ── Handle empty search results (k-mer space saturation) ──────────────────
    # Two cases: 0-byte file, or non-zero zst that decompresses to an empty CSV
    if os.path.getsize(results_zst) == 0:
        write_empty_outputs(ksize, reason='0-byte file')
        sys.exit(0)

    # ── Load ortholog ground truth ─────────────────────────────────────────────
    ortho = (
        pl.read_csv(ortholog_pairs, separator='\t')
        .with_columns(
            pl.col('human_gene').str.to_uppercase(),
            pl.lit(True).alias('is_ortholog'),
        )
    )

    lazy_base = build_lazy_base(results_zst, ortho)
    try:
        schema = lazy_base.collect_schema()
    except pl.exceptions.NoDataError:
        write_empty_outputs(ksize, reason='empty CSV after decompression')
        sys.exit(0)
    metrics = [c for c in METRIC_COLS if c in schema]

    # ── Row counts ─────────────────────────────────────────────────────────────
    # For the pre-filtered pathway, polars' streaming engine can OOM on the
    # join even for a filtered file (streaming join buffers more than hash join).
    # Instead: skip the streaming count, run MHT first (uses a regular hash join
    # collect), and get n_orth from the collected MHT data.
    # For the non-prefiltered pathway, use the original streaming count.
    if prefiltered:
        n_total = n_total_full
        n_orth  = None  # filled in after compute_mht below
    else:
        counts  = lazy_base.select([
            pl.len().alias('n_total'),
            pl.col('is_ortholog').sum().alias('n_orth'),
        ]).collect(engine='streaming')
        n_total = counts['n_total'].item()
        n_orth  = counts['n_orth'].item()

    # ── Full evaluation TSV ────────────────────────────────────────────────────
    eval_path = f'ortholog_evaluation.{ENCODING}.k{ksize}.tsv'
    if prefiltered:
        # For pre-filtered data the join prevents true streaming in sink_csv and
        # would try to materialize the full file (OOM). Sink without join instead;
        # is_ortholog is absent but all search metrics are present.
        (
            scan_results(results_zst)
            .with_columns([
                pl.col('query_name').str.split('|').list.get(6).str.to_uppercase().alias('human_gene'),
                pl.col('target_name').str.split('|').list.get(6).alias('mouse_gene'),
            ])
            .sink_csv(eval_path, separator='\t')
        )
    else:
        lazy_base.sink_csv(eval_path, separator='\t')
    zst_compress(eval_path)

    # ── MHT corrections ────────────────────────────────────────────────────────
    # Run before histograms so n_orth is known for the pre-filtered pathway.
    # compute_mht uses a regular (non-streaming) hash join collect — polars
    # builds a hash table on the small ortho side (~20K rows) and scans
    # through the left side in chunks, so RAM ≈ O(data_cols × n_rows) not O(full_row).
    mht_summary, mht_path, n_orth_alpha = compute_mht(lazy_base, n_total, ksize, schema)
    zst_compress(mht_path)

    if prefiltered:
        n_orth = n_orth_alpha  # orthologs with p < 0.001 (bash cutoff)
    n_non = n_total - n_orth

    prefilter_note = ' [pre-filtered p<0.001; n_orth/recall relative to p<0.001 hits]' if prefiltered else ''
    print(f'k={ksize}: {n_total:,} total tests | {n_orth:,} sig. orthologs | {n_non:,} other pairs{prefilter_note}')

    # ── Histograms ─────────────────────────────────────────────────────────────
    # streaming=False for prefiltered: polars streaming engine + join can silently
    # materialise all rows; hash-join collect keeps peak RAM = O(select_cols × n_rows).
    compute_histograms(lazy_base, metrics, ksize, n_orth, n_non, streaming=not prefiltered)

    # ── Metric statistics ──────────────────────────────────────────────────────
    metric_stats = compute_metric_stats(lazy_base, metrics, n_orth, n_non, streaming=not prefiltered)

    # ── ROC data (2-column collect + sample) ──────────────────────────────────
    roc_path = compute_roc_data(lazy_base, n_total, n_orth, n_non, ksize, schema)
    zst_compress(roc_path)

    # ── Summary text + JSON ───────────────────────────────────────────────────
    summary_json = {
        'ksize': ksize, 'encoding': ENCODING,
        'total_hits': n_total, 'n_ortholog': n_orth, 'n_non_ortholog': n_non,
        'prefiltered_pvalue': prefiltered,
        'mht': mht_summary, 'metric_stats': metric_stats,
    }

    with open(f'ortholog_evaluation.{ENCODING}.k{ksize}.summary.txt', 'w') as f:
        f.write(f'K-size: {ksize}\nEncoding: {ENCODING}\n')
        if prefiltered:
            f.write('Note: results pre-filtered to poisson_pvalue < 0.001 to avoid OOM.\n'
                    '      n_total is from full file; n_ortholog/recall are among p<0.001 hits.\n'
                    '      BH cutoff safe: with n~364M, BH threshold <<1e-3 for any realistic M.\n')
        f.write(f'\nTotal tests: {n_total:,}  |  Sig. orthologs: {n_orth:,}  |  Other pairs: {n_non:,}\n\n')

        f.write('=== MHT Rejections (alpha=0.05) ===\n')
        for method, s in mht_summary.items():
            f.write(f'  {method:15s}: {s["rejected"]:>10,} rejected  '
                    f'TP={s["TP"]:>8,}  prec={s["precision"]:.4f}  rec={s["recall"]:.4f}\n')

        f.write('\n=== Metric Means (orthologs vs non-orthologs) ===\n')
        for m, st in metric_stats.items():
            ov, nv = st['ortholog'], st['non_ortholog']
            f.write(f'  {m}:\n')
            f.write(f'    ortholog     mean={ov["mean"]:.4f}  median={ov["median"]:.4f}  n={ov["n"]:,}\n')
            f.write(f'    non-ortholog mean={nv["mean"]:.4f}  median={nv["median"]:.4f}  n={nv["n"]:,}\n')

        f.write('\n=== JSON SUMMARY ===\n')
        f.write(json.dumps(summary_json, indent=2) + '\n')


if __name__ == '__main__':
    main()
