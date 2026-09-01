"""
Shared analysis code for notebook 066: KmerSeek HP SCOPe40 all-metrics AUC sweep.

Imported by two thin notebooks that differ only in the exclude_gray_zone flag:
  066_scope_all_metrics_excl_gz.ipynb   — Foldseek/TEA paper convention (default)
  066_scope_all_metrics_all_fps.ipynb   — inclusive FPs (legacy)
"""

import gc
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.integrate import trapezoid

_here = next(
    p for p in [Path.cwd(), Path.cwd() / 'notebooks']
    if (p / 'scope_kmerseek_utils.py').exists()
)
sys.path.insert(0, str(_here))

from scope_kmerseek_utils import (
    EVAL_USECOLS, SCOP_LEVELS,
    BENCH_DIR, TEA_DIR,
    SCOPE_SCORE_COLS, SCOPE_CLASS_DESCRIPTIONS,
    load_eval, load_baselines, add_composite_scores_scope, recompute_bonferroni,
    eval_tsv_to_rocx, rocx_restrict, sensitivity_stats, compute_auc,
    plot_sensitivity_from_rocx,
    precision_recall_from_eval, pr_operating_point_from_rocx,
)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 120

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KMIN, KMAX = 15, 45
KSIZES = list(range(KMIN, KMAX + 1))

BONF_THRESHOLD  = 0.05
SWEEP_NQUERY_MIN = 150

K_COMPARE      = [24, 27]
COLORS_COMPARE = ['royalblue', 'tomato']
LABEL_COMPARE  = {k: f'Bonferroni<{BONF_THRESHOLD} | Jaccard' for k in K_COMPARE}
PVAL_COMPARE   = {k: 1e-3 for k in K_COMPARE}

# score_col, ascending, label
METRICS = [m for m in SCOPE_SCORE_COLS if m[0] not in {'mean_matched_kmer_freq'}]

_LABEL_TO_COL = {
    'Jaccard':                   ('jaccard',             False),
    'Containment':               ('containment',         False),
    'BH (FDR)':                  ('bh',                  True),
    'TF-IDF':                    ('query_tfidf',         False),
    '−log10(BH q)×Containment':  ('neg_log10_bh_x_cont', False),
}

# Section-9 threshold sweep parameters
N_SCOPE40        = 11_211
NQUERY_MIN       = 150
TFIDF_MINS       = [None, 100, 500, 1_000, 5_000, 10_000]
BH_MAXES         = [None, 0.05, 1e-3, 1e-5]
PVAL_MAXES       = [None, 0.05, 1e-3, 1e-5]

THRESH_METRICS = [
    ('jaccard',             False, 'Jaccard'),
    ('containment',        False, 'Containment'),
    ('bh',                  True, 'BH (FDR)'),
    ('query_tfidf',        False, 'TF-IDF'),
    ('neg_log10_bh_x_cont',False, '−log10(BH q)×Containment'),
]
THRESH_KSIZES = list(range(15, 46))


def _cache_suffix(exclude_gray_zone: bool) -> str:
    return 'excl_gz' if exclude_gray_zone else 'all_fps'


def _thresh_label(tfidf_min, bh_max, pval_max):
    parts = []
    if tfidf_min is not None:
        parts.append(f'TF-IDF>{tfidf_min:g}')
    if bh_max is not None:
        parts.append(f'BH<{bh_max:g}')
    if pval_max is not None:
        parts.append(f'pval<{pval_max:g}')
    return ' & '.join(parts) if parts else 'No filter'


def _build_filter_combos():
    combos = []
    for tfidf_min in TFIDF_MINS:
        for bh_max in BH_MAXES:
            for pval_max in PVAL_MAXES:
                if bh_max is not None and pval_max is not None:
                    continue
                combos.append(dict(
                    tfidf_min=tfidf_min, bh_max=bh_max, pval_max=pval_max,
                    filter_label=_thresh_label(tfidf_min, bh_max, pval_max),
                ))
    return combos


# ---------------------------------------------------------------------------
# Section 2: Load baselines
# ---------------------------------------------------------------------------

def load_scope_baselines():
    """Return (foldseek, tea_all, ref_queries)."""
    foldseek, tea_all = load_baselines(TEA_DIR)
    ref_queries = set(foldseek['NAME'].to_list())
    print(f'FoldSeek: {len(foldseek):,} queries')
    print(f'TEA:      {len(tea_all):,} queries')
    print(f'Benchmarkable query set: {len(ref_queries):,} queries')
    for name, df in [('FoldSeek', foldseek), ('TEA', tea_all)]:
        _, _, sfam = sensitivity_stats(df, 'SFAM')
        _, _, fam  = sensitivity_stats(df, 'FAM')
        _, _, fold = sensitivity_stats(df, 'FOLD')
        print(f'{name}: FAM={fam:.4f}  SFAM={sfam:.4f}  FOLD={fold:.4f}')
    return foldseek, tea_all, ref_queries


# ---------------------------------------------------------------------------
# Section 3: Full AUC sweep (cached)
# ---------------------------------------------------------------------------

def run_auc_sweep(exclude_gray_zone: bool, ref_queries: set,
                  force_recompute: bool = False) -> pl.DataFrame:
    """Run the all-metrics × all-k AUC sweep; cache by gray-zone setting."""
    suffix = _cache_suffix(exclude_gray_zone)
    cache = BENCH_DIR / f'scope_all_metrics_auc_sweep_{suffix}.parquet'
    if not force_recompute and cache.exists():
        auc_df = pl.read_parquet(cache)
        print(f'Loaded cached sweep ({suffix}): {len(auc_df):,} rows')
        return auc_df

    auc_rows = []
    for k in KSIZES:
        print(f'k={k}:', end=' ', flush=True)
        try:
            eval_df = load_eval(k, columns=EVAL_USECOLS)
            eval_df = add_composite_scores_scope(eval_df)
        except FileNotFoundError as e:
            print(f'SKIP ({e})')
            continue

        for col, ascending, label in METRICS:
            if col not in eval_df.columns:
                continue
            rocx   = eval_tsv_to_rocx(eval_df, score_col=col, ascending=ascending,
                                       exclude_gray_zone=exclude_gray_zone)
            rocx_r = rocx.filter(pl.col('NAME').is_in(ref_queries))

            def _aucs(r):
                return (
                    sensitivity_stats(r, 'CLASS')[2],
                    sensitivity_stats(r, 'SFAM')[2],
                    sensitivity_stats(r, 'FAM')[2],
                    sensitivity_stats(r, 'FOLD')[2],
                )

            c_all, s_all, f_all, fo_all = _aucs(rocx)
            c_ref, s_ref, f_ref, fo_ref = _aucs(rocx_r)
            auc_rows.append({
                'ksize': k, 'score_col': col, 'label': label,
                'auc_class_all': round(c_all, 5), 'auc_sfam_all': round(s_all, 5),
                'auc_fam_all':   round(f_all, 5), 'auc_fold_all': round(fo_all, 5),
                'n_queries_all': len(rocx),
                'auc_class': round(c_ref, 5), 'auc_sfam': round(s_ref, 5),
                'auc_fam':   round(f_ref, 5), 'auc_fold': round(fo_ref, 5),
                'n_queries': len(rocx_r),
            })
            print('.', end='', flush=True)

        del eval_df; gc.collect()
        print()

    auc_df = pl.DataFrame(auc_rows)
    auc_df.write_parquet(cache)
    print(f'\nSwept {len(auc_df):,} combinations — saved to {cache.name}')
    return auc_df


# ---------------------------------------------------------------------------
# Section 4: AUC heatmap
# ---------------------------------------------------------------------------

def plot_auc_heatmap(auc_df: pl.DataFrame, foldseek, tea_all,
                     fig_prefix: str) -> None:
    _, _, fs_sfam = sensitivity_stats(foldseek, 'SFAM')
    _, _, tea_sfam = sensitivity_stats(tea_all,  'SFAM')
    _, _, fs_fam  = sensitivity_stats(foldseek, 'FAM')
    _, _, tea_fam = sensitivity_stats(tea_all,  'FAM')

    def _build_matrix(auc_col):
        pivot = auc_df.pivot(values=auc_col, index='label', on='ksize').sort('label')
        labels = pivot['label'].to_list()
        ksizes = sorted([int(c) for c in pivot.columns if c != 'label'])
        mat = np.array([
            [pivot.filter(pl.col('label') == lbl)[str(k)][0]
             if str(k) in pivot.columns else np.nan
             for k in ksizes]
            for lbl in labels
        ])
        return labels, ksizes, mat

    labels, ksizes_present, sfam_mat = _build_matrix('auc_sfam')
    _,      _,              fam_mat  = _build_matrix('auc_fam')

    fig, axes = plt.subplots(1, 2, figsize=(32, 8))
    for ax, matrix, level_label, fs_auc, tea_auc, col_label in [
        (axes[0], sfam_mat, 'Superfamily', fs_sfam, tea_sfam, 'Superfamily AUC'),
        (axes[1], fam_mat,  'Family',      fs_fam,  tea_fam,  'Family AUC'),
    ]:
        vmax = max(fs_auc, tea_auc, np.nanmax(matrix))
        im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=vmax)
        plt.colorbar(im, ax=ax, label=col_label)
        ax.set_xticks(range(len(ksizes_present)))
        ax.set_xticklabels([str(k) for k in ksizes_present], fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('K-size (HP encoding)', fontsize=12)
        ax.set_title(
            f'{level_label} AUC: All Metrics × K-size (k=15–45)\n'
            f'FoldSeek: {fs_auc:.3f}  |  TEA: {tea_auc:.3f}',
            fontsize=13, fontweight='bold')
        for i, lbl in enumerate(labels):
            for j, k in enumerate(ksizes_present):
                v = matrix[i, j]
                if not np.isnan(v) and v > fs_auc:
                    ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                               fill=False, edgecolor='gold', lw=2))
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.3f}', ha='center', va='center', fontsize=5.5,
                            color='black' if v < 0.6 else 'white')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_all_metrics_auc_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f'Gold boxes = beats FoldSeek (SFAM: {fs_sfam:.3f}, FAM: {fs_fam:.3f})')


# ---------------------------------------------------------------------------
# Section 5: Best metric
# ---------------------------------------------------------------------------

def get_best_metric(auc_df: pl.DataFrame, foldseek, tea_all) -> dict:
    _, _, fs_sfam  = sensitivity_stats(foldseek, 'SFAM')
    _, _, tea_sfam = sensitivity_stats(tea_all,  'SFAM')

    jac_df = auc_df.filter(pl.col('label') == 'Jaccard')
    filtered = jac_df.filter(pl.col('n_queries') >= SWEEP_NQUERY_MIN)

    best_fam_row = filtered.sort('auc_fam', descending=True).row(0, named=True)
    best_sfam_row = filtered.sort('auc_sfam', descending=True).row(0, named=True)

    print('=== BEST COMBINATION (Jaccard only) ===')
    print(f'\n--- Best by Family AUC (k={best_fam_row["ksize"]}) ---')
    print(f'  n_queries:       {best_fam_row["n_queries"]:,}')
    print(f'  Family AUC:      {best_fam_row["auc_fam"]:.4f}')
    print(f'  Superfamily AUC: {best_fam_row["auc_sfam"]:.4f}  '
          f'(FoldSeek: {fs_sfam:.4f}, TEA: {tea_sfam:.4f})')
    print(f'  Fold AUC:        {best_fam_row["auc_fold"]:.4f}')

    print(f'\n--- Best by Superfamily AUC (k={best_sfam_row["ksize"]}) ---')
    print(f'  n_queries:       {best_sfam_row["n_queries"]:,}')
    print(f'  Family AUC:      {best_sfam_row["auc_fam"]:.4f}')
    print(f'  Superfamily AUC: {best_sfam_row["auc_sfam"]:.4f}  '
          f'(FoldSeek: {fs_sfam:.4f}, TEA: {tea_sfam:.4f})')
    print(f'  Fold AUC:        {best_sfam_row["auc_fold"]:.4f}')

    return best_fam_row


# ---------------------------------------------------------------------------
# Section 6: Best-metric sensitivity curve
# ---------------------------------------------------------------------------

def plot_best_metric_sensitivity(foldseek, tea_all, ref_queries: set,
                                  exclude_gray_zone: bool, fig_prefix: str,
                                  best_covered=None):
    """Load best metric (from section 9 if available, else k=27/Jaccard) and plot."""
    if best_covered is not None:
        best_k   = best_covered['ksize']
        best_col, best_asc = _LABEL_TO_COL[best_covered['score_label']]
        best_label = f'Bonferroni<{BONF_THRESHOLD} | {best_covered["score_label"]}'
    else:
        best_k, best_col, best_asc = 27, 'jaccard', False
        best_label = f'Bonferroni<{BONF_THRESHOLD} | Jaccard'

    print(f'Best k={best_k}  metric={best_label}')
    best_eval = load_eval(best_k)
    best_eval = add_composite_scores_scope(best_eval)
    best_eval = recompute_bonferroni(best_eval)
    best_eval = best_eval.filter(pl.col('bonferroni_correct') < BONF_THRESHOLD)
    best_rocx = eval_tsv_to_rocx(best_eval, score_col=best_col, ascending=best_asc,
                                  exclude_gray_zone=exclude_gray_zone)
    n_best_all = best_rocx['NAME'].n_unique()
    _, _, auc_sfam_best = sensitivity_stats(best_rocx, 'SFAM')
    print(f'  AUC (all covered): {auc_sfam_best:.4f}  N covered: {n_best_all}')

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    for ax, (lc, ll) in zip(axes, [(lc, ll) for lc, _, ll in SCOP_LEVELS]):
        if lc != 'CLASS':
            frac, sens, auc = sensitivity_stats(foldseek, lc)
            ax.plot(frac, sens, 'k-', lw=2.5, label=f'FoldSeek (AUC={auc:.3f})')
            frac, sens, auc = sensitivity_stats(tea_all, lc)
            ax.plot(frac, sens, 'k--', lw=2.5, label=f'TEA (AUC={auc:.3f})')
        frac, sens, auc = sensitivity_stats(best_rocx, lc)
        ax.plot(frac, sens, color='tomato', lw=2.5,
                label=f'KmerSeek HP k={best_k} | {best_label}\nAUC={auc:.3f} ({n_best_all} covered)')
        ax.set_xlabel('Fraction of Queries', fontsize=12)
        ax.set_ylabel(f'Sensitivity to First FP ({ll})', fontsize=12)
        ax.set_title(f'{ll}-level Sensitivity', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle(
        f'KmerSeek HP k={best_k} ({best_label}) vs FoldSeek & TEA — SCOPe40\n'
        f'each method over its own covered queries',
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_best_metric_sensitivity_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    return best_rocx


# ---------------------------------------------------------------------------
# Section 6b: k=24 vs k=27 sensitivity comparison
# ---------------------------------------------------------------------------

def compute_k_comparison_rocx(foldseek, tea_all, ref_queries: set,
                                exclude_gray_zone: bool):
    """Load k=24 and k=27 rocx dicts; plot sensitivity curves."""
    _compare_col, _compare_asc = 'jaccard', False
    _rocx_by_k = {}
    for kc in K_COMPARE:
        print(f'Loading k={kc}...')
        _ev = load_eval(kc)
        _ev = add_composite_scores_scope(_ev)
        _ev = recompute_bonferroni(_ev)
        _ev = _ev.filter(pl.col('bonferroni_correct') < BONF_THRESHOLD)
        _rocx_by_k[kc] = eval_tsv_to_rocx(_ev, score_col=_compare_col,
                                            ascending=_compare_asc,
                                            exclude_gray_zone=exclude_gray_zone)
        del _ev
    return _rocx_by_k


def plot_k_comparison(rocx_by_k: dict, foldseek, tea_all, ref_queries: set,
                       fig_prefix: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    for ax, (lc, ll) in zip(axes, [(lc, ll) for lc, _, ll in SCOP_LEVELS]):
        if lc != 'CLASS':
            frac, sens, auc = sensitivity_stats(foldseek, lc)
            ax.plot(frac, sens, 'k-', lw=2.5, label=f'FoldSeek (AUC={auc:.3f})')
            frac, sens, auc = sensitivity_stats(tea_all, lc)
            ax.plot(frac, sens, 'k--', lw=2.5, label=f'TEA (AUC={auc:.3f})')
        for kc, color in zip(K_COMPARE, COLORS_COMPARE):
            _rocx = rocx_by_k[kc]
            n_all = _rocx['NAME'].n_unique()
            n_ref = _rocx.filter(pl.col('NAME').is_in(ref_queries))['NAME'].n_unique()
            frac, sens, auc = sensitivity_stats(_rocx, lc)
            ax.plot(frac, sens, color=color, lw=2.5,
                    label=f'KmerSeek k={kc} | {LABEL_COMPARE[kc]}\n'
                          f'AUC={auc:.3f} ({n_all} covered / {n_ref} in ref)')
        ax.set_xlabel('Fraction of Queries', fontsize=12)
        ax.set_ylabel(f'Sensitivity to First FP ({ll})', fontsize=12)
        ax.set_title(f'{ll}-level Sensitivity', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle(
        f'KmerSeek k=24 vs k=27 | Bonferroni<{BONF_THRESHOLD} | Jaccard vs FoldSeek & TEA\n'
        f'each method over its own covered queries',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_k24_k27_sensitivity_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Section 6c: Shared queries sensitivity
# ---------------------------------------------------------------------------

def plot_shared_queries(foldseek, tea_all, ref_queries: set,
                         exclude_gray_zone: bool, fig_prefix: str) -> None:
    fs_covered = set(foldseek.filter(pl.col('SFAM') > 0)['NAME'].to_list())
    _compare_col, _compare_asc = 'jaccard', False
    _rocx_shared_by_k, _shared_by_k = {}, {}
    for kc in K_COMPARE:
        _ev = load_eval(kc)
        _ev = add_composite_scores_scope(_ev)
        _ev = _ev.filter(pl.col('poisson_pvalue') < PVAL_COMPARE[kc])
        _rocx = eval_tsv_to_rocx(_ev, score_col=_compare_col, ascending=_compare_asc,
                                  exclude_gray_zone=exclude_gray_zone)
        _shared = fs_covered & set(_rocx['NAME'].to_list())
        _rocx_shared_by_k[kc] = _rocx.filter(pl.col('NAME').is_in(_shared))
        _shared_by_k[kc]      = _shared
        n_all = _rocx['NAME'].n_unique()
        n_ref = _rocx.filter(pl.col('NAME').is_in(ref_queries))['NAME'].n_unique()
        print(f'k={kc}: {n_all} covered | {n_ref} in ref | {len(_shared)} shared w/ FoldSeek')
        del _ev

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    for ax, (lc, ll) in zip(axes, [(lc, ll) for lc, _, ll in SCOP_LEVELS]):
        for kc, color in zip(K_COMPARE, COLORS_COMPARE):
            shared = _shared_by_k[kc]
            frac, sens, auc = sensitivity_stats(_rocx_shared_by_k[kc], lc)
            ax.plot(frac, sens, color=color, lw=2.5,
                    label=f'KmerSeek k={kc} | {LABEL_COMPARE[kc]}\n'
                          f'AUC={auc:.3f} ({len(shared)} shared w/ FoldSeek)')
        if lc != 'CLASS':
            shared_k27 = _shared_by_k[27]
            frac, sens, auc = sensitivity_stats(
                foldseek.filter(pl.col('NAME').is_in(shared_k27)), lc)
            ax.plot(frac, sens, 'k-', lw=2.5, label=f'FoldSeek (AUC={auc:.3f})')
            frac, sens, auc = sensitivity_stats(
                tea_all.filter(pl.col('NAME').is_in(shared_k27)), lc)
            ax.plot(frac, sens, 'k--', lw=2.5, label=f'TEA (AUC={auc:.3f})')
        ax.set_xlabel('Fraction of Queries', fontsize=12)
        ax.set_ylabel(f'Sensitivity to First FP ({ll})', fontsize=12)
        ax.set_title(f'{ll}-level Sensitivity', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    shared_k27 = _shared_by_k[27]
    plt.suptitle(
        f'Head-to-Head on Shared Queries (FoldSeek-detected ∩ KmerSeek-covered) — k=27\n'
        f'{len(shared_k27)} queries — baselines omitted from Class panel',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_shared_queries_sensitivity.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Section 7: Metric category AUC bar charts
# ---------------------------------------------------------------------------

def plot_metric_category_auc(auc_df: pl.DataFrame, foldseek, tea_all,
                               fig_prefix: str) -> None:
    _, _, fs_sfam  = sensitivity_stats(foldseek, 'SFAM')
    _, _, tea_sfam = sensitivity_stats(tea_all,  'SFAM')
    pval_metrics = ['Poisson p-value (raw)', 'Bonferroni', 'BH (FDR)', 'BY']
    sim_metrics  = ['Containment', 'Max Containment', 'Jaccard', 'TF-IDF', 'Enrichment']
    comp_metrics = ['−log10(BH q)', '−log10(BH q) × Containment',
                    'Enrichment × Containment', 'TF-IDF × Containment']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    for ax, (group_name, group_labels, color) in zip(axes, [
        ('P-value corrections', pval_metrics, 'steelblue'),
        ('Similarity metrics',  sim_metrics,  'seagreen'),
        ('Composite scores',    comp_metrics, 'darkorange'),
    ]):
        group_data = (
            auc_df.filter(pl.col('label').is_in(group_labels))
            .group_by('label')
            .agg(pl.col('auc_sfam').max().alias('best_sfam_auc'))
            .sort('best_sfam_auc', descending=True)
        )
        ax.barh(group_data['label'].to_list(),
                group_data['best_sfam_auc'].to_list(), color=color, alpha=0.8)
        ax.axvline(fs_sfam,  color='black', ls='-',  lw=2, label=f'FoldSeek ({fs_sfam:.3f})')
        ax.axvline(tea_sfam, color='black', ls='--', lw=2, label=f'TEA ({tea_sfam:.3f})')
        ax.set_xlabel('Best Superfamily AUC (over all k)', fontsize=11)
        ax.set_title(group_name, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlim(0, 1)
    plt.suptitle('Best AUC per Metric Category vs FoldSeek & TEA', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_metric_category_auc.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Section 8: AUC vs k-size line plots
# ---------------------------------------------------------------------------

def plot_auc_vs_ksize(auc_df: pl.DataFrame, foldseek, tea_all,
                       fig_prefix: str) -> None:
    _, _, fs_sfam  = sensitivity_stats(foldseek, 'SFAM')
    _, _, tea_sfam = sensitivity_stats(tea_all,  'SFAM')
    selected = ['BH (FDR)', '−log10(BH q) × Containment', 'Containment', 'TF-IDF', 'Jaccard']
    palette  = ['tomato', 'steelblue', 'seagreen', 'darkorange', 'mediumpurple']
    _level_specs = [
        ('Family',      'auc_fam_all',  'auc_fam',
         sensitivity_stats(foldseek, 'FAM')[2], sensitivity_stats(tea_all, 'FAM')[2]),
        ('Superfamily', 'auc_sfam_all', 'auc_sfam', fs_sfam, tea_sfam),
        ('Fold',        'auc_fold_all', 'auc_fold',
         sensitivity_stats(foldseek, 'FOLD')[2], sensitivity_stats(tea_all, 'FOLD')[2]),
    ]

    fig1, axes1 = plt.subplots(2, 3, figsize=(24, 9),
                                gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08})
    for col, (level_name, auc_all_col, _, _, _) in enumerate(_level_specs):
        ax_auc, ax_nq = axes1[0, col], axes1[1, col]
        for label, color in zip(selected, palette):
            sub = auc_df.filter(pl.col('label') == label).sort('ksize')
            if len(sub) == 0 or auc_all_col not in auc_df.columns:
                continue
            ax_auc.plot(sub['ksize'].to_numpy(), sub[auc_all_col].to_numpy(),
                        'o-', color=color, lw=2, markersize=5, label=label)
            ax_nq.plot(sub['ksize'].to_numpy(), sub['n_queries_all'].to_numpy(),
                       'o-', color=color, lw=2, markersize=4)
        ax_auc.set_ylabel(f'{level_name} AUC', fontsize=12)
        ax_auc.set_title(level_name, fontsize=13, fontweight='bold')
        ax_auc.legend(fontsize=8)
        ax_auc.set_xlim(KMIN - 0.5, KMAX + 0.5); ax_auc.set_ylim(0, 1)
        ax_auc.set_xticklabels([])
        ax_nq.set_xlabel('K-size (HP encoding)', fontsize=12)
        ax_nq.set_ylabel('Covered queries', fontsize=10)
        ax_nq.set_xlim(KMIN - 0.5, KMAX + 0.5); ax_nq.set_yscale('log')
    fig1.suptitle('KmerSeek HP — AUC over own covered queries (all k-sizes)',
                  fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_kmerseek_only_auc_vs_ksize.png', dpi=150, bbox_inches='tight')
    plt.show()

    fig2, axes2 = plt.subplots(2, 3, figsize=(24, 9),
                                gridspec_kw={'height_ratios': [3, 1.2], 'hspace': 0.08})
    for col, (level_name, _, auc_ref_col, fs_auc, tea_auc) in enumerate(_level_specs):
        ax_auc, ax_nq = axes2[0, col], axes2[1, col]
        ax_auc.axhline(fs_auc,  color='black', ls='-',  lw=2.5,
                       label=f'FoldSeek ({fs_auc:.3f})', zorder=5)
        ax_auc.axhline(tea_auc, color='black', ls='--', lw=2.5,
                       label=f'TEA ({tea_auc:.3f})',     zorder=5)
        for label, color in zip(selected, palette):
            sub = auc_df.filter(pl.col('label') == label).sort('ksize')
            if len(sub) == 0 or auc_ref_col not in auc_df.columns:
                continue
            ax_auc.plot(sub['ksize'].to_numpy(), sub[auc_ref_col].to_numpy(),
                        'o-', color=color, lw=2, markersize=5, label=label)
            ax_nq.plot(sub['ksize'].to_numpy(), sub['n_queries'].to_numpy(),
                       'o-', color=color, lw=2, markersize=4)
        ax_nq.axhline(SWEEP_NQUERY_MIN, color='grey', ls=':', lw=1.5)
        ax_auc.set_ylabel(f'{level_name} AUC', fontsize=12)
        ax_auc.set_title(level_name, fontsize=13, fontweight='bold')
        ax_auc.legend(fontsize=8)
        ax_auc.set_xlim(KMIN - 0.5, KMAX + 0.5); ax_auc.set_ylim(0, 1)
        ax_auc.set_xticklabels([])
        ax_nq.set_xlabel('K-size (HP encoding)', fontsize=12)
        ax_nq.set_ylabel('Covered queries\n(in ref set)', fontsize=10)
        ax_nq.set_xlim(KMIN - 0.5, KMAX + 0.5); ax_nq.set_yscale('log')
    fig2.suptitle('KmerSeek HP vs FoldSeek & TEA — AUC over benchmarkable ref queries',
                  fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_sfam_auc_vs_ksize.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Section 8b: Precision-recall curves
# ---------------------------------------------------------------------------

def plot_precision_recall_curves(foldseek, tea_all, exclude_gray_zone: bool,
                                  fig_prefix: str) -> None:
    _rocx_to_eval = {'FAM': 'same_family', 'SFAM': 'same_superfamily', 'FOLD': 'same_fold'}
    _eval_to_rocx = {v: k for k, v in _rocx_to_eval.items()}
    _pr_levels    = [(ec, ll) for _, ec, ll in SCOP_LEVELS]

    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    for kc, color in zip(K_COMPARE, COLORS_COMPARE):
        print(f'Loading k={kc} for PR curves...')
        _ev = load_eval(kc, columns=EVAL_USECOLS)
        _ev = add_composite_scores_scope(_ev)
        _ev = recompute_bonferroni(_ev)
        _ev = _ev.filter(pl.col('bonferroni_correct') < BONF_THRESHOLD)
        for ax, (lc, ll) in zip(axes, _pr_levels):
            if lc not in _ev.columns:
                ax.text(0.5, 0.5, f'{ll} n/a', ha='center', va='center',
                        transform=ax.transAxes)
                continue
            rec, prec, ap = precision_recall_from_eval(
                _ev, score_col='jaccard', ascending=False, level_col=lc,
                exclude_gray_zone=exclude_gray_zone)
            ax.plot(rec, prec, color=color, lw=2.5,
                    label=f'KmerSeek k={kc} | {LABEL_COMPARE[kc]}\nAP={ap:.3f}')
        del _ev

    _baseline_styles = [(foldseek, 'Foldseek', 'k', 'o'), (tea_all, 'TEA', 'k', '^')]
    for ax, (lc, ll) in zip(axes, _pr_levels):
        _rc = _eval_to_rocx.get(lc)
        if _rc is None:
            continue
        for _df, _name, _color, _marker in _baseline_styles:
            if _rc not in _df.columns:
                continue
            _recall, _prec = pr_operating_point_from_rocx(_df, level_col=_rc)
            ax.plot(_recall, _prec, marker=_marker, color=_color, ms=10, ls='none',
                    label=f'{_name} (recall={_recall:.3f}, prec={_prec:.3f})')

    gz_label = 'gray-zone excluded' if exclude_gray_zone else 'all FPs'
    for ax, (_, ll) in zip(axes, _pr_levels):
        ax.set_xlabel('Recall', fontsize=12); ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'{ll}-level PR Curve', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle(
        f'Precision-Recall Curves — KmerSeek HP k=24 vs k=27 vs Foldseek & TEA\n'
        f'({gz_label}; Foldseek/TEA shown as operating points at first FP)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_precision_recall_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Section 8c: Bonferroni sweep (AUC vs k-size, Jaccard)
# ---------------------------------------------------------------------------

def run_bonferroni_sweep(foldseek, tea_all, exclude_gray_zone: bool,
                          fig_prefix: str) -> pl.DataFrame:
    fs_sfam_auc = sensitivity_stats(foldseek, 'SFAM')[2]
    fs_fam_auc  = sensitivity_stats(foldseek, 'FAM')[2]
    fs_fold_auc = sensitivity_stats(foldseek, 'FOLD')[2]
    N_QUERIES_MIN = 150

    _bonf_rows = []
    for k in KSIZES:
        try:
            _ev = load_eval(k, columns=EVAL_USECOLS)
            _ev = add_composite_scores_scope(_ev)
            _ev = recompute_bonferroni(_ev)
            _ev = _ev.filter(pl.col('bonferroni_correct') < BONF_THRESHOLD)
            _rocx = eval_tsv_to_rocx(_ev, score_col='jaccard', ascending=False,
                                     exclude_gray_zone=exclude_gray_zone)
            del _ev; gc.collect()

            n_q    = _rocx['NAME'].n_unique()
            scops  = _rocx['SCOP']
            n_fold = scops.str.split('.').list.slice(0, 2).list.join('.').n_unique()
            n_sfam = scops.str.split('.').list.slice(0, 3).list.join('.').n_unique()
            n_fam  = scops.n_unique()
            _, _, auc_sfam = sensitivity_stats(_rocx, 'SFAM')
            _, _, auc_fam  = sensitivity_stats(_rocx, 'FAM')
            _, _, auc_fold = sensitivity_stats(_rocx, 'FOLD')
            del _rocx
            _bonf_rows.append(dict(ksize=k, n_queries=n_q,
                                   n_fold=n_fold, n_sfam=n_sfam, n_fam=n_fam,
                                   auc_sfam=auc_sfam, auc_fam=auc_fam, auc_fold=auc_fold))
            print(f'k={k}: n={n_q}, fam={auc_fam:.3f}, sfam={auc_sfam:.3f}, fold={auc_fold:.3f}')
        except FileNotFoundError:
            print(f'k={k}: SKIP')

    bonf_df = pl.DataFrame(_bonf_rows)
    ks = bonf_df['ksize'].to_numpy()

    _covered = bonf_df.filter(pl.col('n_queries') >= N_QUERIES_MIN)

    _first_beats = {}
    _peak_k      = {}
    for lvl_name, auc_col, fs_auc in [
        ('Family',      'auc_fam',  fs_fam_auc),
        ('Superfamily', 'auc_sfam', fs_sfam_auc),
        ('Fold',        'auc_fold', fs_fold_auc),
    ]:
        _beats = bonf_df.filter(pl.col(auc_col) > fs_auc).sort('ksize')
        _first_beats[lvl_name] = _beats['ksize'][0] if len(_beats) > 0 else None
        _pk = _covered.sort(auc_col, descending=True)
        _peak_k[lvl_name] = _pk['ksize'][0] if len(_pk) > 0 else None

    for lvl_name in ('Family', 'Superfamily', 'Fold'):
        print(f'{lvl_name}-level first beats FoldSeek: k={_first_beats[lvl_name]}')
        print(f'{lvl_name}-level peak (n≥{N_QUERIES_MIN}):       k={_peak_k[lvl_name]}')

    _specs = [
        ('Family',      'auc_fam',  'n_fam',  fs_fam_auc,  'Unique families'),
        ('Superfamily', 'auc_sfam', 'n_sfam', fs_sfam_auc, 'Unique superfamilies'),
        ('Fold',        'auc_fold', 'n_fold', fs_fold_auc, 'Unique folds'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(24, 9),
                              gridspec_kw={'height_ratios': [3, 1.5], 'hspace': 0.08})
    for col_i, (level_name, auc_col, cnt_col, fs_auc, cnt_label) in enumerate(_specs):
        ax_auc, ax_cnt = axes[0, col_i], axes[1, col_i]
        ys_auc = bonf_df[auc_col].to_numpy()
        ys_cnt = bonf_df[cnt_col].to_numpy()
        ax_auc.axhline(fs_auc, color='black', ls='-', lw=2.5, label=f'FoldSeek ({fs_auc:.3f})')
        ax_auc.plot(ks, ys_auc, 'o-', color='tomato', lw=2, markersize=5,
                    label=f'KmerSeek Bonferroni<{BONF_THRESHOLD}')
        first_k = _first_beats[level_name]
        best_k  = _peak_k[level_name]
        if first_k is not None:
            _y = bonf_df.filter(pl.col('ksize') == first_k)[auc_col][0]
            ax_auc.axvline(first_k, color='seagreen', ls='--', lw=1.5, alpha=0.8)
            ax_auc.annotate(f'first beats\nFoldSeek\nk={first_k}',
                            xy=(first_k, _y), xytext=(first_k+0.8, _y-0.05),
                            fontsize=8, color='seagreen',
                            arrowprops=dict(arrowstyle='->', color='seagreen'))
        if best_k is not None and best_k != first_k:
            _y = bonf_df.filter(pl.col('ksize') == best_k)[auc_col][0]
            ax_auc.axvline(best_k, color='steelblue', ls=':', lw=1.5, alpha=0.8)
            ax_auc.annotate(f'peak\n(n≥{N_QUERIES_MIN})\nk={best_k}',
                            xy=(best_k, _y), xytext=(best_k+0.8, _y+0.03),
                            fontsize=8, color='steelblue',
                            arrowprops=dict(arrowstyle='->', color='steelblue'))
        ax_auc.axhline(0, color='grey', ls=':', lw=0.8, alpha=0.5)
        ax_auc.set_title(level_name, fontsize=13, fontweight='bold')
        ax_auc.set_ylabel(f'{level_name} AUC', fontsize=11)
        ax_auc.legend(fontsize=9)
        ax_auc.set_xlim(KMIN - 0.5, KMAX + 0.5); ax_auc.set_ylim(0, 1)
        ax_auc.set_xticklabels([])
        ax_cnt.plot(ks, ys_cnt, 'o-', color='tomato', lw=2, markersize=4)
        ax_cnt.axhline(N_QUERIES_MIN, color='grey', ls=':', lw=1.5, label=f'n={N_QUERIES_MIN}')
        ax_cnt.set_xlabel('K-size (HP encoding)', fontsize=11)
        ax_cnt.set_ylabel(cnt_label, fontsize=10)
        ax_cnt.legend(fontsize=8)
        ax_cnt.set_xlim(KMIN - 0.5, KMAX + 0.5); ax_cnt.set_ylim(bottom=0)
    fig.suptitle(
        f'KmerSeek HP — AUC vs K-size | Bonferroni<{BONF_THRESHOLD} | Jaccard\n'
        f'each method over its own covered queries',
        fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_bonf_auc_vs_ksize_fam_sfam_fold.png', dpi=150, bbox_inches='tight')
    plt.show()
    return bonf_df


# ---------------------------------------------------------------------------
# Section 9a: Bonferroni + Jaccard combo analysis (replaces threshold sweep)
# ---------------------------------------------------------------------------

def run_bonf_combo_analysis(bonf_df: pl.DataFrame, foldseek, tea_all,
                              ref_queries: set, exclude_gray_zone: bool,
                              fig_prefix: str) -> None:
    """Run class-enrichment analysis for best-FAM-k and best-SFAM-k from bonf sweep."""
    N_QUERIES_MIN_LOCAL = 150
    _, _, fs_sfam = sensitivity_stats(foldseek, 'SFAM')

    _covered = bonf_df.filter(pl.col('n_queries') >= N_QUERIES_MIN_LOCAL)
    if len(_covered) == 0:
        print('No k-sizes with n_queries >= 150; skipping combo analysis.')
        return

    best_fam_k  = _covered.sort('auc_fam',  descending=True)['ksize'][0]
    best_sfam_k = _covered.sort('auc_sfam', descending=True)['ksize'][0]

    ks_to_run = list(dict.fromkeys([best_fam_k, best_sfam_k]))  # deduplicate, preserve order

    fs_rocx_ref = foldseek.filter(pl.col('NAME').is_in(ref_queries))
    base_det    = detected(fs_rocx_ref)
    fs_cls      = class_dist(base_det, fs_rocx_ref)

    COMBOS = []
    combo_results = {}
    for k in ks_to_run:
        row = bonf_df.filter(pl.col('ksize') == k).row(0, named=True)
        label = f'k={k} Bonferroni<{BONF_THRESHOLD} Jaccard'
        ev    = load_eval(k, columns=EVAL_USECOLS)
        ev    = add_composite_scores_scope(ev)
        ev    = recompute_bonferroni(ev)
        ev    = ev.filter(pl.col('bonferroni_correct') < BONF_THRESHOLD)
        rocx  = eval_tsv_to_rocx(ev, score_col='jaccard', ascending=False,
                                  exclude_gray_zone=exclude_gray_zone)
        rocx_cov = rocx.filter(pl.col('NAME').is_in(ref_queries))

        _, _, auc_fam_cov  = sensitivity_stats(rocx_cov, 'FAM')
        _, _, auc_sfam_cov = sensitivity_stats(rocx_cov, 'SFAM')
        km_det  = detected(rocx_cov)
        km_uniq = km_det - base_det

        def auc_restricted(rocx_df, query_names, level='SFAM'):
            sub = rocx_df.filter(pl.col('NAME').is_in(query_names))
            return sensitivity_stats(sub, level)[2] if len(sub) > 0 else float('nan')

        fs_auc_shared = auc_restricted(foldseek, km_det)
        print(f'k={k}: n_queries={row["n_queries"]}, fam={auc_fam_cov:.4f}, '
              f'sfam={auc_sfam_cov:.4f}, unique={len(km_uniq)}')

        COMBOS.append(dict(k=k, score_col='jaccard', ascending=False, label=label))
        combo_results[label] = dict(
            rocx_cov=rocx_cov, km_det=km_det, km_uniq=km_uniq,
            auc_cov=auc_sfam_cov, fs_auc_shared=fs_auc_shared, label=label)
        del ev, rocx; gc.collect()

    plot_class_enrichment(COMBOS, combo_results, fs_cls, fig_prefix)


# ---------------------------------------------------------------------------
# Section 9: Threshold combo sweep (cached)
# ---------------------------------------------------------------------------

def run_threshold_sweep(ref_queries: set, exclude_gray_zone: bool,
                         force_recompute: bool = False) -> pl.DataFrame:
    suffix  = _cache_suffix(exclude_gray_zone)
    cache   = BENCH_DIR / f'scope_thresh_combo_auc_sweep_k15_45_{suffix}.parquet'
    filter_combos = _build_filter_combos()

    if not force_recompute and cache.exists():
        thresh_auc_df = pl.read_parquet(cache)
        print(f'Loaded cached threshold sweep ({suffix}): {len(thresh_auc_df):,} rows')
        return thresh_auc_df

    thresh_rows = []
    for k in THRESH_KSIZES:
        print(f'k={k}:', end=' ', flush=True)
        try:
            eval_df = load_eval(k, columns=EVAL_USECOLS)
            eval_df = add_composite_scores_scope(eval_df)
        except FileNotFoundError as e:
            print(f'SKIP ({e})'); continue

        for fc in filter_combos:
            fdf = eval_df
            if fc['tfidf_min'] is not None:
                fdf = fdf.filter(pl.col('query_tfidf') > fc['tfidf_min'])
            if fc['bh_max'] is not None:
                fdf = fdf.filter(pl.col('bh') < fc['bh_max'])
            if fc['pval_max'] is not None:
                fdf = fdf.filter(pl.col('poisson_pvalue') < fc['pval_max'])
            n_pairs = len(fdf)

            for score_col, ascending, slabel in THRESH_METRICS:
                if score_col not in fdf.columns:
                    continue
                if n_pairs == 0:
                    thresh_rows.append(dict(
                        ksize=k, filter_label=fc['filter_label'],
                        tfidf_min=fc['tfidf_min'] or 0.0,
                        bh_max=fc['bh_max'] or 1.0, pval_max=fc['pval_max'] or 1.0,
                        score_label=slabel, auc_sfam_covered=0.0,
                        coverage=0.0, n_pairs=0, n_queries=0))
                    continue
                rocx         = eval_tsv_to_rocx(fdf, score_col=score_col,
                                                ascending=ascending,
                                                exclude_gray_zone=exclude_gray_zone)
                rocx_covered = rocx.filter(pl.col('NAME').is_in(ref_queries))
                n_covered    = rocx_covered.height
                _, _, sfam_covered = (sensitivity_stats(rocx_covered, 'SFAM')
                                      if n_covered > 0 else (None, None, 0.0))
                thresh_rows.append(dict(
                    ksize=k, filter_label=fc['filter_label'],
                    tfidf_min=float(fc['tfidf_min']) if fc['tfidf_min'] is not None else 0.0,
                    bh_max=float(fc['bh_max'])    if fc['bh_max']   is not None else 1.0,
                    pval_max=float(fc['pval_max']) if fc['pval_max'] is not None else 1.0,
                    score_label=slabel, auc_sfam_covered=round(sfam_covered, 5),
                    coverage=round(n_covered / N_SCOPE40, 5),
                    n_pairs=n_pairs, n_queries=n_covered))
            print('.', end='', flush=True)

        del eval_df; gc.collect()
        print()

    thresh_auc_df = pl.DataFrame(thresh_rows)
    thresh_auc_df.write_parquet(cache)
    print(f'Saved {len(thresh_auc_df):,} rows → {cache.name}')
    return thresh_auc_df


# ---------------------------------------------------------------------------
# Section 9 plots
# ---------------------------------------------------------------------------

def _nq_style(nq, nq_10pct, nq_1pct):
    if nq >= nq_10pct: return dict(ls='-', lw=5)
    if nq >= nq_1pct:  return dict(ls='-', lw=2)
    return                     dict(ls=':', lw=1)


def _plot_with_nq_width(ax, ks, auc, nq, color, label, nq_10pct, nq_1pct):
    cats  = [_nq_style(n, nq_10pct, nq_1pct) for n in nq]
    first = True
    for _, grp in itertools.groupby(range(len(ks)),
                                    key=lambda i: (cats[i]['ls'], cats[i]['lw'])):
        idx = list(grp)
        lo  = max(0, idx[0] - 1)
        hi  = min(len(ks) - 1, idx[-1] + 1)
        style = cats[idx[0]]
        ax.plot(ks[lo:hi+1], auc[lo:hi+1], color=color,
                marker='o', markersize=3 if style['lw'] < 3 else 5,
                label=label if first else '_nolegend_', **style)
        first = False


def plot_threshold_auc_by_filter(thresh_auc_df: pl.DataFrame, foldseek, tea_all,
                                   fig_prefix: str) -> None:
    _, _, fs_sfam  = sensitivity_stats(foldseek, 'SFAM')
    _, _, tea_sfam = sensitivity_stats(tea_all,  'SFAM')
    NQ_10PCT = int(N_SCOPE40 * 0.10)
    NQ_1PCT  = int(N_SCOPE40 * 0.01)
    palette5 = ['tomato', 'steelblue', 'seagreen', 'darkorange', 'mediumpurple']
    all_filter_labels = ['No filter',
                         'pval<0.05', 'pval<0.001', 'pval<1e-05',
                         'BH<0.05',   'BH<0.001',   'BH<1e-05']
    ncols = 3
    nrows = -(-len(all_filter_labels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharey=True)
    axes_flat = list(axes.flat)
    for ax, fl in zip(axes_flat, all_filter_labels):
        sub = thresh_auc_df.filter(pl.col('filter_label') == fl).sort('ksize')
        ax.axhline(fs_sfam,  color='black', ls='-',  lw=1.5, label=f'FoldSeek ({fs_sfam:.3f})', zorder=5)
        ax.axhline(tea_sfam, color='black', ls='--', lw=1.5, label=f'TEA ({tea_sfam:.3f})',     zorder=5)
        for (_, _, slabel), color in zip(THRESH_METRICS, palette5):
            sl = sub.filter(pl.col('score_label') == slabel).sort('ksize')
            if len(sl) == 0: continue
            _plot_with_nq_width(ax, sl['ksize'].to_numpy(),
                                sl['auc_sfam_covered'].to_numpy(),
                                sl['n_queries'].to_numpy(),
                                color, slabel, NQ_10PCT, NQ_1PCT)
        max_nq = int(sub['n_queries'].max()) if sub.height > 0 else 0
        ax.set_title(f'"{fl}"  (max n={max_nq:,})', fontsize=10, fontweight='bold')
        ax.set_xlabel('K-size (HP)', fontsize=9)
        ax.set_ylabel('Covered-query SFAM AUC', fontsize=9)
        ax.set_xlim(THRESH_KSIZES[0] - 0.5, THRESH_KSIZES[-1] + 0.5)
        ax.set_ylim(0, 1)
    for ax in axes_flat[len(all_filter_labels):]:
        ax.set_visible(False)
    metric_handles = [Line2D([0],[0], color=c, lw=3, marker='o', markersize=4, label=l)
                      for (_, _, l), c in zip(THRESH_METRICS, palette5)]
    style_handles = [
        Line2D([0],[0], color='grey', lw=5, ls='-',  label=f'n_queries ≥ 10% ({NQ_10PCT:,})'),
        Line2D([0],[0], color='grey', lw=2, ls='-',  label=f'n_queries ≥ 1% ({NQ_1PCT:,})'),
        Line2D([0],[0], color='grey', lw=1, ls=':',  label=f'n_queries < 1% ({NQ_1PCT:,})'),
        Line2D([0],[0], color='black', lw=1.5, ls='-',  label=f'FoldSeek ({fs_sfam:.3f})'),
        Line2D([0],[0], color='black', lw=1.5, ls='--', label=f'TEA ({tea_sfam:.3f})'),
    ]
    fig.legend(handles=metric_handles + style_handles, fontsize=9,
               loc='lower right', bbox_to_anchor=(0.98, 0.01), ncol=2)
    fig.suptitle(
        f'Covered-query AUC vs K-size  '
        f'(lw=5: ≥10% [{NQ_10PCT:,}], lw=2: ≥1% [{NQ_1PCT:,}], dotted: <1%)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_tfidf_threshold_auc.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_threshold_auc_by_metric(thresh_auc_df: pl.DataFrame, foldseek, tea_all,
                                   fig_prefix: str) -> None:
    _, _, fs_sfam  = sensitivity_stats(foldseek, 'SFAM')
    _, _, tea_sfam = sensitivity_stats(tea_all,  'SFAM')
    NQ_10PCT = int(N_SCOPE40 * 0.10)
    NQ_1PCT  = int(N_SCOPE40 * 0.01)
    all_filter_labels = ['No filter',
                         'pval<0.05', 'pval<0.001', 'pval<1e-05',
                         'BH<0.05',   'BH<0.001',   'BH<1e-05']
    filter_palette = ['#888888','#fdb462','#e6550d','#a63603',
                      '#9ecae1','#3182bd','#08519c']
    filter_color = dict(zip(all_filter_labels, filter_palette))
    n_metrics = len(THRESH_METRICS)
    ncols = 3
    nrows = -(-n_metrics // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows), sharey=True)
    axes_flat = list(axes.flat)
    for ax, (_, _, slabel) in zip(axes_flat, THRESH_METRICS):
        ax.axhline(fs_sfam,  color='black', ls='-',  lw=1.5, zorder=5,
                   label=f'FoldSeek ({fs_sfam:.3f})')
        ax.axhline(tea_sfam, color='black', ls='--', lw=1.5, zorder=5,
                   label=f'TEA ({tea_sfam:.3f})')
        for fl in all_filter_labels:
            sl = (thresh_auc_df
                  .filter(pl.col('filter_label') == fl)
                  .filter(pl.col('score_label')  == slabel)
                  .sort('ksize'))
            if len(sl) == 0: continue
            _plot_with_nq_width(ax, sl['ksize'].to_numpy(),
                                sl['auc_sfam_covered'].to_numpy(),
                                sl['n_queries'].to_numpy(),
                                filter_color[fl], fl, NQ_10PCT, NQ_1PCT)
        ax.set_title(slabel, fontsize=11, fontweight='bold')
        ax.set_xlabel('K-size (HP)', fontsize=9)
        ax.set_ylabel('Covered-query SFAM AUC', fontsize=9)
        ax.set_xlim(THRESH_KSIZES[0] - 0.5, THRESH_KSIZES[-1] + 0.5)
        ax.set_ylim(0, 1)
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)
    filter_handles = [Line2D([0],[0], color=filter_color[fl], lw=2, marker='o',
                             markersize=4, label=fl)
                      for fl in all_filter_labels]
    style_handles = [
        Line2D([0],[0], color='grey', lw=5, ls='-',  label=f'n_queries ≥ 10% ({NQ_10PCT:,})'),
        Line2D([0],[0], color='grey', lw=2, ls='-',  label=f'n_queries ≥ 1% ({NQ_1PCT:,})'),
        Line2D([0],[0], color='grey', lw=1, ls=':',  label=f'n_queries < 1% ({NQ_1PCT:,})'),
        Line2D([0],[0], color='black', lw=1.5, ls='-',  label=f'FoldSeek ({fs_sfam:.3f})'),
        Line2D([0],[0], color='black', lw=1.5, ls='--', label=f'TEA ({tea_sfam:.3f})'),
    ]
    fig.legend(handles=filter_handles + style_handles, fontsize=9,
               loc='lower right', bbox_to_anchor=(0.98, 0.01), ncol=2)
    fig.suptitle(
        f'Covered-query AUC vs K-size by Metric  '
        f'(lw=5: ≥10% [{NQ_10PCT:,}], lw=2: ≥1% [{NQ_1PCT:,}], dotted: <1%)',
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_sensitivity_curves_k25.png', dpi=150, bbox_inches='tight')
    plt.show()


def run_threshold_comparison(thresh_auc_df: pl.DataFrame, foldseek, tea_all,
                               ref_queries: set, exclude_gray_zone: bool,
                               fig_prefix: str):
    """Evaluate top 30 combos head-to-head vs FoldSeek/TEA; return cmp_df, best_covered."""
    _, _, fs_sfam = sensitivity_stats(foldseek, 'SFAM')
    _score_map = {lbl: (col, asc) for col, asc, lbl in THRESH_METRICS}

    def auc_restricted(rocx_df, query_names, level='SFAM'):
        sub = rocx_df.filter(pl.col('NAME').is_in(query_names))
        return sensitivity_stats(sub, level)[2] if len(sub) > 0 else float('nan')

    TOP_N = 30
    top_combos = (thresh_auc_df
                  .filter(pl.col('n_queries') >= NQUERY_MIN)
                  .sort('auc_sfam_covered', descending=True)
                  .head(TOP_N))
    comparison_rows = []
    for row in top_combos.iter_rows(named=True):
        try:
            eval_df = load_eval(row['ksize'], columns=EVAL_USECOLS)
            eval_df = add_composite_scores_scope(eval_df)
            fdf = eval_df
            if row['tfidf_min'] and row['tfidf_min'] > 0:
                fdf = fdf.filter(pl.col('query_tfidf') > row['tfidf_min'])
            if row['bh_max'] and row['bh_max'] < 1:
                fdf = fdf.filter(pl.col('bh') < row['bh_max'])
            if row['pval_max'] and row['pval_max'] < 1:
                fdf = fdf.filter(pl.col('poisson_pvalue') < row['pval_max'])
            score_col, ascending = next(
                (c, a) for c, a, l in THRESH_METRICS if l == row['score_label'])
            rocx_km = eval_tsv_to_rocx(fdf, score_col=score_col, ascending=ascending,
                                        exclude_gray_zone=exclude_gray_zone)
            covered_names = set(rocx_km.filter(pl.col('NAME').is_in(ref_queries))['NAME'].to_list())
            del eval_df, fdf, rocx_km; gc.collect()
        except Exception as e:
            print(f'  skip k={row["ksize"]}: {e}'); continue

        fs_auc  = auc_restricted(foldseek, covered_names)
        tea_auc = auc_restricted(tea_all,  covered_names)
        comparison_rows.append(dict(
            ksize=row['ksize'], filter_label=row['filter_label'],
            score_label=row['score_label'], n_covered=len(covered_names),
            coverage=row['coverage'], km_auc=row['auc_sfam_covered'],
            fs_auc_shared=fs_auc, tea_auc_shared=tea_auc,
            km_beats_fs=bool(row['auc_sfam_covered'] > fs_auc),
            km_beats_tea=bool(row['auc_sfam_covered'] > tea_auc),
        ))
        print(f"k={row['ksize']:2d} n={len(covered_names):4d} | "
              f"{row['filter_label']:<25} | {row['score_label']:<30} | "
              f"KM={row['auc_sfam_covered']:.3f}  FS={fs_auc:.3f}  TEA={tea_auc:.3f}  "
              f"{'✓ BEATS FS' if row['auc_sfam_covered'] > fs_auc else ''}")

    cmp_df = pl.DataFrame(comparison_rows)
    print(f"\nKmerSeek beats FoldSeek: {cmp_df['km_beats_fs'].sum()} / {len(cmp_df)}")
    print(f"KmerSeek beats TEA:      {cmp_df['km_beats_tea'].sum()} / {len(cmp_df)}")

    best_covered = (thresh_auc_df
                    .filter(pl.col('n_queries') >= NQUERY_MIN)
                    .sort('auc_sfam_covered', descending=True)
                    .row(0, named=True))
    print(f"\nbest_covered: k={best_covered['ksize']}  {best_covered['filter_label']}  "
          f"score={best_covered['score_label']}  "
          f"AUC={best_covered['auc_sfam_covered']:.4f}  n={best_covered['n_queries']:,}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    xs = range(len(cmp_df))
    labels = [f"k={r['ksize']}\n{r['filter_label'][:18]}\n{r['score_label'][:15]}"
              for r in cmp_df.iter_rows(named=True)]
    ax.plot(xs, cmp_df['km_auc'].to_list(),        'o-', color='tomato',    lw=2, ms=6, label='KmerSeek')
    ax.plot(xs, cmp_df['fs_auc_shared'].to_list(),  's-', color='black',     lw=2, ms=6, label='FoldSeek (same queries)')
    ax.plot(xs, cmp_df['tea_auc_shared'].to_list(), '^-', color='steelblue', lw=2, ms=6, label='TEA (same queries)')
    for i, row in enumerate(cmp_df.iter_rows(named=True)):
        if row['km_beats_fs']:
            ax.axvspan(i - 0.4, i + 0.4, alpha=0.15, color='gold', zorder=0)
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('Superfamily AUC (on KmerSeek-covered queries)', fontsize=11)
    ax.set_title('Head-to-head on shared queries: KmerSeek vs FoldSeek vs TEA\n'
                 '(gold = KmerSeek beats FoldSeek)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_bonferroni_correct_auc.png', dpi=150, bbox_inches='tight')
    plt.show()

    return cmp_df, best_covered


def plot_coverage_bins(thresh_auc_df: pl.DataFrame, foldseek, tea_all,
                        fig_prefix: str) -> None:
    _, _, fs_sfam  = sensitivity_stats(foldseek, 'SFAM')
    _, _, tea_sfam = sensitivity_stats(tea_all,  'SFAM')
    cov_bins = [(0.10, 1.01, '≥10% coverage'),
                (0.05, 0.10, '5–10% coverage'),
                (0.01, 0.05, '1–5% coverage')]
    fig, axes = plt.subplots(1, len(cov_bins), figsize=(15, 5), sharey=False)
    for ax, (lo, hi, label) in zip(axes, cov_bins):
        sub = thresh_auc_df.filter((pl.col('coverage') >= lo) & (pl.col('coverage') < hi))
        if len(sub) == 0:
            ax.set_title(label + '\n(no data)'); continue
        best = (sub.group_by('score_label')
                   .agg(pl.col('auc_sfam_covered').max())
                   .sort('auc_sfam_covered', descending=True))
        ax.barh(best['score_label'].to_list(), best['auc_sfam_covered'].to_list(),
                color='steelblue', alpha=0.8)
        ax.axvline(fs_sfam,  color='black', ls='-',  lw=1.5, label=f'FoldSeek ({fs_sfam:.3f})')
        ax.axvline(tea_sfam, color='black', ls='--', lw=1.5, label=f'TEA ({tea_sfam:.3f})')
        ax.set_title(f'{label}\n(n combos: {len(sub)})', fontsize=11, fontweight='bold')
        ax.set_xlabel('Best covered-AUC (max over k)', fontsize=10)
        ax.set_xlim(0, 1.05); ax.legend(fontsize=8)
    plt.suptitle('Best Ranking Metric by Coverage Level', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_best_combo_sensitivity(best_covered: dict, foldseek, tea_all,
                                 ref_queries: set, exclude_gray_zone: bool,
                                 fig_prefix: str) -> None:
    bt = best_covered
    print(f'Loading k={bt["ksize"]} for best-combo sensitivity curve...')
    eval_best = load_eval(bt['ksize'], columns=EVAL_USECOLS)
    eval_best = add_composite_scores_scope(eval_best)
    fdf_best  = eval_best
    if bt['tfidf_min'] and bt['tfidf_min'] > 0:
        fdf_best = fdf_best.filter(pl.col('query_tfidf') > bt['tfidf_min'])
    if bt['bh_max'] and bt['bh_max'] < 1:
        fdf_best = fdf_best.filter(pl.col('bh') < bt['bh_max'])
    if bt['pval_max'] and bt['pval_max'] < 1:
        fdf_best = fdf_best.filter(pl.col('poisson_pvalue') < bt['pval_max'])

    best_score_col, best_ascending = next(
        (col, asc) for col, asc, lbl in THRESH_METRICS if lbl == bt['score_label'])
    rocx_best     = eval_tsv_to_rocx(fdf_best, score_col=best_score_col,
                                      ascending=best_ascending,
                                      exclude_gray_zone=exclude_gray_zone)
    rocx_best_cov = rocx_best.filter(pl.col('NAME').is_in(ref_queries))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, lc, ll in zip(axes, ['SFAM', 'FAM'], ['Superfamily', 'Family']):
        frac, sens, auc = sensitivity_stats(foldseek, lc)
        ax.plot(frac, sens, 'k-',  lw=2.5, label=f'FoldSeek (AUC={auc:.3f})')
        frac, sens, auc = sensitivity_stats(tea_all, lc)
        ax.plot(frac, sens, 'k--', lw=2.5, label=f'TEA (AUC={auc:.3f})')
        frac, sens, auc = sensitivity_stats(rocx_best_cov, lc)
        cov_pct = bt['coverage'] * 100
        ax.plot(frac, sens, color='tomato', lw=2.5,
                label=f'KmerSeek k={bt["ksize"]} | {bt["filter_label"]} | {bt["score_label"]}\n'
                      f'AUC={auc:.3f} ({bt["n_queries"]} queries, {cov_pct:.1f}% of ref)')
        ax.set_xlabel('Fraction of Queries', fontsize=12)
        ax.set_ylabel(f'Sensitivity to First FP ({ll})', fontsize=12)
        ax.set_title(f'{ll}-level Sensitivity', fontsize=13, fontweight='bold')
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle('Best KmerSeek Threshold Combo vs FoldSeek & TEA',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_bonferroni_collapse.png', dpi=150, bbox_inches='tight')
    plt.show()
    del eval_best; gc.collect()


# ---------------------------------------------------------------------------
# Section 10 helpers (class enrichment analysis)
# ---------------------------------------------------------------------------

def detected(rocx: pl.DataFrame) -> set:
    return set(rocx.filter(pl.col('SFAM') > 0)['NAME'].to_list())


def class_dist(query_set: set, rocx: pl.DataFrame) -> dict:
    sub = (rocx.filter(pl.col('NAME').is_in(query_set))
               .select(['NAME', 'SCOP']).unique()
               .with_columns(pl.col('SCOP').str.slice(0, 1).alias('cls')))
    total = sub.height
    return {row['cls']: (row['n'], row['n'] / total if total > 0 else 0.0)
            for row in sub.group_by('cls').agg(pl.len().alias('n')).sort('cls').iter_rows(named=True)}


def sfam_from_lineage(scop: str) -> str:
    if not scop: return ''
    parts = scop.split('.')
    return '.'.join(parts[:3]) if len(parts) >= 3 else scop


def run_combo_analysis(thresh_auc_df: pl.DataFrame, foldseek, tea_all,
                        ref_queries: set, exclude_gray_zone: bool):
    """Auto-select top 3 combos that beat FoldSeek and run per-combo analysis."""
    _, _, fs_sfam = sensitivity_stats(foldseek, 'SFAM')
    _score_map = {lbl: (col, asc) for col, asc, lbl in THRESH_METRICS}

    fs_rocx_ref = foldseek.filter(pl.col('NAME').is_in(ref_queries))
    base_det    = detected(fs_rocx_ref)
    fs_cls      = class_dist(base_det, fs_rocx_ref)

    def auc_restricted(rocx_df, query_names, level='SFAM'):
        sub = rocx_df.filter(pl.col('NAME').is_in(query_names))
        return sensitivity_stats(sub, level)[2] if len(sub) > 0 else float('nan')

    _top = (thresh_auc_df
            .filter(pl.col('n_queries') >= NQUERY_MIN)
            .filter(pl.col('auc_sfam_covered') > fs_sfam)
            .sort('auc_sfam_covered', descending=True)
            .head(3))
    if _top.height == 0:
        _top = (thresh_auc_df
                .filter(pl.col('n_queries') >= NQUERY_MIN)
                .sort('auc_sfam_covered', descending=True)
                .head(3))

    COMBOS = []
    for row in _top.iter_rows(named=True):
        score_col, ascending = _score_map.get(row['score_label'], ('query_tfidf', False))
        COMBOS.append(dict(
            k=row['ksize'],
            tfidf_min=row['tfidf_min'] if row['tfidf_min'] > 0 else None,
            bh_max=row['bh_max']       if row['bh_max']   < 1 else None,
            pval_max=row['pval_max']   if row['pval_max'] < 1 else None,
            score_col=score_col, ascending=ascending,
            label=f"k={row['ksize']} {row['filter_label']} ({row['score_label']})",
        ))

    combo_results = {}
    for cfg in COMBOS:
        ev  = load_eval(cfg['k'], columns=EVAL_USECOLS)
        ev  = add_composite_scores_scope(ev)
        fdf = ev
        if cfg['tfidf_min']: fdf = fdf.filter(pl.col('query_tfidf') > cfg['tfidf_min'])
        if cfg['bh_max']:    fdf = fdf.filter(pl.col('bh') < cfg['bh_max'])
        if cfg['pval_max']:  fdf = fdf.filter(pl.col('poisson_pvalue') < cfg['pval_max'])
        rocx     = eval_tsv_to_rocx(fdf, score_col=cfg['score_col'],
                                     ascending=cfg['ascending'],
                                     exclude_gray_zone=exclude_gray_zone)
        rocx_cov = rocx.filter(pl.col('NAME').is_in(ref_queries))
        _, _, auc_cov = sensitivity_stats(rocx_cov, 'SFAM')
        km_det   = detected(rocx_cov)
        km_uniq  = km_det - base_det
        fs_auc_shared = auc_restricted(foldseek, km_det)
        combo_results[cfg['label']] = dict(
            rocx_cov=rocx_cov, km_det=km_det, km_uniq=km_uniq,
            auc_cov=auc_cov, fs_auc_shared=fs_auc_shared, label=cfg['label'])
        del ev, fdf, rocx; gc.collect()

    return COMBOS, combo_results, base_det, fs_cls


def plot_class_enrichment(COMBOS: list, combo_results: dict, fs_cls: dict,
                           fig_prefix: str) -> None:
    all_cls  = sorted(set(list(fs_cls) +
                          [c for r in combo_results.values()
                           for c in class_dist(r['km_det'], r['rocx_cov'])]))
    desc_map = {c: SCOPE_CLASS_DESCRIPTIONS.get(c, c) for c in all_cls}
    x = np.arange(len(all_cls))
    n_combos  = len(COMBOS)
    total_bars = n_combos * 2 + 1
    w = 0.75 / total_bars
    fig, ax = plt.subplots(figsize=(14, 5))
    colors_uniq = ['tomato',   'darkorange', 'firebrick']
    colors_all  = ['salmon',   'peachpuff',  'lightcoral']
    for ci, cfg in enumerate(COMBOS):
        r = combo_results[cfg['label']]
        u_cls = class_dist(r['km_uniq'], r['rocx_cov'])
        a_cls = class_dist(r['km_det'],  r['rocx_cov'])
        offset_u = (ci * 2)     * w - (total_bars / 2) * w + w / 2
        offset_a = (ci * 2 + 1) * w - (total_bars / 2) * w + w / 2
        ax.bar(x + offset_u, [u_cls.get(c,(0,0))[1]*100 for c in all_cls],
               w, color=colors_uniq[ci], alpha=0.9, label=f'{cfg["label"]} — KM unique')
        ax.bar(x + offset_a, [a_cls.get(c,(0,0))[1]*100 for c in all_cls],
               w, color=colors_all[ci],  alpha=0.7, label=f'{cfg["label"]} — KM all')
    offset_fs = (n_combos * 2) * w - (total_bars / 2) * w + w / 2
    ax.bar(x + offset_fs, [fs_cls.get(c,(0,0))[1]*100 for c in all_cls],
           w, color='steelblue', alpha=0.8, label='FoldSeek all')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n{desc_map[c][:20]}' for c in all_cls], fontsize=8)
    ax.set_ylabel('% of detected queries in SCOPe class', fontsize=11)
    ax.set_title('SCOPe class enrichment: KmerSeek unique vs FoldSeek', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{fig_prefix}_unique_families_by_class.png', dpi=150, bbox_inches='tight')
    plt.show()


# ---------------------------------------------------------------------------
# Section 11: Summary AUC table
# ---------------------------------------------------------------------------

def print_auc_comparison_table(ref_queries: set, foldseek, tea_all,
                                 exclude_gray_zone: bool) -> None:
    combos = [
        ('n_queries ≥ 150', {
            'ksize': 27, 'filter_label': 'pval<0.001', 'score_label': 'Jaccard',
            'tfidf_min': None, 'bh_max': None, 'pval_max': 1e-3,
            'n_queries': 156, 'coverage': 0.014,
        }),
        ('most queries, beats FoldSeek', {
            'ksize': 24, 'filter_label': 'pval<1e-05', 'score_label': 'Jaccard',
            'tfidf_min': None, 'bh_max': None, 'pval_max': 1e-5,
            'n_queries': 268, 'coverage': 0.024,
        }),
    ]

    def km_auc_row(row):
        score_col, ascending = next(
            (c, a) for c, a, l in THRESH_METRICS if l == row['score_label'])
        ev = load_eval(row['ksize'], columns=EVAL_USECOLS)
        ev = add_composite_scores_scope(ev)
        fdf = ev
        if row.get('tfidf_min'): fdf = fdf.filter(pl.col('query_tfidf') > row['tfidf_min'])
        if row.get('bh_max'):    fdf = fdf.filter(pl.col('bh') < row['bh_max'])
        if row.get('pval_max'):  fdf = fdf.filter(pl.col('poisson_pvalue') < row['pval_max'])
        rocx     = eval_tsv_to_rocx(fdf, score_col=score_col, ascending=ascending,
                                     exclude_gray_zone=exclude_gray_zone)
        rocx_cov = rocx.filter(pl.col('NAME').is_in(ref_queries))
        aucs = {level: (sensitivity_stats(rocx_cov, level)[2] if rocx_cov.height > 0 else 0.0)
                for level in ('FAM', 'SFAM', 'FOLD')}
        del ev, fdf, rocx, rocx_cov
        return aucs

    print('=== AUC COMPARISON TABLE (covered-query AUC) ===\n')
    print(f'{"Method":<60} {"FAM AUC":>9} {"SFAM AUC":>9} {"FOLD AUC":>9}  n_queries')
    print('-' * 105)
    for name, df in [('FoldSeek', foldseek), ('TEA', tea_all)]:
        _, _, fam  = sensitivity_stats(df, 'FAM')
        _, _, sfam = sensitivity_stats(df, 'SFAM')
        _, _, fold = sensitivity_stats(df, 'FOLD')
        print(f'{name:<60} {fam:>9.4f} {sfam:>9.4f} {fold:>9.4f}  {df["NAME"].n_unique():,}')
    print()
    for suffix, row in combos:
        label = f'KmerSeek ({suffix}): k={row["ksize"]} {row["filter_label"]} {row["score_label"]}'
        print(f'  computing {label} ...')
        aucs = km_auc_row(row)
        print(f'{label:<60} {aucs["FAM"]:>9.4f} {aucs["SFAM"]:>9.4f} {aucs["FOLD"]:>9.4f}'
              f'  {row["n_queries"]:,}  ({row["coverage"]:.1%})')
    gc.collect()
