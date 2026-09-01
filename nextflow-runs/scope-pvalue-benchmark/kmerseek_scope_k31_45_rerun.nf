#!/usr/bin/env nextflow

/*
 * Nextflow pipeline: SCOPe p-value benchmark — k=31–45 rerun
 *
 * Runs only k=31–45 with two key fixes over the original pipeline:
 *
 * 1. threshold=0.0 (was 0.01)
 *    At k=35+ with HP encoding, proteins need to share ≥1% of their 35-mers
 *    to pass threshold=0.01.  That yields only ~200 pairs (almost nothing).
 *    Setting threshold=0.0 reports ALL non-zero containment pairs.
 *
 * 2. Correct Bonferroni (was multiplied by n_reported_pairs)
 *    The original pipeline computed bonferroni = pvalue × n_reported_pairs,
 *    which is ~137,000× too permissive for the SCOPe40 all-vs-all benchmark.
 *    Correct: bonferroni = pvalue × N_SCOPE40_PAIRS = pvalue × 230,326,152.
 *
 * Assumes:
 *   - scope_domains.tsv already exists in params.outdir (from original run)
 *   - kmerseek binary exists at params.kmerseek
 *   - k=15–30 parquets already exist; this adds k=31–45
 *
 * Usage:
 *   cd nextflow-runs/scope-pvalue-benchmark
 *   nextflow run kmerseek_scope_k31_45_rerun.nf
 */

params.fasta_gd = "${System.getProperty('user.home')}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir = "${System.getProperty('user.home')}/data/scope/results-scope-pvalue-benchmark"
params.kmerseek = "${System.getProperty('user.home')}/code/kmerseek/target/release/kmerseek-rust"
params.python = "/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3"

params.bcl2_id = "d7jgwa1"
params.ced9_id = "d1ohua_"

// threshold=0.0: report ALL non-zero containment pairs.
// At k=35 HP, threshold=0.01 produced only 217 rows (basically BCL2/Ced9 only).
// threshold=0.0 will capture all pairs with any shared k-mer.
params.threshold = 0.0

// Total all-vs-all comparisons for correct Bonferroni: 15177 × 15176
params.n_comparisons = 230326152


process parseScopeHeaders {
    publishDir params.outdir, mode: 'copy'

    input:
    path fasta

    output:
    path 'scope_domains.tsv'

    script:
    """
    #!/usr/bin/env python3
    import re, sys

    # Header: >d1dlwa_ a.1.1.1 (A:) Protozoan/bacterial hemoglobin {Ciliate ...} [TaxId: 5885]
    pat = re.compile(r'^>(\\S+)\\s+(\\S+)\\s+\\([^)]*\\)\\s+(.*?)\\s+\\{(.*?)\\}\\s*\$')

    n = 0
    with open('${fasta}') as fi, open('scope_domains.tsv', 'w') as fo:
        fo.write('domain_id\\tscop_id\\tscop_class\\tscop_fold\\tscop_superfamily\\tscop_family\\tprotein_name\\tspecies\\n')
        for line in fi:
            if not line.startswith('>'):
                continue
            m = pat.match(line.rstrip())
            if m:
                domain_id, scop_id, pname, species = m.group(1), m.group(2), m.group(3), m.group(4)
            else:
                parts = line[1:].rstrip().split(None, 2)
                domain_id = parts[0] if parts else ''
                scop_id   = parts[1] if len(parts) > 1 else ''
                pname, species = '', ''

            sp = scop_id.split('.')
            scop_class = sp[0]            if len(sp) >= 1 else scop_id
            scop_fold  = '.'.join(sp[:2]) if len(sp) >= 2 else scop_id
            scop_sf    = '.'.join(sp[:3]) if len(sp) >= 3 else scop_id
            scop_fam   = scop_id

            fo.write(f'{domain_id}\\t{scop_id}\\t{scop_class}\\t{scop_fold}\\t'
                     f'{scop_sf}\\t{scop_fam}\\t{pname.replace(chr(9)," ")}\\t{species.replace(chr(9)," ")}\\n')
            n += 1
    print(f'Parsed {n} SCOP domains', file=sys.stderr)
    """
}


process indexDatabase {
    tag "hp_k${ksize}"
    publishDir "${params.outdir}", mode: 'copy', pattern: '*.index.log.gz'
    publishDir "${params.outdir}/indices", mode: 'copy', pattern: '*.rocksdb', type: 'dir'

    input:
    tuple path(fasta), val(ksize)

    output:
    tuple val(ksize), path("${fasta}.hp.k${ksize}.scaled1.kmerseek.rocksdb", type: 'dir')
    path "${fasta}.hp.k${ksize}.scaled1.kmerseek.index.log.gz"

    script:
    def log_file = "${fasta}.hp.k${ksize}.scaled1.kmerseek.index.log"
    """
    echo "=== Indexing: hp k=${ksize} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    _START=\$(date +%s)
    /usr/bin/time -l ${params.kmerseek} index \\
        --encoding hp --ksize ${ksize} --scaled 1 --input ${fasta} \\
        2>&1 | tee -a ${log_file}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Elapsed: \$(( \$(date +%s) - _START )) seconds" | tee -a ${log_file}
    gzip ${log_file}
    """
}


process searchAllVsAll {
    tag "hp_k${ksize}"
    publishDir params.outdir, mode: 'copy', pattern: '*.results.csv.gz'
    publishDir params.outdir, mode: 'copy', pattern: '*.search.log.gz'

    input:
    tuple val(ksize), path(index)

    output:
    tuple val(ksize), path("scope40.hp.k${ksize}.results.csv.gz")
    path "scope40.hp.k${ksize}.search.log.gz"

    script:
    def output_csv = "scope40.hp.k${ksize}.results.csv"
    def log_file = "scope40.hp.k${ksize}.search.log"
    """
    echo "=== Searching all-vs-all: hp k=${ksize} (threshold=${params.threshold}) ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    _START=\$(date +%s)
    /usr/bin/time -l ${params.kmerseek} search \\
        --encoding hp --ksize ${ksize} \\
        --threshold ${params.threshold} \\
        --query-is-index --query ${index} --target ${index} \\
        > ${output_csv} 2>> ${log_file}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Elapsed: \$(( \$(date +%s) - _START )) seconds" | tee -a ${log_file}
    echo "Results: \$(wc -l < ${output_csv}) rows" | tee -a ${log_file}
    gzip ${log_file} ${output_csv}
    """
}


process evaluateScopeHierarchy {
    tag "hp_k${ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(ksize), path(results_csv_gz), path(scop_domains)

    output:
    tuple val(ksize), path("scope_eval.hp.k${ksize}.parquet")
    path "scope_eval.hp.k${ksize}.tsv.gz"
    path "scope_eval.hp.k${ksize}.summary.txt.gz"
    path "scope_eval.hp.k${ksize}.pr_data.tsv.gz"
    path "scope_eval.hp.k${ksize}.bcl2ced9.tsv.gz"
    path "scope_eval.hp.k${ksize}.*.png", optional: true

    script:
    """
    #!/opt/conda/envs/2025-kmerseek-analysis/bin/python3
    import os, sys, gzip, json, time as _time
    import polars as pl
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _t0     = _time.time()
    ksize   = ${ksize}
    bcl2_id = '${params.bcl2_id}'
    ced9_id = '${params.ced9_id}'
    # Correct Bonferroni denominator: all possible ordered non-self pairs
    N_COMPARISONS = ${params.n_comparisons}

    results_gz = '${results_csv_gz}'
    if os.path.getsize(results_gz) == 0:
        print(f'Warning: empty results for k={ksize}', file=sys.stderr)
        for stem in ['parquet', 'tsv.gz', 'summary.txt.gz', 'pr_data.tsv.gz', 'bcl2ced9.tsv.gz']:
            open(f'scope_eval.hp.k{ksize}.{stem}', 'wb').close()
        sys.exit(0)

    scop = pl.read_csv('${scop_domains}', separator='\\t')

    raw = (
        pl.scan_csv(results_gz)
        .with_columns([
            pl.col('query_name').str.split(' ').list.get(0).alias('query_domain'),
            pl.col('target_name').str.split(' ').list.get(0).alias('target_domain'),
        ])
        .filter(pl.col('query_domain') != pl.col('target_domain'))
        .collect()
    )
    # Keep best hit per pair (lowest p-value, then highest containment)
    df = (
        raw
        .sort(['query_domain', 'target_domain', 'poisson_pvalue', 'containment'],
              descending=[False, False, False, True])
        .unique(subset=['query_domain', 'target_domain'], keep='first')
    )
    n_pairs = len(df)
    print(f'k={ksize}: {len(raw):,} raw rows -> {n_pairs:,} unique pairs', file=sys.stderr)

    # Join SCOP hierarchy
    scop_q = scop.rename({c: f'q_{c}' for c in scop.columns}).rename({'q_domain_id': 'query_domain'})
    scop_t = scop.rename({c: f't_{c}' for c in scop.columns}).rename({'t_domain_id': 'target_domain'})
    df = df.join(scop_q, on='query_domain', how='left').join(scop_t, on='target_domain', how='left')

    LEVELS = ['class', 'fold', 'superfamily', 'family']
    for lvl in LEVELS:
        df = df.with_columns(
            (pl.col(f'q_scop_{lvl}') == pl.col(f't_scop_{lvl}')).alias(f'same_{lvl}')
        )

    # MHT corrections
    # IMPORTANT: Bonferroni uses N_COMPARISONS (all possible pairs), NOT n_pairs (reported only).
    pvals = df['poisson_pvalue'].fill_null(1.0).to_numpy()
    n     = len(pvals)

    def bh_adj(p):
        idx = np.argsort(p); sp = p[idx]; m = len(p)
        adj = np.minimum(1.0, sp * m / np.arange(1, m + 1))
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(m); out[idx] = adj; return out

    def by_adj(p):
        idx = np.argsort(p); sp = p[idx]; m = len(p)
        c   = np.sum(1.0 / np.arange(1, m + 1))
        adj = np.minimum(1.0, sp * m * c / np.arange(1, m + 1))
        adj = np.minimum.accumulate(adj[::-1])[::-1]
        out = np.empty(m); out[idx] = adj; return out

    df = df.with_columns([
        # bonferroni: WRONG (n_reported), kept for backward compatibility
        pl.Series('bonferroni',         np.minimum(pvals * n,              1.0)),
        # bonferroni_correct: uses all possible pairs — this is statistically valid
        pl.Series('bonferroni_correct', np.minimum(pvals * N_COMPARISONS,  1.0)),
        pl.Series('bh',                 bh_adj(pvals)),
        pl.Series('by',                 by_adj(pvals)),
    ])

    # Save parquet (primary output, replaces tsv for k=31+)
    df.write_parquet(f'scope_eval.hp.k{ksize}.parquet', compression='zstd')

    # Save gzipped TSV for compatibility
    import io
    buf = io.BytesIO()
    df.write_csv(buf, separator='\\t')
    with gzip.open(f'scope_eval.hp.k{ksize}.tsv.gz', 'wb') as fout:
        fout.write(buf.getvalue())

    # BCL2/Ced9 spotlight
    bcl2_rows = df.filter(
        ((pl.col('query_domain') == bcl2_id) & (pl.col('target_domain') == ced9_id)) |
        ((pl.col('query_domain') == ced9_id) & (pl.col('target_domain') == bcl2_id))
    )
    spot_cols = [c for c in [
        'query_domain', 'target_domain', 'q_scop_id', 't_scop_id',
        'q_scop_family', 't_scop_family',
        'same_class', 'same_fold', 'same_superfamily', 'same_family',
        'containment', 'jaccard', 'n_intersecting_hashes',
        'poisson_pvalue', 'bonferroni', 'bonferroni_correct', 'bh', 'by',
        'query_subseq', 'target_subseq', 'moltype_seq',
    ] if c in df.columns]
    bcl2_buf = io.BytesIO()
    bcl2_rows.select(spot_cols).write_csv(bcl2_buf, separator='\\t')
    with gzip.open(f'scope_eval.hp.k{ksize}.bcl2ced9.tsv.gz', 'wb') as fout:
        fout.write(bcl2_buf.getvalue())
    if len(bcl2_rows) > 0:
        r = bcl2_rows.row(0, named=True)
        print(f"k={ksize} BCL2/Ced9: p={r.get('poisson_pvalue','NA'):.2e} "
              f"bh={r.get('bh','NA'):.2e} bonf_correct={r.get('bonferroni_correct','NA'):.2e} "
              f"contain={r.get('containment','NA'):.4f}", file=sys.stderr)
    else:
        print(f'k={ksize} BCL2/Ced9 pair NOT FOUND', file=sys.stderr)

    # Precision-recall evaluation
    P_METHODS = [
        ('raw',               'poisson_pvalue'),
        ('bonferroni_correct','bonferroni_correct'),
        ('bh',                'bh'),
        ('by',                'by'),
    ]
    alpha   = 0.05
    pr_rows = []
    mht_sum = {}
    lines   = [
        f'=== SCOPe Hierarchy Benchmark  HP k={ksize} ===\\n',
        f'Total pairs (reported):      {n_pairs:,}\\n',
        f'Total comparisons (n^2):     {N_COMPARISONS:,}\\n',
        f'Bonferroni alpha threshold:  {alpha / N_COMPARISONS:.2e}\\n\\n',
    ]

    for lvl in LEVELS:
        is_pos = df[f'same_{lvl}'].fill_null(False).to_numpy()
        n_pos  = int(is_pos.sum())
        lines.append(f'--- {lvl.upper():<15s} positives={n_pos:,} negatives={n_pairs-n_pos:,} ---\\n')
        mht_sum[lvl] = {}

        for mname, col in P_METHODS:
            scores   = df[col].fill_null(1.0).to_numpy()
            sidx     = np.argsort(scores)
            spos     = is_pos[sidx]
            sscores  = scores[sidx]
            cum_tp   = np.cumsum(spos)
            cum_fp   = np.cumsum(~spos)
            recall   = cum_tp / max(n_pos, 1)
            prec_arr = cum_tp / np.arange(1, n_pairs + 1)

            step = max(1, n_pairs // 5000)
            for i in range(0, n_pairs, step):
                pr_rows.append({
                    'ksize': ksize, 'level': lvl, 'method': mname,
                    'threshold': float(sscores[i]),
                    'recall':    float(recall[i]),
                    'precision': float(prec_arr[i]),
                    'n_tp': int(cum_tp[i]), 'n_fp': int(cum_fp[i]),
                })

            rej   = scores <= alpha
            rej_n = int(rej.sum())
            tp_n  = int((rej & is_pos).sum())
            prec_a = round(tp_n / rej_n if rej_n else 0.0, 4)
            rec_a  = round(tp_n / n_pos if n_pos else 0.0, 4)
            mht_sum[lvl][mname] = {'rejected': rej_n, 'TP': tp_n, 'precision': prec_a, 'recall': rec_a}
            lines.append(f'  {mname:<20s}: {rej_n:>8,} rejected  TP={tp_n:>7,}  prec={prec_a:.4f}  rec={rec_a:.4f}\\n')
        lines.append('\\n')

    if len(bcl2_rows) > 0:
        lines.append('=== BCL2/Ced9 ===\\n')
        r = bcl2_rows.row(0, named=True)
        for col in ['poisson_pvalue','bonferroni','bonferroni_correct','bh','by',
                    'containment','jaccard','n_intersecting_hashes',
                    'same_class','same_fold','same_superfamily','same_family']:
            if col in r:
                lines.append(f'  {col:<28s}: {r[col]}\\n')

    eval_elapsed = _time.time() - _t0
    summary_json = {
        'ksize': ksize, 'encoding': 'hp', 'n_pairs': n_pairs,
        'n_comparisons': N_COMPARISONS, 'threshold': ${params.threshold},
        'eval_seconds': round(eval_elapsed, 1),
        'mht': mht_sum,
    }
    lines += [f'\\nElapsed: {eval_elapsed:.1f} seconds\\n',
              '\\n=== JSON SUMMARY ===\\n', json.dumps(summary_json, indent=2) + '\\n']

    summary_bytes = ''.join(lines).encode()
    with gzip.open(f'scope_eval.hp.k{ksize}.summary.txt.gz', 'wb') as fout:
        fout.write(summary_bytes)

    pr_buf = io.BytesIO()
    pl.DataFrame(pr_rows).write_csv(pr_buf, separator='\\t')
    with gzip.open(f'scope_eval.hp.k{ksize}.pr_data.tsv.gz', 'wb') as fout:
        fout.write(pr_buf.getvalue())

    # Precision-recall plots
    pr_df  = pl.DataFrame(pr_rows)
    colors = {'raw': 'grey', 'bonferroni_correct': 'steelblue', 'bh': 'tomato', 'by': 'goldenrod'}

    for lvl in LEVELS:
        fig, ax = plt.subplots(figsize=(7, 5))
        lvl_df  = pr_df.filter(pl.col('level') == lvl)
        is_pos  = df[f'same_{lvl}'].fill_null(False).to_numpy()
        n_pos_l = int(is_pos.sum())

        for mname, _ in P_METHODS:
            md = lvl_df.filter(pl.col('method') == mname).sort('recall')
            if len(md) == 0:
                continue
            ax.plot(md['recall'].to_numpy(), md['precision'].to_numpy(),
                    label=mname, color=colors.get(mname, 'black'), linewidth=1.5)

        if len(bcl2_rows) > 0:
            r = bcl2_rows.row(0, named=True)
            for mname, col in P_METHODS:
                pv = r.get(col)
                if pv is None: continue
                rej = df[col].fill_null(1.0).to_numpy() <= pv
                tp, tot = int((rej & is_pos).sum()), int(rej.sum())
                if tot > 0 and n_pos_l > 0:
                    ax.plot(tp / n_pos_l, tp / tot, marker='*', markersize=12,
                            zorder=5, color=colors.get(mname, 'black'))

        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'SCOPe {lvl}  |  HP k={ksize}  (threshold={${params.threshold}})')
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'scope_eval.hp.k{ksize}.{lvl}.png', dpi=150)
        plt.close()

    print(f'k={ksize} done in {eval_elapsed:.1f}s', file=sys.stderr)
    """
}


workflow {
    fasta_ch = channel.fromPath(params.fasta_gd)
    scop_domains = parseScopeHeaders(fasta_ch)

    hp_ksizes = channel.of(31..45)

    indexed = indexDatabase(fasta_ch.combine(hp_ksizes))
    search_outputs = searchAllVsAll(indexed[0])
    eval_outputs = evaluateScopeHierarchy(search_outputs[0].combine(scop_domains))

    eval_outputs[0].subscribe { ksize, parquet ->
        println("Evaluated: HP k=${ksize} -> ${parquet}")
    }
}
