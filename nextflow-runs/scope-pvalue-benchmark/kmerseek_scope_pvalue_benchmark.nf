#!/usr/bin/env nextflow

/*
 * Nextflow pipeline: SCOPe p-value benchmark
 *
 * Evaluates kmerseek's ability to detect structurally similar proteins across
 * the SCOP hierarchy (class, fold, superfamily, family) using Poisson p-value
 * (raw, Bonferroni, BH) as the ranking metric.
 *
 * SCOP hierarchy from the 4-component SCOP ID (e.g. "f.1.4.1"):
 *   class       = f          (secondary structure content)
 *   fold        = f.1        (structural similarity, no homology implied)
 *   superfamily = f.1.4      (common ancestor inferred)
 *   family      = f.1.4.1    (sequence similarity + function)
 *
 * BCL2/Ced9 focal pair (both f.1.4.1 = same family):
 *   d7jgwa1  Apoptosis regulator Bcl-xL  {Human}
 *   d1ohua_  Apoptosis regulator ced-9   {C. elegans}
 *
 * Usage:
 *   nextflow run kmerseek_scope_pvalue_benchmark.nf
 */

params.fasta_gd = "${System.getProperty('user.home')}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir   = "${System.getProperty('user.home')}/data/scope/results-scope-pvalue-benchmark"
params.kmerseek = "${System.getProperty('user.home')}/code/kmerseek/target/release/kmerseek-rust"

params.bcl2_id   = "d7jgwa1"
params.ced9_id   = "d1ohua_"

// Minimum containment threshold for search.
// At small k (e.g. k=15 HP) with threshold=0.0, nearly every pair shares a k-mer
// → ~115M candidate pairs → OOM. 0.01 keeps only pairs with ≥1% containment.
params.threshold = 0.01


process buildRelease {
    output:
    path 'kmerseek-rust'

    script:
    def kmerseek_dir = "${System.getProperty('user.home')}/code/kmerseek"
    """
    WORK_DIR=\$PWD
    cd ${kmerseek_dir}
    cargo build --release
    cp ${kmerseek_dir}/target/release/kmerseek-rust \$WORK_DIR/kmerseek-rust
    """
}


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

    # Header: >d1dlwa_ a.1.1.1 (A:) Protozoan/bacterial hemoglobin {Ciliate ... [TaxId: 5885]}
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
    publishDir "${params.outdir}/indices", mode: 'copy', pattern: '*.rocksdb', type: 'dir'
    publishDir params.outdir,             mode: 'copy', pattern: '*.index.log'

    input:
    path  kmerseek
    tuple path(fasta), val(ksize)

    output:
    tuple val(ksize), path("${fasta}.hp.k${ksize}.scaled1.kmerseek.rocksdb", type: 'dir')
    path  "${fasta}.hp.k${ksize}.scaled1.kmerseek.index.log"

    script:
    def log_file = "${fasta}.hp.k${ksize}.scaled1.kmerseek.index.log"
    """
    echo "=== Indexing: hp k=${ksize} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    _START=\$(date +%s)
    /usr/bin/time -l ${kmerseek} index \\
        --encoding hp --ksize ${ksize} --scaled 1 --input ${fasta} \\
        2>&1 | tee -a ${log_file}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Elapsed: \$(( \$(date +%s) - _START )) seconds" | tee -a ${log_file}
    """
}


process searchAllVsAll {
    tag "hp_k${ksize}"
    publishDir params.outdir, mode: 'copy', pattern: '*.csv'
    publishDir params.outdir, mode: 'copy', pattern: '*.search.log'

    input:
    path  kmerseek
    tuple val(ksize), path(index)

    output:
    tuple val(ksize), path("scope40.hp.k${ksize}.results.csv")
    path  "scope40.hp.k${ksize}.search.log"

    script:
    def output_csv = "scope40.hp.k${ksize}.results.csv"
    def log_file   = "scope40.hp.k${ksize}.search.log"
    """
    echo "=== Searching all-vs-all: hp k=${ksize} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    _START=\$(date +%s)
    /usr/bin/time -l ${kmerseek} search \\
        --encoding hp --ksize ${ksize} \\
        --threshold ${params.threshold} \\
        --query-is-index --query ${index} --target ${index} \\
        > ${output_csv} 2>> ${log_file}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Elapsed: \$(( \$(date +%s) - _START )) seconds" | tee -a ${log_file}
    echo "Results: \$(wc -l < ${output_csv}) rows" | tee -a ${log_file}
    """
}


process evaluateScopeHierarchy {
    tag "hp_k${ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(ksize), path(results_csv), path(scop_domains)

    output:
    tuple val(ksize), path("scope_eval.hp.k${ksize}.tsv")
    path "scope_eval.hp.k${ksize}.summary.txt"
    path "scope_eval.hp.k${ksize}.pr_data.tsv"
    path "scope_eval.hp.k${ksize}.bcl2ced9.tsv"
    path "scope_eval.hp.k${ksize}.*.png", optional: true

    script:
    """
    #!/usr/bin/env python3
    import os, sys, json, time as _time
    import polars as pl
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    _t0     = _time.time()
    ksize   = ${ksize}
    bcl2_id = '${params.bcl2_id}'
    ced9_id = '${params.ced9_id}'

    if os.path.getsize('${results_csv}') == 0:
        print(f'Warning: empty results for k={ksize}', file=sys.stderr)
        for stem in ['tsv', 'summary.txt', 'pr_data.tsv', 'bcl2ced9.tsv']:
            open(f'scope_eval.hp.k{ksize}.{stem}', 'w').close()
        sys.exit(0)

    scop = pl.read_csv('${scop_domains}', separator='\\t')

    # One row per matched region in the CSV — deduplicate to best p-value per pair
    raw = (
        pl.scan_csv('${results_csv}')
        .with_columns([
            pl.col('query_name').str.split(' ').list.get(0).alias('query_domain'),
            pl.col('target_name').str.split(' ').list.get(0).alias('target_domain'),
        ])
        .filter(pl.col('query_domain') != pl.col('target_domain'))
        .collect()
    )
    df = (
        raw
        .sort(['query_domain', 'target_domain', 'poisson_pvalue', 'containment'],
              descending=[False, False, False, True])
        .unique(subset=['query_domain', 'target_domain'], keep='first')
    )
    n_pairs = len(df)
    print(f'k={ksize}: {len(raw):,} raw rows -> {n_pairs:,} unique pairs', file=sys.stderr)

    # Join SCOP hierarchy for both sides
    scop_q = scop.rename({c: f'q_{c}' for c in scop.columns}).rename({'q_domain_id': 'query_domain'})
    scop_t = scop.rename({c: f't_{c}' for c in scop.columns}).rename({'t_domain_id': 'target_domain'})
    df = df.join(scop_q, on='query_domain', how='left').join(scop_t, on='target_domain', how='left')

    # SCOP level labels (broad -> specific)
    LEVELS = ['class', 'fold', 'superfamily', 'family']
    for lvl in LEVELS:
        df = df.with_columns(
            (pl.col(f'q_scop_{lvl}') == pl.col(f't_scop_{lvl}')).alias(f'same_{lvl}')
        )

    # MHT corrections on poisson_pvalue
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
        pl.Series('bonferroni', np.minimum(pvals * n, 1.0)),
        pl.Series('bh',         bh_adj(pvals)),
        pl.Series('by',         by_adj(pvals)),
    ])

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
        'poisson_pvalue', 'bonferroni', 'bh', 'by',
        'query_subseq', 'target_subseq', 'moltype_seq',
    ] if c in df.columns]
    bcl2_rows.select(spot_cols).write_csv(f'scope_eval.hp.k{ksize}.bcl2ced9.tsv', separator='\\t')
    if len(bcl2_rows) > 0:
        r = bcl2_rows.row(0, named=True)
        print(f"k={ksize} BCL2/Ced9: p={r.get('poisson_pvalue','NA'):.2e} "
              f"bh={r.get('bh','NA'):.2e} contain={r.get('containment','NA'):.4f}", file=sys.stderr)
    else:
        print(f'k={ksize} BCL2/Ced9 pair NOT FOUND', file=sys.stderr)

    # Precision-recall evaluation across SCOP levels x correction methods
    P_METHODS = [('raw','poisson_pvalue'), ('bonferroni','bonferroni'), ('bh','bh'), ('by','by')]
    alpha     = 0.05
    pr_rows   = []
    mht_sum   = {}
    lines     = [f'=== SCOPe Hierarchy Benchmark  HP k={ksize} ===\\n',
                 f'Total pairs: {n_pairs:,}\\n\\n']

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

            rej = scores <= alpha
            rej_n, tp_n = int(rej.sum()), int((rej & is_pos).sum())
            prec_a = round(tp_n / rej_n if rej_n else 0.0, 4)
            rec_a  = round(tp_n / n_pos if n_pos else 0.0, 4)
            mht_sum[lvl][mname] = {'rejected': rej_n, 'TP': tp_n, 'precision': prec_a, 'recall': rec_a}
            lines.append(f'  {mname:<12s}: {rej_n:>8,} rejected  TP={tp_n:>7,}  prec={prec_a:.4f}  rec={rec_a:.4f}\\n')
        lines.append('\\n')

    if len(bcl2_rows) > 0:
        lines.append('=== BCL2/Ced9 ===\\n')
        r = bcl2_rows.row(0, named=True)
        for col in ['poisson_pvalue','bonferroni','bh','by','containment','jaccard','n_intersecting_hashes',
                    'same_class','same_fold','same_superfamily','same_family']:
            if col in r:
                lines.append(f'  {col:<26s}: {r[col]}\\n')

    eval_elapsed = _time.time() - _t0
    summary_json = {
        'ksize': ksize, 'encoding': 'hp', 'n_pairs': n_pairs,
        'eval_seconds': round(eval_elapsed, 1),
        'mht': mht_sum,
    }
    lines += [f'\\nElapsed: {eval_elapsed:.1f} seconds\\n',
              '\\n=== JSON SUMMARY ===\\n', json.dumps(summary_json, indent=2) + '\\n']
    with open(f'scope_eval.hp.k{ksize}.summary.txt', 'w') as f:
        f.writelines(lines)

    pl.DataFrame(pr_rows).write_csv(f'scope_eval.hp.k{ksize}.pr_data.tsv', separator='\\t')
    df.write_csv(f'scope_eval.hp.k{ksize}.tsv', separator='\\t')

    # Precision-recall plots (one per SCOP level)
    pr_df  = pl.DataFrame(pr_rows)
    colors = {'raw': 'grey', 'bonferroni': 'steelblue', 'bh': 'tomato', 'by': 'goldenrod'}

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

        # Star marker for BCL2/Ced9 pair location on each curve
        if len(bcl2_rows) > 0:
            r = bcl2_rows.row(0, named=True)
            for mname, col in P_METHODS:
                pv = r.get(col)
                if pv is None:
                    continue
                rej = df[col].fill_null(1.0).to_numpy() <= pv
                tp, tot = int((rej & is_pos).sum()), int(rej.sum())
                if tot > 0 and n_pos_l > 0:
                    ax.plot(tp / n_pos_l, tp / tot, marker='*', markersize=12,
                            zorder=5, color=colors.get(mname, 'black'))

        ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
        ax.set_title(f'SCOPe {lvl}  |  HP k={ksize}  (* = BCL2/Ced9)')
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f'scope_eval.hp.k{ksize}.{lvl}.png', dpi=150)
        plt.close()

    print(f'k={ksize} done', file=sys.stderr)
    """
}


process aggregateScopeResults {
    publishDir params.outdir, mode: 'copy'

    input:
    path summaries   // eval summary.txt files (collected)
    path logs        // index + search .log files (collected)

    output:
    path 'scope_benchmark_summary.tsv'
    path 'scope_benchmark_summary.json'
    path 'scope_benchmark_timing.tsv'

    script:
    """
    #!/usr/bin/env python3
    import glob, json, re

    # ── Parse evaluation summaries ──────────────────────────────────────────
    results = []
    for f in sorted(glob.glob('*.summary.txt')):
        m = re.search(r'k(\\d+)', f)
        if not m:
            continue
        with open(f) as fh:
            content = fh.read()
        jm = re.search(r'=== JSON SUMMARY ===\\n(\\{[\\s\\S]+\\})', content)
        if jm:
            try:
                results.append(json.loads(jm.group(1)))
                continue
            except json.JSONDecodeError:
                pass
        results.append({'ksize': int(m.group(1))})

    results.sort(key=lambda x: x.get('ksize', 0))

    LEVELS  = ['class', 'fold', 'superfamily', 'family']
    METHODS = ['raw', 'bonferroni', 'bh', 'by']
    headers = ['ksize', 'n_pairs']
    for lvl in LEVELS:
        for mth in METHODS:
            headers += [f'{lvl}_{mth}_rejected', f'{lvl}_{mth}_precision', f'{lvl}_{mth}_recall']

    with open('scope_benchmark_summary.tsv', 'w') as f:
        f.write('\\t'.join(headers) + '\\n')
        for r in results:
            row = [str(r.get('ksize', '')), str(r.get('n_pairs', ''))]
            for lvl in LEVELS:
                lm = r.get('mht', {}).get(lvl, {})
                for mth in METHODS:
                    s = lm.get(mth, {})
                    row += [str(s.get('rejected','')), f"{s.get('precision',0):.4f}", f"{s.get('recall',0):.4f}"]
            f.write('\\t'.join(row) + '\\n')

    with open('scope_benchmark_summary.json', 'w') as f:
        json.dump({'encoding': 'hp', 'results': results}, f, indent=2)

    # ── Parse timing from log files + eval summaries ────────────────────────
    # Log files for index and search contain lines: "Elapsed: N seconds"
    # Eval summaries contain "eval_seconds" in the JSON block.

    def parse_elapsed(path):
        # Return seconds (int) from 'Elapsed: N seconds' line, or None if not found.
        try:
            with open(path) as fh:
                for line in fh:
                    m = re.search(r'Elapsed:\\s+(\\d+)\\s+seconds', line)
                    if m:
                        return int(m.group(1))
        except FileNotFoundError:
            pass
        return None

    # Build ksize -> timing dict from eval summaries (eval_seconds already parsed above)
    eval_timing = {r['ksize']: r.get('eval_seconds') for r in results}

    # Gather index/search timing from any .log files present
    index_timing  = {}
    search_timing = {}
    for f in glob.glob('*.index.log'):
        m = re.search(r'\\.k(\\d+)\\.', f)
        if m:
            index_timing[int(m.group(1))] = parse_elapsed(f)
    for f in glob.glob('*.search.log'):
        m = re.search(r'\\.k(\\d+)\\.', f)
        if m:
            search_timing[int(m.group(1))] = parse_elapsed(f)

    all_ksizes = sorted({r['ksize'] for r in results} | set(index_timing) | set(search_timing))

    with open('scope_benchmark_timing.tsv', 'w') as f:
        f.write('ksize\\tindex_seconds\\tsearch_seconds\\teval_seconds\\ttotal_seconds\\n')
        for k in all_ksizes:
            idx  = index_timing.get(k)
            srch = search_timing.get(k)
            evl  = eval_timing.get(k)
            tots = [x for x in [idx, srch, evl] if x is not None]
            total = sum(tots) if tots else ''
            f.write(f"{k}\\t{idx or ''}\\t{srch or ''}\\t{evl or ''}\\t{total}\\n")

    print(f'Aggregated {len(results)} k-sizes')
    """
}


workflow {
    kmerseek_bin = buildRelease()

    fasta_ch     = channel.fromPath(params.fasta_gd)
    scop_domains = parseScopeHeaders(fasta_ch)

    hp_ksizes = channel.of(15..50)

    indexed        = indexDatabase(kmerseek_bin, fasta_ch.combine(hp_ksizes))
    search_outputs = searchAllVsAll(kmerseek_bin, indexed[0])
    eval_outputs   = evaluateScopeHierarchy(search_outputs[0].combine(scop_domains))

    // Collect eval summaries + index/search logs for timing aggregation
    all_summaries = eval_outputs[1].collect()
    all_logs      = indexed[1].mix(search_outputs[1]).collect()
    aggregateScopeResults(all_summaries, all_logs)

    eval_outputs[0].subscribe { ksize, tsv ->
        println("Evaluated: HP k=${ksize} -> ${tsv}")
    }
}
