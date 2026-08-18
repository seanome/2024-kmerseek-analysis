#!/usr/bin/env nextflow

/*
 * Nextflow pipeline: kmerseek HP alphabet robustness sweep
 *
 * Benchmarks all HP alphabet variants (named + 10 seeded shuffled controls)
 * against the SCOPe 40% nr dataset across k-sizes 15-45.
 *
 * Alphabets tested:
 *   Named:    hp_thomas_dill, hp_kyte_doolittle, hp_thomas_dill_no_c,
 *             hp_lehninger_plus_c, hp_pbotc_1st_ed, hp_shuffled_control
 *   Seeded:   hp_shuffled_control_1 .. hp_shuffled_control_10
 *             (via --encoding hp_shuffled_control --shuffled-seed N)
 *
 * kmerseek v0.3.0, branch: olgabot/more-hp-alphabets
 * Docker image: kmerseek:0.3.0  (build with the Dockerfile in this directory)
 *
 * Usage:
 *   nextflow run main.nf
 *   nextflow run main.nf --fasta_gd /path/to/scope40.fa --outdir /path/to/results
 */

params.fasta_gd  = "${System.getProperty('user.home')}/data/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.outdir    = "${System.getProperty('user.home')}/data/scope/results-hp-alphabet-sweep"
params.threshold = 0.01   // minimum containment; 0.01 avoids OOM at small k


// Named HP alphabets: [cli_flag, label] pairs.
// CLI flag uses clap's kebab-case; label uses snake_case for filenames.
def NAMED_ENCODINGS = [
    ['hp-thomas-dill',     'hp_thomas_dill'],
    ['hp-kyte-doolittle',  'hp_kyte_doolittle'],
    ['hp-thomas-dill-no-c','hp_thomas_dill_no_c'],
    ['hp-lehninger-plus-c','hp_lehninger_plus_c'],
    ['hp-pbotc-1st-ed',    'hp_pbotc_1st_ed'],
    ['hp-shuffled-control','hp_shuffled_control'],
]

// Seeded shuffled controls: [cli_flag, seed] pairs.
// kmerseek stores the moltype as hp_shuffled_control_N in the index.
def SHUFFLED_SEEDS = (1..10).collect { seed -> ['hp-shuffled-control', seed] }

// k-sizes to sweep.
def KSIZES = (20..40).toList()


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
    tag "${label}_k${ksize}"
    publishDir "${params.outdir}/indices", mode: 'copy', pattern: '*.rocksdb', type: 'dir'
    publishDir params.outdir,             mode: 'copy', pattern: '*.index.log'

    input:
    tuple path(fasta), val(encoding), val(seed), val(ksize), val(label)

    output:
    tuple val(label), val(ksize), path("*.rocksdb", type: 'dir'), val(encoding)
    path "*.index.log"

    script:
    def seed_flag  = seed != null ? "--shuffled-seed ${seed}" : ""
    def db_name    = "${fasta}.${label}.k${ksize}.scaled1.kmerseek.rocksdb"
    def log_file   = "${fasta}.${label}.k${ksize}.scaled1.kmerseek.index.log"
    """
    kmerseek index \\
        --encoding ${encoding} ${seed_flag} \\
        --ksize ${ksize} --scaled 1 \\
        --input ${fasta} \\
        --output ${db_name} \\
        2> ${log_file}
    """
}


process searchAllVsAll {
    tag "${label}_k${ksize}"
    publishDir params.outdir, mode: 'copy', pattern: '*.csv'
    publishDir params.outdir, mode: 'copy', pattern: '*.search.log'

    input:
    tuple val(label), val(ksize), path(index), val(encoding)

    output:
    tuple val(label), val(ksize), path("scope40.${label}.k${ksize}.results.csv")
    path "scope40.${label}.k${ksize}.search.log"

    script:
    def output_csv = "scope40.${label}.k${ksize}.results.csv"
    def log_file   = "scope40.${label}.k${ksize}.search.log"
    """
    kmerseek search \\
        --encoding ${encoding} \\
        --ksize ${ksize} \\
        --threshold ${params.threshold} \\
        --query-is-index --query ${index} --target ${index} \\
        > ${output_csv} 2> ${log_file}
    echo "Results: \$(wc -l < ${output_csv}) rows" >> ${log_file}
    """
}


process evaluateScopeHierarchy {
    tag "${label}_k${ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(label), val(ksize), path(results_csv), path(scop_domains)

    output:
    tuple val(label), val(ksize), path("scope_eval.${label}.k${ksize}.tsv")
    path "scope_eval.${label}.k${ksize}.summary.txt"
    path "scope_eval.${label}.k${ksize}.pr_data.tsv"
    path "scope_eval.${label}.k${ksize}.bcl2ced9.tsv"

    script:
    """
    #!/usr/bin/env python3
    import os, sys, json
    import polars as pl
    import numpy as np

    label  = '${label}'
    ksize  = ${ksize}
    bcl2_id = 'd7jgwa1'
    ced9_id = 'd1ohua_'

    if os.path.getsize('${results_csv}') == 0:
        print(f'Warning: empty results for {label} k={ksize}', file=sys.stderr)
        for stem in ['tsv', 'summary.txt', 'pr_data.tsv', 'bcl2ced9.tsv']:
            open(f'scope_eval.{label}.k{ksize}.{stem}', 'w').close()
        sys.exit(0)

    scop = pl.read_csv('${scop_domains}', separator='\\t')

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
    print(f'{label} k={ksize}: {len(raw):,} raw rows -> {n_pairs:,} unique pairs', file=sys.stderr)

    scop_q = scop.rename({c: f'q_{c}' for c in scop.columns}).rename({'q_domain_id': 'query_domain'})
    scop_t = scop.rename({c: f't_{c}' for c in scop.columns}).rename({'t_domain_id': 'target_domain'})
    df = df.join(scop_q, on='query_domain', how='left').join(scop_t, on='target_domain', how='left')

    LEVELS = ['class', 'fold', 'superfamily', 'family']
    for lvl in LEVELS:
        df = df.with_columns(
            (pl.col(f'q_scop_{lvl}') == pl.col(f't_scop_{lvl}')).alias(f'same_{lvl}')
        )

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
    bcl2_rows.select(spot_cols).write_csv(f'scope_eval.{label}.k{ksize}.bcl2ced9.tsv', separator='\\t')

    P_METHODS = [('raw','poisson_pvalue'), ('bonferroni','bonferroni'), ('bh','bh'), ('by','by')]
    alpha     = 0.05
    pr_rows   = []
    mht_sum   = {}
    lines     = [f'=== SCOPe Hierarchy Benchmark  {label} k={ksize} ===\\n',
                 f'Total pairs: {n_pairs:,}\\n\\n']

    for lvl in LEVELS:
        is_pos = df[f'same_{lvl}'].fill_null(False).to_numpy()
        n_pos  = int(is_pos.sum())
        lines.append(f'--- {lvl.upper():<15s} positives={n_pos:,} negatives={n_pairs-n_pos:,} ---\\n')
        mht_sum[lvl] = {}

        for mname, col in P_METHODS:
            scores  = df[col].fill_null(1.0).to_numpy()
            sidx    = np.argsort(scores)
            spos    = is_pos[sidx]
            sscores = scores[sidx]
            cum_tp  = np.cumsum(spos)
            recall  = cum_tp / max(n_pos, 1)
            prec_arr = cum_tp / np.arange(1, n_pairs + 1)

            step = max(1, n_pairs // 5000)
            for i in range(0, n_pairs, step):
                pr_rows.append({
                    'label': label, 'ksize': ksize, 'level': lvl, 'method': mname,
                    'threshold': float(sscores[i]),
                    'recall':    float(recall[i]),
                    'precision': float(prec_arr[i]),
                    'n_tp': int(cum_tp[i]),
                })

            rej   = scores <= alpha
            rej_n = int(rej.sum()); tp_n = int((rej & is_pos).sum())
            prec_a = round(tp_n / rej_n if rej_n else 0.0, 4)
            rec_a  = round(tp_n / n_pos if n_pos else 0.0, 4)
            mht_sum[lvl][mname] = {'rejected': rej_n, 'TP': tp_n, 'precision': prec_a, 'recall': rec_a}
            lines.append(f'  {mname:<12s}: {rej_n:>8,} rejected  TP={tp_n:>7,}  prec={prec_a:.4f}  rec={rec_a:.4f}\\n')
        lines.append('\\n')

    summary_json = {
        'label': label, 'ksize': ksize, 'n_pairs': n_pairs,
        'mht': mht_sum,
    }
    lines += ['\\n=== JSON SUMMARY ===\\n', json.dumps(summary_json, indent=2) + '\\n']
    with open(f'scope_eval.{label}.k{ksize}.summary.txt', 'w') as f:
        f.writelines(lines)

    pl.DataFrame(pr_rows).write_csv(f'scope_eval.{label}.k{ksize}.pr_data.tsv', separator='\\t')
    df.write_csv(f'scope_eval.{label}.k{ksize}.tsv', separator='\\t')
    print(f'{label} k={ksize} done', file=sys.stderr)
    """
}


process aggregateScopeResults {
    publishDir params.outdir, mode: 'copy'

    input:
    path summaries
    path logs

    output:
    path 'hp_alphabet_sweep_summary.tsv'
    path 'hp_alphabet_sweep_summary.json'

    script:
    """
    #!/usr/bin/env python3
    import glob, json, re

    results = []
    for f in sorted(glob.glob('*.summary.txt')):
        m = re.search(r'scope_eval\\.(.+)\\.k(\\d+)\\.summary\\.txt', f)
        if not m:
            continue
        label, k = m.group(1), int(m.group(2))
        with open(f) as fh:
            content = fh.read()
        jm = re.search(r'=== JSON SUMMARY ===\\n(\\{[\\s\\S]+\\})', content)
        if jm:
            try:
                results.append(json.loads(jm.group(1)))
                continue
            except json.JSONDecodeError:
                pass
        results.append({'label': label, 'ksize': k})

    results.sort(key=lambda x: (x.get('label',''), x.get('ksize', 0)))

    LEVELS  = ['class', 'fold', 'superfamily', 'family']
    METHODS = ['raw', 'bonferroni', 'bh', 'by']
    headers = ['label', 'ksize', 'n_pairs']
    for lvl in LEVELS:
        for mth in METHODS:
            headers += [f'{lvl}_{mth}_rejected', f'{lvl}_{mth}_precision', f'{lvl}_{mth}_recall']

    with open('hp_alphabet_sweep_summary.tsv', 'w') as f:
        f.write('\\t'.join(headers) + '\\n')
        for r in results:
            row = [str(r.get('label','')), str(r.get('ksize','')), str(r.get('n_pairs',''))]
            for lvl in LEVELS:
                lm = r.get('mht', {}).get(lvl, {})
                for mth in METHODS:
                    s = lm.get(mth, {})
                    row += [str(s.get('rejected','')),
                            f"{s.get('precision',0):.4f}",
                            f"{s.get('recall',0):.4f}"]
            f.write('\\t'.join(row) + '\\n')

    with open('hp_alphabet_sweep_summary.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

    print(f'Aggregated {len(results)} results across {len({r["label"] for r in results})} alphabets')
    """
}


workflow {
    fasta_ch     = channel.fromPath(params.fasta_gd)
    scop_domains = parseScopeHeaders(fasta_ch)

    // Named encoding channels: [fasta, cli_flag, seed=null, ksize, label]
    // channel.of(*NAMED_ENCODINGS) spreads inner [cli_flag, label] lists -> 3 items after combine.
    named_ch = fasta_ch
        .combine(channel.of(*NAMED_ENCODINGS))
        .combine(channel.of(*KSIZES))
        .map { fasta, cli_flag, lbl, k -> tuple(fasta, cli_flag, null, k, lbl) }

    // Seeded shuffled control channels: [fasta, cli_flag, seed, ksize, label]
    // channel.of(*SHUFFLED_SEEDS) spreads inner [cli_flag, seed] lists -> 3 items after combine.
    seeded_ch = fasta_ch
        .combine(channel.of(*SHUFFLED_SEEDS))
        .combine(channel.of(*KSIZES))
        .map { fasta, cli_flag, seed, k ->
            def lbl = "hp_shuffled_control_${seed}"
            tuple(fasta, cli_flag, seed, k, lbl)
        }

    all_inputs = named_ch.mix(seeded_ch)

    indexed        = indexDatabase(all_inputs)
    search_outputs = searchAllVsAll(indexed[0])

    eval_inputs    = search_outputs[0].combine(scop_domains)
    eval_outputs   = evaluateScopeHierarchy(eval_inputs)

    all_summaries  = eval_outputs[1].collect()
    all_logs       = indexed[1].mix(search_outputs[1]).collect()
    aggregateScopeResults(all_summaries, all_logs)
}
