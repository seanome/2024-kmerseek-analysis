#!/usr/bin/env nextflow

/*
 * Nextflow pipeline to evaluate kmerseek 0.4.0 ortholog detection: all HP alphabet
 * encodings x k=18-19 against mouse GENCODE proteins.
 *
 * Sibling of ../human-mouse-gencode-orthologs/kmerseek_human_mouse_orthologs.nf, kept
 * fully separate (own outdir, own workDir, own storeDir indices) so it can never
 * invalidate that pipeline's Nextflow resume cache. Two differences from that pipeline:
 *
 *   1. Uses the local kmerseek 0.4.0 binary (params.kmerseek_bin) directly instead of
 *      the kmerseek:0.3.1 Docker container.
 *   2. kmerseek 0.4.0's `search` natively drops matches below --min-shared-kmers /
 *      above --max-pvalue, so results are pre-filtered at the source. There is no bash
 *      awk prefilter step here (unlike the sibling pipeline) — n_total (the MHT
 *      denominator) is instead recovered from the query/target counts each process
 *      already logs to stderr.
 *
 * Usage:
 *   nextflow run kmerseek_human_mouse_orthologs_hp_v040.nf
 */

params.human_fasta = "${System.getProperty('user.home')}/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
params.mouse_fasta = "${System.getProperty('user.home')}/data/gencode/mouse/m38/gencode.vM38.pc_translations.canonical.fa"
params.outdir = "${System.getProperty('user.home')}/data/gencode/results-human-mouse-orthologs-hp-v040"

params.ortholog_url = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"

// Minimum containment threshold — 0.0 keeps all hits; the real filtering now happens
// natively in kmerseek search via min_shared_kmers/max_pvalue below.
params.threshold = 0.0

// kmerseek 0.4.0 native search filters (see olgabot/bump-version-0.4.0 branch of
// the kmerseek repo). Defaults match the kmerseek CLI's own defaults.
params.min_shared_kmers = 2
params.max_pvalue = 0.05

// Absolute path to the locally-built kmerseek 0.4.0 release binary — no Docker container
// for this pipeline (that container is still pinned to kmerseek:0.3.1).
params.kmerseek_bin = "/Users/olga/code/kmerseek/target/release/kmerseek"

process downloadOrthologMapping {
    publishDir params.outdir, mode: 'copy'

    output:
    path 'HOM_MouseHumanSequence.rpt'

    script:
    """
    curl -o HOM_MouseHumanSequence.rpt "${params.ortholog_url}"
    """
}

process parseOrthologMapping {
    publishDir params.outdir, mode: 'copy'

    input:
    path ortholog_file

    output:
    path 'ortholog_pairs.tsv'
    path 'ortholog_stats.txt'

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import csv
    from collections import defaultdict

    # Parse ortholog file and extract human-mouse gene symbol pairs
    # Group by DB Class Key to identify ortholog groups
    ortholog_groups = defaultdict(lambda: {'human': set(), 'mouse': set()})

    with open('${ortholog_file}', 'r') as f:
        reader = csv.DictReader(f, delimiter='\\t')
        for row in reader:
            db_class_key = row['DB Class Key']
            organism = row['Common Organism Name']
            symbol = row['Symbol']

            if organism == 'human':
                ortholog_groups[db_class_key]['human'].add(symbol.upper())
            elif organism == 'mouse, laboratory':
                ortholog_groups[db_class_key]['mouse'].add(symbol)

    human_to_mouse = defaultdict(set)
    mouse_to_human = defaultdict(set)

    for group in ortholog_groups.values():
        for human_gene in group['human']:
            for mouse_gene in group['mouse']:
                human_to_mouse[human_gene].add(mouse_gene)
                mouse_to_human[mouse_gene].add(human_gene)

    with open('ortholog_pairs.tsv', 'w') as f:
        f.write('human_gene\\tmouse_gene\\n')
        for human_gene, mouse_genes in sorted(human_to_mouse.items()):
            for mouse_gene in sorted(mouse_genes):
                f.write(f'{human_gene}\\t{mouse_gene}\\n')

    with open('ortholog_stats.txt', 'w') as f:
        f.write(f'Number of ortholog groups: {len(ortholog_groups)}\\n')
        f.write(f'Number of human genes with mouse orthologs: {len(human_to_mouse)}\\n')
        f.write(f'Number of mouse genes with human orthologs: {len(mouse_to_human)}\\n')
        f.write(f'Total human-mouse pairs: {sum(len(v) for v in human_to_mouse.values())}\\n')

        one_to_one = sum(1 for v in human_to_mouse.values() if len(v) == 1)
        one_to_many = sum(1 for v in human_to_mouse.values() if len(v) > 1)
        f.write(f'Human genes with exactly one mouse ortholog: {one_to_one}\\n')
        f.write(f'Human genes with multiple mouse orthologs: {one_to_many}\\n')
    """
}

process indexDatabase {
    tag "${species}_${encoding}_k${ksize}"
    storeDir "${params.outdir}/indices"
    publishDir params.outdir, mode: 'copy', pattern: '*.index.log'

    input:
    tuple val(species), path(fasta), val(encoding), val(ksize)

    output:
    tuple val(species), val(encoding), val(ksize),
        path("${fasta}.${encoding.replace('-', '_')}.k${ksize}.scaled1.kmerseek.rocksdb", type: 'dir'),
        path("${fasta}.${encoding.replace('-', '_')}.k${ksize}.scaled1.kmerseek.index.log")

    script:
    // kmerseek normalises the encoding to underscores when auto-naming the database
    // (e.g. hp-thomas-dill -> hp_thomas_dill). BOTH declared outputs (rocksdb dir AND
    // log) must use that normalised name — see the sibling pipeline for why.
    def enc_fname = encoding.replace('-', '_')
    def log_file = "${fasta}.${enc_fname}.k${ksize}.scaled1.kmerseek.index.log"
    def db_dir   = "${fasta}.${enc_fname}.k${ksize}.scaled1.kmerseek.rocksdb"
    """
    echo "=== Indexing (kmerseek 0.4.0 local): ${species} ${encoding} k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    ${params.kmerseek_bin} index \\
        --encoding ${encoding} \\
        --ksize ${ksize} \\
        --input ${fasta} \\
        2>&1 | tee -a ${log_file} || true

    # Ensure output directory exists even if k-mer space is saturated
    mkdir -p ${db_dir}

    echo "" | tee -a ${log_file}
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    """
}

process searchHumanVsMouse {
    tag "${encoding}_k${ksize}"
    storeDir params.outdir
    publishDir params.outdir, mode: 'copy', pattern: '*.search.log'

    input:
    tuple val(encoding), val(ksize), path(human_fasta), path(mouse_index), path(index_log)

    output:
    tuple val(encoding), val(ksize),
        path("human_vs_mouse.${encoding}.k${ksize}.results.parquet"),
        path("human_vs_mouse.${encoding}.k${ksize}.search.log"),
        path(index_log)

    script:
    def output_zst     = "human_vs_mouse.${encoding}.k${ksize}.results.csv.zst"
    def output_parquet = "human_vs_mouse.${encoding}.k${ksize}.results.parquet"
    def log_file       = "human_vs_mouse.${encoding}.k${ksize}.search.log"
    """
    echo "=== Searching (kmerseek 0.4.0 local): human vs mouse ${encoding} k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    ${params.kmerseek_bin} search \\
        --encoding ${encoding} \\
        --ksize ${ksize} \\
        --query ${human_fasta} \\
        --target ${mouse_index} \\
        --threshold ${params.threshold} \\
        --min-shared-kmers ${params.min_shared_kmers} \\
        --max-pvalue ${params.max_pvalue} \\
        2>> ${log_file} \\
        | zstd -T2 -o ${output_zst} \\
        || true

    # Ensure output exists even if no matches
    touch ${output_zst}

    echo "" | tee -a ${log_file}
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Compressed size (pre-parquet): \$(du -sh ${output_zst} | cut -f1)" | tee -a ${log_file}

    # Convert straight to parquet (dropping the two unused md5 columns) and delete the raw
    # CSV in the SAME task -- storeDir persists whatever this task declares as output forever,
    # so keeping both formats around is exactly the ~530GB blow-up that made this necessary
    # (see scripts/convert_results_to_parquet.py for the one-off backfill of pre-existing runs).
    if [ -s ${output_zst} ]; then
        /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3 << 'PYEOF'
import polars as pl
DROP_COLUMNS = ["query_md5", "target_md5"]
lf = pl.scan_csv("${output_zst}", ignore_errors=True)
cols = [c for c in lf.collect_schema().names() if c not in DROP_COLUMNS]
lf.select(cols).sink_parquet("${output_parquet}", compression="zstd", compression_level=9)
PYEOF
        rm ${output_zst}
    else
        # 0-byte input (search produced nothing) -- keep the empty-file signal
        # evaluate_orthologs.py checks via os.path.getsize == 0, just under the .parquet name.
        touch ${output_parquet}
    fi

    echo "Compressed size (parquet): \$(du -sh ${output_parquet} | cut -f1)" | tee -a ${log_file}
    """
}

process evaluateOrthologs {
    tag "${encoding}_k${ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(encoding), val(ksize), path(results_zst), path(search_log), path(index_log), path(ortholog_pairs)

    output:
    tuple val(encoding), val(ksize), path("ortholog_evaluation.${encoding}.k${ksize}.tsv.zst")
    path "ortholog_evaluation.${encoding}.k${ksize}.summary.txt"
    path "ortholog_evaluation.${encoding}.k${ksize}.roc_data.tsv.zst"
    path "ortholog_evaluation.${encoding}.k${ksize}.mht.csv.zst"
    path "metrics_*.${encoding}.k${ksize}.png"

    script:
    """
    # results_zst is already filtered at the source (kmerseek 0.4.0 --min-shared-kmers/
    # --max-pvalue), so n_total (all query x target comparisons, the MHT denominator)
    # can't be counted from the file — recover it from what each process already logged:
    #   search_log: "Total queries: N"        (human sequences streamed)
    #   index_log:  "Building inverted index for N signatures" (mouse targets indexed)
    query_count=\$(grep -oE 'Total queries: [0-9]+' ${search_log} | tail -1 | grep -oE '[0-9]+' || echo 0)
    target_count=\$(grep -oE 'Building inverted index for [0-9]+ signatures' ${index_log} | tail -1 | grep -oE '[0-9]+' || echo 0)
    n_total=\$((query_count * target_count))
    n_total=\${n_total:-0}

    evaluate_orthologs.py ${ksize} ${results_zst} ${ortholog_pairs} "\${n_total}" "${encoding}" "${params.max_pvalue}"
    """
}

process aggregateResults {
    publishDir params.outdir, mode: 'copy'

    input:
    path summaries

    output:
    path 'kmer_sweep_summary.tsv'
    path 'kmer_sweep_summary.json'

    script:
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import glob, json, re

    results = []
    for f in sorted(glob.glob('*.summary.txt')):
        m = re.search(r'\\.([^.]+)\\.k(\\d+)\\.summary', f)
        if not m:
            continue
        encoding, ksize = m.group(1), int(m.group(2))
        with open(f) as fh:
            content = fh.read()
        jm = re.search(r'=== JSON SUMMARY ===\\n(\\{[\\s\\S]+\\})', content)
        if jm:
            try:
                d = json.loads(jm.group(1))
                d['encoding'] = encoding
                results.append(d)
                continue
            except json.JSONDecodeError:
                pass
        results.append({'encoding': encoding, 'ksize': ksize})

    results.sort(key=lambda x: (x.get('encoding', ''), x.get('ksize', 0)))

    MHT_METHODS = ['bonferroni', 'bh', 'by', 'two_stage_bh']

    headers = ['encoding', 'ksize', 'total_hits', 'n_ortholog', 'n_non_ortholog']
    for method in MHT_METHODS:
        headers += [f'{method}_rejected', f'{method}_precision', f'{method}_recall']

    with open('kmer_sweep_summary.tsv', 'w') as f:
        f.write('\\t'.join(headers) + '\\n')
        for r in results:
            row = [
                r.get('encoding', ''),
                str(r.get('ksize', '')),
                str(r.get('total_hits', '')),
                str(r.get('n_ortholog', '')),
                str(r.get('n_non_ortholog', '')),
            ]
            mht = r.get('mht', {})
            for method in MHT_METHODS:
                s = mht.get(method, {})
                row += [
                    str(s.get('rejected', '')),
                    f"{s.get('precision', 0):.4f}",
                    f"{s.get('recall', 0):.4f}",
                ]
            f.write('\\t'.join(row) + '\\n')

    with open('kmer_sweep_summary.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)
    """
}

process multiQC {
    publishDir params.outdir, mode: 'copy'

    input:
    path sweep_json

    output:
    path "multiqc_report.html"
    path "multiqc_data/"

    script:
    """
    make_multiqc_input.py ${sweep_json} mqc_input/

    /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/multiqc \\
        mqc_input/ \\
        --config mqc_input/multiqc_config.yaml \\
        --outdir . \\
        --filename multiqc_report.html \\
        --force \\
        --no-megaqc-upload
    """
}

workflow {
    ortholog_file = downloadOrthologMapping()
    (ortholog_pairs, ortholog_stats) = parseOrthologMapping(ortholog_file)

    // Re-run of the 17 TRUNCATED combos, not a fresh sweep.
    //
    // Their existing results in ../results-human-mouse-orthologs (and, for kyte-doolittle k18,
    // in this pipeline's own outdir) contain only the first ~1000 human query sequences: each
    // file's gene set is exactly FASTA gene indices 0..999, a contiguous prefix, even though the
    // search logs show all 19732 queries were read. The writes were cut off during the disk-full
    // period (the volume hit 260 MB free). Row counts still look healthy — 26M to 160M rows — so
    // the truncation is invisible without counting DISTINCT query genes: ~999 vs the ~19,696 a
    // complete run produces. They surfaced as "0 MHC hits" in notebook 201.
    //
    // Listed explicitly rather than as an encoding x ksize cross-product, because the affected
    // (encoding, ksize) pairs are irregular — re-running the full grid would redo ~60 combos
    // that are already complete.
    hp_enc_ksize = Channel.of(
        ['hp-kyte-doolittle',   18], ['hp-kyte-doolittle',   22], ['hp-kyte-doolittle',   23],
        ['hp-kyte-doolittle',   24], ['hp-kyte-doolittle',   25], ['hp-kyte-doolittle',   26],
        ['hp-kyte-doolittle',   27],
        ['hp-pbotc-1st-ed',     18],
        ['hp-thomas-dill',      19], ['hp-thomas-dill',      20], ['hp-thomas-dill',      21],
        ['hp-thomas-dill',      22],
        ['hp-thomas-dill-no-c', 21], ['hp-thomas-dill-no-c', 22], ['hp-thomas-dill-no-c', 23],
        ['hp-thomas-dill-no-c', 24], ['hp-thomas-dill-no-c', 25]
    ).map { enc, k -> tuple(enc, k) }

    human_decompressed = channel.of(tuple('human', file(params.human_fasta)))
    mouse_decompressed = channel.of(tuple('mouse', file(params.mouse_fasta)))

    mouse_index_params = mouse_decompressed
        .combine(hp_enc_ksize)
        .map { species, fasta, encoding, ksize -> tuple(species, fasta, encoding, ksize) }

    indexed = indexDatabase(mouse_index_params)
    // (species, encoding, ksize, index_dir, index_log)

    mouse_indexes = indexed.map { species, encoding, ksize, index, index_log -> tuple(encoding, ksize, index, index_log) }

    human_fasta_enc_ksize = human_decompressed
        .combine(hp_enc_ksize)
        .map { species, fasta, encoding, ksize -> tuple(encoding, ksize, fasta) }

    search_inputs = human_fasta_enc_ksize.join(mouse_indexes, by: [0, 1])
    // (encoding, ksize, human_fasta, mouse_index, index_log)

    search_outputs = searchHumanVsMouse(search_inputs)
    // (encoding, ksize, results_csv, search_log, index_log)

    eval_inputs = search_outputs.combine(ortholog_pairs)
    // (encoding, ksize, results_csv, search_log, index_log, ortholog_pairs)
    eval_outputs = evaluateOrthologs(eval_inputs)

    summaries = eval_outputs[1].collect()
    agg_out = aggregateResults(summaries)

    multiQC(agg_out[1])

    eval_outputs[0].subscribe { encoding, ksize, eval_file ->
        println("Completed evaluation: ${encoding} k=${ksize} -> ${eval_file}")
    }
}
