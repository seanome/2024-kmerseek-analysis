#!/usr/bin/env nextflow

/*
 * Nextflow pipeline to evaluate kmerseek ortholog detection using human-mouse orthologs
 *
 * This pipeline:
 * 1. Downloads the JAX ortholog mapping file
 * 2. Indexes human and mouse GENCODE protein sequences
 * 3. Searches human proteins against mouse proteins for k=15-30
 * 4. Evaluates ortholog detection accuracy at each k-size
 *
 * Usage:
 *   nextflow run kmerseek_human_mouse_orthologs.nf
 */

// FASTA files for human and mouse protein sequences
params.human_fasta = "${System.getProperty('user.home')}/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
params.mouse_fasta = "${System.getProperty('user.home')}/data/gencode/mouse/m38/gencode.vM38.pc_translations.canonical.fa"
params.outdir = "${System.getProperty('user.home')}/data/gencode/results-human-mouse-orthologs"

// Ortholog mapping URL
params.ortholog_url = "https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt"

// Minimum containment threshold — 0.0 keeps all hits (large CSVs; polars handles them)
params.threshold = 0.0

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
                # Human gene symbols are uppercase in GENCODE
                ortholog_groups[db_class_key]['human'].add(symbol.upper())
            elif organism == 'mouse, laboratory':
                # Mouse gene symbols are capitalized (first letter uppercase) in GENCODE
                # But the JAX file already has proper capitalization
                ortholog_groups[db_class_key]['mouse'].add(symbol)

    # Create pairs: for each human gene, list all mouse orthologs
    human_to_mouse = defaultdict(set)
    mouse_to_human = defaultdict(set)

    for group in ortholog_groups.values():
        for human_gene in group['human']:
            for mouse_gene in group['mouse']:
                human_to_mouse[human_gene].add(mouse_gene)
                mouse_to_human[mouse_gene].add(human_gene)

    # Write pairs file (human_gene, mouse_gene)
    with open('ortholog_pairs.tsv', 'w') as f:
        f.write('human_gene\\tmouse_gene\\n')
        for human_gene, mouse_genes in sorted(human_to_mouse.items()):
            for mouse_gene in sorted(mouse_genes):
                f.write(f'{human_gene}\\t{mouse_gene}\\n')

    # Write stats
    with open('ortholog_stats.txt', 'w') as f:
        f.write(f'Number of ortholog groups: {len(ortholog_groups)}\\n')
        f.write(f'Number of human genes with mouse orthologs: {len(human_to_mouse)}\\n')
        f.write(f'Number of mouse genes with human orthologs: {len(mouse_to_human)}\\n')
        f.write(f'Total human-mouse pairs: {sum(len(v) for v in human_to_mouse.values())}\\n')

        # Check for one-to-many and many-to-many relationships
        one_to_one = sum(1 for v in human_to_mouse.values() if len(v) == 1)
        one_to_many = sum(1 for v in human_to_mouse.values() if len(v) > 1)
        f.write(f'Human genes with exactly one mouse ortholog: {one_to_one}\\n')
        f.write(f'Human genes with multiple mouse orthologs: {one_to_many}\\n')
    """
}

process decompressFasta {
    tag "${species}_${fasta.name}"

    input:
    tuple val(species), path(fasta)

    output:
    tuple val(species), path("${fasta.baseName}")

    script:
    """
    gunzip -c ${fasta} > ${fasta.baseName}
    """
}

process indexDatabase {
    tag "${species}_hp_k${ksize}"
    container 'kmerseek:0.2.1'
    containerOptions '--entrypoint ""'
    // storeDir: if the rocksdb directory already exists in outdir/indices/, skip indexing.
    // Indices for k=15-23 (mouse canonical) are already published there.
    storeDir "${params.outdir}/indices"
    publishDir params.outdir, mode: 'copy', pattern: '*.index.log'

    input:
    tuple val(species), path(fasta), val(ksize)

    output:
    tuple val(species), val(ksize), path("${fasta}.hp.k${ksize}.scaled1.kmerseek.rocksdb", type: 'dir')
    path "${fasta}.hp.k${ksize}.scaled1.kmerseek.index.log"

    script:
    def log_file = "${fasta}.hp.k${ksize}.scaled1.kmerseek.index.log"
    def db_dir = "${fasta}.hp.k${ksize}.scaled1.kmerseek.rocksdb"
    """
    echo "=== Indexing: ${species} hp k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    kmerseek index \\
        --encoding hp \\
        --ksize ${ksize} \\
        --scaled 1 \\
        --input ${fasta} \\
        2>&1 | tee -a ${log_file} || true

    # Ensure output directory exists even if k-mer space is saturated (k=15-20)
    mkdir -p ${db_dir}

    echo "" | tee -a ${log_file}
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    """
}

process searchHumanVsMouse {
    tag "hp_k${ksize}"
    container 'kmerseek:0.2.1'
    containerOptions '--entrypoint ""'
    // storeDir: search CSVs (hundreds of GB each) are stored directly in outdir.
    // If human_vs_mouse.hp.k${ksize}.results.csv.zst already exists there, skip.
    // Results for k=15-23 are already present as of 2026-04-30.
    // This prevents the work/ directory from ballooning to 300+ GB again.
    storeDir params.outdir
    publishDir params.outdir, mode: 'copy', pattern: '*.search.log'

    input:
    tuple val(ksize), path(human_fasta), path(mouse_index)

    output:
    tuple val(ksize), path("human_vs_mouse.hp.k${ksize}.results.csv.zst")
    path "human_vs_mouse.hp.k${ksize}.search.log"

    script:
    def output_zst = "human_vs_mouse.hp.k${ksize}.results.csv.zst"
    def log_file = "human_vs_mouse.hp.k${ksize}.search.log"
    """
    echo "=== Searching: human vs mouse hp k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    kmerseek search \\
        --encoding hp \\
        --ksize ${ksize} \\
        --query ${human_fasta} \\
        --target ${mouse_index} \\
        2>> ${log_file} \\
        | zstd -T2 -o ${output_zst} \\
        || true

    # Ensure output exists even if no matches (empty index from k-mer space saturation)
    touch ${output_zst}

    echo "" | tee -a ${log_file}
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Compressed size: \$(du -sh ${output_zst} | cut -f1)" | tee -a ${log_file}
    """
}

process evaluateOrthologs {
    tag "hp_k${ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(ksize), path(results_zst), path(ortholog_pairs)

    output:
    tuple val(ksize), path("ortholog_evaluation.hp.k${ksize}.tsv.zst")
    path "ortholog_evaluation.hp.k${ksize}.summary.txt"
    path "ortholog_evaluation.hp.k${ksize}.roc_data.tsv.zst"
    path "ortholog_evaluation.hp.k${ksize}.mht.csv.zst"
    path "metrics_*.hp.k${ksize}.png"

    script:
    """
    # Stream-decompress once: count all rows and filter to poisson_pvalue (col 23) < 0.001.
    # With n~364M tests, BH can only reject rows with p < 0.05*M/n. Even at M=2M rejections
    # (biologically impossible: only ~20K ortholog gene pairs exist), BH threshold ~2.7e-4.
    # Cutoff 1e-3 safely captures all BH-rejectable rows while shrinking the file ~1000x
    # (364M*1e-3 = ~364K expected FPs + true positives). n_total is passed to Python so
    # MHT correction denominators stay correct.
    if [ ! -s ${results_zst} ]; then
        ln -sf ${results_zst} filtered_results.csv.zst
        n_total=0
    else
        zstdcat ${results_zst} | awk -F',' '
            NR==1 { print; next }
            { n++ }
            \$23+0 < 0.001 { print }
            END { print n > "n_total.txt" }
        ' | zstd -T2 -q -f -o filtered_results.csv.zst
        n_total=\$(cat n_total.txt 2>/dev/null || echo 0)
    fi
    evaluate_orthologs.py ${ksize} filtered_results.csv.zst ${ortholog_pairs} \${n_total}
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

    results.sort(key=lambda x: x['ksize'])

    MHT_METHODS = ['bonferroni', 'bh', 'by', 'two_stage_bh']

    # Build TSV header
    headers = ['ksize', 'total_hits', 'n_ortholog', 'n_non_ortholog']
    for method in MHT_METHODS:
        headers += [f'{method}_rejected', f'{method}_precision', f'{method}_recall']

    with open('kmer_sweep_summary.tsv', 'w') as f:
        f.write('\\t'.join(headers) + '\\n')
        for r in results:
            row = [
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
        json.dump({'encoding': 'hp', 'results': results}, f, indent=2)
    """
}


workflow {
    // Download and parse ortholog mapping
    ortholog_file = downloadOrthologMapping()
    (ortholog_pairs, ortholog_stats) = parseOrthologMapping(ortholog_file)

    // k=15-30 ALL COMPLETE as of 2026-04-30. Results published to params.outdir.
    // storeDir on indexDatabase and searchHumanVsMouse will skip completed k-sizes on rerun.
    // k=24-30 search results are in outdir as .csv.gz (older format); only k=15-23 are .csv.zst.
    // Only indexing k=15-23 here since those are the ones with storeDir-compatible rocksdb dirs.
    ksizes = Channel.of(15, 16, 17, 18, 19, 20, 21, 22, 23)

    // FASTA files are already uncompressed - use directly
    human_decompressed = channel.of(tuple('human', file(params.human_fasta)))
    mouse_decompressed = channel.of(tuple('mouse', file(params.mouse_fasta)))

    // Only index mouse (target) database - human queries are processed from FASTA
    // to avoid loading both full indices into memory simultaneously
    mouse_index_params = mouse_decompressed.combine(ksizes).map { species, fasta, ksize -> tuple(species, fasta, ksize) }

    // Index mouse database
    indexed = indexDatabase(mouse_index_params)
    index_only = indexed[0]
    // (species, ksize, index_path)

    // Get mouse indexes by ksize
    mouse_indexes = index_only.map { species, ksize, index -> tuple(ksize, index) }

    // Combine human FASTA with mouse indexes by ksize
    // Human queries are processed from FASTA on-the-fly to save memory
    human_fasta_with_ksize = human_decompressed
        .combine(ksizes)
        .map { species, fasta, ksize -> tuple(ksize, fasta) }

    search_inputs = human_fasta_with_ksize.join(mouse_indexes)
    // (ksize, human_fasta, mouse_index)

    // Search human against mouse
    search_outputs = searchHumanVsMouse(search_inputs)
    search_results = search_outputs[0]
    // (ksize, results_csv)

    // Evaluate ortholog detection
    eval_inputs = search_results.combine(ortholog_pairs)
    // (ksize, results_csv, ortholog_pairs)
    eval_outputs = evaluateOrthologs(eval_inputs)

    // Collect all summary files for aggregation
    // eval_outputs: [0] = (ksize, eval_tsv), [1] = summary.txt, [2] = roc_data.tsv
    summaries = eval_outputs[1].collect()

    // Aggregate results across all k-sizes
    aggregateResults(summaries)

    // Print progress
    eval_outputs[0].subscribe { ksize, eval_file ->
        println("Completed evaluation: k=${ksize} -> ${eval_file}")
    }
}
