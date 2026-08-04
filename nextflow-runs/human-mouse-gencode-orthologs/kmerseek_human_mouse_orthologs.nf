#!/usr/bin/env nextflow

/*
 * Nextflow pipeline to evaluate kmerseek ortholog detection using human-mouse orthologs
 *
 * Runs all HP alphabet encodings × k=20-30 against mouse GENCODE proteins.
 * (k=15-19 HP is covered by the sibling kmerseek 0.4.0 pipeline instead — see
 * ../human-mouse-gencode-orthologs-hp-v040/.)
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

process indexDatabase {
    tag "${species}_${encoding}_k${ksize}"
    container 'kmerseek:0.3.1'
    containerOptions '--entrypoint ""'
    storeDir "${params.outdir}/indices"
    publishDir params.outdir, mode: 'copy', pattern: '*.index.log'

    input:
    tuple val(species), path(fasta), val(encoding), val(ksize)

    output:
    tuple val(species), val(encoding), val(ksize), path("${fasta}.${encoding.replace('-', '_')}.k${ksize}.scaled1.kmerseek.rocksdb", type: 'dir')
    path "${fasta}.${encoding.replace('-', '_')}.k${ksize}.scaled1.kmerseek.index.log"

    script:
    // kmerseek normalises the encoding to underscores when auto-naming the database
    // (e.g. hp-thomas-dill -> hp_thomas_dill). BOTH declared outputs (rocksdb dir AND
    // log) must use that normalised name, otherwise storeDir's "all outputs present"
    // check never passes: the dir alone existing isn't enough, and every run re-executes
    // the task, rebuilds an identical rocksdb dir, then crashes trying to `mv` it onto
    // the already-populated storeDir target ("Directory not empty").
    def enc_fname = encoding.replace('-', '_')
    def log_file = "${fasta}.${enc_fname}.k${ksize}.scaled1.kmerseek.index.log"
    def db_dir   = "${fasta}.${enc_fname}.k${ksize}.scaled1.kmerseek.rocksdb"
    """
    echo "=== Indexing: ${species} ${encoding} k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    kmerseek index \\
        --encoding ${encoding} \\
        --ksize ${ksize} \\
        --scaled 1 \\
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
    container 'kmerseek:0.3.1'
    containerOptions '--entrypoint ""'
    storeDir params.outdir
    publishDir params.outdir, mode: 'copy', pattern: '*.search.log'

    input:
    tuple val(encoding), val(ksize), path(human_fasta), path(mouse_index)

    output:
    tuple val(encoding), val(ksize), path("human_vs_mouse.${encoding}.k${ksize}.results.csv.zst")
    path "human_vs_mouse.${encoding}.k${ksize}.search.log"

    script:
    def output_zst = "human_vs_mouse.${encoding}.k${ksize}.results.csv.zst"
    def log_file   = "human_vs_mouse.${encoding}.k${ksize}.search.log"
    """
    set -o pipefail

    echo "=== Searching: human vs mouse ${encoding} k=${ksize} ===" | tee ${log_file}
    echo "Start time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    # NOTE: no `|| true` and no fallback `touch` here anymore -- both used to swallow a
    # crashed/OOM-killed `kmerseek search` (e.g. SIGKILL, no stderr) and let the task exit 0
    # with a truncated or empty results file, which storeDir then cached as a permanent
    # "success". Some HP encodings (hp-kyte-doolittle, hp-thomas-dill, hp-thomas-dill-no-c) at
    # low ksize produce dense hub k-mers that exceed real available memory (Docker Desktop's
    # VM, not the `memory` directive below, is the actual ceiling) and get killed. Now that
    # failure propagates (`pipefail` + Nextflow's default `set -e`) so the task is marked
    # FAILED and produces no cached output -- see withName: searchHumanVsMouse's
    # errorStrategy 'ignore' in nextflow.config, which lets the rest of the sweep continue.
    kmerseek search \\
        --encoding ${encoding} \\
        --ksize ${ksize} \\
        --query ${human_fasta} \\
        --target ${mouse_index} \\
        2>> ${log_file} \\
        | zstd -T2 -o ${output_zst}

    echo "" | tee -a ${log_file}
    echo "End time: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "Compressed size: \$(du -sh ${output_zst} | cut -f1)" | tee -a ${log_file}
    """
}

// Converts each raw human_vs_mouse.*.results.csv.zst to parquet (dropping the two unused
// md5 columns) and deletes the raw file from searchHumanVsMouse's storeDir -- kept as its own
// (non-containerized) process because kmerseek:0.3.1's container has no python/polars, so this
// can't be folded into searchHumanVsMouse's script the way the hp-v040 sibling pipeline does it.
//
// Tradeoff, stated explicitly: because searchHumanVsMouse's storeDir output no longer exists
// on disk after this runs, a future `-resume` that needs a NEW (not-yet-searched) combo is
// unaffected, but re-touching an ALREADY-converted combo would force kmerseek search to redo
// that (expensive) work rather than finding a cached hit -- same tradeoff as manually running
// `nextflow clean`, just automatic. Acceptable here since this pipeline is being superseded by
// the hp-v040 sibling for new sweeps (see that .nf's header comment); this one's raw csv.zst
// files were the ~228GB that prompted this change in the first place.
process convertResultsToParquet {
    tag "${encoding}_k${ksize}"
    storeDir params.outdir

    input:
    tuple val(encoding), val(ksize), path(results_csv_zst)

    output:
    tuple val(encoding), val(ksize), path("human_vs_mouse.${encoding}.k${ksize}.results.parquet")

    script:
    def parquet = "human_vs_mouse.${encoding}.k${ksize}.results.parquet"
    """
    #!/Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
    import polars as pl
    from pathlib import Path

    src = Path("${results_csv_zst}")
    dst = Path("${parquet}")

    if src.stat().st_size == 0:
        dst.touch()  # keep the empty-file signal evaluate_orthologs.py checks for
    else:
        DROP_COLUMNS = ["query_md5", "target_md5"]
        lf = pl.scan_csv(str(src), ignore_errors=True)
        cols = [c for c in lf.collect_schema().names() if c not in DROP_COLUMNS]
        lf.select(cols).sink_parquet(str(dst), compression="zstd", compression_level=9)

    # ${results_csv_zst} here is nextflow's symlink into THIS task's work dir -- the real file
    # searchHumanVsMouse's storeDir wrote lives directly in params.outdir under the same name.
    real_src = Path("${params.outdir}") / "human_vs_mouse.${encoding}.k${ksize}.results.csv.zst"
    if real_src.exists():
        real_src.unlink()
    """
}

process evaluateOrthologs {
    tag "${encoding}_k${ksize}"
    publishDir params.outdir, mode: 'copy'

    input:
    tuple val(encoding), val(ksize), path(results_zst), path(ortholog_pairs)

    output:
    tuple val(encoding), val(ksize), path("ortholog_evaluation.${encoding}.k${ksize}.tsv.zst")
    path "ortholog_evaluation.${encoding}.k${ksize}.summary.txt"
    path "ortholog_evaluation.${encoding}.k${ksize}.roc_data.tsv.zst"
    path "ortholog_evaluation.${encoding}.k${ksize}.mht.csv.zst"
    path "metrics_*.${encoding}.k${ksize}.png"

    script:
    """
    # Pre-filter: detect column indices from CSV header, then keep rows where
    # n_intersecting_hashes > expected_shared_kmers * 1000 (enrichment > 1000).
    # This safely captures all rows with Poisson p < ~0.001 regardless of n_total,
    # while shrinking the file ~1000x. n_total (full row count) is tracked for MHT.
    if [ ! -s ${results_zst} ]; then
        ln -sf ${results_zst} filtered_results.csv.zst
        n_total=0
    else
        zstdcat ${results_zst} | awk -F',' '
            NR==1 {
                for (i=1; i<=NF; i++) {
                    if (\$i == "n_intersecting_hashes") k_col = i
                    if (\$i == "expected_shared_kmers")  lam_col = i
                }
                print; next
            }
            { n++ }
            (k_col && lam_col && \$k_col+0 >= 2 && (\$lam_col+0 == 0 || \$k_col+0 / \$lam_col+0 >= 100)) { print }
            END { print n+0 > "n_total.txt" }
        ' | zstd -T2 -q -f -o filtered_results.csv.zst
        n_total=\$(cat n_total.txt 2>/dev/null || echo 0)
    fi
    # Guard against an empty/missing n_total (e.g. a header-only results file with 0 data rows):
    # an unquoted empty \${n_total} would drop the argument and shift ${encoding} into its place.
    n_total=\${n_total:-0}
    evaluate_orthologs.py ${ksize} filtered_results.csv.zst ${ortholog_pairs} "\${n_total}" "${encoding}"
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

    # Build TSV header
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
    path "multiqc_report_data/"

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
    // Download and parse ortholog mapping
    ortholog_file = downloadOrthologMapping()
    (ortholog_pairs, ortholog_stats) = parseOrthologMapping(ortholog_file)

    // Encoding × ksize ranges — alphabet size determines useful ksize:
    //   hp variants  k=20-30  (2-letter; 'hp' storeDir results reused)
    //                (k=15-19 now covered by the sibling kmerseek 0.4.0 pipeline in
    //                 ../human-mouse-gencode-orthologs-hp-v040/, so dropped here)
    //   dayhoff      k=10-20  (6-letter)
    //   protein      k=5-15   (20-letter)
    hp_ksizes      = Channel.of(20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
    dayhoff_ksizes = Channel.of(10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
    protein_ksizes = Channel.of(5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)

    hp_enc_ksize = Channel.of(
        'hp-lehninger',
        'hp-thomas-dill',
        'hp-kyte-doolittle',
        'hp-thomas-dill-no-c',
        'hp-lehninger-plus-c',
        'hp-pbotc-1st-ed'
    ).combine(hp_ksizes)

    dayhoff_enc_ksize = Channel.of('dayhoff').combine(dayhoff_ksizes)
    protein_enc_ksize = Channel.of('protein').combine(protein_ksizes)

    all_enc_ksize = hp_enc_ksize.mix(dayhoff_enc_ksize).mix(protein_enc_ksize)

    // FASTA files are already uncompressed - use directly
    human_decompressed = channel.of(tuple('human', file(params.human_fasta)))
    mouse_decompressed = channel.of(tuple('mouse', file(params.mouse_fasta)))

    // Index mouse database for each (encoding, ksize)
    mouse_index_params = mouse_decompressed
        .combine(all_enc_ksize)
        .map { species, fasta, encoding, ksize -> tuple(species, fasta, encoding, ksize) }

    indexed = indexDatabase(mouse_index_params)
    index_only = indexed[0]
    // (species, encoding, ksize, index_path)

    // Get mouse indexes by (encoding, ksize)
    mouse_indexes = index_only.map { species, encoding, ksize, index -> tuple(encoding, ksize, index) }

    // Combine human FASTA with mouse indexes by (encoding, ksize)
    human_fasta_enc_ksize = human_decompressed
        .combine(all_enc_ksize)
        .map { species, fasta, encoding, ksize -> tuple(encoding, ksize, fasta) }

    search_inputs = human_fasta_enc_ksize.join(mouse_indexes, by: [0, 1])
    // (encoding, ksize, human_fasta, mouse_index)

    // Search human against mouse
    search_outputs = searchHumanVsMouse(search_inputs)
    search_results = search_outputs[0]
    // (encoding, ksize, results_csv_zst)

    // Convert to parquet immediately and drop the raw csv.zst (see process comment) --
    // everything downstream reads the parquet.
    parquet_results = convertResultsToParquet(search_results)
    // (encoding, ksize, results_parquet)

    // Evaluate ortholog detection
    eval_inputs = parquet_results.combine(ortholog_pairs)
    // (encoding, ksize, results_parquet, ortholog_pairs)
    eval_outputs = evaluateOrthologs(eval_inputs)

    // Collect all summary files for aggregation
    summaries = eval_outputs[1].collect()
    agg_out = aggregateResults(summaries)

    // MultiQC: single-document summary of the whole encoding x ksize sweep
    multiQC(agg_out[1])

    eval_outputs[0].subscribe { encoding, ksize, eval_file ->
        println("Completed evaluation: ${encoding} k=${ksize} -> ${eval_file}")
    }
}
