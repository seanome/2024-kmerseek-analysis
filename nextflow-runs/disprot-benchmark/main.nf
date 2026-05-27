#!/usr/bin/env nextflow

/*
 * disprot_benchmark.nf
 *
 * Measure whether Kmerseek detects homology through intrinsically disordered
 * regions where Foldseek fails (no stable 3D structure to encode).
 *
 * Strategy: filter the existing Pfam QfO benchmark to DisProt-annotated human
 * proteins, then compare Kmerseek, Foldseek, and MMseqs2 recall stratified
 * by disorder level.
 *
 * Key design: reuses existing Pfam ground truth pairs — DO NOT rebuild them.
 * DisProt proteins are a subset of the human proteins already benchmarked.
 *
 * Usage:
 *   nextflow run main.nf
 *   nextflow run main.nf --resume
 *   nextflow run main.nf --skip_foldseek          # skip AlphaFold download
 *   nextflow run main.nf --kmerseek_k_values 20,24
 */

nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

def home = System.getProperty('user.home')

params.qfo_dir          = "${home}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.pfam_pairs_dir   = "${projectDir}/../../results/pfam_benchmark/pairs"
params.outdir           = "${home}/data/disprot-benchmark/results"
params.alphafold_cache  = "${home}/data/alphafold_structures"

// DisProt data — null triggers download from API in first process
params.disprot_json     = null

// Kmerseek settings
params.kmerseek_k_values = [20, 24, 27]
params.kmerseek_scaled   = 1
params.kmerseek_encoding = "hp"      // hp | dayhoff | protein

// Foldseek
params.skip_foldseek     = false
params.evalue_report     = 10.0

// MMseqs2
params.mmseqs2_sensitivity = 7
params.mmseqs2_iterations  = 1       // seq-seq only for speed

// ---------------------------------------------------------------------------
// Species: identical to Pfam benchmark pipeline
// ---------------------------------------------------------------------------

def SPECIES = [
    [label: "mouse",       taxon: "10090",  proteome: "UP000000589", subdir: "Eukaryota", mya: 100],
    [label: "chicken",     taxon: "9031",   proteome: "UP000000539", subdir: "Eukaryota", mya: 300],
    [label: "zebrafish",   taxon: "7955",   proteome: "UP000000437", subdir: "Eukaryota", mya: 430],
    [label: "ciona",       taxon: "7719",   proteome: "UP000008144", subdir: "Eukaryota", mya: 550],
    [label: "fly",         taxon: "7227",   proteome: "UP000000803", subdir: "Eukaryota", mya: 600],
    [label: "worm",        taxon: "6239",   proteome: "UP000001940", subdir: "Eukaryota", mya: 650],
    [label: "yeast",       taxon: "559292", proteome: "UP000002311", subdir: "Eukaryota", mya: 900],
    [label: "arabidopsis", taxon: "3702",   proteome: "UP000006548", subdir: "Eukaryota", mya: 1500],
    [label: "ecoli",       taxon: "83333",  proteome: "UP000000625", subdir: "Bacteria",  mya: 2000],
]

// ---------------------------------------------------------------------------
// PROCESS 1 — Download DisProt JSON and parse human proteins
// ---------------------------------------------------------------------------

process downloadDisprot {
    label 'python'
    publishDir "${params.outdir}/disprot", mode: 'copy'

    input:
    val disprot_json_path   // null → download; path string → use file

    output:
    path "disprot_human.tsv"

    script:
    def use_local = (disprot_json_path != null && disprot_json_path != "null")
    def local_arg = use_local ? "--local ${disprot_json_path}" : ""
    """
    ${projectDir}/bin/parse_disprot.py ${local_arg} disprot_human.tsv
    """
}

// ---------------------------------------------------------------------------
// PROCESS 2 — Map DisProt proteins to Pfam ground truth
// ---------------------------------------------------------------------------

process mapDisprotToPfam {
    label 'python'
    publishDir "${params.outdir}/disprot", mode: 'copy'

    input:
    path disprot_tsv
    path pfam_pairs_dir

    output:
    path "disprot_pfam_mapping.tsv"

    script:
    """
    ${projectDir}/bin/map_disprot_pfam.py \\
        ${disprot_tsv} \\
        ${pfam_pairs_dir} \\
        disprot_pfam_mapping.tsv
    """
}

// ---------------------------------------------------------------------------
// PROCESS 3 — Build DisProt ground truth per species + query FASTA
// ---------------------------------------------------------------------------

process buildDisprotGroundTruth {
    label 'python'
    publishDir "${params.outdir}/disprot", mode: 'copy'

    input:
    path disprot_pfam_mapping
    path pfam_pairs_dir
    path human_fasta

    output:
    path "gt/human_vs_*_ground_truth.parquet", emit: gt_parquets
    path "disprot_benchmark_queries.fasta",    emit: query_fasta
    path "benchmark_stats.txt",                emit: stats

    script:
    """
    ${projectDir}/bin/build_disprot_ground_truth.py \\
        ${disprot_pfam_mapping} \\
        ${pfam_pairs_dir} \\
        ${human_fasta} \\
        gt/ \\
        disprot_benchmark_queries.fasta \\
        benchmark_stats.txt
    """
}

// ---------------------------------------------------------------------------
// PROCESS 4 — Predict per-residue disorder with metapredict
// ---------------------------------------------------------------------------

process predictDisorder {
    label 'python'
    publishDir "${params.outdir}/disprot", mode: 'copy'

    input:
    path query_fasta

    output:
    path "query_disorder_scores.tsv"

    script:
    """
    ${projectDir}/bin/predict_disorder.py ${query_fasta} query_disorder_scores.tsv
    """
}

// ---------------------------------------------------------------------------
// PROCESS 5 — Kmerseek: build index then search (one per species × k)
// ---------------------------------------------------------------------------

process kmerseekIndex {
    tag "${species} k=${k}"
    container 'kmerseek:0.2.1'
    label 'medium_cpu'

    input:
    tuple val(species), path(species_fasta), val(k)

    output:
    tuple val(species), val(k), path("${species}_k${k}.db")

    script:
    """
    kmerseek index \\
        --input    ${species_fasta} \\
        --output   ${species}_k${k}.db \\
        --ksize    ${k} \\
        --scaled   ${params.kmerseek_scaled} \\
        --encoding ${params.kmerseek_encoding} \\
        --threads  ${task.cpus}
    """
}

process kmerseekSearch {
    tag "human_vs_${species} k=${k}"
    container 'kmerseek:0.2.1'
    label 'medium_cpu'
    publishDir "${params.outdir}/kmerseek_k${k}", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), val(k), path(target_db), path(query_fasta)

    output:
    tuple val(species), val("kmerseek_k${k}"), path("human_vs_${species}.kmerseek_k${k}.tsv.gz")

    script:
    // kmerseek CSV columns (header row):
    //   query_name, query_md5, target_name, target_md5, containment,
    //   n_intersecting_hashes, ..., max_containment, ..., poisson_pvalue, ...
    // We use max_containment as the similarity score (higher = more similar).
    // Output: 4-col TSV (query_name, target_name, max_containment, poisson_pvalue)
    """
    kmerseek search \\
        --query    ${query_fasta} \\
        --target   ${target_db} \\
        --output   raw_results.csv \\
        --ksize    ${k} \\
        --scaled   ${params.kmerseek_scaled} \\
        --encoding ${params.kmerseek_encoding} \\
        --threads  ${task.cpus}

    # Reformat: pick best max_containment per (query, target) pair,
    # emit 4-col TSV: query_name <TAB> target_name <TAB> max_containment <TAB> poisson_pvalue
    python3 - <<'PYEOF'
import csv, sys, gzip
from collections import defaultdict

best = {}   # (query, target) -> (max_containment, poisson_pvalue)
with open('raw_results.csv') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        key = (row['query_name'], row['target_name'])
        mc  = float(row.get('max_containment', 0) or 0)
        pv  = float(row.get('poisson_pvalue', 1) or 1)
        if key not in best or mc > best[key][0]:
            best[key] = (mc, pv)

with gzip.open('human_vs_${species}.kmerseek_k${k}.tsv.gz', 'wt') as out:
    for (q, t), (mc, pv) in best.items():
        out.write(f"{q}\\t{t}\\t{mc}\\t{pv}\\n")
PYEOF
    """
}

// ---------------------------------------------------------------------------
// PROCESS 6a — Download AlphaFold structures (batch, cached)
// ---------------------------------------------------------------------------

process downloadAlphaFoldStructures {
    tag "${label}"
    label 'python'
    publishDir "${params.alphafold_cache}", mode: 'copy', saveAs: { f -> f }

    input:
    tuple val(label), path(fasta)

    output:
    tuple val(label), path("structures/${label}/")

    script:
    // Downloads AF v4 CIF files; skips proteins already in cache.
    // Rate-limited to 5 concurrent per process (maxForks in config).
    """
    mkdir -p structures/${label}
    ${projectDir}/bin/download_alphafold.py \\
        --fasta        ${fasta} \\
        --outdir       structures/${label} \\
        --cache        ${params.alphafold_cache} \\
        --max-workers  5
    """
}

// ---------------------------------------------------------------------------
// PROCESS 6b — Foldseek search
// ---------------------------------------------------------------------------

process foldseekSearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/foldseek:9.427df8a--pl5321h6a68c12_3'
    label 'medium_cpu'
    publishDir "${params.outdir}/foldseek", mode: 'copy', pattern: '*.tsv.gz'

    when:
    !params.skip_foldseek

    input:
    tuple val(species), path(query_structs), path(target_structs)

    output:
    tuple val(species), val("foldseek"), path("human_vs_${species}.foldseek.tsv.gz")

    script:
    """
    mkdir -p foldseek_tmp

    foldseek easy-search \\
        ${query_structs} \\
        ${target_structs} \\
        human_vs_${species}.foldseek.tsv \\
        foldseek_tmp \\
        --format-output "query,target,bits,evalue" \\
        --threads ${task.cpus} \\
        -e ${params.evalue_report}

    gzip -c human_vs_${species}.foldseek.tsv > human_vs_${species}.foldseek.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESS 7 — MMseqs2 easy-search (same process as Pfam pipeline)
// ---------------------------------------------------------------------------

process mmseqs2EasySearch {
    tag "human_vs_${species}"
    container 'quay.io/biocontainers/mmseqs2:18.8cc5c--hd6d6fdc_0'
    label 'medium_cpu'
    publishDir "${params.outdir}/mmseqs2", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), path(species_fasta), path(query_fasta)

    output:
    tuple val(species), val("mmseqs2"), path("human_vs_${species}.mmseqs2.tsv.gz")

    script:
    """
    mkdir -p mmseqs_tmp
    mmseqs easy-search \\
        ${query_fasta} \\
        ${species_fasta} \\
        human_vs_${species}.mmseqs2.tsv \\
        mmseqs_tmp \\
        --threads ${task.cpus} \\
        -s ${params.mmseqs2_sensitivity} \\
        --max-seqs 1000 \\
        --format-output "query,target,bits,evalue"
    gzip -c human_vs_${species}.mmseqs2.tsv > human_vs_${species}.mmseqs2.tsv.gz
    """
}

// ---------------------------------------------------------------------------
// PROCESS 8 — Evaluate: standard metrics + disorder stratification
// ---------------------------------------------------------------------------

process evaluateDisprotBenchmark {
    tag "${tool} vs ${species}"
    label 'python'
    publishDir "${params.outdir}/metrics", mode: 'copy', pattern: '*.parquet'

    input:
    tuple val(species), val(tool), path(results_tsv_gz), path(ground_truth), path(disorder_scores)

    output:
    path "${tool}.${species}.disprot_metrics.parquet"
    path "${tool}.${species}.disprot_pr_curve.parquet"

    script:
    """
    ${projectDir}/bin/evaluate_disprot_benchmark.py \\
        ${results_tsv_gz} \\
        ${species} \\
        ${ground_truth} \\
        ${tool} \\
        ${disorder_scores} \\
        ${tool}.${species}.disprot_metrics.parquet \\
        ${tool}.${species}.disprot_pr_curve.parquet
    """
}

// ---------------------------------------------------------------------------
// PROCESS 9 — Aggregate metrics + produce figures
// ---------------------------------------------------------------------------

process aggregateDisprotMetrics {
    label 'python'
    publishDir params.outdir, mode: 'copy'

    input:
    path 'metrics/*'

    output:
    path "all_disprot_metrics.parquet"
    path "all_disprot_pr_curves.parquet"
    path "figures/"

    script:
    """
    ${projectDir}/bin/aggregate_disprot_metrics.py \\
        metrics \\
        all_disprot_metrics.parquet \\
        all_disprot_pr_curves.parquet \\
        figures/
    """
}

// ---------------------------------------------------------------------------
// PROCESS 10 — Generate markdown report
// ---------------------------------------------------------------------------

process generateReport {
    label 'python'
    publishDir params.outdir, mode: 'copy'

    input:
    path metrics_parquet
    path figures_dir

    output:
    path "disprot_benchmark_report.md"

    script:
    """
    ${projectDir}/bin/generate_report.py \\
        ${metrics_parquet} \\
        ${figures_dir} \\
        disprot_benchmark_report.md
    """
}

// ===========================================================================
// WORKFLOW
// ===========================================================================

workflow {

    def human_fasta = file("${params.qfo_dir}/Eukaryota/UP000005640_9606.fasta")

    // Species channel: (label, fasta)
    species_ch = Channel.fromList(
        SPECIES.collect { s ->
            tuple(s.label, file("${params.qfo_dir}/${s.subdir}/${s.proteome}_${s.taxon}.fasta"))
        }
    )

    // -----------------------------------------------------------------------
    // Steps 1-3: Build DisProt benchmark dataset
    // -----------------------------------------------------------------------
    disprot_raw = downloadDisprot(params.disprot_json ?: "null")

    disprot_mapping = mapDisprotToPfam(
        disprot_raw,
        file(params.pfam_pairs_dir)
    )

    gt_out = buildDisprotGroundTruth(
        disprot_mapping,
        file(params.pfam_pairs_dir),
        human_fasta
    )

    // Emit benchmark stats to log
    gt_out.stats.subscribe { f -> log.info "Benchmark stats:\n" + f.text }

    // Per-species ground truth channel: (species_label, parquet_file)
    disprot_gt_ch = gt_out.gt_parquets
        .flatten()
        .map { f ->
            def m = (f.name =~ /human_vs_(.+)_ground_truth\.parquet/)
            def label = m ? m[0][1] : f.baseName
            tuple(label, f)
        }

    // Single query FASTA (DisProt human proteins that passed filtering)
    query_fasta = gt_out.query_fasta

    // -----------------------------------------------------------------------
    // Step 4: Predict disorder scores
    // -----------------------------------------------------------------------
    disorder_scores = predictDisorder(query_fasta)

    // -----------------------------------------------------------------------
    // Step 5: Kmerseek (species × k combinations)
    // -----------------------------------------------------------------------
    k_ch = Channel.fromList(params.kmerseek_k_values)

    // Build one index per species × k
    kmerseek_index_input = species_ch.combine(k_ch)
        .map { species, fasta, k -> tuple(species, fasta, k) }
    kmerseek_index_ch = kmerseekIndex(kmerseek_index_input)

    // Search with query FASTA against each index
    // combine() broadcasts the single query_fasta element across all index tuples
    kmerseek_search_input = kmerseek_index_ch
        .combine(query_fasta)
        .map { species, k, db, qf -> tuple(species, k, db, qf) }
    kmerseek_results = kmerseekSearch(kmerseek_search_input)

    // -----------------------------------------------------------------------
    // Step 6: Foldseek (optional)
    // -----------------------------------------------------------------------
    foldseek_results = Channel.empty()
    if (!params.skip_foldseek) {
        // Download structures for human queries AND all species in one channel.
        // Human is tagged "human"; species use their species label.
        all_for_af = query_fasta.map { f -> tuple("human", f) }.mix(species_ch)
        all_structs_ch = downloadAlphaFoldStructures(all_for_af)

        human_structs_ch   = all_structs_ch.filter { label, _structs -> label == "human" }
        species_structs_ch = all_structs_ch.filter { label, _structs -> label != "human" }

        // Combine: for each species emit (species, human_structs_dir, species_structs_dir)
        foldseek_input = species_structs_ch
            .combine(human_structs_ch.map { _label, structs -> structs })
            .map { sp_label, sp_structs, hu_structs ->
                tuple(sp_label, hu_structs, sp_structs)
            }

        foldseek_results = foldseekSearch(foldseek_input)
    }

    // -----------------------------------------------------------------------
    // Step 7: MMseqs2
    // -----------------------------------------------------------------------
    mmseqs2_input = species_ch
        .combine(query_fasta)
        .map { species, fasta, qf -> tuple(species, fasta, qf) }
    mmseqs2_results = mmseqs2EasySearch(mmseqs2_input)

    // -----------------------------------------------------------------------
    // Step 8: Evaluate — join tool results with ground truth + disorder scores
    // -----------------------------------------------------------------------
    all_results = kmerseek_results
        .mix(foldseek_results)
        .mix(mmseqs2_results)

    // Join each (species, tool, tsv) with its ground truth parquet
    eval_input = all_results
        .join(disprot_gt_ch, by: 0)
        .combine(disorder_scores)
        .map { species, tool, tsv, gt, disorder ->
            tuple(species, tool, tsv, gt, disorder)
        }

    eval_out = evaluateDisprotBenchmark(eval_input)

    // -----------------------------------------------------------------------
    // Step 9: Aggregate all metrics
    // -----------------------------------------------------------------------
    all_parquets = eval_out[0].mix(eval_out[1]).collect()
    agg_out = aggregateDisprotMetrics(all_parquets)

    // -----------------------------------------------------------------------
    // Step 10: Generate report
    // -----------------------------------------------------------------------
    generateReport(agg_out[0], agg_out[2])

    // Summary log
    all_results.subscribe { species, tool, _tsv ->
        log.info "Completed: ${tool} human_vs_${species}"
    }
}
