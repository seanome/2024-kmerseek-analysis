#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------
params.human_fasta = "${System.getProperty('user.home')}/data/gencode/human/v49/gencode.v49.pc_translations.canonical.fa"
params.mouse_fasta = "${System.getProperty('user.home')}/data/gencode/mouse/m38/gencode.vM38.pc_translations.canonical.fa"

// OrthoFinder uses full (non-canonical) translations placed in a single dir
params.orthofinder_input_dir = "${System.getProperty('user.home')}/data/gencode/data-for-orthofinder"

params.ksize = 24
params.scaled = 1
params.moltype = 'hp'
params.outdir = "${launchDir}/results"

// ---------------------------------------------------------------------------
// INDEX the mouse proteome with kmerseek
// ---------------------------------------------------------------------------
process KMERSEEK_INDEX {
    container 'kmerseek:0.2.0'
    tag "mouse_hp_k${params.ksize}"

    publishDir "${params.outdir}/kmerseek", mode: 'copy', pattern: '*.index.log'

    input:
    path mouse_fasta

    output:
    path "${mouse_fasta}.hp.k${params.ksize}.scaled${params.scaled}.kmerseek.rocksdb", type: 'dir'
    path "${mouse_fasta}.hp.k${params.ksize}.scaled${params.scaled}.kmerseek.index.log"

    script:
    def index_log = "${mouse_fasta}.hp.k${params.ksize}.scaled${params.scaled}.kmerseek.index.log"
    """
    echo "=== Kmerseek INDEX: mouse hp k=${params.ksize} ===" | tee ${index_log}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')"              | tee -a ${index_log}
    echo ""                                                   | tee -a ${index_log}

    kmerseek-rust index \\
        --encoding ${params.moltype} \\
        --ksize    ${params.ksize}   \\
        --scaled   ${params.scaled}  \\
        --input    ${mouse_fasta}    \\
        2>&1 | tee -a ${index_log}

    echo "" | tee -a ${index_log}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${index_log}
    """
}

// ---------------------------------------------------------------------------
// SEARCH human proteins against the indexed mouse database
// ---------------------------------------------------------------------------
process KMERSEEK_SEARCH {
    container 'kmerseek:0.2.0'
    tag "human_vs_mouse_hp_k${params.ksize}"

    publishDir "${params.outdir}/kmerseek", mode: 'copy'

    input:
    path human_fasta
    path mouse_index

    output:
    path "human_vs_mouse.hp.k${params.ksize}.results.csv.gz"
    path "human_vs_mouse.hp.k${params.ksize}.search.log"

    script:
    def out_csv = "human_vs_mouse.hp.k${params.ksize}.results.csv.gz"
    def log_file = "human_vs_mouse.hp.k${params.ksize}.search.log"
    """
    echo "=== Kmerseek SEARCH: human vs mouse hp k=${params.ksize} ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')"                         | tee -a ${log_file}
    echo ""                                                              | tee -a ${log_file}

    kmerseek search \\
        --encoding ${params.moltype} \\
        --ksize    ${params.ksize}   \\
        --query    ${human_fasta}    \\
        --target   ${mouse_index}    \\
        2>> ${log_file} | gzip > ${out_csv}

    echo ""                                                                          | tee -a ${log_file}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')"                                       | tee -a ${log_file}
    echo "Rows: \$(zcat ${out_csv} | wc -l)"                                        | tee -a ${log_file}
    """
}

// ---------------------------------------------------------------------------
// RUN OrthoFinder on human + mouse protein FASTAs
// search_mode: 'diamond' | 'diamond_ultra_sens' | 'mmseqs_ultra_sens'
// ---------------------------------------------------------------------------
process ORTHOFINDER {
    container 'orthofinder-procps:latest'
    tag "human_vs_mouse_${search_mode}"

    publishDir "${params.outdir}/orthofinder_${search_mode}", mode: 'copy'

    input:
    path input_dir
    val  search_mode

    output:
    path "OrthoFinder/Results_*", type: 'dir'
    path "orthofinder_${search_mode}.log"

    script:
    def log_file = "orthofinder_${search_mode}.log"
    """
    echo "=== OrthoFinder: human vs mouse (${search_mode}) ===" | tee ${log_file}
    echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    echo "" | tee -a ${log_file}

    orthofinder \\
        -f  ${input_dir} \\
        -t  ${task.cpus} \\
        -a  1            \\
        -og              \\
        -S  ${search_mode} \\
        2>&1 | tee -a ${log_file}

    echo "" | tee -a ${log_file}
    echo "End: \$(date '+%Y-%m-%d %H:%M:%S')" | tee -a ${log_file}
    """
}

// ---------------------------------------------------------------------------
// Workflow
// ---------------------------------------------------------------------------
workflow {
    human_fasta_ch = Channel.fromPath(params.human_fasta, checkIfExists: true)
    mouse_fasta_ch = Channel.fromPath(params.mouse_fasta, checkIfExists: true)
    orthofinder_dir_ch = channel.fromPath(params.orthofinder_input_dir, checkIfExists: true).first()

    // Kmerseek: index mouse, then search human vs mouse
    (mouse_index, _index_log) = KMERSEEK_INDEX(mouse_fasta_ch)
    KMERSEEK_SEARCH(human_fasta_ch, mouse_index)

    // OrthoFinder: run three search modes in parallel
    orthofinder_modes_ch = channel.of('diamond', 'diamond_ultra_sens', 'mmseqs_ultra_sens')
    ORTHOFINDER(orthofinder_dir_ch, orthofinder_modes_ch)
}
