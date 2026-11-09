#!/usr/bin/env nextflow
/*
 * Notebook 361: reads per gene, with no genome, for the three RNA-seq series of Domen et al. 2026
 * Fig. 4C-D (GSE120862, GSE99963, GSE144304). Plan and rules:
 * analysis/pec-domen2026-splash/design/fig4_score_from_reads/README.md.
 *
 * It reuses the fastp-trimmed reads and the GENCODE v50 proteins that ../pec-domen2026-human made.
 * Each series is counted on every read and on the first 1/10, 1/100 and 1/1000 of each run's reads.
 * The score itself is computed from the count tables by scripts/fig4_score_from_reads.py.
 * Results are kept with storeDir under params.out_root, so a relaunch reuses them.
 */

params.pec_root = null   // $SCRATCH/nb361-pec
params.out_root = null   // $SCRATCH/nb361-pec/human/fig4_score
params.code_dir = null   // .../analysis/pec-domen2026-splash
params.series   = 'GSE99963,GSE144304,GSE120862'
params.shares   = '1000,100,10,1'   // 1/N of each run's reads; 1 = every read

if (!params.pec_root || !params.out_root || !params.code_dir) {
    error "set --pec_root, --out_root and --code_dir"
}

ENV     = "${params.pec_root}/env/bin"
SCRIPTS = "${params.code_dir}/scripts"
TRIMMED = "${params.pec_root}/human/trimmed"
GENCODE = "${params.pec_root}/human/db/gencode"
O       = params.out_root

// Built from this checkout, so the counter is the committed code. The single-end test needs pigz,
// which is in the environment here and not on the laptop.
process build_tool {
    label 'build'

    input:
    path src, stageAs: 'pec-tools-src'

    output:
    path 'pec-tools'

    script:
    """
    export PATH=${ENV}:\$PATH
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS
    export CARGO_HOME=${params.pec_root}/cargo
    export CARGO_TARGET_DIR=\$PWD/target
    cargo test --release --manifest-path ${src}/Cargo.toml gene_counts > cargo_test.log 2>&1
    grep -q 'single_end_reads_are_counted_without_a_second_mate ... ok' cargo_test.log
    if grep -q 'skipped: pigz' cargo_test.log; then echo 'pigz missing: the single-end test did not run'; exit 4; fi
    cargo build --release --manifest-path ${src}/Cargo.toml > cargo_build.log 2>&1
    cp target/release/pec-tools pec-tools
    """
}

process gene_map {
    label 'build'
    storeDir "${O}"

    output:
    path 'gene_map.tsv'

    script:
    """
    ${ENV}/python ${SCRIPTS}/fig4_reads_inputs.py gene-map ${GENCODE}/gencode.id_map.tsv gene_map.tsv
    """
}

process gene_counts {
    tag "${series} 1/${share}"
    label 'count'
    storeDir "${O}/${series}/share_${share}"
    // sizing evidence: this folder's README
    time   { (share.toInteger() == 1 ? 6.h : share.toInteger() == 10 ? 1.h : 30.min) * task.attempt }

    input:
    tuple val(series), val(share), path(tool), path(map)

    output:
    tuple val(series), val(share), path('gene.counts.tsv'), path('gene.summary.tsv'), path('gene_counts.time'), path('manifest.tsv')

    script:
    """
    export PATH=${ENV}:\$PATH
    python ${SCRIPTS}/fig4_reads_inputs.py manifest ${TRIMMED}/${series} ${share} manifest.tsv
    /usr/bin/time -v -o gene_counts.time ./${tool} gene-counts ${GENCODE}/gencode.faa ${map} manifest.tsv gene ${task.cpus} 2> gene_counts.log
    test -s gene.counts.tsv && test -s gene.summary.tsv
    """
}

workflow {
    tool = build_tool(file("${params.code_dir}/pec-tools", type: 'dir'))
    map  = gene_map()
    jobs = Channel.fromList(params.series.tokenize(','))
        .combine(Channel.fromList(params.shares.toString().tokenize(',')))
        .combine(tool)
        .combine(map)
    gene_counts(jobs)
}
