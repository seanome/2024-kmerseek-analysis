#!/usr/bin/env nextflow
/*
 * Run KaMRaT and SPLASH on each array series' raw probe intensities, with no gene annotation.
 * KaMRaT asks which 25-letter k-mers differ between pre and post, the k-mer form of differential expression.
 * SPLASH asks which anchors change their following sequence between samples, as alternative splicing does.
 *
 *   nextflow run fig4_splash_kamrat.nf -profile small --raw <folder> --probes <folder> --series <folder> \
 *       --rscript <path> --splash <folder> --kamrat <sif> --outdir <folder>
 *
 * --raw holds one folder per raw series, from fig4/fetch_raw.py. --probes and --series are fig4.nf's.
 * Each comparison also runs on --shuffles random groupings, to show how many k-mers pass by chance.
 * --only limits the run to one series.
 */
nextflow.enable.dsl = 2

params.outdir   = null
params.raw      = null
params.probes   = null
params.series   = null
params.rscript  = null
params.splash   = null
params.kamrat   = null
params.shuffles = 5
params.only     = null
params.python   = "/scratch/users/olgabot/geo2kmers-test/env/bin/python"
params.groups   = "${projectDir}/results/fig4/corrected_labels/groups.tsv"
params.proteins = "${projectDir}/results/fig4/signature_proteins.fasta"

if (!params.outdir || !params.raw || !params.probes || !params.series) error "set --outdir, --raw, --probes and --series"

def killedFromOutside(status) {
    status == null || status == Integer.MAX_VALUE || status in 128..143
}

// series, its raw folder, and its platform
SERIES = [
    ['GSE104079', 'GSE104079', 'GPL13912'],
    ['GSE27536', 'GSE27536', 'GPL570'],
    ['GSE28998', 'GSE28998', 'GPL570'],
    ['GSE83578', 'GSE83578', 'GPL13667'],
    ['GSE111555', 'GSE111552', 'GPL17586'],
]
COMPARISONS = [
    ['GSE104079', 'main'],
    ['GSE27536', 'main'],
    ['GSE28998', 'main'],
    ['GSE83578', 'pre_vs_30min'],
    ['GSE83578', 'pre_vs_3h'],
    ['GSE83578', 'pre_vs_24h'],
    ['GSE111555', 'main'],
]

process PROBE_INTENSITIES {
    tag "${gse}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    memory 13.GB
    time 2.hour
    storeDir "${params.outdir}/probe_intensities"

    input:
    tuple val(gse), val(raw), val(gpl)

    output:
    tuple val(gse), val(gpl), path("${gse}.probes.tsv.gz"), path("${gse}.features.tsv.gz")

    script:
    def agilent = gpl == 'GPL13912'
    """
    export PYTHONPATH=${projectDir}
    if [ "${agilent}" = "true" ]; then
        ${params.python} ${projectDir}/fig4/agilent_probe_intensities.py ${params.raw}/${raw} ${params.series}/${gse}/${gpl}.txt ${gse}.probes.tsv.gz
    else
        ${params.rscript} ${projectDir}/fig4/affymetrix_probe_intensities.R ${params.raw}/${raw} ${gse}.probes.tsv.gz
    fi
    ${params.python} ${projectDir}/fig4/feature_table.py ${gpl} ${params.probes} ${params.series}/${gse} ${gse}.features.tsv.gz
    """
}

process KAMRAT {
    tag "${gse} ${comparison} ${seed}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 1
    memory 12.GB
    time 1.hour
    maxForks 1
    publishDir "${params.outdir}/kamrat", mode: 'copy', pattern: '*.tsv'

    input:
    tuple val(gse), val(comparison), val(seed), path(probes)

    output:
    path "${gse}.${comparison}.${seed}.kamrat_*.tsv"

    script:
    def name = "${gse}.${comparison}.${seed}"
    def k = "apptainer exec -B \$PWD ${params.kamrat} kamrat"
    """
    ${params.python} ${projectDir}/fig4/kamrat_table.py ${probes} ${params.groups} ${gse} ${comparison} ${seed} table.tsv design.tsv
    mkdir -p index
    ${k} index -intab \$PWD/table.tsv -outdir \$PWD/index -klen 25
    ${k} score -idxdir \$PWD/index -scoreby ttest.padj -design \$PWD/design.tsv -outfmt tab -counts float -outpath \$PWD/scores.tsv
    ${params.python} ${projectDir}/fig4/kamrat_summary.py scores.tsv ${gse} ${comparison} ${seed} ${name}.kamrat_summary.tsv ${name}.kamrat_top.tsv
    if [ "${seed}" = "0" ]; then
        ${k} score -idxdir \$PWD/index -scoreby ttest.padj -design \$PWD/design.tsv -seltop 10000 -outfmt bin -outpath \$PWD/top.bin
        ${k} merge -idxdir \$PWD/index -overlap 24-13 -with \$PWD/top.bin:min -outfmt tab -counts mean:float -outpath \$PWD/${name}.kamrat_contigs.tsv
    fi
    rm -rf index table.tsv scores.tsv
    """
}

process SPLASH {
    tag "${gse}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 4
    memory 13.GB
    time 2.hour
    maxForks 1
    publishDir "${params.outdir}/splash", mode: 'copy', pattern: '*.tsv'

    input:
    tuple val(gse), val(gpl), path(probes), path(features)

    output:
    path "${gse}.splash_*.tsv"

    script:
    """
    export PYTHONPATH=${projectDir}
    ${params.python} ${projectDir}/fig4/splash_reads.py ${probes} ${params.groups} ${gse} reads
    ${params.splash}/splash --bin_path ${params.splash} --anchor_len 15 --gap_len 0 --target_len 10 \
        --anchor_sample_counts_threshold 0 --without_compactors --dump_sample_anchor_target_count_txt \
        --kmc_max_mem_GB 8 --outname_prefix ${gse} reads/input.txt
    ${params.python} ${projectDir}/fig4/splash_groups.py ${gse}.after_correction.scores.tsv ${gse}_dumps ${params.groups} \
        ${gse} ${features} ${params.proteins} .
    rm -rf reads ${gse}_dumps
    """
}

workflow {
    def wanted = { row -> !params.only || row[0] == params.only }
    intensities = PROBE_INTENSITIES(Channel.fromList(SERIES.findAll(wanted)))
    seeds = Channel.fromList(COMPARISONS.findAll(wanted)).flatMap { c -> (0..params.shuffles).collect { s -> tuple(c[0], c[1], s) } }
    KAMRAT(seeds.combine(intensities.map { tuple(it[0], it[2]) }, by: 0))
    SPLASH(intensities)
}
