#!/usr/bin/env nextflow
/*
 * The Fig. 4C-D test: land each array series on the signature proteins and score its samples, with no
 * gene annotation.
 *
 *   nextflow run fig4.nf -profile small --probes <folder> --rscript <path> --outdir results/fig4
 *
 * --rscript is the Rscript of an environment built from fig4/environment-r.yml. When --probes holds
 * GPL570.probes.tsv.gz, GPL13667.probes.tsv.gz and GPL17586.raw.tsv.gz, PROBE_TABLES is skipped.
 * Set NCBI_EMAIL first. Each process runs Python by the absolute path in --python.
 */
nextflow.enable.dsl = 2

params.outdir  = null
params.probes  = null
params.rscript = ''
params.python  = "/scratch/users/olgabot/nb361-pec/env/bin/python"
params.proteins  = "${projectDir}/results/fig4/signature_proteins.fasta"
params.signature = "${projectDir}/results/fig4/signature_proteins.tsv"

if (!params.outdir || !params.probes) error "set --outdir and --probes"

def killedFromOutside(status) {
    status == null || status == Integer.MAX_VALUE || status in 128..143
}

// the array series, with the file that holds each platform's probe sequences
SERIES = [
    ['GSE104079', 'GPL13912', null],
    ['GSE27536', 'GPL570', 'GPL570.probes.tsv.gz'],
    ['GSE28998', 'GPL570', 'GPL570.probes.tsv.gz'],
    ['GSE83578', 'GPL13667', 'GPL13667.probes.tsv.gz'],
    ['GSE111555', 'GPL17586', 'GPL17586.raw.tsv.gz'],
]

process PROBE_TABLES {
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'terminate' }
    maxRetries 3
    memory { 12.GB * task.attempt }
    time { 1.hour * task.attempt }
    storeDir "${params.probes}"

    output:
    path "*.tsv.gz"

    script:
    """
    ${params.rscript} ${projectDir}/fig4/export_probe_tables.R .
    """
}

process FETCH {
    tag "${gse}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'terminate' }
    maxRetries 3
    storeDir "${params.outdir}/series"

    input:
    val gse

    output:
    path "${gse}"

    script:
    """
    #!${params.python}
    import sys
    sys.path.insert(0, "${projectDir}")
    from geo2kmers.cli import main
    raise SystemExit(main(["fetch", "${gse}", "--outdir", "${gse}"]))
    """.stripIndent()
}

process LAND {
    tag "${gse}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'terminate' }
    maxRetries 3
    memory { 12.GB * task.attempt }
    time { 30.min * task.attempt }
    publishDir "${params.outdir}/landing", mode: 'copy'

    input:
    tuple val(gse), val(gpl), val(table), path(series)
    path probes

    output:
    path "${gse}.landing.tsv", emit: landing
    path "${gse}.landing_share.tsv", emit: share

    script:
    def source = table ? "${probes}/${table}" : "${series}/${gpl}.txt"
    """
    PYTHONPATH=${projectDir} ${params.python} ${projectDir}/fig4/land.py ${gse} ${gpl} ${source} ${params.proteins} .
    """
}

process SCORE {
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'terminate' }
    maxRetries 3
    memory { 12.GB * task.attempt }
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path shares, stageAs: 'shares/*'
    path landings, stageAs: 'landing/*'
    path folders, stageAs: 'series/*'

    output:
    path "platforms.tsv"
    path "auc.tsv", emit: auc
    path "sample_scores.tsv"

    script:
    """
    export PYTHONPATH=${projectDir}
    ${params.python} ${projectDir}/fig4/platforms.py shares/* platforms.tsv
    ${params.python} ${projectDir}/fig4/score.py ${params.signature} ${projectDir}/fig4/groups.tsv \
        ${projectDir}/fig4/published_fig4.tsv platforms.tsv series landing .
    """
}

// Runs only after scoring, which never reads the annotation.
process ANNOTATION_CONTROL {
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 3 ? 'retry' : 'terminate' }
    maxRetries 3
    publishDir "${params.outdir}", mode: 'copy'

    input:
    path auc
    path landings, stageAs: 'landing/*'
    path folders, stageAs: 'series/*'

    output:
    path "annotation_control.tsv"

    script:
    """
    PYTHONPATH=${projectDir} ${params.python} ${projectDir}/fig4/annotation_control.py series landing annotation_control.tsv
    """
}

workflow.onComplete {
    file(params.outdir).mkdirs()
    file("${params.outdir}/wall_time.txt").text = "wall_seconds ${workflow.duration.toSeconds()} exit ${workflow.exitStatus}\n"
}

workflow {
    probes = PROBE_TABLES().collect().map { file(params.probes) }.first()
    series = FETCH(Channel.fromList(SERIES.collect { it[0] }))
    tables = Channel.fromList(SERIES).join(series.map { tuple(it.name, it) })
    LAND(tables, probes)
    SCORE(LAND.out.share.collect(), LAND.out.landing.collect(), series.collect())
    ANNOTATION_CONTROL(SCORE.out.auc, LAND.out.landing.collect(), series.collect())
}
