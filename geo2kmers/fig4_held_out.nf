#!/usr/bin/env nextflow
/*
 * Test whether k-mers chosen without one person tell that person's pre samples from post, under the rules in
 * fig4/held_out_plan.md, name the genes behind the SPLASH anchors inside one gene cluster, and score each U133 Plus 2
 * series with k-mers chosen on the other, under fig4/cross_series_plan.md.
 *
 *   nextflow run fig4_held_out.nf -profile small --intensities <folder> --series <folder> --splash_hits <file> \
 *       --rscript <path> --outdir <folder>
 *
 * --intensities is fig4_splash_kamrat.nf's probe_intensities folder. --only limits the run to one series.
 *
 * With --hp <folder of hp_kmer_table.py's tables>, it runs instead the same two tests on 30-letter HP k-mers from the
 * proteins behind each probe set, under fig4/hp_tests_plan.md.
 */
nextflow.enable.dsl = 2

params.outdir      = null
params.intensities = null
params.series      = null
params.splash_hits = null
params.rscript     = null
params.shuffles    = 200
params.only        = null
params.hp          = null
params.python      = "/scratch/users/olgabot/geo2kmers-test/env/bin/python"
params.groups      = "${projectDir}/results/fig4/corrected_labels/groups.tsv"
params.signature   = "${projectDir}/results/fig4/signature_proteins.tsv"

if (!params.outdir || !params.series) error "set --outdir and --series"
if (!params.hp && !params.intensities) error "set --intensities, or --hp for the HP k-mer tests"

def killedFromOutside(status) {
    status == null || status == Integer.MAX_VALUE || status in 128..143
}

COMPARISONS = [
    ['GSE28998', 'main'],
    ['GSE83578', 'pre_vs_30min'],
    ['GSE111555', 'main'],
]

// Trained on, its comparison, scored, its comparison: the two series on the U133 Plus 2 array
CROSS = [
    ['GSE28998', 'main', 'GSE27536', 'main'],
    ['GSE27536', 'main', 'GSE28998', 'main'],
]

// Array, series: each series' HP values come from its array's k-mers
HP_SERIES = [
    ['U133_Plus_2', 'GSE28998'],
    ['U133_Plus_2', 'GSE27536'],
    ['HTA_2.0', 'GSE111555'],
]
HP_ALPHABETS = ['hp_lehninger2', 'hp_thomas_dill2']
HP_HELD_OUT_SERIES = ['GSE28998', 'GSE111555']

process HP_VALUES {
    tag "${gse} ${alphabet}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 1
    memory 15.GB
    time 1.hour
    maxForks 2

    input:
    tuple val(array), val(gse), val(alphabet)
    path code

    output:
    tuple val(gse), val(alphabet), path("${gse}.${alphabet}.values.tsv.gz")

    script:
    """
    ${params.python} hp_value_table.py ${params.hp}/${array}.${alphabet}.kmers.npz ${params.hp}/${gse}.probe_set_values.tsv.gz ${gse}.${alphabet}.values.tsv.gz
    """
}

process HP_HELD_OUT {
    tag "${gse} ${alphabet}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 4
    memory 15.GB
    time 3.hour
    maxForks 2
    publishDir "${params.outdir}/hp_held_out", mode: 'copy', pattern: '*.tsv'

    input:
    tuple val(gse), val(alphabet), path(table)
    path code

    output:
    path "${gse}.main.${alphabet}.held_out*.tsv"

    script:
    """
    export PYTHONPATH=${projectDir} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    ${params.python} write_design.py ${params.groups} ${gse} main design.tsv
    ${params.python} held_out_auc.py ${table} design.tsv ${params.series} ${gse} main ${task.cpus} ${params.shuffles} ${gse}.main.${alphabet}
    """
}

process HP_CROSS {
    tag "${train} to ${scored} ${alphabet}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 1
    memory 15.GB
    time 2.hour
    maxForks 2
    publishDir "${params.outdir}/hp_cross_series", mode: 'copy', pattern: '*.tsv'

    input:
    tuple val(alphabet), val(train), path(train_table), val(scored), path(scored_table)
    path code

    output:
    path "${train}_to_${scored}.${alphabet}.cross*.tsv"

    script:
    """
    export PYTHONPATH=${projectDir} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    ${params.python} write_design.py ${params.groups} ${train} main train_design.tsv
    ${params.python} write_design.py ${params.groups} ${scored} main scored_design.tsv
    ${params.python} cross_series_auc.py ${train_table} train_design.tsv ${scored_table} scored_design.tsv ${params.series} ${train} ${scored} main ${params.shuffles} ${train}_to_${scored}.${alphabet}
    """
}

process CROSS_SERIES {
    tag "${train} to ${scored}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 1
    memory 15.GB
    time 2.hour
    maxForks 1
    publishDir "${params.outdir}/cross_series", mode: 'copy', pattern: '*.tsv'

    input:
    tuple val(train), val(train_comparison), val(scored), val(scored_comparison)
    // The scripts are inputs so that editing one reruns this step on -resume
    path code

    output:
    path "${train}_to_${scored}.cross*.tsv"

    script:
    """
    export PYTHONPATH=${projectDir} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    ${params.python} kamrat_table.py ${params.intensities}/${train}.probes.tsv.gz ${params.groups} ${train} ${train_comparison} 0 train.tsv train_design.tsv
    ${params.python} kamrat_table.py ${params.intensities}/${scored}.probes.tsv.gz ${params.groups} ${scored} ${scored_comparison} 0 scored.tsv scored_design.tsv
    ${params.python} cross_series_auc.py train.tsv train_design.tsv scored.tsv scored_design.tsv ${params.series} ${train} ${scored} ${scored_comparison} ${params.shuffles} ${train}_to_${scored}
    rm train.tsv scored.tsv
    """
}

process HELD_OUT {
    tag "${gse} ${comparison}"
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 4
    memory 15.GB
    time 3.hour
    maxForks 1
    publishDir "${params.outdir}/held_out", mode: 'copy', pattern: '*.tsv'

    input:
    tuple val(gse), val(comparison)

    output:
    path "${gse}.${comparison}.held_out*.tsv"

    script:
    """
    export PYTHONPATH=${projectDir} OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    ${params.python} ${projectDir}/fig4/kamrat_table.py ${params.intensities}/${gse}.probes.tsv.gz ${params.groups} ${gse} ${comparison} 0 table.tsv design.tsv
    ${params.python} ${projectDir}/fig4/held_out_auc.py table.tsv design.tsv ${params.series} ${gse} ${comparison} ${task.cpus} ${params.shuffles} ${gse}.${comparison}
    rm table.tsv
    """
}

process NAME_ANCHORS {
    errorStrategy { killedFromOutside(task.exitStatus) && task.attempt <= 2 ? 'retry' : 'terminate' }
    cpus 1
    memory 15.GB
    time 1.hour
    maxForks 1
    publishDir "${params.outdir}/held_out", mode: 'copy', pattern: '*.tsv'

    input:
    // The scripts are inputs so that editing one reruns this step on -resume
    path code

    output:
    path "GSE111555.anchor_*.tsv"

    script:
    """
    export PYTHONPATH=${projectDir}
    ${params.rscript} ${projectDir}/fig4/hta_probe_design.R design.tsv.gz
    ${params.python} ${projectDir}/fig4/name_anchors.py ${params.splash_hits} ${params.intensities}/GSE111555.probes.tsv.gz design.tsv.gz \
        ${params.series}/GSE111555/GPL17586.txt ${params.groups} ${params.signature} GSE111555 main GSE111555
    rm design.tsv.gz
    """
}

workflow {
    if (params.hp) {
        // The scripts are inputs so that editing one reruns the steps that use it on -resume
        code = Channel.fromPath(["${projectDir}/fig4/hp_value_table.py", "${projectDir}/fig4/write_design.py", "${projectDir}/fig4/held_out_auc.py", "${projectDir}/fig4/cross_series_auc.py", "${projectDir}/geo2kmers/held_out.py", "${projectDir}/geo2kmers/people.py"]).collect()
        values = HP_VALUES(Channel.fromList(HP_SERIES).combine(Channel.fromList(HP_ALPHABETS)), code)
        HP_HELD_OUT(values.filter { it[0] in HP_HELD_OUT_SERIES }, code)
        u133 = values.filter { it[0] in ['GSE28998', 'GSE27536'] }
        pairs = u133.combine(u133).filter { it[1] == it[4] && it[0] != it[3] }.map { [it[1], it[0], it[2], it[3], it[5]] }
        HP_CROSS(pairs, code)
    } else {
        HELD_OUT(Channel.fromList(COMPARISONS.findAll { !params.only || it[0] == params.only }))
        if (!params.only) CROSS_SERIES(Channel.fromList(CROSS), Channel.fromPath(["${projectDir}/fig4/kamrat_table.py", "${projectDir}/fig4/cross_series_auc.py", "${projectDir}/geo2kmers/held_out.py", "${projectDir}/geo2kmers/people.py"]).collect())
        if (params.splash_hits && !params.only) NAME_ANCHORS(Channel.fromPath(["${projectDir}/fig4/hta_probe_design.R", "${projectDir}/fig4/name_anchors.py", "${projectDir}/geo2kmers/hta.py"]).collect())
    }
}
