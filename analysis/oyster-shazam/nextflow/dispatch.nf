nextflow.enable.dsl=2

// Group short operations into one Slurm allocation. Preserve each attempt separately.
process BUILD_CORRECTED_COLLECTIONS {
    cache 'deep'
    input:
    path code
    path source_inputs, stageAs: 'sources/*'
    output:
    tuple path('attempt-*/completed.json'), path('attempt-*/*.tar.gz'), path('attempt-*/build-inputs.json')
    script:
    """
    set -euo pipefail
    attempt="\$PWD/attempt-\$SLURM_JOB_ID"
    mkdir "\$attempt"
    '${params.python}' '${code}/nextflow/build_provenance.py' "\$attempt" '${params.sourmash}' ${task.cpus}
    """
}

workflow {
    if (!workflow.profile.tokenize(',').contains('sherlock'))
        error 'Slurm dispatch test requires sherlock profile'
    if (!params.code || !params.python || !params.sourmash || !params.task_time || !params.task_memory || !params.task_cpus || !params.partition)
        error 'Explicit code, executables, and measured Slurm resources required'
    def group = System.getenv('GROUP_SCRATCH')
    def scratch = System.getenv('SCRATCH')
    def selection = file("${scratch}/oyster-dekupl-queries-20261024/selection.json", checkIfExists: true)
    def selected = new groovy.json.JsonSlurper().parseText(selection.text)
    def inputs = [
        file("${group}/oyster-first-public-linear-20261025/manifest.json", checkIfExists: true),
        selection,
        file("${scratch}/oyster-dekupl-queries-20261024/all-significant-contigs.tsv.gz", checkIfExists: true),
        file(selected.source, checkIfExists: true),
        file("${scratch}/oyster-full-extensions-20261023/splash-trimmed-all/batch-000/compactors.tsv", checkIfExists: true)
    ]
    BUILD_CORRECTED_COLLECTIONS(file(params.code, checkIfExists: true), inputs)
}
