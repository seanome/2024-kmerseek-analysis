nextflow.enable.dsl=2

process ADOPT_REFERENCE {
    cache 'deep'
    input:
    path manifest
    path sketch, stageAs: 'reference/*'
    path validator
    output:
    tuple path('reference/*', includeInputs: true), path('reference-receipt.json')
    script:
    """
    '${params.python}' '${validator}' adopt reference '${manifest}' '${sketch}' reference-receipt.json
    """
}

process ADOPT_QUERY {
    cache 'deep'
    input:
    path manifest
    path sketch, stageAs: 'query/*'
    path validator
    output:
    tuple path('query/*', includeInputs: true), path('query-receipt.json')
    script:
    """
    '${params.python}' '${validator}' adopt query '${manifest}' '${sketch}' query-receipt.json
    """
}

process INDEX {
    cache 'deep'
    input:
    tuple path(reference), path(receipt)
    path validator
    output:
    tuple path('reference.rocksdb'), path(reference), path(receipt)
    path 'index.log'
    path 'versions.json'
    script:
    """
    '${params.python}' '${validator}' versions '${params.sourmash}' > versions.json
    printf '%s\\n' '${reference}' > reference-paths.txt
    '${params.sourmash}' scripts index reference-paths.txt -o reference.rocksdb -k 31 -m DNA --scaled 100 -c ${task.cpus} > index.log 2>&1
    """
}

process SEARCH {
    cache 'deep'
    input:
    tuple path(query), path(query_receipt)
    tuple path(index), path(reference), path(reference_receipt)
    path validator
    output:
    tuple path(query), path(reference), path(query_receipt), path(reference_receipt), path('matches.csv'), path('search.log')
    path 'versions.json'
    script:
    """
    if ${params.test_stop_before_search}; then
        echo 'Intentional bounded-test stop after upstream completion' >&2
        exit 75
    fi
    '${params.python}' '${validator}' versions '${params.sourmash}' > versions.json
    '${params.sourmash}' scripts manysearch '${query}' '${index}' -o matches.csv -k 31 -m DNA -s 100 -t 0 -A -N --ignore-abundance -c ${task.cpus} > search.log 2>&1
    """
}

process AUDIT {
    cache 'deep'
    input:
    tuple path(query), path(reference), path(query_receipt), path(reference_receipt), path(matches), path(log)
    path validator
    output:
    path 'validated.json'
    script:
    """
    '${params.python}' '${validator}' audit '${query}' '${reference}' '${query_receipt}' '${reference_receipt}' '${matches}' '${log}' validated.json
    """
}

workflow {
    if (!params.access_confirmed || !params.resources_reviewed || !params.ownership_reviewed)
        error 'Blocked: renewed Sherlock access, current resource history, and ownership review required'
    def grouped = workflow.profile.tokenize(',').contains('allocation_test')
    if (grouped && !System.getenv('SLURM_JOB_ID'))
        error 'allocation_test requires an existing Slurm allocation'
    if (!grouped && !workflow.profile.tokenize(',').contains('sherlock'))
        error 'Use the sherlock profile, or allocation_test inside Slurm'
    if (!params.python || !params.sourmash || !params.reference_manifest || !params.query_manifest)
        error 'Explicit interpreter, sourmash and adoption manifest paths are required'
    if (!params.python.startsWith('/') || !params.sourmash.startsWith('/') || params.python.contains("'") || params.sourmash.contains("'"))
        error 'Use absolute executable paths without single quotes'
    if (params.test_stop_before_search && !grouped)
        error 'Intentional test stop is restricted to allocation_test'
    if (!params.task_cpus || !params.task_memory || !params.task_time || !params.partition)
        error 'Measured CPU, memory, time and partition are required'
    def reference_manifest = file(params.reference_manifest, checkIfExists: true)
    def query_manifest = file(params.query_manifest, checkIfExists: true)
    def reference = new groovy.json.JsonSlurper().parseText(reference_manifest.text)
    def query = new groovy.json.JsonSlurper().parseText(query_manifest.text)
    if (!grouped && [reference, query].any { it.provenance_scope == 'historical-build-code-unrecorded-test-only' })
        error 'Historical code provenance is incomplete: production adoption is blocked'
    def validator = file("${projectDir}/artifacts.py")
    ADOPT_REFERENCE(reference_manifest, file(reference.artifact_path, checkIfExists: true), validator)
    ADOPT_QUERY(query_manifest, file(query.artifact_path, checkIfExists: true), validator)
    INDEX(ADOPT_REFERENCE.out, validator)
    SEARCH(ADOPT_QUERY.out, INDEX.out[0], validator)
    AUDIT(SEARCH.out[0], validator)
}
