/*
 * MMseqs2 best-hit search: the human queries against the clade-excluded Swiss-Prot,
 * -s 7.5, one round (no --num-iterations; PR 121 found three rounds give shuffled decoy
 * queries hits at E 1e-22). MMseqs2's other defaults: E-value 0.001, --max-seqs 300.
 *
 * -a keeps the alignment backtrace, without which convertalis cannot write qaln/taln.
 *
 * Same image main.nf's mmseqs2Search uses. Resource history: mmseqs2Search seq-seq,
 * 998 midi-plus queries against mouse (20_707 proteins), took 9 s on 16 CPUs at 1.1 GB
 * (run-midi-plus trace 2026-09-01, Slurm job 41609970). Swiss-Prot minus Mammalia is
 * 507_740 proteins, about 25 times mouse: minutes, and a few GB for the prefilter.
 * Asked: 16 CPUs, 16 GB, 1 h. No history at this size: record the peak with sacct.
 */

include { retryOnKill } from './annotator_common'

process mmseqsBestHitDb {
    tag "mmseqs createdb ${name}"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    cpus 2
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    tuple val(name), path(fasta)

    output:
    tuple val(name), path("${name}_db")

    script:
    """
    mkdir -p ${name}_db
    mmseqs createdb ${fasta} ${name}_db/db
    """

    stub:
    """
    mkdir -p ${name}_db && touch ${name}_db/db
    """
}

process mmseqsBestHitSearch {
    tag "mmseqs2 ${setting}"
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    cpus 16
    memory { 16.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/hits", mode: 'copy'

    input:
    tuple val(setting), val(sensitivity), path(query_db), path(target_db)

    output:
    tuple val("mmseqs2_besthit"), val(setting), path("mmseqs2_besthit.${setting}.hits.tsv.gz"), emit: hits
    path "mmseqs2_besthit.${setting}.run.txt", emit: run_info

    script:
    def search = "mmseqs search ${query_db}/db ${target_db}/db result tmp " +
                 "-s ${sensitivity} -a --threads ${task.cpus}"
    def convert = "mmseqs convertalis ${query_db}/db ${target_db}/db result hits.tsv " +
                  "--format-output query,target,qstart,qend,tstart,tend,bits,evalue,pident,qaln,taln " +
                  "--threads ${task.cpus}"
    """
    set -euo pipefail
    mkdir -p tmp
    ${search}
    ${convert}
    gzip -nc hits.tsv > mmseqs2_besthit.${setting}.hits.tsv.gz
    {
        echo "command: ${search} && ${convert}"
        echo "version: \$(mmseqs version)"
        echo "cpu_model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
    } > mmseqs2_besthit.${setting}.run.txt
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/annotator/mmseqs2_besthit.${setting}.hits.tsv.gz .
    printf 'command: stub\\nversion: stub\\ncpu_model: stub\\n' > mmseqs2_besthit.${setting}.run.txt
    """
}

workflow MMSEQS_BEST_HIT {
    take:
    reference     // path: clade-excluded reference FASTA
    queries       // path: query FASTA
    sensitivity   // val: MMseqs2 -s, e.g. 7.5

    main:
    // One process call for both databases: DSL2 allows a process to be called once per
    // workflow.
    dbs = mmseqsBestHitDb(
        Channel.of('query').combine(queries)
            .mix(Channel.of('reference').combine(reference))
    )
    qdb = dbs.filter { it[0] == 'query' }
    tdb = dbs.filter { it[0] == 'reference' }
    mmseqsBestHitSearch(
        Channel.of(["s${sensitivity}", sensitivity])
            .combine(qdb.map { it[1] })
            .combine(tdb.map { it[1] })
    )

    emit:
    hits = mmseqsBestHitSearch.out.hits
    run_info = mmseqsBestHitSearch.out.run_info
}
