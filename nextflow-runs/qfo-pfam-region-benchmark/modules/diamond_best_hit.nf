/*
 * DIAMOND best-hit search: the human queries against the clade-excluded Swiss-Prot.
 *
 * DIAMOND's defaults except the sensitivity mode: E-value 0.001, 25 targets per query
 * (-k 25), one HSP per target. The best of those 25 is chosen later by
 * annotator_best_hit.py (highest bit score), so DIAMOND and MMseqs2 rows are picked by
 * one rule rather than by each tool's own -k / --max-seqs behaviour.
 *
 * Output columns are the annotator hits table: qseq_gapped and sseq_gapped are the
 * aligned residues with gaps, so the domain placement reads the alignment itself.
 *
 * Resource history: the throughput benchmark's ledger has DIAMOND at 17 s and 1.2 GB per
 * 1_000 Swiss-Prot queries (the default mode). --ultra-sensitive is slower; 8 GB and 1 h
 * covers both with room. Record the peak with sacct after the run.
 */

include { retryOnKill } from './annotator_common'

process diamondMakedb {
    tag "diamond makedb"
    container 'quay.io/biocontainers/diamond@sha256:e02013280712f1295e59339dc6ea162adf8a6bbdd730e73a9c00700c03c3e556' // 2.2.8--he361c42_0
    cpus 4
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path reference

    output:
    path "reference.dmnd"

    script:
    """
    diamond makedb --in ${reference} -d reference --threads ${task.cpus}
    """

    stub:
    """
    touch reference.dmnd
    """
}

process diamondBestHitSearch {
    tag "diamond ${setting}"
    container 'quay.io/biocontainers/diamond@sha256:e02013280712f1295e59339dc6ea162adf8a6bbdd730e73a9c00700c03c3e556' // 2.2.8--he361c42_0
    cpus 8
    memory { 8.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/hits", mode: 'copy'

    input:
    tuple val(setting), path(db), path(queries)

    output:
    tuple val("diamond_besthit"), val(setting), path("diamond_besthit.${setting}.hits.tsv.gz"), emit: hits
    path "diamond_besthit.${setting}.run.txt", emit: run_info

    script:
    def cmd = "diamond blastp --query ${queries} --db ${db} --${setting} " +
              "--threads ${task.cpus} --out hits.tsv " +
              "--outfmt 6 qseqid sseqid qstart qend sstart send bitscore evalue pident qseq_gapped sseq_gapped"
    """
    set -euo pipefail
    ${cmd}
    gzip -nc hits.tsv > diamond_besthit.${setting}.hits.tsv.gz
    {
        echo "command: ${cmd}"
        echo "version: \$(diamond version)"
        echo "cpu_model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
    } > diamond_besthit.${setting}.run.txt
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/annotator/diamond_besthit.${setting}.hits.tsv.gz .
    printf 'command: stub\\nversion: stub\\ncpu_model: stub\\n' > diamond_besthit.${setting}.run.txt
    """
}

workflow DIAMOND_BEST_HIT {
    take:
    reference   // path: clade-excluded reference FASTA
    queries     // path: query FASTA
    settings    // list: DIAMOND sensitivity modes, e.g. ['sensitive', 'ultra-sensitive']

    main:
    db = diamondMakedb(reference)
    diamondBestHitSearch(Channel.fromList(settings).combine(db).combine(queries))

    emit:
    hits = diamondBestHitSearch.out.hits
    run_info = diamondBestHitSearch.out.run_info
}
