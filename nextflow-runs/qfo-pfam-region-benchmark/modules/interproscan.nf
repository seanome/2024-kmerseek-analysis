/*
 * InterProScan 5.78-109.0 with every member database it ships switched on (no -appl),
 * on the human queries. A reference-database tool: it annotates against its own models,
 * and InterPro's Pfam member IS the Pfam truth's source, so the row is labelled
 * "reference-database tool, graded on its own models", as hmmscan is.
 *
 * The call is its Pfam matches with their coordinates (bin/refdb_calls.py interproscan);
 * the other members' matches are counted in the QC json, not scored.
 *
 * --disable-precalc: the match lookup service is a web call to EBI, which compute nodes
 * may not reach, and a lookup would hide the compute cost the card reports.
 *
 * Data: the full distribution (interproscan-5.78-109.0-64-bit.tar.gz, 7_037_746_512
 * bytes from EBI's Content-Length, 2026-10-16), unpacked and set up by
 * `make interproscan-data`; its data/ is bound over the image's /opt/interproscan/data.
 *
 * No history: 16 CPUs, 32 GB, 8 h (prompt G's ask). Record the peak with sacct.
 */

include { retryOnKill } from './annotator_common'

process interproscanSearch {
    tag "interproscan all_members"
    container 'docker.io/interpro/interproscan@sha256:dc58b7c147fbbf00c2dd4f5ced42121fc1e8841fcbc7cc2c484380248ff76d11' // 5.78-109.0
    containerOptions "--bind ${params.interproscan_dir}/data:/opt/interproscan/data"
    cpus 16
    memory { 32.GB * task.attempt }
    time { 8.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/hits", mode: 'copy'

    input:
    path queries

    output:
    tuple val("interproscan"), val("all_members"), path("interproscan.all_members.tsv.gz"), emit: hits
    path "interproscan.all_members.run.txt", emit: run_info

    script:
    def cmd = "/opt/interproscan/interproscan.sh --input ${queries} --formats tsv " +
              "--outfile interproscan.tsv --cpu ${task.cpus} --disable-precalc " +
              "--iprlookup --tempdir ips_tmp"
    """
    set -euo pipefail
    mkdir -p ips_tmp
    ${cmd}
    gzip -nc interproscan.tsv > interproscan.all_members.tsv.gz
    {
        echo "command: ${cmd}"
        echo "version: \$(/opt/interproscan/interproscan.sh --version 2>&1 | grep -m1 -i version)"
        echo "cpu_model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
        echo "database_size: \$(du -sh ${params.interproscan_dir}/data | cut -f1)"
    } > interproscan.all_members.run.txt
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/refdb/interproscan.all_members.tsv.gz .
    printf 'command: stub\\nversion: stub\\ncpu_model: stub\\ndatabase_size: stub\\n' > interproscan.all_members.run.txt
    """
}

process interproscanCalls {
    tag "interproscan all_members"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 4.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/regions", mode: 'copy'

    input:
    tuple val(tool), val(setting), path(tsv), path(queries)

    output:
    tuple val(tool), val(setting), path("${tool}.${setting}.regions.tsv.gz"), emit: regions
    path "${tool}.${setting}.calls_qc.json", emit: qc

    script:
    """
    refdb_calls.py interproscan --tsv ${tsv} --queries ${queries} --setting ${setting} \\
        --out ${tool}.${setting}.regions.tsv.gz --qc-out ${tool}.${setting}.calls_qc.json
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/refdb/${tool}.${setting}.regions.tsv.gz .
    cp ${projectDir}/tests/fixtures/refdb/${tool}.${setting}.calls_qc.json .
    """
}

workflow INTERPROSCAN {
    take:
    queries

    main:
    interproscanSearch(queries)
    interproscanCalls(interproscanSearch.out.hits.combine(queries))

    emit:
    regions = interproscanCalls.out.regions
}
