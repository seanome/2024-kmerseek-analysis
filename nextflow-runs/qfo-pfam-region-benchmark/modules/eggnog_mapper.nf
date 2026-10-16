/*
 * eggNOG-mapper 2.1.15 with its default search (DIAMOND against eggnog_proteins.dmnd)
 * on the human queries. A reference-database tool: eggNOG 5 contains human, so the row
 * is labelled "reference-database tool, graded on its own models".
 *
 * The call is the PFAMs column of .emapper.annotations: Pfam family names copied from
 * the seed ortholog's orthologous group, with no coordinates. Each family becomes one
 * call over the whole query (bin/refdb_calls.py eggnog), so only the family-level scores
 * mean anything; the card marks the position scores "not applicable" and the tool has
 * no pile-up line. (eggNOG-mapper's --pfam_realign would add coordinates; it is not the
 * default and is not run here.)
 *
 * Data: emapperdb 5.0.2 in $SCRATCH/data/eggnog/emapperdb-5.0.2, from
 * assets/download_eggnog_data.sbatch (`make eggnog-data`): 12_058_580_877 bytes
 * compressed (sum of the three files' Content-Length, 2026-10-16); the unpacked size is
 * written to the run info by du.
 *
 * No history: 8 CPUs, 32 GB, 4 h (prompt G's ask). Record the peak with sacct.
 */

include { retryOnKill } from './annotator_common'

process eggnogMapperSearch {
    tag "eggnog_mapper diamond"
    container 'quay.io/biocontainers/eggnog-mapper@sha256:f70babaf681ff4b6b2fc8e8e76754bf989f01dd4910ca91f156c22aa88ea70d3' // 2.1.15--pyhdfd78af_0
    cpus 8
    memory { 32.GB * task.attempt }
    time { 4.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/hits", mode: 'copy'

    input:
    tuple path(queries), path(data_dir)

    output:
    tuple val("eggnog_mapper"), val("diamond"), path("eggnog_mapper.diamond.emapper.annotations"), emit: hits
    path "eggnog_mapper.diamond.run.txt", emit: run_info

    script:
    def cmd = "emapper.py -i ${queries} --itype proteins -m diamond --data_dir ${data_dir} " +
              "--cpu ${task.cpus} -o eggnog_mapper.diamond --output_dir . " +
              "--temp_dir emapper_tmp --override"
    """
    set -euo pipefail
    mkdir -p emapper_tmp
    ${cmd}
    {
        echo "command: ${cmd}"
        echo "version: \$(emapper.py --version 2>&1 | head -1)"
        echo "cpu_model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
        echo "database_size: \$(du -shL ${data_dir} | cut -f1)"
    } > eggnog_mapper.diamond.run.txt
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/refdb/eggnog_mapper.diamond.emapper.annotations .
    printf 'command: stub\\nversion: stub\\ncpu_model: stub\\ndatabase_size: stub\\n' > eggnog_mapper.diamond.run.txt
    """
}

process pfamNameMap {
    /* Pfam family NAME -> accession, from the same Pfam-A.hmm the other rows use. */
    tag "pfam names"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 2.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path pfam_hmm

    output:
    path "pfam_names.tsv"

    script:
    """
    refdb_calls.py pfam-names --pfam-hmm ${pfam_hmm} --out pfam_names.tsv
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/refdb/pfam_names.tsv .
    """
}

process eggnogCalls {
    tag "eggnog_mapper diamond"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 4.GB * task.attempt }
    time { 1.h * task.attempt }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/regions", mode: 'copy'

    input:
    tuple val(tool), val(setting), path(annotations), path(pfam_names), path(queries)

    output:
    tuple val(tool), val(setting), path("${tool}.${setting}.regions.tsv.gz"), emit: regions
    path "${tool}.${setting}.calls_qc.json", emit: qc

    script:
    """
    refdb_calls.py eggnog --annotations ${annotations} --pfam-names ${pfam_names} \\
        --queries ${queries} --setting ${setting} \\
        --out ${tool}.${setting}.regions.tsv.gz --qc-out ${tool}.${setting}.calls_qc.json
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/refdb/${tool}.${setting}.regions.tsv.gz .
    cp ${projectDir}/tests/fixtures/refdb/${tool}.${setting}.calls_qc.json .
    """
}

workflow EGGNOG_MAPPER {
    take:
    queries
    data_dir
    pfam_hmm

    main:
    eggnogMapperSearch(queries.combine(data_dir))
    names = pfamNameMap(pfam_hmm)
    eggnogCalls(eggnogMapperSearch.out.hits.combine(names).combine(queries))

    emit:
    regions = eggnogCalls.out.regions
}
