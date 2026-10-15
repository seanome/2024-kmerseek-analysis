#!/usr/bin/env nextflow
/*
 * Short disulfide-rich peptides by cysteine framework.
 *
 * A disulfide-rich peptide is a short secreted peptide, 10 to 60 residues here, held in
 * shape by at least two disulfide bonds. Its cysteine framework is the pattern of
 * cysteine positions, written C-C-CC-C-C and so on, and venom peptide families are named
 * by it. The residues between the cysteines change fast between species, so phmmer often
 * has too little sequence left to align.
 *
 * The question: does a kmerseek setting that gives cysteine a letter of its own recover
 * homologous cysteine-rich peptides across lineages where phmmer cannot?
 *
 * The falsification test, and the reason this pipeline runs the regex at all: if a regular
 * expression built from the cysteine spacing alone recovers the same homologs, the k-mers
 * add nothing. Every kmerseek setting is scored against that regex, framework by
 * framework.
 *
 * One combination of alphabet, k-size, scaled, extension penalty and mask is a "setting"
 * (assets/settings.csv). Masking is off everywhere: under cys2 a stretch with no cysteine
 * is a run of one letter, so the mask would drop the spacers the framework is made of.
 *
 * A reference is Swiss-Prot with one clade removed, so no hit can come from the query's
 * own genus (--level genus) or its own phylum (--level phylum). A true positive is a
 * reference entry carrying UniProt's Toxin keyword, KW-0800.
 *
 * Cost: a reference is indexed once per setting, and an index over all of Swiss-Prot is
 * the expensive step. --level phylum is 3 references, so 3 x settings indexes. --level
 * genus is one reference per genus that has at least --genus_min_queries queries, which
 * is many more. Start with phylum.
 */

nextflow.enable.dsl = 2

params.queries           = "${projectDir}/../../tables/cys_frameworks/queries.csv"
params.reference_entries = "${HOME}/data/cys-frameworks/reference_entries.csv"
params.settings          = "${projectDir}/../../tables/cys_frameworks/settings.csv"
params.swissprot_fasta   = null          // uniprot_sprot.fasta.gz
params.level             = 'phylum'      // 'phylum' or 'genus'
params.genus_min_queries = 20
params.regex_slack       = [0, 2]
params.shuffle_seed      = 0
params.pfam_hmm          = null          // optional, for the hmmsearch comparator
params.best_setting      = null          // 'alphabet:ksize'; the controls use this one
params.outdir            = "${launchDir}/results"

// The shared module the bin/ scripts import. It is staged into each task folder rather
// than imported by path, because notebooks/ does not exist inside the container.
params.utils_module      = "${projectDir}/../../notebooks/cys_framework_utils.py"

// Pilot knobs. Both are 0 for a real run, which is what the defaults are. A pilot proves
// the pipeline runs end to end; its numbers are not the experiment's numbers, because a
// sampled reference holds few of the entries a search could wrongly hit, so precision
// comes out better than it is.
params.only                  = null   // run one clade only, e.g. 'Cnidaria'
params.sample_non_toxins     = 0      // keep all toxins, only this many other entries
params.queries_per_framework = 0      // at most this many queries per framework

// Container images. The pipeline's own image holds kmerseek and polars; phmmer comes
// from its own pinned biocontainer, as the deep-twilight controls do it. nextflow.config
// pins both by digest.
params.pipeline_image = null
params.hmmer_image    = null

/*
 * Write one held-out reference: Swiss-Prot with `excluded` removed at `level`, plus the
 * list of Toxin-keyword accessions left in it and the queries that belong to it.
 */
process heldOutReference {
    tag "${level}=${excluded}"
    publishDir "${params.outdir}/references", mode: 'copy', pattern: '*.tsv'

    container params.pipeline_image

    input:
    tuple val(level), val(excluded)
    path queries
    path reference_entries
    path swissprot
    path utils_module, stageAs: 'cys_framework_utils.py'

    output:
    tuple val(level), val(excluded), path("reference.fasta.gz"), path("truth.txt"),
          path("queries.fasta"), path("query_table.tsv"), emit: reference
    path "counts.${level}.${excluded}.tsv"

    script:
    """
    set -uo pipefail
    export PYTHONPATH="\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
    build_reference.py \\
        --queries ${queries} \\
        --reference-entries ${reference_entries} \\
        --swissprot ${swissprot} \\
        --level ${level} \\
        --excluded '${excluded}' \\
        --out-fasta reference.fasta.gz \\
        --out-truth truth.txt \\
        --out-queries queries.fasta \\
        --out-query-table query_table.tsv \\
        --out-counts counts.${level}.${excluded}.tsv \\
        --sample-non-toxins ${params.sample_non_toxins} \\
        --queries-per-framework ${params.queries_per_framework}
    """
}

/*
 * Index one reference under one setting.
 *
 * Ends at `finish` when it is out of retries: every search against this reference needs
 * the index, so losing it loses that whole branch, and `finish` lets the tasks already
 * running end rather than cancelling them.
 */
process indexReference {
    tag "${level}=${excluded} ${alphabet} k${ksize}"
    container params.pipeline_image
    errorStrategy {
        (task.exitStatus == null
         || task.exitStatus == Integer.MAX_VALUE
         || task.exitStatus in 128..143) && task.attempt <= 3 ? 'retry' : 'finish'
    }
    maxRetries 3
    memory { 32.GB * task.attempt }
    time { 6.h * task.attempt }

    input:
    tuple val(level), val(excluded), path(reference), path(truth), path(queries),
          path(query_table), val(alphabet), val(ksize), val(moltype), val(scaled)

    output:
    tuple val(level), val(excluded), val(alphabet), val(ksize), val(moltype),
          path("index.db"), path(truth), path(queries), path(query_table), emit: index

    script:
    // --remove-low-complexity is a bare flag on this kmerseek, and it is left off on
    // purpose: under cys2 a stretch holding no cysteine is a run of one letter, so the
    // mask would drop exactly the spacers a cysteine framework is made of.
    """
    set -uo pipefail
    kmerseek index \\
        --input ${reference} \\
        --output index.db \\
        --ksize ${ksize} \\
        --alphabet ${moltype} \\
        --scaled ${scaled}
    """
}

/*
 * Search one reference's queries under one setting.
 *
 * Ends at `ignore` when it is out of retries. One setting against one reference is one
 * cell of the results table; the scoring step records a missing cell as missing rather
 * than as a zero, so losing one does not stop the run. That is the opposite of
 * indexReference above, which everything downstream of it depends on.
 */
process searchQueries {
    tag "${level}=${excluded} ${alphabet} k${ksize}"
    container params.pipeline_image
    errorStrategy {
        (task.exitStatus == null
         || task.exitStatus == Integer.MAX_VALUE
         || task.exitStatus in 128..143) && task.attempt <= 3 ? 'retry' : 'ignore'
    }
    maxRetries 3
    memory { 48.GB * task.attempt }
    time { 8.h * task.attempt }
    publishDir "${params.outdir}/kmerseek", mode: 'copy'

    input:
    tuple val(level), val(excluded), val(alphabet), val(ksize), val(moltype), path(index),
          path(truth), path(queries), path(query_table)

    output:
    tuple val(level), val(excluded), val(alphabet), val(ksize),
          path("${level}.${excluded}.${alphabet}.k${ksize}.hits.csv"), emit: hits

    script:
    def stem = "${level}.${excluded}.${alphabet}.k${ksize}"
    """
    set -uo pipefail
    kmerseek search \\
        --query ${queries} \\
        --target ${index} \\
        --ksize ${ksize} \\
        --alphabet ${moltype} \\
        --output ${stem}.hits.csv
    """
}

/*
 * Scan one reference with the regular expression each query's cysteine spacing implies.
 * This is the control the verdict turns on, so it runs on the same reference and is
 * scored by the same rule as the searches.
 */
process regexScan {
    tag "${level}=${excluded} slack=${slack}"
    errorStrategy {
        (task.exitStatus == null
         || task.exitStatus == Integer.MAX_VALUE
         || task.exitStatus in 128..143) && task.attempt <= 3 ? 'retry' : 'ignore'
    }
    maxRetries 3
    memory { 16.GB * task.attempt }
    time { 8.h * task.attempt }
    publishDir "${params.outdir}/regex", mode: 'copy'

    container params.pipeline_image

    input:
    tuple val(level), val(excluded), path(reference), path(truth), path(queries),
          path(query_table), val(slack)
    path utils_module, stageAs: 'cys_framework_utils.py'

    output:
    tuple val(level), val(excluded), val(slack),
          path("${level}.${excluded}.regex.d${slack}.hits.csv"), emit: hits

    script:
    """
    set -uo pipefail
    export PYTHONPATH="\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
    regex_scan.py \\
        --query-table ${query_table} \\
        --reference ${reference} \\
        --slack ${slack} \\
        --out ${level}.${excluded}.regex.d${slack}.hits.csv
    """
}

/*
 * phmmer, each query against the same reference, reported at every E-value so the
 * notebook can cut at 0.01 and at 1 without a second run.
 */
process phmmerSearch {
    tag "${level}=${excluded}"
    container params.hmmer_image
    errorStrategy {
        (task.exitStatus == null
         || task.exitStatus == Integer.MAX_VALUE
         || task.exitStatus in 128..143) && task.attempt <= 3 ? 'retry' : 'ignore'
    }
    maxRetries 3
    cpus 8
    memory { 16.GB * task.attempt }
    time { 12.h * task.attempt }
    publishDir "${params.outdir}/phmmer", mode: 'copy'

    input:
    tuple val(level), val(excluded), path(reference), path(truth), path(queries),
          path(query_table)

    output:
    tuple val(level), val(excluded), path("${level}.${excluded}.phmmer.tbl"), emit: hits

    script:
    """
    set -uo pipefail
    gunzip -c ${reference} > reference.fasta
    phmmer --cpu ${task.cpus} -E 1000 --domE 1000 \\
        --tblout ${level}.${excluded}.phmmer.tbl \\
        -o /dev/null \\
        ${queries} reference.fasta
    """
}

workflow {
    if (!params.swissprot_fasta) {
        error "--swissprot_fasta is required (uniprot_sprot.fasta.gz)"
    }

    queries = file(params.queries)
    reference_entries = file(params.reference_entries)
    swissprot = file(params.swissprot_fasta)

    // Which clades to hold out, read from the query table itself so the pipeline never
    // reads a notebook's output (the lists are derived from files, not pasted in).
    clades = Channel
        .fromPath(params.queries)
        .splitCsv(header: true)
        // One row per query, counted per clade, so --genus_min_queries can be applied
        // without any list of clades written down anywhere. The 1 is what groupTuple
        // collects: its length is the number of queries in that clade.
        .map { row -> tuple(params.level, row[params.level], 1) }
        .filter { level, value, one -> value && value != '' }
        .filter { level, value, one -> !params.only || value == params.only }
        .groupTuple(by: [0, 1])
        .filter { level, value, ones ->
            params.level == 'phylum' ? true : ones.size() >= params.genus_min_queries
        }
        .map { level, value, ones -> tuple(level, value) }

    utils = file(params.utils_module)
    references = heldOutReference(clades, queries, reference_entries, swissprot, utils)
        .reference

    settings = Channel
        .fromPath(params.settings)
        .splitCsv(header: true)
        .map { row ->
            tuple(row.alphabet, row.ksize as Integer, row.moltype, row.scaled as Integer)
        }

    indexes = indexReference(references.combine(settings)).index
    kmerseek_hits = searchQueries(indexes).hits

    regex_hits = regexScan(references.combine(Channel.fromList(params.regex_slack)), utils).hits
    phmmer_hits = phmmerSearch(references).hits

    kmerseek_hits.collectFile(name: 'kmerseek_hits.index.txt', newLine: true,
                              storeDir: "${params.outdir}") { it[4].name }
    regex_hits.collectFile(name: 'regex_hits.index.txt', newLine: true,
                           storeDir: "${params.outdir}") { it[3].name }
    phmmer_hits.collectFile(name: 'phmmer_hits.index.txt', newLine: true,
                            storeDir: "${params.outdir}") { it[2].name }
}
