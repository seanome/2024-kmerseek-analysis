#!/usr/bin/env nextflow
/*
 * How much of an invertebrate proteome does conventional sequence search miss?
 *
 * The dark set is every protein in the query proteome that phmmer, jackhmmer and mmseqs2
 * ALL fail to place into the reference at default cutoffs. It is the denominator of the
 * whole proteome-annotate claim: if phmmer already finds the homolog, kmerseek finding it
 * too is not a result. Only the dark set is territory kmerseek can add anything in.
 *
 * The point of this pipeline is that the dark set needs NO answer key. C. gigas has 16
 * reviewed Swiss-Prot entries and Botryllus has 0, so neither can be scored until the
 * structural key exists -- but "no arm hit this protein" is a property of the searches
 * alone. So this runs now, while Chainsaw and InterProScan are still going, and it decides
 * whether the expensive half is worth running at all.
 *
 * Two entry points, because they cost very different amounts:
 *
 *   -entry darkSet     reference + three sequence arms + the count. Cheap, no kmerseek,
 *                      no index. This is the number that comes back first.
 *   -entry kmerseekDark  adds kmerseek on top, mask ON and OFF as a pair.
 *
 * DIRECTION. Every other pipeline here has human as the query and a proteome as the target.
 * This one is the deployment direction: the invertebrate proteome is the QUERY and reviewed
 * Swiss-Prot minus its clade is the reference. kmerseek indexes the TARGET, so the index
 * here is built over ~572_700 reference sequences rather than one proteome -- which is why
 * the mini run is an index-cost probe rather than a small search.
 */

nextflow.enable.dsl = 2

def home = System.getProperty('user.home')

params.species        = null          // cgigas | botryllus
params.registry       = "${projectDir}/../../data/species_metadata.json"
params.qfo_dir        = "${home}/data/quest-for-orthologs/QfO_release_2020_04_with_updated_UP000008143"
params.swissprot_dat  = "${home}/data/uniprot/uniprot_sprot.dat.gz"
params.outdir         = "${home}/data/invertebrate-dark-set/results"
params.reference_cache = null         // shared across species; see buildReference
params.query_fasta    = null          // overrides the registry path
params.exclude_clade  = null          // overrides the registry default rung

params.evalue_report        = 10.0
params.evalue_call          = 1e-3    // what counts as "found something" for the dark set
params.jackhmmer_iterations = 3
params.mmseqs2_sensitivity  = 7

// Queries per task. 45_339 botryllus proteins against 572_700 reference sequences is
// ~1-3 s/query for phmmer and several times that for jackhmmer at 3 iterations, so the
// whole proteome as one task is 12-38h for the cheapest arm and well past that for the
// dearest -- past the 12h walltime, and a scheduler kill reports exitStatus
// Integer.MAX_VALUE, so it would retry into the same wall. 2_000 gives 23 tasks that each
// finish inside an hour or two.
params.query_chunk_size = 2000

// alphabet:ksize pairs. Both defaults are arms with a measured identity-axis row, so a
// zero here can be read against a known baseline rather than guessed at. hp_pbotc is the
// designated best HP arm but was renamed in PR #43 and its post-rename CLI flag has not
// been resolved against the binary, so it is not defaulted in -- add it once confirmed.
params.kmerseek_alphabets = 'hp_thomas_dill2:23,protein20:10'

// The low-complexity mask runs ON and OFF as a PAIR, always, not as a sweep dimension.
// BHF's seven flagship matches included polar-biased low-complexity segments (ZNF292
// pppphpppppppphhppp), so a dark-set hit that does not survive masking is not a finding.
// Reporting one setting alone makes that unfalsifiable.
params.kmerseek_lowcomp = 'true,false'

params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_query_pvalue = 0.05
params.min_region_score = 1.3
params.index_cache        = null
params.with_kmerseek      = false

HMMER   = 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
MMSEQS  = 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'

// The reference is keyed by the CLADE REMOVED, not by the species asking for it. Chordata
// is one file shared by six query species and Metazoa by nine, so a second species costs
// nothing once the first has built it.
process buildReference {
    tag "minus_${clade}"
    storeDir { params.reference_cache ?: "${params.outdir}/reference" }
    label 'python_scoring'
    cpus 2
    memory '8 GB'

    input:
    tuple val(clade), path(swissprot)

    output:
    path "minus_${clade}"

    script:
    """
    set -euo pipefail
    mkdir -p minus_${clade}
    build_clade_excluded_reference.py \\
        --swissprot-dat ${swissprot} \\
        --exclude-clade ${clade} \\
        --out-prefix minus_${clade}/reference \\
        --summary-out minus_${clade}/summary.json
    """
}

process splitQuery {
    tag "${species}"
    label 'python_scoring'
    cpus 1
    memory '4 GB'

    input:
    tuple val(species), path(query)

    output:
    tuple val(species), path("chunks/chunk_*.fasta")

    script:
    """
    set -euo pipefail
    mkdir -p chunks
    split_query_fasta.py --in ${query} --outdir chunks \\
        --chunk-size ${params.query_chunk_size}
    """
}

process phmmerSearch {
    tag "${species}.${chunk.simpleName}_vs_minus_${clade}"
    container HMMER
    label 'high_cpu'
    publishDir "${params.outdir}/${species}/hits", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), val(clade), path(chunk), path(ref_dir)

    output:
    tuple val(species), val('phmmer'), path("${species}.${chunk.simpleName}.phmmer.tsv.gz")

    script:
    """
    set -euo pipefail
    phmmer --domtblout /dev/stdout --tblout /dev/null -o /dev/stderr --noali \\
        -E ${params.evalue_report} --cpu ${task.cpus} \\
        ${chunk} ${ref_dir}/reference.fasta \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$4 "\\t" \$1 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > ${species}.${chunk.simpleName}.phmmer.tsv.gz
    """
}

process jackhmmerSearch {
    tag "${species}.${chunk.simpleName}_vs_minus_${clade}"
    container HMMER
    label 'high_cpu'
    publishDir "${params.outdir}/${species}/hits", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), val(clade), path(chunk), path(ref_dir)

    output:
    tuple val(species), val('jackhmmer'), path("${species}.${chunk.simpleName}.jackhmmer.tsv.gz")

    script:
    """
    set -euo pipefail
    jackhmmer -N ${params.jackhmmer_iterations} \\
        --domtblout /dev/stdout --tblout /dev/null -o /dev/stderr --noali \\
        -E ${params.evalue_report} --cpu ${task.cpus} \\
        ${chunk} ${ref_dir}/reference.fasta \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$4 "\\t" \$1 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > ${species}.${chunk.simpleName}.jackhmmer.tsv.gz
    """
}

process mmseqs2Search {
    tag "${species}.${chunk.simpleName}_vs_minus_${clade}"
    container MMSEQS
    label 'high_cpu'
    publishDir "${params.outdir}/${species}/hits", mode: 'copy', pattern: '*.tsv.gz'

    input:
    tuple val(species), val(clade), path(chunk), path(ref_dir)

    output:
    tuple val(species), val('mmseqs2'), path("${species}.${chunk.simpleName}.mmseqs2.tsv.gz")

    script:
    """
    set -euo pipefail
    mmseqs createdb ${chunk} qdb
    mmseqs createdb ${ref_dir}/reference.fasta tdb
    # --num-iterations 3 is the iterative arm. The dark set is defined against the STRONGEST
    # sequence search available, not the cheapest -- a protein that iterative search reaches
    # is not dark, and counting it as dark would inflate the headline.
    mmseqs search qdb tdb res tmp -s ${params.mmseqs2_sensitivity} \\
        --num-iterations 3 -e ${params.evalue_report} --threads ${task.cpus}
    mmseqs convertalis qdb tdb res out.tsv \\
        --format-output 'query,target,tstart,tend,bits,evalue'
    gzip -c out.tsv > ${species}.${chunk.simpleName}.mmseqs2.tsv.gz
    """
}

// Indexes the SAME clade-excluded reference the sequence arms search. Indexing all of
// reviewed Swiss-Prot instead and dropping self-clade hits afterwards is safe for these two
// species -- Ascidiacea is 0.02% of the database and Bivalvia 0.05%, so the index-time IDF
// barely moves -- but it would leave kmerseek searching a target set the other arms never
// saw, and a win from a hit the baselines had no access to is not a win. With only two
// query clades the saving is one index build; the confound is permanent.
process kmerseekIndex {
    tag "minus_${clade}.${alphabet}.k${ksize}.lc${lowcomp}"
    storeDir { params.index_cache ?: "${params.outdir}/kmerseek_index" }
    container params.kmerseek_image
    cpus 8
    // The 2-letter alphabets at high k over 572_700 sequences are the corner where index
    // memory has never been measured on a reference this size. Doubling on retry rather
    // than guessing a ceiling; an exhausted ladder ends at finish, not at a dead run.
    memory { 64.GB * task.attempt }
    time   { 12.h * task.attempt }

    input:
    tuple val(clade), path(ref_dir), val(alphabet), val(ksize), val(lowcomp)

    output:
    tuple val(clade), val(alphabet), val(ksize), val(lowcomp),
          path("minus_${clade}.${alphabet}.k${ksize}.lc${lowcomp}.kmerseek.rocksdb")

    script:
    def idx = "minus_${clade}.${alphabet}.k${ksize}.lc${lowcomp}.kmerseek.rocksdb"
    def lc  = lowcomp == 'true' ? '--remove-low-complexity' : ''
    """
    set -euo pipefail
    kmerseek index \\
        --alphabet ${alphabet} --ksize ${ksize} \\
        --input  ${ref_dir}/reference.fasta \\
        --output ${idx} ${lc} \\
        --kmer-stats-out ${idx}/spectrum.csv.gz 2>&1 | tee index.log
    """
}

process kmerseekSearch {
    tag "${chunk.simpleName}.${alphabet}.k${ksize}.lc${lowcomp}"
    container params.kmerseek_image
    label 'high_cpu'
    publishDir "${params.outdir}/${species}/kmerseek", mode: 'copy', pattern: '*.zst'

    input:
    tuple val(species), val(clade), path(chunk), val(alphabet), val(ksize), val(lowcomp), path(index_dir)

    output:
    tuple val(species), val(alphabet), val(ksize), val(lowcomp),
          path("${chunk.simpleName}.${alphabet}.k${ksize}.lc${lowcomp}.queries.txt")
    path "*.regions.csv.zst"

    script:
    def slug = "${chunk.simpleName}.${alphabet}.k${ksize}.lc${lowcomp}"
    def lc   = lowcomp == 'true' ? '--remove-low-complexity' : ''
    """
    set -euo pipefail
    kmerseek search \\
        --alphabet ${alphabet} --ksize ${ksize} \\
        --query  ${chunk} --target ${index_dir} ${lc} \\
        --threshold        ${params.threshold} \\
        --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} \\
        --min-region-score ${params.min_region_score} \\
        2> ${slug}.log | zstd -T2 -o ${slug}.regions.csv.zst || true

    # The queries that got any region, as a plain list. Extracted with zstd + awk rather
    # than by reading the .zst in polars: scan_csv on a zstd CSV inflates the whole file in
    # RAM (1.9 GB -> 184.7 GB once, on this project). The query column is found by HEADER
    # NAME, never by position, so a column order change cannot silently shift it.
    if [ -s ${slug}.regions.csv.zst ]; then
        zstd -dc ${slug}.regions.csv.zst \\
        | awk -F, 'NR==1 { for (i=1;i<=NF;i++) if (\$i=="query_name") c=i;
                           if (!c) { print "no query_name column" > "/dev/stderr"; exit 3 }
                           next }
                   c { print \$c }' \\
        | sort -u > ${slug}.queries.txt
    else
        : > ${slug}.queries.txt
    fi
    """
}

process kmerseekDarkGain {
    tag "${species}"
    label 'python_scoring'
    memory '16 GB'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(dark_parquet), path(query_lists)

    output:
    tuple path("${species}_kmerseek_dark_gain.parquet"),
          path("${species}_kmerseek_dark_gain.json")

    script:
    """
    set -euo pipefail
    kmerseek_dark_gain.py \\
        --dark ${dark_parquet} --species ${species} \\
        --queries ${query_lists} \\
        --out ${species}_kmerseek_dark_gain.parquet \\
        --summary-out ${species}_kmerseek_dark_gain.json
    """
}

process computeDarkSet {
    tag "${species}"
    label 'python_scoring'
    memory '16 GB'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(query), path(hits)

    output:
    tuple val(species), path("${species}_dark_set.parquet"), path("${species}_dark_summary.json")

    script:
    """
    set -euo pipefail
    compute_dark_set.py \\
        --query ${query} --species ${species} \\
        --evalue-call ${params.evalue_call} \\
        --hits ${hits} \\
        --out ${species}_dark_set.parquet \\
        --summary-out ${species}_dark_summary.json
    """
}

def registryRow(species) {
    def f = file(params.registry)
    if (!f.exists()) error "no registry at ${params.registry}"
    def reg = new groovy.json.JsonSlurper().parse(f.toFile())
    def row = reg[species]
    if (!row) {
        def known = reg.findAll { k, v -> v instanceof Map && v.annotate_query }.keySet()
        error "unknown species '${species}'. Registry query species: ${known.join(', ')}"
    }
    return row
}

workflow darkSet {
    if (!params.species) error "--species is required (cgigas, botryllus, or any annotate_query row)"

    def row   = registryRow(params.species)
    def clade = params.exclude_clade ?: row.annotate_clade
    if (!clade) error "no clade to exclude for ${params.species}; pass --exclude_clade"

    def qpath = params.query_fasta
        ?: (row.staged_fasta
              ? "${params.qfo_dir}/${row.staged_fasta}"
              : error("no query FASTA for ${params.species}; pass --query_fasta"))
    def query = file(qpath)
    if (!query.exists()) error "query proteome missing: ${query}"

    log.info "  query    : ${params.species} (${query.name})"
    log.info "  reference: reviewed Swiss-Prot minus ${clade}"

    ref_ch = buildReference(Channel.of(tuple(clade, file(params.swissprot_dat))))

    // flatten() so each chunk is its own task rather than all of them arriving as one
    // list to a single task -- which is the whole point of splitting.
    chunks = splitQuery(Channel.of(tuple(params.species, query)))
        .map { sp, cs -> cs }
        .flatten()

    in_ch = chunks.combine(ref_ch).map { c, r -> tuple(params.species, clade, c, r) }

    // Every chunk's hits from every arm land in one list, so computeDarkSet still sees the
    // whole proteome at once. A protein is dark only if NO arm placed it in ANY chunk, and
    // that judgement cannot be made per chunk.
    hits = phmmerSearch(in_ch)
        .mix(jackhmmerSearch(in_ch))
        .mix(mmseqs2Search(in_ch))
        .map { sp, arm, f -> f }
        .collect()

    dark = computeDarkSet(hits.map { h -> tuple(params.species, query, h) })

    // kmerseek is opt-in. The dark set is defined by the sequence arms alone and is worth
    // having on its own; adding kmerseek costs an index over 572_700 sequences per
    // alphabet/ksize/mask, which is the expensive part of this pipeline.
    if (params.with_kmerseek) {
        combos = Channel.fromList(
            params.kmerseek_alphabets.tokenize(',')*.trim().findAll { it }.collectMany { spec ->
                def (a, k) = spec.tokenize(':')
                params.kmerseek_lowcomp.tokenize(',')*.trim().findAll { it }.collect { lc ->
                    tuple(a, k as Integer, lc)
                }
            }
        )

        idx = kmerseekIndex(
            ref_ch.combine(combos).map { r, a, k, lc -> tuple(clade, r, a, k, lc) })

        ks_in = chunks.combine(idx).map { c, cl, a, k, lc, i ->
            tuple(params.species, cl, c, a, k, lc, i)
        }
        ks = kmerseekSearch(ks_in)

        // Every chunk and every combo reaches the gain step together: a protein counts as
        // rescued only against the whole dark set, and the dark set is proteome-wide.
        q_lists = ks[0].map { sp, a, k, lc, f -> f }.collect()
        kmerseekDarkGain(dark.map { sp, dp, _j -> tuple(sp, dp) }.combine(q_lists)
                             .map { sp, dp, q -> tuple(sp, dp, q) })
    }
}

workflow { darkSet() }
