#!/usr/bin/env nextflow
// Time-split test, notebook 280: would kmerseek have placed Pfam domains before Pfam did?
//
// Queries: proteins with no Pfam-A domain in Pfam 33.1 (t0) that gained one by Pfam 38.2
// (t1), plus one dipeptide-shuffled decoy each (bin/make_search_inputs.py). Every tool
// searches Swiss-Prot as of UniProt 2019_08, the t0 reference; bin/score_transfers.py
// scores the family each hit transfers against the t1 domains.
//
// Tools:
//   kmerseek  every alphabet, each k of its range (every other k when the range has more
//             than 5), scaled 1, mask off; two search settings per index: exact, and
//             extend with the alphabet's C_opt and X-drop 4 x C_opt
//   MMseqs2   -s 7 --num-iterations 3, as the dark-set run
//   phmmer    --domE 0.001
//   hmmscan   Pfam 33.1 models, -E 0.001 --domE 0.001, no gathering thresholds
//
// The kmerseek index and search processes, and their memory sizing, are copied from
// nextflow-runs/invertebrate-dark-set/main.nf on olgabot/dark-set-kmerseek-0.4 (PR 57,
// commit ec9846d). That directory is not edited by this pipeline.

nextflow.enable.dsl = 2

params.inputs          = null   // dir from make_search_inputs.py queries: queries.fasta, queries.tsv, instances.tsv
params.reference_dir   = null   // dir from make_search_inputs.py reference: reference.fasta, sprot_domains.tsv
params.settings        = null   // settings.tsv from make_search_inputs.py settings
params.pfam_hmm        = null   // Pfam 33.1 Pfam-A.hmm.gz
params.outdir          = null
params.index_cache     = null   // storeDir for kmerseek indexes, shared by the pilot and the full run
params.hmm_cache       = null
params.query_chunk_size = 2000  // the dark-set run's, so its measured search times carry over
params.read_test_half  = false  // true only for the full run (Stop 3)
params.only_alphabets  = ''     // comma list, for a smaller pilot; empty = every row of settings.tsv

params.kmerseek_image = 'docker.io/olgabot/kmerseek@sha256:7855826572644933e0c71926102b90174e650c144ae90b3641b760bd9f463afb'  // 0.4.0-rc5
HMMER  = 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
MMSEQS = 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'

// kmerseek search filters: the dark-set run's values, so the two Figure 4 experiments match.
params.threshold        = 0.0
params.min_shared_kmers = 2
params.max_query_pvalue = 0.05
params.min_region_score = 1.3
params.kmerseek_extend_xdrop_per_penalty = 4
params.kmerseek_chain_max_gap   = 30
params.kmerseek_chain_max_shift = 10
params.mmseqs2_sensitivity = 7
params.mmseqs2_evalue      = 10.0

// Karlin-Altschul fit queries (dark-set kaQueriesFor): 500 up to 30 bits of keyspace,
// doubling every 3 bits after that, at most 3_000, rounded to 50.
params.kmerseek_ka_queries                    = 500
params.kmerseek_ka_queries_bits_base          = 30
params.kmerseek_ka_queries_bits_per_doubling  = 3
params.kmerseek_ka_queries_max                = 3000

// Memory model, dark-set values; see the long comment above kmerseekIndexMemory there.
params.kmerseek_memory_max          = '500 GB'
params.kmerseek_memory_retry_factor = 2.0
params.kmerseek_index_memory_floor  = '48 GB'
params.kmerseek_search_memory_floor = '24 GB'
params.kmerseek_index_build_gb              = 45
params.kmerseek_ka_fit_memory_gb_at_12_bits = 300
params.kmerseek_ka_fit_memory_halving_bits  = 2.5
params.kmerseek_search_memory_base     = 36
params.kmerseek_search_memory_slope    = 2.34
params.kmerseek_search_memory_headroom = 1.5

def CLASSES = [protein20: 20, uniprot18: 18, hsdm17: 17, wass14: 14, mmseqs12: 12, sdm12: 12,
               funcgroups8: 8, gbmr7: 7, dayhoff6: 6, wwmj5: 5, gbmr4: 4, polarity4: 4,
               hp_lehninger_hpc3: 3]
def classesOf(String a) { CLASSES.containsKey(a) ? CLASSES[a] : (a.startsWith('hp_') ? 2 : null) }
def keyspaceBits(String a, int k) { k * (Math.log(classesOf(a) as double) / Math.log(2.0d)) }

def kaQueriesFor(String alphabet, int ksize) {
    double bits = keyspaceBits(alphabet, ksize)
    int nq = params.kmerseek_ka_queries as int
    if (bits <= (params.kmerseek_ka_queries_bits_base as double)) return nq
    double n = nq * Math.pow(2.0d, (bits - (params.kmerseek_ka_queries_bits_base as double))
                                   / (params.kmerseek_ka_queries_bits_per_doubling as double))
    (int) (Math.round(Math.min(params.kmerseek_ka_queries_max as double, n) / 50.0d) * 50)
}

def memoryLadder(double firstGb, int attempt) {
    long capMb = MemoryUnit.of(params.kmerseek_memory_max).toMega()
    long askMb = (long) (firstGb * 1024L * Math.pow(params.kmerseek_memory_retry_factor as double, attempt - 1))
    MemoryUnit.of("${Math.min(capMb, askMb)} MB")
}

// First asks measured by the dark-set runs: index builds (alphabet|k|scaled -> GB, from its
// assets and runs/ladder-0.4-midi/measured) and searches (alphabet|k -> largest peak RSS and
// longest run time of a 2_000-query chunk, from the work dirs of ladder-0.4 and
// ladder-0.4-midi at scaled 1, read 2026-10-12).
def INDEX_MEASURED = [:]
file("${projectDir}/assets/kmerseek_index_memory_measured.tsv").readLines().drop(1).each { l ->
    def c = l.split('\t'); def key = "${c[0]}|${c[1]}|${c[2]}".toString()
    INDEX_MEASURED[key] = Math.max((INDEX_MEASURED[key] ?: 0) as double, c[3] as double)
}
def SEARCH_MEASURED = [:]
file("${projectDir}/assets/darkset_search_measured_s1.csv").readLines().drop(1).each { l ->
    def c = l.split(','); SEARCH_MEASURED["${c[0]}|${c[1]}".toString()] = [gb: c[6] as double, h: c[5] as double]
}

def kmerseekIndexMemory(String alphabet, int ksize, int attempt) {
    double bits = keyspaceBits(alphabet, ksize)
    int nq      = kaQueriesFor(alphabet, ksize)
    double fit  = (params.kmerseek_ka_fit_memory_gb_at_12_bits as double) *
                  Math.pow(2.0d, -(bits - 12.0d) / (params.kmerseek_ka_fit_memory_halving_bits as double)) * (nq / 500.0d)
    double gb   = 1.4d * Math.max(params.kmerseek_index_build_gb as double, fit)
    double floorGb  = MemoryUnit.of(params.kmerseek_index_memory_floor).toGiga()
    double measured = (INDEX_MEASURED["${alphabet}|${ksize}|1".toString()] ?: 0) as double
    memoryLadder(Math.max(Math.max(floorGb, gb), measured), attempt)
}

// Spectrum load of an index (dark-set spectrumLoad): sum(occ^2 n) / sum(occ n) over its
// spectrum.csv.gz. Read in the workflow, where the real path is known, and cached by name.
def SPECTRUM_LOAD = java.util.Collections.synchronizedMap([:])
def spectrumLoad(Path index_dir) {
    double num = 0.0d, den = 0.0d
    new java.util.zip.GZIPInputStream(index_dir.resolve('spectrum.csv.gz').newInputStream()).withReader('UTF-8') { r ->
        r.eachLine { line ->
            if (line.startsWith('#') || line.startsWith('moltype')) return
            def c = line.split(',')
            double occ = c[2] as double, n = c[3] as double
            num += occ * occ * n; den += occ * n
        }
    }
    den > 0 ? num / den : 0.0d
}

// A search asks 1.5x the largest peak the same alphabet and k reached on a dark-set chunk
// of the same size; with no measurement, the dark-set spectrum-load model.
def searchFirstGb(String alphabet, int ksize, String index_name) {
    def m = SEARCH_MEASURED["${alphabet}|${ksize}".toString()]
    double floorGb = MemoryUnit.of(params.kmerseek_search_memory_floor).toGiga()
    if (m) return Math.max(floorGb, 1.5d * m.gb)
    double load = (SPECTRUM_LOAD[index_name] ?: 0.0d) as double
    Math.max(floorGb, (params.kmerseek_search_memory_headroom as double) *
             ((params.kmerseek_search_memory_base as double) + (params.kmerseek_search_memory_slope as double) * Math.sqrt(load)))
}
def searchHours(String alphabet, int ksize, int attempt) {
    def m = SEARCH_MEASURED["${alphabet}|${ksize}".toString()]
    int first = m ? Math.max(2, (int) Math.ceil(1.5d * m.h)) : 4
    "${Math.min(first * (1 << (attempt - 1)), 24)}h".toString()
}

process kmerseekIndex {
    tag "${alphabet}.k${ksize}"
    storeDir params.index_cache
    container params.kmerseek_image
    cpus 8
    memory { kmerseekIndexMemory(alphabet, ksize as int, task.attempt) }
    time   { "${Math.min(4 * task.attempt, 24)}h" }

    input:
    tuple val(alphabet), val(ksize), val(copt)
    path reference

    output:
    tuple val(alphabet), val(ksize), val(copt), path("sprot2019_08.${alphabet}.k${ksize}.kmerseek.rocksdb")

    script:
    def idx = "sprot2019_08.${alphabet}.k${ksize}.kmerseek.rocksdb"
    def xd  = (params.kmerseek_extend_xdrop_per_penalty as double) * (copt as double)
    def nq  = kaQueriesFor(alphabet, ksize as int)
    """
    set -euo pipefail
    stamp() { while IFS= read -r line; do printf '%(%Y-%m-%dT%H:%M:%S)T %s\\n' -1 "\$line"; done; }
    echo start | stamp | tee index.log
    kmerseek index --alphabet ${alphabet} --ksize ${ksize} --scaled 1 \\
        --input ${reference} --output ${idx} \\
        --extend-mismatch-penalty ${copt} --extend-xdrop ${xd} \\
        --ka-queries ${nq} --ka-survival-out ${idx}/ka_survival.C${copt}.csv \\
        --kmer-stats-out ${idx}/spectrum.csv.gz 2>&1 | stamp | tee -a index.log
    cp index.log ${idx}/index.log
    # Immutable once built (see the dark-set note): read-only files, writable directory.
    chmod -R a-w ${idx}
    chmod u+w ${idx}
    """
}

process kmerseekSearch {
    tag "${alphabet}.k${ksize}.${arm}.${chunk.simpleName}"
    container params.kmerseek_image
    publishDir "${params.outdir}/kmerseek", mode: 'copy'
    cpus 8
    memory { memoryLadder(searchFirstGb(alphabet, ksize as int, index_dir.name), task.attempt) }
    time   { searchHours(alphabet, ksize as int, task.attempt) }

    input:
    tuple val(alphabet), val(ksize), val(copt), path(index_dir), val(arm), path(chunk)
    path domains
    path compact_awk

    output:
    path "${slug}.transfers.tsv.gz", emit: transfers
    path "${slug}.stats.tsv", emit: stats

    script:
    slug = "${alphabet}.k${ksize}.${arm}.${chunk.simpleName}"
    def xd = (params.kmerseek_extend_xdrop_per_penalty as double) * (copt as double)
    // --ka-queries 0: the E-value comes from the fit stored at index time for this penalty
    // and X-drop, or the search is refused (handled below).
    def flags = arm == 'exact' ? '' :
        "--extend-mismatch-penalty ${copt} --extend-xdrop ${xd} --ka-queries 0 " +
        "--chain-max-gap ${params.kmerseek_chain_max_gap} --chain-max-shift ${params.kmerseek_chain_max_shift}"
    """
    set -uo pipefail
    kmerseek search --alphabet ${alphabet} --ksize ${ksize} \\
        --query ${chunk} --target ${index_dir} ${flags} \\
        --threshold ${params.threshold} --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} --min-region-score ${params.min_region_score} \\
        2> ${slug}.log \\
      | tee >(wc -l > raw_rows.txt) \\
      | awk -F, -v domains=${domains} -f ${compact_awk} \\
      | gzip -c > ${slug}.transfers.tsv.gz
    status=(\${PIPESTATUS[@]})
    fit=yes
    if [ "\${status[0]}" -ne 0 ]; then
        if grep -q "no Karlin-Altschul fit" ${slug}.log; then
            # The index-time fit was refused for this penalty: the extend setting cannot be
            # searched on this index. Recorded as unfitted, not as zero transfers.
            fit=no
            printf 'query\\tqstart\\tqend\\ttarget\\ttstart\\ttend\\tpfam_acc\\n' | gzip -c > ${slug}.transfers.tsv.gz
        else
            cat ${slug}.log >&2; exit 1
        fi
    elif [ "\${status[2]}" -ne 0 ] || [ "\${status[3]}" -ne 0 ]; then
        exit 1
    fi
    sleep 1  # let the wc in the process substitution finish writing
    rows=\$(zcat ${slug}.transfers.tsv.gz | tail -n +2 | wc -l)
    printf 'alphabet\\tksize\\tarm\\tchunk\\tc_opt\\tfitted\\traw_rows\\ttransfer_rows\\tbytes\\n' > ${slug}.stats.tsv
    printf '%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\t%s\\n' ${alphabet} ${ksize} ${arm} ${chunk.simpleName} ${copt} \$fit \\
        "\$(cat raw_rows.txt 2>/dev/null || echo NA)" \$rows \$(stat -c %s ${slug}.transfers.tsv.gz) >> ${slug}.stats.tsv
    """
}

process mmseqsSearch {
    tag "${chunk.simpleName}"
    container MMSEQS
    publishDir "${params.outdir}/mmseqs2", mode: 'copy'
    cpus 16
    memory '64 GB'
    time { "${Math.min(8 * task.attempt, 48)}h" }

    input:
    path chunk
    path reference

    output:
    path "${chunk.simpleName}.mmseqs2.tsv.gz"

    script:
    """
    set -euo pipefail
    mmseqs createdb ${chunk} qdb
    mmseqs createdb ${reference} tdb
    # --num-iterations 3: the dark-set run's iterative profile search.
    mmseqs search qdb tdb res tmp -s ${params.mmseqs2_sensitivity} --num-iterations 3 \\
        -e ${params.mmseqs2_evalue} --threads ${task.cpus}
    mmseqs convertalis qdb tdb res out.tsv \\
        --format-output 'query,target,qstart,qend,tstart,tend,pident,bits,evalue'
    gzip -c out.tsv > ${chunk.simpleName}.mmseqs2.tsv.gz
    """
}

process phmmerSearch {
    tag "${chunk.simpleName}"
    container HMMER
    publishDir "${params.outdir}/phmmer", mode: 'copy'
    cpus 8
    memory '16 GB'
    time { "${Math.min(12 * task.attempt, 48)}h" }

    input:
    path chunk
    path reference

    output:
    path "${chunk.simpleName}.phmmer.domtbl.gz"

    script:
    """
    set -euo pipefail
    phmmer --cpu ${task.cpus} --noali -o /dev/null --domtblout out.domtbl --domE 0.001 ${chunk} ${reference}
    gzip -c out.domtbl > ${chunk.simpleName}.phmmer.domtbl.gz
    """
}

process hmmpressPfam {
    storeDir params.hmm_cache
    container HMMER
    cpus 1
    memory '8 GB'
    time '2h'

    input:
    path hmm_gz

    output:
    path "Pfam33.1-A.hmm*"

    script:
    """
    set -euo pipefail
    gzip -dc ${hmm_gz} > Pfam33.1-A.hmm
    hmmpress Pfam33.1-A.hmm
    """
}

process hmmscanSearch {
    tag "${chunk.simpleName}"
    container HMMER
    publishDir "${params.outdir}/hmmscan", mode: 'copy'
    cpus 8
    memory '16 GB'
    time { "${Math.min(12 * task.attempt, 48)}h" }

    input:
    path chunk
    path hmm_files

    output:
    path "${chunk.simpleName}.hmmscan.domtbl.gz"

    script:
    // No --cut_ga: the question is whether Pfam's own t0 model saw the domain and only the
    // gathering threshold kept it out. -E alone filters whole-sequence scores; --domE
    // filters each domain's i-Evalue.
    """
    set -euo pipefail
    hmmscan --cpu ${task.cpus} --noali -o /dev/null --domtblout out.domtbl -E 0.001 --domE 0.001 \\
        Pfam33.1-A.hmm ${chunk}
    gzip -c out.domtbl > ${chunk.simpleName}.hmmscan.domtbl.gz
    """
}

process scoreTransfers {
    container params.kmerseek_image
    publishDir "${params.outdir}/scores", mode: 'copy'
    cpus 4
    memory '64 GB'
    time '8h'

    input:
    path kmerseek_files, stageAs: 'kmerseek/*'
    path kmerseek_stats, stageAs: 'stats/*'
    path mmseqs_files, stageAs: 'mmseqs2/*'
    path phmmer_files, stageAs: 'phmmer/*'
    path hmmscan_files, stageAs: 'hmmscan/*'
    path inputs_dir
    path reference_dir
    path settings
    path score_py
    path common_py

    output:
    path "*"

    script:
    """
    python3 ${score_py} --kmerseek kmerseek --stats stats --mmseqs2 mmseqs2 --phmmer phmmer \\
        --hmmscan hmmscan --inputs ${inputs_dir} --reference ${reference_dir} \\
        --settings ${settings} --out . ${params.read_test_half ? '--read-test-half' : ''}
    """
}

workflow {
    def inputs = file(params.inputs, checkIfExists: true)
    def refdir = file(params.reference_dir, checkIfExists: true)
    def reference = file("${refdir}/reference.fasta", checkIfExists: true)
    def domains   = file("${refdir}/sprot_domains.tsv", checkIfExists: true)
    def only = params.only_alphabets.toString().tokenize(',')*.trim().findAll { it }

    chunks = Channel.fromPath("${inputs}/queries.fasta", checkIfExists: true)
        .splitFasta(by: params.query_chunk_size as int, file: 'chunk')
        .toSortedList { a, b -> a.name <=> b.name }.flatMap { it }

    settings = Channel.fromPath(params.settings, checkIfExists: true)
        .splitCsv(header: true, sep: '\t')
        .filter { !only || it.alphabet in only }
        .map { [it.alphabet, it.ksize as int, it.c_opt] }

    indexes = kmerseekIndex(settings, reference)
        .map { a, k, c, idx -> SPECTRUM_LOAD[idx.name] = spectrumLoad(idx); [a, k, c, idx] }

    searches = indexes
        .combine(Channel.of('exact', 'extend'))
        .combine(chunks)
        .map { a, k, c, idx, arm, ch -> [a, k, c, idx, arm, ch] }
    kmerseekSearch(searches, domains, file("${projectDir}/bin/compact_kmerseek.awk"))

    mm  = mmseqsSearch(chunks, reference)
    ph  = phmmerSearch(chunks, reference)
    hmm = hmmscanSearch(chunks, hmmpressPfam(file(params.pfam_hmm, checkIfExists: true)).collect())

    scoreTransfers(
        kmerseekSearch.out.transfers.collect(), kmerseekSearch.out.stats.collect(),
        mm.collect(), ph.collect(), hmm.collect(),
        inputs, refdir, file(params.settings),
        file("${projectDir}/bin/score_transfers.py"), file("${projectDir}/bin/time_split_common.py"))
}
