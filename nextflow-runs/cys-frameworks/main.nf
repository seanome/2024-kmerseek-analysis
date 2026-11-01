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
 * (tables/cys_frameworks/settings.csv). The low-complexity mask is on for cys2 and off for
 * cys_hp3. Under cys2 it drops only the k-mer with no cysteine in it; every k-mer holding a
 * cysteine is kept. See SETTINGS in notebooks/cys_framework_utils.py for why.
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
// pins all three.
//
// kmerseek_image runs indexReference and searchQueries; pipeline_image runs the Python
// steps (heldOutReference, regexScan). They are two parameters because the image is part
// of every task's cache key: a new kmerseek build named in pipeline_image would rebuild
// every reference, and so rerun every regex scan and phmmer search (2 h for Mollusca),
// none of which runs kmerseek.
params.pipeline_image = null
params.kmerseek_image = null
params.hmmer_image    = null

// First-attempt memory, in GB, for a cys2 kmerseek index task. Each retry doubles it.
// 112 is sized for the pilot's reference (27_570 entries: every toxin plus 20_000 other
// entries); see indexGb below for how. A full reference is about 21 times larger, so the
// run-* targets in the Makefile make you set it from the pilot's peak_rss rather than
// inherit this one.
params.cys2_memory_gb = 112

// How many times larger this run's reference is than the pilot's, which every memory ask
// below the cys2 one is multiplied by.
//
// The pilot's reference is 27_570 entries and 8_358_903 residues; a full one is about
// 575_700 entries and 209_017_843 residues, so 25. Both an index's Karlin-Altschul fit and
// a search hold matching regions, and the number of those follows how many seed hits the
// sequences meet, which grows with the reference's residue count. Left at 1 the asks are
// pilot-sized, and the run-* targets in the Makefile make you set it.
params.reference_scale = 1

// Settings to leave out of this run, as 'alphabet:ksize', comma-separated. Empty runs all
// of settings.csv. This is the lever for getting results sooner when one setting is what
// the run is waiting on: on a full reference cys_hp3 k9 asks about 1.6 TB, which only
// three of Sherlock's thirteen bigmem nodes can take, while every other setting fits a
// node there are many of. `--drop_settings cys_hp3:9` runs the other eight.
params.drop_settings = ''

/*
 * Memory for one kmerseek index task, in GB.
 *
 * Most of the memory is not the index itself. `kmerseek index` writes the index in
 * seconds and then fits the Karlin-Altschul lambda and K by searching 200 of the
 * database's own sequences against it (--ka-queries, default 200). The fit's memory
 * follows the number of matching regions it holds.
 *
 * Measured on Sherlock, pilot reference (27_570 entries), 2026-10-15, peak_rss:
 *   cys_hp3 k9   49.8 GB  (453_552_601 fit regions; 32 GB was killed, 64 GB finished)
 *   cys_hp3 k11  11.8 GB
 *   cys_hp3 k13   4.8 GB  (27_017_189 fit regions)
 *   cys_hp3 k15   2.8 GB
 *   cys2 k12-k28 killed in the fit at 180 GB, 360 GB, 720 GB and 1.4 TB. Under cys2 the
 *                k-mer made only of non-cysteine letters is 70-83% of the reference's
 *                k-mers (5.3-6.7 million copies), so the 200 fit sequences meet about
 *                2-3 x 10^11 seed hits. No node holds that; more memory is not the fix.
 *
 * So cys2 runs with the low-complexity mask, which drops that one k-mer. The masked fit
 * meets 7.2-7.5 x 10^8 seed hits at every k from 12 to 28 (counted in Python on the pilot
 * reference, same 200-sequence sample size). cys_hp3 k9 held 4.5 x 10^8 regions in 49.8 GB,
 * about 110 bytes each, and a fit holds at most one region per seed hit, so masked cys2
 * should peak at or below about 82 GB. It asks 112 GB, which still fits a normal node.
 * cys_hp3 asks the measured peak with about a third more on top. A full reference is
 * about 21 times the pilot's, so the run-* targets need these remeasured or scaled.
 */
//
// Measured again on the full references (run-phylum, 2026-10-17, reference_scale 25),
// peak_rss over the three held-out phyla:
//   cys_hp3 k11  545.7-577.2 GB  (48 times the pilot: the fit's regions grow faster than
//                                 the reference at this k; 400 GB would have been killed)
//   cys_hp3 k13  113.1-122.4 GB  (25 times)
//   cys_hp3 k15   62.6-63.2 GB   (22 times)
// So per reference-multiple: k11 30 GB (750 GB at 25), k13 7 GB (175 GB, an hns node),
// k15 4 GB (100 GB, any node), about a third over the largest measured peak. k9 keeps the
// pilot-based 64; at k11's growth it would need well over 2 TB, which is why it is left out.
//
// cys2 was killed at 700 GB on every full reference (run-phylum, 2026-10-16): the fit kept
// every region it found. kmerseek 05b4131 (seanome/kmerseek PR 154) counts the regions per
// score bin as it finds them instead. Measured with that build, cys2 k16, 8 CPUs, Sherlock
// job 48247581:
//   pilot Cnidaria reference          0.58 GB, 1 min 13 s,   85_440_891 fit regions
//   full, Arthropoda held out         9.3 GB, 28 min,   2_087_781_588 fit regions
// The pilot's fit is the same as the old build's (which peaked at 27.1 GB there). Memory
// now follows the reference (16 times the pilot for 25 times the residues), not the
// number of fit regions (24 times), so k12, which meets about as many seed hits as k16
// (pilot peaks with the old build: k12 21.4 GB, k16 27.1 GB), should not need more. The
// run-* targets set CYS2_MEMORY_GB=32: 3.4 times the measured peak, still a normal node.
def indexGb(alphabet, ksize, attempt) {
    def perScale = ksize <= 9 ? 64 : ksize <= 11 ? 30 : ksize <= 13 ? 7 : 4
    def first = alphabet == 'cys2'
              ? params.cys2_memory_gb
              : perScale * params.reference_scale
    bigmemFloor(first * (1 << (attempt - 1)))
}

/*
 * Walltime for one kmerseek index task, in hours.
 *
 * Measured on Sherlock, realtime:
 *              pilot reference   Arthropoda held out (full, reference_scale 25)
 *   cys_hp3 k9   11 min 49 s       not run
 *   cys_hp3 k11   7 min 49 s       killed at the 1 h limit (exit 140)
 *   cys_hp3 k13   4 min 29 s       49 min 5 s
 *   cys_hp3 k15   2 min 39 s       31 min 34 s
 * So a full reference took 11-12 times the pilot's time, about half of reference_scale.
 * Asking the pilot's minutes times the whole reference_scale gives twice the measured
 * time: k11 about 3.3 h for a task expected to take about 1.6 h. cys2 never finished an
 * index on a full reference, so its 4 h is a guess with room in it.
 */
//
// cys2 with the per-bin fit (kmerseek 05b4131): k16 on the full Arthropoda-held-out
// reference took 28 min on 8 CPUs, so cys2 index tasks ask 8 CPUs (cpus below). The old
// build ran k24 and k28 for 52-56 min on 2 CPUs before reaching 700 GB, against 18-20 min
// for k12 and k16, so the larger k may take up to three times longer. 4 h is about eight
// times the k16 measurement and about three times that guess for k24 and k28.
def indexHours(alphabet, ksize, attempt) {
    def pilotMinutes = ksize <= 9 ? 12 : ksize <= 11 ? 8 : ksize <= 13 ? 5 : 3
    def first = alphabet == 'cys2'
              ? 4
              : Math.max(1, Math.ceil(pilotMinutes * params.reference_scale / 60) as int)
    first * attempt
}

// Memory for one kmerseek search task, in GB.
//
// A search holds the query's matching regions, so its memory follows how many seed hits the
// queries meet, which grows with the reference the same way an index's fit does.
//
// Measured on Sherlock over the pilot's 27_570-entry reference, 2026-10-15, peak_rss: every
// search of the pilot's 59 queries peaked between 0.46 GB (cys2 k28) and 3.1 GB (cys2 k12),
// cys_hp3 between 0.85 and 2.7 GB. So 4 GB per reference-multiple covers the largest of them
// with room on top, and the two alphabets are not sized apart, because their peaks overlap.
//
// 4 rather than the index's 8 or 16 on purpose: this is scaled from a measured peak, not
// from an ask that already carried headroom. Scaling the ask instead put every search at
// 400 GB, which routes to bigmem, and bigmem has 13 nodes to hns's 95 -- an over-ask that
// serialises 27 searches behind a handful of nodes costs more wall-clock than it saves.
//
// Measured on the full references (run-phylum, 2026-10-17, reference_scale 25), cys_hp3
// searches peaked at 33.2-92.3 GB, 27-50 times the same setting's pilot peak (largest:
// Mollusca k15, 42.7 GB against 0.85 GB). A full run also searches every query of the
// held-out phylum, not 10 per framework, which is part of that growth. No cys2 search has
// run on a full reference, so cys2 asks its pilot peak times 50, plus about a sixth to a
// third, per k:
//   k    pilot peak   x 50      ask at reference_scale 25
//   12   3.1 GB       155 GB    180 GB (7.2 per multiple; the most an hns node takes)
//   16   1.8 GB        90 GB    120 GB (4.8)
//   20   1.1 GB        55 GB     75 GB (3)
//   24   0.80 GB       40 GB     55 GB (2.2)
//   28   0.46 GB       23 GB     33 GB (1.3)
def searchGb(alphabet, ksize, attempt) {
    def perScale = alphabet != 'cys2' ? 4
                 : ksize <= 12 ? 7.2 : ksize <= 16 ? 4.8 : ksize <= 20 ? 3
                 : ksize <= 24 ? 2.2 : 1.3
    bigmemFloor(Math.ceil(perScale * params.reference_scale * (1 << (attempt - 1))) as int)
}

// The least memory Sherlock's bigmem partition accepts. It refuses anything smaller with
// "ERROR: memory request too low, job not submitted", so an ask between what hns can hold
// and this floor is refused by sbatch rather than queued: on 2026-10-16 a cys_hp3 k13 index
// asking 200 GB was routed to bigmem and rejected. A refused submission is not an exit
// status, so the errorStrategy never saw it, the task went straight to FAILED, and
// indexReference ends at `finish`, which stopped the run from submitting anything new.
BIGMEM_MIN_GB = 256

// Raise an ask that lands on bigmem to bigmem's floor, so the partition and the ask agree.
def bigmemFloor(gb) {
    gb > 180 && gb < BIGMEM_MIN_GB ? BIGMEM_MIN_GB : gb
}

// Sherlock partition that has nodes with this much memory: normal nodes hold about
// 125 GB and hns nodes about 186 GB; past that only bigmem fits.
def sherlockQueue(gb) {
    gb > 180 ? 'bigmem' : gb > 120 ? 'hns' : 'hns,normal'
}

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

    stub:
    """
    touch reference.fasta.gz truth.txt queries.fasta query_table.tsv
    touch counts.${level}.${excluded}.tsv
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
    container params.kmerseek_image
    errorStrategy {
        (task.exitStatus == null
         || task.exitStatus == Integer.MAX_VALUE
         || task.exitStatus in 128..143) && task.attempt <= 3 ? 'retry' : 'finish'
    }
    maxRetries 3
    // 8 for cys2, the CPU count its 28 min full-reference index was measured with.
    cpus { alphabet == 'cys2' ? 8 : 2 }
    memory { indexGb(alphabet, ksize, task.attempt).GB }
    queue { sherlockQueue(indexGb(alphabet, ksize, task.attempt)) }
    time { "${indexHours(alphabet, ksize, task.attempt)}h" }

    input:
    tuple val(level), val(excluded), path(reference), path(truth), path(queries),
          path(query_table), val(alphabet), val(ksize), val(moltype), val(scaled), val(mask)

    output:
    tuple val(level), val(excluded), val(alphabet), val(ksize), val(moltype),
          path("index.db"), path(truth), path(queries), path(query_table), emit: index

    script:
    // --remove-low-complexity is a bare flag on this kmerseek. It is stored in the index,
    // so the search applies the same mask to the queries without being told.
    def mask_flag = mask ? '--remove-low-complexity' : ''
    """
    set -uo pipefail
    kmerseek index \\
        --input ${reference} \\
        --output index.db \\
        --ksize ${ksize} \\
        --alphabet ${moltype} \\
        --scaled ${scaled} ${mask_flag}
    """

    stub:
    """
    mkdir index.db
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
    container params.kmerseek_image
    errorStrategy {
        (task.exitStatus == null
         || task.exitStatus == Integer.MAX_VALUE
         || task.exitStatus in 128..143) && task.attempt <= 3 ? 'retry' : 'ignore'
    }
    maxRetries 3
    memory { searchGb(alphabet, ksize, task.attempt).GB }
    queue { sherlockQueue(searchGb(alphabet, ksize, task.attempt)) }
    // Full-reference cys_hp3 searches took 4 min 47 s to 19 min; pilot cys2 searches took
    // 5-27 s against cys_hp3's 11-31 s, so cys2 should take about as long.
    time { 2.h * task.attempt }
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

    stub:
    """
    touch ${level}.${excluded}.${alphabet}.k${ksize}.hits.csv
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

    stub:
    """
    touch ${level}.${excluded}.regex.d${slack}.hits.csv
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

    stub:
    """
    touch ${level}.${excluded}.phmmer.tbl
    """
}

workflow {
    // kmerseek runs on Sherlock only. Two pilot runs on the laptop under `standard` crashed
    // it, since the local executor does not hold a task to its `memory`.
    if (!workflow.stubRun && !workflow.profile.tokenize(',').contains('sherlock')) {
        error "Run this with -profile sherlock, on Sherlock. Profile '${workflow.profile}' " +
              "is for -stub-run only: kmerseek index under cys2 needs more memory than " +
              "this laptop has."
    }

    if (!workflow.stubRun && !params.kmerseek_image) {
        error "--kmerseek_image is required; nextflow.config pins it"
    }

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
        // quote: '"' so an empty field written as "" reads as empty. Without it the three
        // Opisthacanthus queries, which have no genus or phylum, became a clade named ""
        // (two quote marks) and a reference with nothing held out.
        .splitCsv(header: true, quote: '"')
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

    dropped = params.drop_settings
        ? params.drop_settings.split(',').collect { it.trim() }.findAll { it }
        : []
    settings = Channel
        .fromPath(params.settings)
        .splitCsv(header: true)
        .filter { row -> !dropped.contains("${row.alphabet}:${row.ksize}".toString()) }
        .map { row ->
            tuple(row.alphabet, row.ksize as Integer, row.moltype, row.scaled as Integer,
                  row.mask == 'true')
        }
    if (dropped) {
        log.info "leaving out ${dropped.size()} setting(s): ${dropped.join(', ')}"
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
