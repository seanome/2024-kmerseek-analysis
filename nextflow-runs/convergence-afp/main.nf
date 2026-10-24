#!/usr/bin/env nextflow
/*
 * Convergence search kill test: does any kmerseek setting reach ice-binding (antifreeze)
 * proteins of other lineages from one query better than two methods that need no search?
 *
 * Every method ranks every entry of reviewed Swiss-Prot (one release) for every query.
 * The queries are one real ice-binding protein per lineage plus, per real query, 20
 * residue shuffles (null A) and 20 within-repeat reorderings (null B); see
 * bin/prep_inputs.py. This pipeline only produces the ranked lists. Recall at the first
 * false positive, the lineage grid and the verdict are computed in the notebook.
 *
 * Methods:
 *   kmerseek    every setting in assets/kmerseek_settings.tsv (the dark-set midi run's
 *               230 settings, kmerseek 0.4 with the Karlin-Altschul E-value path). One task
 *               per setting builds the Swiss-Prot index, fits the E-value statistics,
 *               searches all queries and deletes the index.
 *   phmmer, jackhmmer (3 iterations), MMseqs2 (-s 7.5)
 *   Foldseek    against the AFDB Swiss-Prot database: real queries from their AFDB
 *               models; every query, real and null, from ProstT5-predicted 3Di letters
 *               (a shuffled sequence has no structure).
 *   scans       composition, hydropathy window, period and published-motif ranking,
 *               none of which searches (bin/scans.py).
 *
 * In code, a kmerseek setting is called setting_label (other pipelines here call it
 * arm_label); it is the name every kmerseek output file carries.
 *
 * `-entry score` (make score) turns the published ranked lists into the small tables the
 * notebook reads: metrics.tsv, grid.tsv, verdict.json and hits_residues.md.
 */

nextflow.enable.dsl = 2

params.swissprot_fasta = "${System.getProperty('user.home')}/data/uniprot/uniprot_sprot.2026_03.fasta.gz"
params.truth_dir       = "${projectDir}/../../data/convergence_afp"
params.outdir          = "${launchDir}/results"
params.db_cache        = "${params.outdir}/db_cache"
// The interpreter for every Python step, by full path. Inside the pipeline image it is the
// image's own python3, which has polars and numpy; a run without containers passes the
// conda env's, e.g. --python /Users/olga/anaconda3/envs/2025-kmerseek-analysis/bin/python3
params.python          = '/usr/bin/python3'

// Query set
params.n_draws  = 20
params.seed     = 20261014
params.subset   = 0       // test runs only: truth entries plus this many other entries
// Window lengths for the hydropathy window scan (bin/scans.py).
params.window_lengths = '9,19' 

// kmerseek
params.kmerseek_image    = 'docker.io/olgabot/kmerseek:0.4.0-pr141-3c353d4'
params.settings_table    = "${projectDir}/assets/kmerseek_settings.tsv"
// all | pilot | comma-separated setting labels. pilot = each alphabet's lowest and highest
// k at scaled 1 and its lowest k at scaled 10: the settings that bracket memory (lowest k)
// and E-value fit time (highest k, which fits on the most queries).
params.settings          = 'all'
params.resources         = null   // resources.tsv from bin/plan_resources.py
params.kmerseek_default_memory_gb = 128
params.kmerseek_default_time_h    = 8
params.kmerseek_memory_max_gb     = 500
params.kmerseek_time_max_h        = 72
// Search flags of the dark-set run, so the settings mean the same thing here.
params.threshold         = 0.0
params.min_shared_kmers  = 2
params.max_query_pvalue  = 0.05
params.min_region_score  = 1.3
params.chain_max_gap     = 30
params.chain_max_shift   = 10

// Sequence tools. Report thresholds are loose on purpose: every method should rank as
// many entries as it will, and the comparison is made at each method's own first false
// positive, not at an E-value.
params.evalue_report        = 1000
params.max_seqs             = 10000
params.jackhmmer_iterations = 3
params.mmseqs2_sensitivity  = 7.5
params.hmmer_chunk_size     = 25

// Foldseek
params.query_structures  = "${params.truth_dir}/structures"
params.foldseek_db_name  = 'Alphafold/Swiss-Prot'
params.foldseek_target_db = null  // a directory holding <prefix> files from `foldseek databases`
params.prostt5_weights   = null   // a directory from `foldseek databases ProstT5`; null downloads

params.skip_kmerseek = false
params.skip_foldseek = false

params.hmmer_image    = 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
params.mmseqs_image   = 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
params.foldseek_image = 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'

// A scheduler kill reports Integer.MAX_VALUE, not null and not 128..143.
def killedByCluster(task) {
    def s = task.exitStatus
    s == null || s == Integer.MAX_VALUE || (s >= 128 && s <= 143)
}

def readSettings() {
    def lines = file(params.settings_table).readLines().findAll { it.trim() }
    def header = lines[0].tokenize('\t')
    lines.drop(1).collect { [header, it.tokenize('\t')].transpose().collectEntries() }
}

def pickSettings(List rows) {
    def spec = params.settings.toString().trim()
    if (spec == 'all') return rows
    if (spec == 'pilot') {
        def picked = []
        rows.groupBy { it.alphabet }.each { a, rs ->
            def s1 = rs.findAll { it.scaled == '1' }.sort { it.ksize as int }
            def s10 = rs.findAll { it.scaled == '10' }.sort { it.ksize as int }
            if (s1) { picked << s1.first(); picked << s1.last() }
            if (s10) picked << s10.first()
        }
        return picked.unique { it.setting_label }
    }
    def want = spec.tokenize(',')*.trim()
    def unknown = want - rows*.setting_label
    if (unknown) error "--settings names unknown setting(s): ${unknown.join(', ')}"
    rows.findAll { it.setting_label in want }
}

def loadResources() {
    if (!params.resources) return [:]
    def lines = file(params.resources).readLines().findAll { it.trim() }
    def header = lines[0].tokenize('\t')
    lines.drop(1).collect { [header, it.tokenize('\t')].transpose().collectEntries() }
         .collectEntries { [(it.setting_label): [it.memory_gb as int, it.time_h as int]] }
}

process prepInputs {
    label 'python_scoring'
    container params.kmerseek_image
    publishDir "${params.outdir}/inputs", mode: 'copy'
    cpus 2
    memory '16 GB'
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    path swissprot
    path truth_dir
    path motifs
    path classes

    output:
    path 'reference.fasta',        emit: reference
    path 'reference_lengths.tsv',  emit: lengths
    path 'queries.fasta',          emit: queries
    path 'draws.tsv',              emit: draws

    script:
    """
    ${params.python} ${projectDir}/bin/prep_inputs.py \\
        --swissprot-fasta ${swissprot} \\
        --queries ${truth_dir}/queries.tsv --truth ${truth_dir}/truth_set.tsv \\
        --motifs ${motifs} --alphabet-classes ${classes} \\
        --n-draws ${params.n_draws} --seed ${params.seed} --subset ${params.subset} \\
        --outdir .
    """
}

process scans {
    label 'python_scoring'
    container params.kmerseek_image
    publishDir "${params.outdir}/hits/scans", mode: 'copy'
    cpus 4
    memory { "${96 * task.attempt} GB" }
    time { "${12 * task.attempt}h" }
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    // The script is a staged input, not read from projectDir: Nextflow hashes a task from
    // its inputs, so a path interpolated into the script text leaves an edited scan
    // looking identical and the stored result is served instead of recomputed.
    path script
    path reference
    path queries
    path draws
    path truth_dir
    path classes
    path motifs
    path eisenberg
    path kyte_doolittle

    output:
    path 'scans.parquet'

    script:
    """
    ${params.python} ${script} \\
        --reference ${reference} --queries ${queries} --draws ${draws} \\
        --truth ${truth_dir}/truth_set.tsv --alphabet-classes ${classes} \\
        --motifs ${motifs} --eisenberg ${eisenberg} --kyte-doolittle ${kyte_doolittle} \\
        --window-lengths ${params.window_lengths} --out scans.parquet
    """
}

// tblout, not domtblout: the test ranks whole entries, and tblout's full-sequence score
// and E-value are one row per query and target. Columns: $1 target, $3 query,
// $5 full-sequence E-value, $6 full-sequence bit score.
// A chunk from splitFasta is queries.<n>.fasta. Its simpleName is "queries" for every
// chunk, so every chunk's output had one name and each publish overwrote the last: the
// first ice-binding run kept 24 of 246 queries for phmmer and jackhmmer. baseName keeps
// the chunk number (queries.<n>). Fixed 2026-10-20.
process phmmerSearch {
    tag "${chunk.baseName}"
    container params.hmmer_image
    label 'high_cpu'
    publishDir "${params.outdir}/hits/phmmer", mode: 'copy'
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    tuple path(chunk), path(reference)

    output:
    path "${chunk.baseName}.phmmer.tsv.gz"

    script:
    """
    set -euo pipefail
    phmmer --tblout hits.tbl -o /dev/null --noali \\
        -E ${params.evalue_report} --cpu ${task.cpus} ${chunk} ${reference}
    { printf 'query_id\\ttarget\\tscore\\tevalue\\n'
      { grep -v '^#' hits.tbl || true; } | awk 'NF >= 18 {print \$3 "\\t" \$1 "\\t" \$6 "\\t" \$5}'; } \\
      | gzip -c > ${chunk.baseName}.phmmer.tsv.gz
    """
}

// jackhmmer's --tblout holds the hits of the last iteration it ran.
process jackhmmerSearch {
    tag "${chunk.baseName}"
    container params.hmmer_image
    label 'high_cpu'
    publishDir "${params.outdir}/hits/jackhmmer", mode: 'copy'
    time { "${12 * task.attempt}h" }
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    tuple path(chunk), path(reference)

    output:
    path "${chunk.baseName}.jackhmmer.tsv.gz"

    script:
    """
    set -euo pipefail
    jackhmmer -N ${params.jackhmmer_iterations} --tblout hits.tbl -o /dev/null --noali \\
        -E ${params.evalue_report} --cpu ${task.cpus} ${chunk} ${reference}
    { printf 'query_id\\ttarget\\tscore\\tevalue\\n'
      { grep -v '^#' hits.tbl || true; } | awk 'NF >= 18 {print \$3 "\\t" \$1 "\\t" \$6 "\\t" \$5}'; } \\
      | gzip -c > ${chunk.baseName}.jackhmmer.tsv.gz
    """
}

process mmseqs2Search {
    container params.mmseqs_image
    label 'high_cpu'
    publishDir "${params.outdir}/hits/mmseqs2", mode: 'copy'
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    path queries
    path reference

    output:
    path 'mmseqs2.tsv.gz'

    script:
    """
    set -euo pipefail
    mmseqs createdb ${queries} qdb
    mmseqs createdb ${reference} tdb
    mmseqs search qdb tdb res tmp -s ${params.mmseqs2_sensitivity} \\
        -e ${params.evalue_report} --max-seqs ${params.max_seqs} --threads ${task.cpus}
    mmseqs convertalis qdb tdb res out.tsv --format-output 'query,target,bits,evalue'
    { printf 'query_id\\ttarget\\tscore\\tevalue\\n'; cat out.tsv; } | gzip -c > mmseqs2.tsv.gz
    """
}

process prostt5Weights {
    container params.foldseek_image
    storeDir "${params.db_cache}/prostt5"
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    output:
    path 'weights'

    script:
    """
    set -euo pipefail
    mkdir -p tmp
    foldseek databases ProstT5 weights tmp
    """
}

process foldseekTargetDb {
    container params.foldseek_image
    label 'high_cpu'
    storeDir "${params.db_cache}/foldseek"
    time '12h'
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    output:
    path 'afdb_swissprot'

    script:
    """
    set -euo pipefail
    mkdir -p afdb_swissprot tmp
    foldseek databases ${params.foldseek_db_name} afdb_swissprot/db tmp --threads ${task.cpus}
    # The accessions that have a model, so the notebook can count Swiss-Prot entries
    # without one: each is a miss for Foldseek.
    awk '{print \$2}' afdb_swissprot/db.lookup \\
      | sed -E 's/^AF-([A-Z0-9]+)-F[0-9]+.*/\\1/' | sort -u > afdb_swissprot/accessions.txt
    """
}

process foldseekQueryDbs {
    container params.foldseek_image
    label 'high_cpu'
    memory { "${32 * task.attempt} GB" }
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    path structures
    path queries
    path weights

    output:
    tuple val('structure'), path('qdb_structure'), emit: structure
    tuple val('prostt5'),   path('qdb_prostt5'),   emit: prostt5

    script:
    """
    set -euo pipefail
    mkdir -p qdb_structure qdb_prostt5 cifs
    cp ${structures}/*.cif cifs/
    foldseek createdb cifs qdb_structure/db --threads ${task.cpus}
    foldseek createdb ${queries} qdb_prostt5/db --prostt5-model ${weights} --threads ${task.cpus}
    """
}

process foldseekSearch {
    tag "${mode}"
    container params.foldseek_image
    label 'high_cpu'
    publishDir "${params.outdir}/hits/foldseek", mode: 'copy'
    memory { "${64 * task.attempt} GB" }
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    tuple val(mode), path(qdb), path(tdb)

    output:
    path "foldseek_${mode}.tsv.gz"

    script:
    // Structure-mode query names are AF-<acc>-F1-model_v6; they become <acc>__real so
    // every method keys queries the same way. Target names become the bare accession.
    """
    set -euo pipefail
    mkdir -p tmp
    foldseek search ${qdb}/db ${tdb}/db res tmp -e ${params.evalue_report} \\
        --max-seqs ${params.max_seqs} --threads ${task.cpus}
    foldseek convertalis ${qdb}/db ${tdb}/db res out.tsv --format-output 'query,target,bits,evalue'
    { printf 'query_id\\ttarget\\tscore\\tevalue\\n'
      awk -F'\\t' 'BEGIN{OFS="\\t"} {
          if (match(\$1, /^AF-[A-Z0-9]+-F[0-9]+/)) { split(substr(\$1, 1, RLENGTH), p, "-"); \$1 = p[2] "__real" }
          if (match(\$2, /AF-[A-Z0-9]+-F[0-9]+/))  { split(substr(\$2, RSTART, RLENGTH), p, "-"); \$2 = p[2] }
          print }' out.tsv; } | gzip -c > foldseek_${mode}.tsv.gz
    """
}

process kmerseekRun {
    tag "${s.setting_label}"
    label 'kmerseek'
    container params.kmerseek_image
    publishDir "${params.outdir}/hits/kmerseek", mode: 'copy'
    memory { "${Math.min(params.kmerseek_memory_max_gb as int, (mem_gb as int) * (1 << (task.attempt - 1)))} GB" }
    time   { "${Math.min(params.kmerseek_time_max_h as int, (time_h as int) * (1 << (task.attempt - 1)))}h" }
    // A kill retries twice with doubled memory and time; a setting that still cannot run
    // is left out (the notebook lists missing settings), not allowed to stop the run.
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'ignore' }

    input:
    tuple val(s), val(mem_gb), val(time_h), path(reference), path(queries), path(truth_dir)

    output:
    path "${s.setting_label}.status.tsv",    emit: status
    path "${s.setting_label}.best.parquet",  optional: true, emit: best
    path "${s.setting_label}.truth.parquet", optional: true, emit: truth
    path "${s.setting_label}.ka_survival.csv", optional: true
    path "${s.setting_label}.log"

    script:
    def lab = s.setting_label
    def lc  = s.low_complexity_mask == 'true' ? '--remove-low-complexity' : ''
    def ext = "--extend-mismatch-penalty ${s.penalty} --extend-xdrop ${s.xdrop}"
    """
    set -euo pipefail
    stamp() { while IFS= read -r line; do printf '%(%Y-%m-%dT%H:%M:%S)T %s\\n' -1 "\$line"; done; }
    echo "kmerseek \$(cat /etc/kmerseek_git_sha 2>/dev/null || echo unknown-commit) ${lab}" | stamp > ${lab}.log
    kmerseek index --alphabet ${s.alphabet} --ksize ${s.ksize} --scaled ${s.scaled} \\
        --input ${reference} --output idx ${lc} ${ext} --ka-queries ${s.ka_queries} \\
        --ka-survival-out ${lab}.ka_survival.csv 2>&1 | stamp >> ${lab}.log
    echo "index size: \$(du -sh idx | cut -f1)" | stamp >> ${lab}.log

    # --ka-queries 0: use the fit the index just stored, or refuse. A refused fit is a
    # result of the setting (no E-value can be computed), recorded as nofit, not a failure.
    set +e
    kmerseek search --alphabet ${s.alphabet} --ksize ${s.ksize} \\
        --query ${queries} --target idx ${lc} ${ext} --ka-queries 0 \\
        --chain-max-gap ${params.chain_max_gap} --chain-max-shift ${params.chain_max_shift} \\
        --threshold ${params.threshold} --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} --min-region-score ${params.min_region_score} \\
        > regions.csv 2> search.err
    rc=\$?
    set -e
    stamp < search.err >> ${lab}.log
    if [ "\$rc" -ge 128 ]; then
        echo "kmerseek search died by signal (exit \$rc)" >&2; exit "\$rc"
    fi
    if [ "\$rc" -ne 0 ]; then
        if grep -q "no Karlin-Altschul fit" search.err; then
            printf 'setting_label\\tstatus\\n${lab}\\tnofit\\n' > ${lab}.status.tsv
            rm -rf idx regions.csv
            exit 0
        fi
        echo "kmerseek search failed (exit \$rc)" >&2; exit 1
    fi
    ${params.python} ${projectDir}/bin/collapse_kmerseek.py --regions regions.csv \\
        --truth ${truth_dir}/truth_set.tsv --label ${lab} 2>&1 | stamp >> ${lab}.log
    printf 'setting_label\\tstatus\\n${lab}\\tsearched\\n' > ${lab}.status.tsv
    rm -rf idx regions.csv
    """
}

// Scoring reads the published hits rather than taking them from the search channels, so
// it is its own entry: `-entry score` turns the ranked lists into the notebook's tables
// without the searches having to be in the same run. The hits directory is a staged input,
// not a path pasted into the script, so the container sees it and an edited script
// changes the task hash.
process scoreLists {
    label 'python_scoring'
    container params.kmerseek_image
    publishDir "${params.outdir}", mode: 'copy'
    cpus 2
    memory { "${32 * task.attempt} GB" }
    time { "${4 * task.attempt}h" }
    errorStrategy { killedByCluster(task) && task.attempt <= 2 ? 'retry' : 'finish' }

    input:
    path script
    path hits
    path truth_dir
    path draws
    path classes
    path afdb

    output:
    path 'scored'

    script:
    def afdb_flag = afdb.name == 'NO_AFDB' ? '' : "--afdb-accessions ${afdb}"
    """
    ${params.python} ${script} \\
        --hits ${hits} --truth ${truth_dir}/truth_set.tsv \\
        --queries ${truth_dir}/queries.tsv --draws ${draws} \\
        --alphabet-classes ${classes} ${afdb_flag} --outdir scored
    """
}

workflow score {
    def truth_dir = file(params.truth_dir, checkIfExists: true)
    scoreLists(
        file("${projectDir}/bin/score.py", checkIfExists: true),
        file("${params.outdir}/hits", checkIfExists: true, type: 'dir'),
        truth_dir,
        file("${params.outdir}/inputs/draws.tsv", checkIfExists: true),
        file("${projectDir}/assets/alphabet_classes.tsv", checkIfExists: true),
        // The Foldseek database's accession list, when that database was built here. The
        // placeholder keeps the input present for a run that skipped Foldseek.
        file("${params.db_cache}/foldseek/afdb_swissprot/accessions.txt").exists()
            ? file("${params.db_cache}/foldseek/afdb_swissprot/accessions.txt")
            : file("${projectDir}/assets/NO_AFDB"),
    )
}

workflow {
    def truth_dir = file(params.truth_dir, checkIfExists: true)
    def motifs    = file("${projectDir}/assets/known_motifs.tsv", checkIfExists: true)
    def classes   = file("${projectDir}/assets/alphabet_classes.tsv", checkIfExists: true)
    def eisenberg = file("${projectDir}/assets/eisenberg_scale.tsv", checkIfExists: true)
    def kyte_doolittle = file("${projectDir}/assets/kyte_doolittle_scale.tsv", checkIfExists: true)

    prep = prepInputs(file(params.swissprot_fasta, checkIfExists: true), truth_dir, motifs, classes)

    scans(file("${projectDir}/bin/scans.py", checkIfExists: true), prep.reference,
          prep.queries, prep.draws, truth_dir, classes, motifs, eisenberg, kyte_doolittle)

    chunks = prep.queries.splitFasta(by: params.hmmer_chunk_size, file: true)
    hmmer_in = chunks.combine(prep.reference)
    phmmerSearch(hmmer_in)
    jackhmmerSearch(hmmer_in)
    mmseqs2Search(prep.queries, prep.reference)

    if (!params.skip_foldseek) {
        weights = params.prostt5_weights ? channel.value(file(params.prostt5_weights, checkIfExists: true))
                                         : prostt5Weights()
        tdb = params.foldseek_target_db ? channel.value(file(params.foldseek_target_db, checkIfExists: true))
                                        : foldseekTargetDb()
        qdbs = foldseekQueryDbs(file(params.query_structures, checkIfExists: true), prep.queries, weights)
        foldseekSearch(qdbs.structure.mix(qdbs.prostt5).combine(tdb))
    }

    if (!params.skip_kmerseek) {
        def chosen = pickSettings(readSettings())
        def res = loadResources()
        def noPlan = chosen.findAll { c -> !res.containsKey(c.setting_label) }
        log.info "kmerseek: ${chosen.size()} setting(s) (--settings ${params.settings}); " +
                 "${chosen.size() - noPlan.size()} sized from ${params.resources ?: 'no resources file'}, " +
                 "${noPlan.size()} on the default ${params.kmerseek_default_memory_gb} GB / " +
                 "${params.kmerseek_default_time_h} h"
        settings_ch = channel.fromList(chosen.collect { s ->
            def r = res[s.setting_label] ?: [params.kmerseek_default_memory_gb as int,
                                                    params.kmerseek_default_time_h as int]
            [s, r[0], r[1]]
        })
        kmerseekRun(settings_ch.combine(prep.reference).combine(prep.queries)
                               .combine(channel.value(truth_dir)))
    }
}
