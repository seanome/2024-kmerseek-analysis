#!/usr/bin/env nextflow

/*
 * SCOPe40 all-against-all for the leaderboard's SCOPe40 page, keeping every hit list.
 *
 * Every search here writes its full hit list (query, target, bit score, E-value), and
 * bin/score_hitlist.py turns it into one row per (query, level): Foldseek's benchmark rules
 * (bench.noselfhit.awk), checked against that script in tests/test_score_hitlist.py. Both
 * the hit list and the scored rows are published, so a score can always be traced back to
 * the hits it came from. The earlier SCOPe40 pipelines kept only an area per search
 * (scope40-alphabet-ksize-v040) or ran on AWS Batch (scope40-benchmark-aws); this one runs
 * on Sherlock with the containers, commands and settings of qfo-pfam-region-benchmark.
 *
 * Set: ASTRAL SCOPe 2.08, 40% identity, 15,177 domains, searched against itself.
 * Inputs: Foldseek on the domain structures (pdbstyle-2.08, experimental coordinates from
 * the PDB); ProstT5 + Foldseek on the domain sequences; every other tool on the sequences.
 * kmerseek is not here: its SCOPe40 pair tables already exist on Sherlock and are scored by
 * score_existing_kmerseek.sbatch.
 *
 * Chance level (--shuffle, on by default): every sequence tool searches the queries again,
 * each shuffled on its own keeping its dipeptide counts (k = 2, seed 1), against the real
 * set. Foldseek on structures has no shuffled version: a shuffled sequence has no structure.
 *
 * Usage (see the Makefile): make run-sbatch
 */

// $SCRATCH on Sherlock.
params.scratch   = System.getenv('SCRATCH') ?: "${System.getProperty('user.home')}/data"
params.fasta     = "${params.scratch}/scope/astral-scopedom-seqres-gd-sel-gs-bib-40-2.08.fa"
params.pdbstyle  = "${params.scratch}/scope/pdbstyle-2.08"
params.outdir    = "${params.scratch}/scope/results-scope40-hitlists"
params.db_cache  = "${params.scratch}/scope/results-scope40-hitlists/db_cache"
// Python with polars 1.x for the scoring step (no container: Sherlock's module python plus
// the venv the notebooks use).
params.python    = "${params.scratch}/venvs/nb275/bin/python"
params.python_module = 'python/3.12.1'
params.shuffle_script = "${projectDir}/../scope40-lowcomplexity-null/bin/shuffle_fasta.py"
params.shuffle   = true
params.shuffle_k = 2
params.shuffle_seed = 1

// Settings of the Foldseek paper's SCOPe benchmark where it gives them (Methods, "Tools and
// options"): Foldseek -s 9.5 -e 10 --max-seqs 2000; MMseqs2 -s 7.5 --max-seqs 2000. -e 10
// for every tool here so the hit lists reach well past the first false positive.
params.evalue    = 10
params.max_seqs  = 2000
params.jackhmmer_iterations = 3
params.mmseqs_iterations    = 3
params.prostt5_weights = null   // a directory of ProstT5 weights; null = fetch once
params.tools = 'foldseek,prostt5,mmseqs2_seqseq,mmseqs2_iterative,hmmer3_phmmer,hmmer3_jackhmmer,hhblits'

// Containers: the pinned digests of qfo-pfam-region-benchmark.
params.foldseek_image = 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
params.mmseqs_image   = 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
params.hmmer_image    = 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
params.hhsuite_image  = 'quay.io/biocontainers/hhsuite@sha256:4bf9bb5229de18f522a94f4443c19fdcbb0f0cb0e6ea92f5390aa170bcb0a24f'

process prepareScope {
    publishDir params.outdir, mode: 'copy'
    cpus 1
    memory '2 GB'
    time '20m'

    input:
    path fasta

    output:
    path 'scope_domains.tsv', emit: domains
    path 'scope40.ids.fa',    emit: fasta

    script:
    """
    set -euo pipefail
    awk 'BEGIN{print "domain_id\\tscop_id"} /^>/{sub(/^>/, ""); print \$1 "\\t" \$2}' ${fasta} > scope_domains.tsv
    awk '/^>/{split(\$1, a, " "); print a[1]; next} {print toupper(\$0)}' ${fasta} > scope40.ids.fa
    n=\$(grep -c '^>' scope40.ids.fa)
    [ "\$n" -eq "\$((\$(wc -l < scope_domains.tsv) - 1))" ] || { echo "domain count mismatch" >&2; exit 1; }
    echo "\$n domains"
    """
}

process shuffleQueries {
    publishDir params.outdir, mode: 'copy'
    cpus 1
    memory '4 GB'
    time '1h'
    beforeScript "ml ${params.python_module}"

    input:
    path fasta
    path shuffle_script

    output:
    path "scope40.shuffled_k${params.shuffle_k}_seed${params.shuffle_seed}.fa"

    script:
    """
    ${params.python} ${shuffle_script} --input ${fasta} \\
        --output scope40.shuffled_k${params.shuffle_k}_seed${params.shuffle_seed}.fa \\
        --seed ${params.shuffle_seed} --shuffle-k ${params.shuffle_k}
    """
}

// ------------------------------------------------------------------ Foldseek, structures

process foldseekStructureDb {
    container params.foldseek_image
    storeDir "${params.db_cache}/foldseek_structures"
    cpus 8
    memory '8 GB'
    time '2h'

    input:
    path structures

    output:
    path 'scope40_fsdb'

    script:
    """
    set -euo pipefail
    mkdir -p scope40_fsdb
    foldseek createdb ${structures}/ scope40_fsdb/db --threads ${task.cpus}
    # createdb reads the directory recursively; a count short of the set means it did not.
    n=\$(wc -l < scope40_fsdb/db.lookup)
    echo "\$n structures in the database"
    [ "\$n" -ge 15000 ] || { echo "only \$n structures read from ${structures}" >&2; exit 1; }
    """
}

process foldseekSearch {
    tag "real"
    container params.foldseek_image
    publishDir "${params.outdir}/hits", mode: 'copy'
    cpus 16
    memory '8 GB'
    time '2h'

    input:
    path db

    output:
    tuple val('foldseek'), val('default'), val('real'), path('foldseek.default.real.hits.tsv.gz')

    script:
    // Rows are named by structure file; strip the .ent (and a chain suffix Foldseek adds
    // for multi-chain files) so they key on the SCOPe domain id.
    """
    set -euo pipefail
    mkdir -p tmp
    foldseek search ${db}/db ${db}/db result tmp --threads ${task.cpus} \\
        -s 9.5 -e ${params.evalue} --max-seqs ${params.max_seqs}
    foldseek convertalis ${db}/db ${db}/db result out.tsv --threads ${task.cpus} \\
        --format-output "query,target,bits,evalue"
    awk -F'\\t' 'BEGIN{OFS="\\t"} {for (i = 1; i <= 2; i++) sub(/\\.ent(_[A-Za-z0-9]+)?\$/, "", \$i); print}' out.tsv \\
        | gzip -c > foldseek.default.real.hits.tsv.gz
    """
}

// ------------------------------------------------------------------ ProstT5 + Foldseek

process prostt5Weights {
    container params.foldseek_image
    storeDir "${params.db_cache}/prostt5"
    cpus 2
    memory '8 GB'
    time '2h'

    output:
    path 'weights'

    script:
    """
    set -euo pipefail
    mkdir -p tmp
    foldseek databases ProstT5 weights tmp
    """
}

process prostt5Db {
    tag "${qset}"
    container params.foldseek_image
    storeDir "${params.db_cache}/prostt5_db"
    cpus 8
    memory '32 GB'
    time '2h'

    input:
    tuple val(qset), path(fasta), path(weights)

    output:
    tuple val(qset), path("${qset}_prostt5")

    script:
    """
    set -euo pipefail
    mkdir -p ${qset}_prostt5
    foldseek createdb ${fasta} ${qset}_prostt5/db --prostt5-model ${weights} --gpu 1 --threads ${task.cpus}
    """
}

process prostt5Search {
    tag "${qset}"
    container params.foldseek_image
    publishDir "${params.outdir}/hits", mode: 'copy'
    cpus 16
    memory '8 GB'
    time '2h'

    input:
    tuple val(qset), path(query_db), path(target_db)

    output:
    tuple val('prostt5'), val('3di_from_seq'), val(qset), path("prostt5.3di_from_seq.${qset}.hits.tsv.gz")

    script:
    """
    set -euo pipefail
    mkdir -p tmp
    foldseek search ${query_db}/db ${target_db}/db result tmp --threads ${task.cpus} \\
        -s 9.5 -e ${params.evalue} --max-seqs ${params.max_seqs}
    foldseek convertalis ${query_db}/db ${target_db}/db result out.tsv --threads ${task.cpus} \\
        --format-output "query,target,bits,evalue"
    gzip -c out.tsv > prostt5.3di_from_seq.${qset}.hits.tsv.gz
    """
}

// ------------------------------------------------------------------ MMseqs2

process mmseqsDb {
    tag "${qset}"
    container params.mmseqs_image
    storeDir "${params.db_cache}/mmseqs_db"
    cpus 4
    memory '4 GB'
    time '30m'

    input:
    tuple val(qset), path(fasta)

    output:
    tuple val(qset), path("${qset}_mmdb")

    script:
    """
    set -euo pipefail
    mkdir -p ${qset}_mmdb
    mmseqs createdb ${fasta} ${qset}_mmdb/db
    """
}

process mmseqsSearch {
    tag "${variant} ${qset}"
    container params.mmseqs_image
    publishDir "${params.outdir}/hits", mode: 'copy'
    cpus 16
    memory '8 GB'
    time '2h'

    input:
    tuple val(variant), val(qset), path(query_db), path(target_db)

    output:
    tuple val(variant), val('s7.5'), val(qset), path("${variant}.s7.5.${qset}.hits.tsv.gz")

    script:
    def iter = variant == 'mmseqs2_iterative' ? "--num-iterations ${params.mmseqs_iterations}" : ''
    """
    set -euo pipefail
    mkdir -p tmp
    mmseqs search ${query_db}/db ${target_db}/db result tmp --threads ${task.cpus} \\
        -s 7.5 ${iter} --max-seqs ${params.max_seqs} -e ${params.evalue}
    mmseqs convertalis ${query_db}/db ${target_db}/db result out.tsv --threads ${task.cpus} \\
        --format-output "query,target,bits,evalue"
    gzip -c out.tsv > ${variant}.s7.5.${qset}.hits.tsv.gz
    """
}

// ------------------------------------------------------------------ HMMER

process hmmerSearch {
    tag "${tool} ${qset}"
    container params.hmmer_image
    publishDir "${params.outdir}/hits", mode: 'copy'
    cpus 8
    memory '8 GB'
    time { tool == 'hmmer3_jackhmmer' ? '6h' : '2h' }

    input:
    tuple val(tool), val(setting), val(qset), path(queries), path(targets)

    output:
    tuple val(tool), val(setting), val(qset), path("${tool}.${setting}.${qset}.hits.tsv.gz")

    script:
    def prog = tool == 'hmmer3_jackhmmer' ? "jackhmmer -N ${params.jackhmmer_iterations}" : 'phmmer'
    // --tblout: \$1 target, \$3 query, \$5 full-sequence E-value, \$6 full-sequence bit score.
    // jackhmmer's --tblout holds the final round only (as in qfo-pfam-region-benchmark).
    """
    set -euo pipefail
    ${prog} --tblout tbl.txt -o /dev/null --noali -E ${params.evalue} --cpu ${task.cpus} ${queries} ${targets}
    grep -v '^#' tbl.txt | awk 'BEGIN{OFS="\\t"} {print \$3, \$1, \$6, \$5}' > all.tsv
    gzip -c all.tsv > ${tool}.${setting}.${qset}.hits.tsv.gz
    """
}

// ------------------------------------------------------------------ HH-suite

process hhDb {
    tag "${qset}"
    container params.hhsuite_image
    storeDir "${params.db_cache}/hh_db"
    cpus 8
    memory '8 GB'
    time '2h'

    input:
    tuple val(qset), path(fasta), val(is_target)

    output:
    tuple val(qset), path("${qset}_hhdb")

    script:
    """
    set -euo pipefail
    mkdir -p ${qset}_hhdb
    ffindex_from_fasta -s ${qset}_hhdb/seq.ffdata ${qset}_hhdb/seq.ffindex ${fasta}
    ffindex_apply ${qset}_hhdb/seq.ffdata ${qset}_hhdb/seq.ffindex \\
        -d ${qset}_hhdb/db_a3m.ffdata -i ${qset}_hhdb/db_a3m.ffindex \\
        -- awk '/^>/{print} !/^>/{print toupper(\$0)}'
    if [ "${is_target}" = "true" ]; then
        ffindex_apply ${qset}_hhdb/db_a3m.ffdata ${qset}_hhdb/db_a3m.ffindex \\
            -d ${qset}_hhdb/db_hhm.ffdata -i ${qset}_hhdb/db_hhm.ffindex -- hhmake -i stdin -o stdout -v 0
        cstranslate -f -x 0.3 -c 4 -I a3m -i ${qset}_hhdb/db_a3m -o ${qset}_hhdb/db_cs219
    fi
    # ffindex_apply exits 0 even when every child failed, so check the files.
    for f in ${qset}_hhdb/db_a3m.ffdata \$( [ "${is_target}" = "true" ] && echo ${qset}_hhdb/db_hhm.ffdata ${qset}_hhdb/db_cs219.ffdata ); do
        [ -s "\$f" ] || { echo "\$f empty: hhmake or cstranslate failed on every entry" >&2; exit 1; }
    done
    """
}

process hhSearch {
    tag "${qset}"
    container params.hhsuite_image
    publishDir "${params.outdir}/hits", mode: 'copy'
    cpus 16
    memory '48 GB'
    time '12h'

    input:
    tuple val(qset), path(query_db), path(target_db)

    output:
    tuple val('hhblits'), val('single_seq'), val(qset), path("hhblits.single_seq.${qset}.hits.tsv.gz")

    script:
    // -blasttab: 1 query, 2 target, 11 E-value, 12 score. The last ten fields are fixed, so
    // read them from the end (a target name with spaces shifts the front ones).
    """
    set -euo pipefail
    ffindex_apply ${query_db}/db_a3m.ffdata ${query_db}/db_a3m.ffindex -d res.ffdata -i res.ffindex \\
        -- hhsearch -i stdin -d ${target_db}/db -blasttab /dev/stdout -cpu ${task.cpus} -v 0 \\
           -E ${params.evalue} -Z ${params.max_seqs} -B ${params.max_seqs}
    ffindex_apply res.ffdata res.ffindex -- cat \\
        | awk 'NF >= 12 {print \$1 "\\t" \$2 "\\t" \$NF "\\t" \$(NF-1)}' > hits.tsv
    [ -s hits.tsv ] || { echo "hhsearch wrote no hits at all: check one entry by hand" >&2; exit 1; }
    gzip -c hits.tsv > hhblits.single_seq.${qset}.hits.tsv.gz
    """
}

// ------------------------------------------------------------------ scoring

process scoreHits {
    tag "${tool} ${setting} ${qset}"
    publishDir "${params.outdir}/scored", mode: 'copy'
    cpus 2
    memory '16 GB'
    time '1h'
    beforeScript "ml ${params.python_module}"

    input:
    tuple val(tool), val(setting), val(qset), path(hits), path(domains)

    output:
    path "${tool}.${setting}.${qset}.per_query.parquet"
    path "${tool}.${setting}.${qset}.hits_to_first_fp.parquet"

    script:
    def name = qset == 'real' ? tool : "${tool}_shuffled"
    """
    ${params.python} ${projectDir}/bin/score_hitlist.py --domains ${domains} --hits ${hits} \\
        --columns query,target,bits,evalue --score bits --higher-first \\
        --tool ${name} --setting ${setting} --out-prefix ${tool}.${setting}.${qset}
    """
}

workflow {
    def TOOLS = params.tools.tokenize(',') as Set
    prep = prepareScope(file(params.fasta, checkIfExists: true))
    real = prep.fasta.map { f -> tuple('real', f) }
    qsets = real
    if (params.shuffle) {
        shuf = shuffleQueries(prep.fasta, file(params.shuffle_script, checkIfExists: true))
        qsets = real.mix(shuf.map { f -> tuple("shuffled_k${params.shuffle_k}_seed${params.shuffle_seed}", f) })
    }
    target_fa = prep.fasta.first()
    hits = channel.empty()

    if ('foldseek' in TOOLS) {
        fsdb = foldseekStructureDb(file(params.pdbstyle, checkIfExists: true))
        hits = hits.mix(foldseekSearch(fsdb))
    }
    if ('prostt5' in TOOLS) {
        w = params.prostt5_weights ? channel.value(file(params.prostt5_weights)) : prostt5Weights().first()
        pdb = prostt5Db(qsets.combine(w))
        tdb = pdb.filter { q, d -> q == 'real' }.map { q, d -> d }.first()
        hits = hits.mix(prostt5Search(pdb.combine(tdb)))
    }
    if ('mmseqs2_seqseq' in TOOLS || 'mmseqs2_iterative' in TOOLS) {
        mdb = mmseqsDb(qsets)
        mtdb = mdb.filter { q, d -> q == 'real' }.map { q, d -> d }.first()
        variants = channel.fromList(['mmseqs2_seqseq', 'mmseqs2_iterative'].findAll { t -> t in TOOLS })
        hits = hits.mix(mmseqsSearch(variants.combine(mdb).combine(mtdb)))
    }
    hm = ['hmmer3_phmmer', 'hmmer3_jackhmmer'].findAll { t -> t in TOOLS }
    if (hm) {
        hs = channel.fromList(hm).map { t -> tuple(t, t == 'hmmer3_jackhmmer' ? "n${params.jackhmmer_iterations}" : 'default') }
        hits = hits.mix(hmmerSearch(hs.combine(qsets).combine(target_fa)))
    }
    if ('hhblits' in TOOLS) {
        hq = hhDb(qsets.map { q, f -> tuple(q, f, q == 'real') })
        htdb = hq.filter { q, d -> q == 'real' }.map { q, d -> d }.first()
        hits = hits.mix(hhSearch(hq.combine(htdb)))
    }
    scoreHits(hits.combine(prep.domains.first()))
}
