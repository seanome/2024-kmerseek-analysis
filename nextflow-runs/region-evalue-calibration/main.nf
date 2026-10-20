#!/usr/bin/env nextflow
// Notebook 344: are kmerseek's region E-values calibrated on decoy queries?
//
// Every human protein is shuffled 5 times in each of three ways (composition, dipeptide,
// 10-residue window). Each decoy is searched against the human proteome with kmerseek
// exactly as a real query would be: same index, same flags, same report filters and
// region choice. Every region a decoy gets is a chance match, so the number of regions at
// E <= E* over Q decoys should be about Q x E*. The notebook reads results/ and does the
// counting; this pipeline only makes the decoys and searches them.
//
//   make stub   (laptop) the whole graph on a 3-protein FASTA, no kmerseek, no container
//   make pilot  (Sherlock) the largest chunk of each kind, to measure memory and time
//   make run    (Sherlock) everything, sized from the pilot

nextflow.enable.dsl = 2

params.human_fasta  = null
params.human_md5    = ''       // empty skips the check (stub only)
params.replicates   = 5
params.chunk_size   = 2000
params.alphabet     = 'hp_lehninger2'
params.ksize        = 19
params.penalty      = 2.0      // kmerseek's DEFAULT_MISMATCH_PENALTY
params.xdrop        = 8.0      // kmerseek's DEFAULT_XDROP
params.max_evalue   = 1.0
params.max_chunks   = 0        // > 0 searches only that many chunks of each kind, largest first (pilot)
params.outdir       = 'results'

process MAKE_DECOYS {
    tag "${params.replicates} per protein"
    publishDir params.outdir, mode: 'copy', pattern: 'decoys/decoys.tsv'
    publishDir params.outdir, mode: 'copy', pattern: 'decoys/*.fasta'

    input:
    path fasta
    path decoys_module, stageAs: 'decoys.py'

    output:
    path 'decoys/*.chunk_*.fasta', emit: chunks
    path 'decoys/decoys.tsv',      emit: table

    script:
    def check = params.human_md5 ? "echo '${params.human_md5}  ${fasta}' | md5sum -c -" : 'true'
    """
    ${check}
    export PYTHONPATH="\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
    make_decoys.py ${fasta} ${params.replicates} ${params.chunk_size} decoys
    """

    stub:
    """
    export PYTHONPATH="\$PWD\${PYTHONPATH:+:\$PYTHONPATH}"
    make_decoys.py ${fasta} ${params.replicates} ${params.chunk_size} decoys
    """
}

// SEG low-complexity share of each protein. Run on the real proteins (the strata are
// defined on them) and on each kind's decoys (to show what each shuffle keeps).
process SEG {
    tag "${label}"
    publishDir "${params.outdir}/seg", mode: 'copy'

    input:
    tuple val(label), path(fastas)

    output:
    path "${label}.seg.tsv"

    script:
    """
    cat ${fastas} > all.fasta
    segmasker -in all.fasta -infmt fasta -outfmt interval -out intervals.txt
    seg_fraction.py all.fasta intervals.txt ${label}.seg.tsv
    """

    stub:
    """
    cat ${fastas} > all.fasta
    awk '/^>/{print}' all.fasta > intervals.txt
    seg_fraction.py all.fasta intervals.txt ${label}.seg.tsv
    """
}

// The index fits r_database and K for this penalty and give-up margin (kmerseek's default
// 200 database queries), so every search reads the same fit back (--ka-queries 0 below
// refuses to search without it, instead of fitting again per chunk).
process INDEX {
    tag "${params.alphabet} k=${params.ksize}"
    publishDir "${params.outdir}/index", mode: 'copy', pattern: '*.{log,csv}'

    input:
    path fasta

    output:
    path 'human.db',          emit: db
    path 'index.log'
    path 'ka_survival.csv', optional: true

    script:
    """
    kmerseek --version > index.log
    cat /etc/kmerseek_git_sha >> index.log
    kmerseek index --input ${fasta} --output human.db \\
        --alphabet ${params.alphabet} --ksize ${params.ksize} \\
        --extend-mismatch-penalty ${params.penalty} --extend-xdrop ${params.xdrop} \\
        --ka-survival-out ka_survival.csv 2>> index.log
    # A refused fit still exits 0 (kmerseek evalue.rs, refused_fit_message); every search
    # would then fail on --ka-queries 0. Stop here instead, with the reason.
    if ! grep -q 'Stored in the index for' index.log; then
        echo "344: the Karlin-Altschul fit was not stored; index.log says:" >&2
        tail -20 index.log >&2
        exit 1
    fi
    """

    stub:
    """
    mkdir human.db
    echo stub > index.log
    echo stub > ka_survival.csv
    """
}

process SEARCH {
    tag "${chunk.baseName}"

    input:
    tuple val(kind), path(chunk), path(db)

    output:
    tuple val(kind), path("${chunk.baseName}.regions.tsv.gz"), path("${chunk.baseName}.summary.tsv"), emit: regions
    path "${chunk.baseName}.log"

    script:
    """
    set +e
    kmerseek search --query ${chunk} --target ${db} \\
        --alphabet ${params.alphabet} --ksize ${params.ksize} \\
        --extend-mismatch-penalty ${params.penalty} --extend-xdrop ${params.xdrop} \\
        --ka-queries 0 2> ${chunk.baseName}.log \\
      | filter_regions.py ${params.max_evalue} ${chunk.baseName}.regions.tsv.gz \\
          ${chunk.baseName}.summary.tsv ${chunk.baseName}
    status=("\${PIPESTATUS[@]}")
    set -e
    if [ "\${status[0]}" -ne 0 ] || [ "\${status[1]}" -ne 0 ]; then
        echo "344: search pipeline failed, exit codes \${status[*]}; kmerseek log:" >&2
        tail -20 ${chunk.baseName}.log >&2
        # If the filter failed and kmerseek only died on the closed pipe (141, or 1/101 from
        # a Rust binary), report the filter's code: it is the cause, and 141 would be retried.
        case "\${status[0]}" in 0|1|101|141) if [ "\${status[1]}" -ne 0 ]; then exit "\${status[1]}"; fi;; esac
        # Otherwise pass the kmerseek exit code (137 = OOM kill) through, so nextflow.config retries it.
        if [ "\${status[0]}" -ne 0 ]; then exit "\${status[0]}"; fi
        exit "\${status[1]}"
    fi
    """

    stub:
    """
    q=\$(awk '/^>/{print substr(\$1,2); exit}' ${chunk})
    {
      echo 'query_name,target_name,region_start,region_end,target_start,target_end,region_subseq,target_subseq,moltype_seq,region_length,region_n_mismatches,region_n_shared_kmers,region_n_chained,region_run_length,region_pr_same,region_run_evalue,region_ka_bits,region_ka_evalue,region_evalue,region_evalue_source,db_n_targets,region_search_space,extra'
      echo "\$q,sp|P10415|BCL2_HUMAN,0,20,5,25,MAHAGRTGYDNREIVMKYIH,MAHAGRTGYDNREIVMKYIH,hhhh,20,0,2,1,20,0.5,0.001,30,0.0005,0.0005,ka,3,100,x"
      echo "\$q,sp|P10415|BCL2_HUMAN,40,60,45,65,AAAAAAAAAAAAAAAAAAAA,AAAAAAAAAAAAAAAAAAAA,hhhh,20,0,2,1,20,0.9,0.2,,,0.2,run,3,100,x"
      echo "\$q,sp|P10415|BCL2_HUMAN,70,90,75,95,AAAAAAAAAAAAAAAAAAAA,AAAAAAAAAAAAAAAAAAAA,hhhh,20,0,2,1,20,0.9,5,1,3,3,ka,3,100,x"
    } | filter_regions.py ${params.max_evalue} ${chunk.baseName}.regions.tsv.gz \\
          ${chunk.baseName}.summary.tsv ${chunk.baseName}
    echo stub > ${chunk.baseName}.log
    """
}

process COLLECT {
    tag "${kind}"
    publishDir "${params.outdir}/regions", mode: 'copy'

    input:
    tuple val(kind), path(files)

    output:
    path "${kind}.regions.tsv.gz"
    path "${kind}.summary.tsv"

    script:
    """
    collect_regions.py ${kind} . ${files}
    """

    stub:
    """
    collect_regions.py ${kind} . ${files}
    """
}

workflow {
    if (!params.human_fasta) {
        error "set --human_fasta"
    }
    fasta = file(params.human_fasta, checkIfExists: true)
    decoys_module = file("${projectDir}/../shared/decoys.py", checkIfExists: true)

    MAKE_DECOYS(fasta, decoys_module)
    INDEX(fasta)

    chunks = MAKE_DECOYS.out.chunks.flatten()
        .map { f -> tuple(f.baseName.tokenize('.')[0], f) }

    // The pilot measures the biggest chunks (by file size, so by residues): chunks run
    // from 0.31 to 1.57 million residues, and the full run is sized from the pilot.
    picked = params.max_chunks > 0
        ? chunks.groupTuple().flatMap { kind, fs -> fs.sort { -it.size() }.take(params.max_chunks).collect { f -> tuple(kind, f) } }
        : chunks

    SEARCH(picked.combine(INDEX.out.db))

    by_kind = SEARCH.out.regions
        .map { kind, regions, summary -> tuple(kind, [regions, summary]) }
        .groupTuple()
        .map { kind, pairs -> tuple(kind, pairs.flatten()) }
    COLLECT(by_kind)

    seg_inputs = Channel.of(tuple('human', fasta))
        .mix(chunks.groupTuple())
    SEG(seg_inputs)
}
