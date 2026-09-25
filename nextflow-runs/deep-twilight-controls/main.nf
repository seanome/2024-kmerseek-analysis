#!/usr/bin/env nextflow
/*
 * Deep-twilight controls: does each tool carry a family protein's functional label onto
 * its homologs, and down to what pairwise identity?
 *
 * Three hand-curated families (tables/deep_twilight_pairs.tsv): globins, lysozyme /
 * alpha-lactalbumin, cystatins; 16 proteins, 80 ordered pairs within families. Every tool
 * searches the same database, the QfO human proteome plus the 8 non-human family proteins,
 * so their E-values count chance hits in the same search space and one cut-off
 * (--max_evalue) means the same thing for all of them. Only family-to-family alignments
 * are scored.
 *
 *   kmerseek  every alphabet and k in assets/arms.tsv (150 arms), extended with the
 *             per-alphabet penalty and give-up margin from the same table; region_evalue
 *             on every region
 *             gbmr7 k8 and k10 are left out. Both were killed for memory at 8, 16 and
 *             24 GB inside the index's Karlin-Altschul fit, and the fit cannot succeed:
 *             two random gbmr7 letters match 39% of the time, so at penalty 0.32 a
 *             random position scores +0.20 on average and no lambda exists (notebooks
 *             241 and 245). Their search would have fallen back to exact k-mers.
 *   phmmer    --max, E-values from the database size
 *   MMseqs2   -s 7.5 --exhaustive-search 1
 *   Foldseek  AlphaFold DB models, --exhaustive-search 1
 *   needle    global identity of each pair, the x-axis of the figure
 */

nextflow.enable.dsl = 2

params.family_fasta = "${projectDir}/assets/deep_twilight_proteins.fasta"
params.labels       = "${projectDir}/../../tables/deep_twilight_pairs.tsv"
params.arms         = "${projectDir}/assets/arms.tsv"
params.human_fasta      = null
params.human_structures = null
params.max_evalue   = 1000
params.outdir       = "${launchDir}/results"

process stageDatabase {
    container params.kmerseek_image
    publishDir "${params.outdir}/database", mode: 'copy'

    input:
    path human
    path family
    path labels

    output:
    path 'database.fasta', emit: fasta
    path 'database_accessions.txt', emit: accessions
    path 'family_pairs.tsv', emit: pairs

    script:
    """
    stage_database.py ${human} ${family} database.fasta database_accessions.txt
    family_pairs.py ${labels} ${family} family_pairs.tsv
    """
}

process kmerseekArm {
    tag "${alphabet} k${ksize}"
    container params.kmerseek_image
    publishDir "${params.outdir}/kmerseek", mode: 'copy', pattern: '*.{parquet,tsv,log}'

    input:
    tuple val(alphabet), val(ksize), val(penalty), val(xdrop)
    path database
    path family
    path pairs

    output:
    path "${alphabet}.k${ksize}.hits.parquet", emit: hits
    path "${alphabet}.k${ksize}.pairs.tsv", emit: pairs
    path "${alphabet}.k${ksize}.log"

    script:
    def arm = "${alphabet}.k${ksize}"
    """
    set -euo pipefail
    log=${arm}.log
    kmerseek index --input ${database} --output idx --ksize ${ksize} --alphabet ${alphabet} \\
        --extend-mismatch-penalty ${penalty} --extend-xdrop ${xdrop} >> \$log 2>&1

    # An index whose Karlin-Altschul fit was refused makes an extended search refuse too.
    # Such an arm is searched exact instead; region_evalue is then region_run_evalue, which
    # needs no fit, and the `extended` column says which it was.
    extended=true
    if ! kmerseek search -q ${family} -t idx -k ${ksize} -a ${alphabet} \\
            --extend-mismatch-penalty ${penalty} --extend-xdrop ${xdrop} \\
            --threshold 0 --min-shared-kmers 1 --max-query-pvalue 1 \\
            -o search.csv >> \$log 2>&1; then
        grep -q 'no Karlin-Altschul fit' \$log || exit 1
        echo "## no Karlin-Altschul fit: searching exact (penalty 0)" >> \$log
        extended=false
        kmerseek search -q ${family} -t idx -k ${ksize} -a ${alphabet} \\
            --extend-mismatch-penalty 0 \\
            --threshold 0 --min-shared-kmers 1 --max-query-pvalue 1 \\
            -o search.csv >> \$log 2>&1
    fi
    kmerseek_family_hits.py search.csv ${family} ${alphabet} ${ksize} \$extended ${arm}.hits.parquet

    mkdir pair
    while read -r q t; do
        kmerseek pair -q ${family} --query-name "\$q" -t ${family} --target-name "\$t" \\
            -k ${ksize} -a ${alphabet} -o "pair/\${q//|/_}__\${t//|/_}.json" >> \$log 2>&1
    done < ${pairs}
    kmerseek_pair_summary.py ${alphabet} ${ksize} ${arm}.pairs.tsv pair/*.json
    rm -rf idx pair search.csv
    """
}

process phmmerSearch {
    container 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
    cpus 4

    input:
    path database
    path family

    output:
    path 'phmmer/*'

    script:
    """
    set -euo pipefail
    mkdir -p phmmer
    awk '/^>/{n++; f=sprintf("phmmer/q%02d.fasta", n)} {print > f}' ${family}
    for q in phmmer/q*.fasta; do
        b=\${q%.fasta}
        phmmer --cpu ${task.cpus} --max -E ${params.max_evalue} --domE ${params.max_evalue} \\
            --incE ${params.max_evalue} --incdomE ${params.max_evalue} \\
            -A \$b.sto --domtblout \$b.domtbl -o /dev/null \$q ${database}
    done
    """
}

process parsePhmmer {
    container params.kmerseek_image
    publishDir "${params.outdir}/baselines", mode: 'copy'

    input:
    path hits
    path family

    output:
    path 'phmmer.tsv'

    script:
    """
    set -euo pipefail
    for q in q*.fasta; do
        b=\${q%.fasta}
        parse_phmmer.py \$q ${family} \$b.sto \$b.domtbl \$b.tsv
    done
    head -1 q01.tsv > phmmer.tsv
    for f in q*.tsv; do tail -n +2 \$f >> phmmer.tsv; done
    """
}

process mmseqsSearch {
    container 'quay.io/biocontainers/mmseqs2@sha256:3503bfe576d560e550df2872af86a1ad1bcc1c06cfb7caadd3e7a95649f5f0ef'
    cpus 4

    input:
    path database
    path family

    output:
    path 'mmseqs.m8'

    script:
    """
    set -euo pipefail
    mmseqs createdb ${database} tdb > /dev/null
    mmseqs createdb ${family} qdb > /dev/null
    mmseqs search qdb tdb aln tmp -a -s 7.5 --exhaustive-search 1 -e ${params.max_evalue} \\
        --max-seqs 100000 --threads ${task.cpus} > mmseqs.log 2>&1
    mmseqs convertalis qdb tdb aln mmseqs.m8 \\
        --format-output query,target,fident,qstart,qend,tstart,tend,evalue,bits,qaln,taln \\
        > /dev/null
    """
}

process fetchFamilyStructures {
    container params.kmerseek_image
    publishDir "${params.outdir}/structures", mode: 'copy'

    input:
    path family

    output:
    path 'family/*.cif', emit: cifs
    path 'afdb_models.tsv', emit: versions

    script:
    """
    fetch_afdb.py ${family} family afdb_models.tsv
    """
}

process foldseekSearch {
    container 'quay.io/biocontainers/foldseek@sha256:1156a052f31b2afb85257c02e83a962f559c9752273fe1064ab735f90ac29d1a'
    cpus 8

    input:
    path human_structures
    path accessions
    path family_cifs

    output:
    path 'foldseek.m8', emit: m8
    path 'foldseek_database.txt'

    script:
    """
    set -euo pipefail
    mkdir targets queries
    for f in ${family_cifs}; do ln -s ../\$f targets/\$f; ln -s ../\$f queries/\$f; done
    # human models for every database protein that is not a family member
    while read -r acc; do
        [ -e targets/AF-\$acc-F1.cif ] && continue
        [ -e ${human_structures}/AF-\$acc-F1.cif ] && ln -s \$(readlink -f ${human_structures})/AF-\$acc-F1.cif targets/
    done < ${accessions}
    echo "\$(ls targets | wc -l) structures for \$(wc -l < ${accessions}) database proteins" > foldseek_database.txt
    foldseek createdb targets tdb --threads ${task.cpus} > /dev/null
    foldseek createdb queries qdb --threads ${task.cpus} > /dev/null
    foldseek search qdb tdb aln tmp -a --exhaustive-search 1 -e ${params.max_evalue} \\
        --max-seqs 100000 --threads ${task.cpus} > foldseek.log 2>&1
    foldseek convertalis qdb tdb aln foldseek.m8 \\
        --format-output query,target,fident,qstart,qend,tstart,tend,evalue,bits,prob,qaln,taln \\
        > /dev/null
    """
}

process normalizeAlignments {
    container params.kmerseek_image
    publishDir "${params.outdir}/baselines", mode: 'copy'

    input:
    tuple val(tool), path(m8)
    path family

    output:
    path "${tool}.tsv"

    script:
    """
    normalize_alignments.py ${tool} ${m8} ${family} ${tool}.tsv
    """
}

process needleIdentity {
    container 'quay.io/biocontainers/emboss@sha256:a9bf499a690de7950a3f553793ea45daa947c63946a43edaefdbcfb0e9960a97'
    publishDir "${params.outdir}/identity", mode: 'copy'

    input:
    path family
    path pairs

    output:
    path 'needle_identity.tsv', emit: tsv
    path 'needle/*'

    script:
    // EMBOSS needle, EBLOSUM62, gap open 10, gap extend 0.5, end gaps not penalised (the
    // defaults). Identity is identical positions over alignment length, gaps included.
    """
    set -euo pipefail
    mkdir needle seqs
    awk '/^>/{split(substr(\$1,2),a,"|"); f="seqs/" a[2] ".fasta"} {print > f}' ${family}
    printf 'a\\tb\\tneedle_identical\\tneedle_length\\tneedle_identity_pct\\n' > needle_identity.tsv
    while read -r q t; do
        a=\$(echo \$q | cut -d'|' -f2); b=\$(echo \$t | cut -d'|' -f2)
        [ "\$a" \\< "\$b" ] || continue
        needle -asequence seqs/\$a.fasta -bsequence seqs/\$b.fasta -gapopen 10 -gapextend 0.5 \\
            -outfile needle/\$a.\$b.needle -auto
        grep '^# Identity:' needle/\$a.\$b.needle \\
            | sed -E 's/.*: *([0-9]+)\\/([0-9]+) \\( *([0-9.]+)%\\)/\\1\\t\\2\\t\\3/' \\
            | awk -v a=\$a -v b=\$b '{print a"\\t"b"\\t"\$0}' >> needle_identity.tsv
    done < ${pairs}
    """
}

process scoreTransfer {
    container params.kmerseek_image
    publishDir "${params.outdir}", mode: 'copy'
    memory '8 GB'

    input:
    path labels
    path family
    path arms
    path identity
    path kmerseek_hits
    path kmerseek_pairs
    path baselines

    output:
    path 'calls.parquet'
    path 'outcomes.parquet'
    path 'pair_kmers.parquet'
    path 'score_transfer.log'

    script:
    """
    score_transfer.py --labels ${labels} --family ${family} --arms ${arms} \\
        --identity ${identity} --max-evalue ${params.max_evalue} \\
        --kmerseek ${kmerseek_hits} --pairs ${kmerseek_pairs} --baselines ${baselines} \\
        --outdir . > score_transfer.log
    cat score_transfer.log
    """
}

workflow {
    if (!params.human_fasta || !params.human_structures) {
        error "--human_fasta and --human_structures are required (see the Makefile)"
    }
    human  = file(params.human_fasta, checkIfExists: true)
    family = file(params.family_fasta, checkIfExists: true)
    labels = file(params.labels, checkIfExists: true)
    arms   = file(params.arms, checkIfExists: true)

    db = stageDatabase(human, family, labels)

    arm_ch = Channel.fromPath(arms)
        .splitCsv(header: true, sep: '\t')
        .map { r -> tuple(r.alphabet, r.ksize as int, r.penalty, r.xdrop) }
    km = kmerseekArm(arm_ch, db.fasta, family, db.pairs)

    ph = parsePhmmer(phmmerSearch(db.fasta, family), family)
    mm = mmseqsSearch(db.fasta, family)
    st = fetchFamilyStructures(family)
    fs = foldseekSearch(file(params.human_structures, checkIfExists: true), db.accessions, st.cifs)
    norm = normalizeAlignments(
        Channel.of('mmseqs2').combine(mm).mix(Channel.of('foldseek').combine(fs.m8)),
        family)
    idn = needleIdentity(family, db.pairs)

    scoreTransfer(
        labels, family, arms, idn.tsv,
        km.hits.collect(), km.pairs.collect(),
        ph.mix(norm).collect())
}
