#!/usr/bin/env nextflow
/*
 * Notebook 360: do differentially expressed k-mers, landed on Pfam domains by kmerseek,
 * recover the domains that changed as well as assembly + DIAMOND + borrowed Pfam?
 *
 * Every rule is in notebooks/360_data/preregistered.md. Per organism:
 *
 *   answer key transcripts --polyester--> 6 samples of read pairs (3 A, 3 B)
 *     standard path:  Trinity -> salmon -> DESeq2 (10 labellings)
 *                     Trinity -> DIAMOND blastx vs search db -> Pfam of best hit
 *     k-mer path:     jellyfish 31-mers -> DESeq2 (10 labellings) -> changed-k-mer runs
 *                     -> 6-frame peptides -> kmerseek, one search per kmerseek setting
 *                     -> regions landed on Pfam domains
 *     distance:       key proteins -> DIAMOND blastp vs their families' Pfam seeds
 *
 * The answer key and search database come from scripts/nb360/build_answer_key.sbatch and
 * are read from --key_dir; this pipeline never changes them.
 */

nextflow.enable.dsl = 2

params.key_dir     = '/scratch/users/olgabot/360-de-kmer-landing/key'
params.pfam_seed   = '/scratch/users/olgabot/360-de-kmer-landing/downloads/Pfam-A.seed.gz'
params.organisms   = 'drosophila,zebrafish,ciona'
params.settings    = "${projectDir}/assets/kmerseek_settings.tsv"
params.labellings  = "${projectDir}/assets/labellings.tsv"
params.outdir      = "${launchDir}/results"
// Run only these kmerseek settings (comma-separated alphabet.kK), for a smoke run.
params.only_settings = ''

// Dark-set flags (notebooks/360_data/preregistered.md, k-mer path step 5).
params.threshold          = 0.0
params.min_shared_kmers   = 2
params.max_query_pvalue   = 0.05
params.min_region_score   = 1.3
params.chain_max_gap      = 30
params.chain_max_shift    = 10
params.xdrop_per_penalty  = 4
params.ka_queries         = 500
params.ka_queries_bits_base         = 30
params.ka_queries_bits_per_doubling = 3
params.ka_queries_max               = 3000
// Rows kept from a search: region_evalue <= this. Landing uses <= 1; the rest is kept for
// the reported (not ruled) E-value sweep.
params.keep_region_evalue = 100

// ---------------------------------------------------------------------------------------

process simulateReads {
    tag "${org}"
    label 'polyester'
    publishDir "${params.outdir}/${org}/reads", mode: 'copy', pattern: '*.{tsv,txt}'
    cpus 2
    memory { 8.GB * task.attempt }
    time   { 2.h * task.attempt }

    input:
    tuple val(org), path(transcripts), path(key)

    output:
    tuple val(org), path('sample_0*_{1,2}.fasta.gz'), emit: reads
    path 'true_counts.tsv'
    path 'polyester_version.txt'

    script:
    """
    simulate_reads.R ${transcripts} ${key} .
    ls sample_0[1-6]_[12].fasta.gz | wc -l | grep -qx 12
    """
}

process trinity {
    tag "${org}"
    label 'trinity'
    publishDir "${params.outdir}/${org}/standard", mode: 'copy', pattern: '*.{fasta,txt}'
    cpus 8
    memory { 32.GB * task.attempt }
    time   { 8.h * task.attempt }

    input:
    tuple val(org), path(reads)

    output:
    tuple val(org), path("${org}.Trinity.fasta"), emit: fasta
    path 'trinity_version.txt'

    script:
    def left  = (1..6).collect { String.format('sample_%02d_1.fasta.gz', it) }.join(',')
    def right = (1..6).collect { String.format('sample_%02d_2.fasta.gz', it) }.join(',')
    """
    Trinity --seqType fa --left ${left} --right ${right} \\
        --CPU ${task.cpus} --max_memory ${task.memory.toGiga() - 2}G \\
        --min_contig_length 200 --output trinity_out --full_cleanup
    if [ -s trinity_out.Trinity.fasta ]; then mv trinity_out.Trinity.fasta ${org}.Trinity.fasta
    else mv trinity_out/Trinity.fasta ${org}.Trinity.fasta; fi
    Trinity --version > trinity_version.txt 2>&1 || true
    grep -c '>' ${org}.Trinity.fasta
    """
}

process salmonQuant {
    tag "${org}"
    label 'salmon'
    publishDir "${params.outdir}/${org}/standard/salmon", mode: 'copy'
    cpus 4
    memory { 8.GB * task.attempt }
    time   { 2.h * task.attempt }

    input:
    tuple val(org), path(trinity), path(reads)

    output:
    tuple val(org), path('sample_0*.quant.sf'), emit: quant
    path 'salmon_version.txt'

    script:
    """
    salmon index -t ${trinity} -i idx -p ${task.cpus}
    for i in 01 02 03 04 05 06; do
        salmon quant -i idx -l A -1 sample_\${i}_1.fasta.gz -2 sample_\${i}_2.fasta.gz \\
            -p ${task.cpus} --validateMappings -o q_\${i}
        cp q_\${i}/quant.sf sample_\${i}.quant.sf
    done
    salmon --version > salmon_version.txt
    """
}

process jellyfishCount {
    tag "${org}"
    label 'jellyfish'
    cpus 4
    memory { 16.GB * task.attempt }
    time   { 2.h * task.attempt }

    input:
    tuple val(org), path(reads)

    output:
    tuple val(org), path('sample_0*.kmers.tsv'), emit: kmers
    path 'jellyfish_version.txt'

    script:
    """
    for i in 01 02 03 04 05 06; do
        zcat sample_\${i}_1.fasta.gz sample_\${i}_2.fasta.gz \\
          | jellyfish count -C -m 31 -s 200M -t ${task.cpus} -o s\${i}.jf /dev/stdin
        jellyfish dump -c -t -L 1 s\${i}.jf > sample_\${i}.kmers.tsv
        rm s\${i}.jf
    done
    jellyfish --version > jellyfish_version.txt
    """
}

process countMatrix {
    tag "${org}.${kind}"
    label 'python'
    publishDir "${params.outdir}/${org}/counts", mode: 'copy'
    cpus 2
    memory { 16.GB * task.attempt }
    time   { 1.h * task.attempt }

    input:
    tuple val(org), val(kind), path(files)

    output:
    tuple val(org), val(kind), path("${org}.${kind}.counts.tsv"), emit: counts

    script:
    def mode = kind == 'transcripts' ? 'salmon' : 'jellyfish'
    """
    count_matrix.py ${mode} ${org}.${kind}.counts.tsv ${files}
    """
}

process deseq2 {
    tag "${org}.${kind}"
    label 'deseq2'
    publishDir "${params.outdir}/${org}/deseq2", mode: 'copy'
    cpus { kind == 'kmers' ? 8 : 2 }
    memory { (kind == 'kmers' ? 32.GB : 8.GB) * task.attempt }
    time   { (kind == 'kmers' ? 8.h : 2.h) * task.attempt }

    input:
    tuple val(org), val(kind), path(counts), path(labellings)

    output:
    tuple val(org), val(kind), path("${org}.${kind}.*.deseq2.tsv.gz"), emit: results
    path "${org}.${kind}.deseq2_version.txt"

    script:
    """
    deseq2_labellings.R ${counts} ${labellings} ${org}.${kind} ${task.cpus}
    """
}

process diamondBlastx {
    tag "${org}"
    label 'diamond'
    publishDir "${params.outdir}/${org}/standard", mode: 'copy'
    cpus 8
    memory { 32.GB * task.attempt }
    time   { 4.h * task.attempt }

    input:
    tuple val(org), path(trinity), path(db)

    output:
    tuple val(org), path("${org}.blastx.tsv"), emit: hits
    path 'diamond_version.txt'

    script:
    """
    diamond makedb --in ${db} -d db -p ${task.cpus}
    diamond blastx -d db -q ${trinity} -o ${org}.blastx.tsv --sensitive -e 1e-5 -k 1 \\
        -p ${task.cpus} --outfmt 6 qseqid sseqid pident length evalue bitscore qstart qend sstart send
    diamond version > diamond_version.txt
    """
}

process standardCalls {
    tag "${org}"
    label 'python'
    publishDir "${params.outdir}/${org}/standard", mode: 'copy'
    cpus 1
    memory { 8.GB * task.attempt }
    time   { 1.h * task.attempt }

    input:
    tuple val(org), path(hits), path(domains), path(key), path(deseq)

    output:
    path "${org}.standard.summary.tsv", emit: summary
    path "${org}.standard.calls.tsv"

    script:
    """
    standard_calls.py --diamond ${hits} --domains ${domains} --key ${key} \\
        --organism ${org} --out-prefix ${org}.standard ${deseq}
    """
}

process deKmerQueries {
    tag "${org}"
    label 'python'
    publishDir "${params.outdir}/${org}/kmer_path", mode: 'copy'
    cpus 1
    memory { 16.GB * task.attempt }
    time   { 2.h * task.attempt }

    input:
    tuple val(org), path(deseq)

    output:
    tuple val(org), path("${org}.queries.fasta"), path("${org}.peptide_map.tsv"), emit: queries
    path "${org}.runs.tsv"

    script:
    """
    de_kmer_contigs.py ${org} ${deseq}
    """
}

process kmerseekIndex {
    tag "${org}.${s.alphabet}.k${s.ksize}"
    label 'kmerseek'
    cpus 8
    memory { (s.index_memory_gb as int).GB * task.attempt }
    time   { 12.h * task.attempt }

    input:
    tuple val(org), val(s), path(db)

    output:
    tuple val(org), val(s), path("${org}.${s.alphabet}.k${s.ksize}.kmerseek.rocksdb"), emit: index

    script:
    def idx = "${org}.${s.alphabet}.k${s.ksize}.kmerseek.rocksdb"
    def c   = s.mismatch_penalty
    """
    set -euo pipefail
    kmerseek index --alphabet ${s.alphabet} --ksize ${s.ksize} --scaled 1 \\
        --input ${db} --output ${idx} \\
        --extend-mismatch-penalty ${c} --extend-xdrop ${xdrop(c)} \\
        --ka-queries ${kaQueries(s.alphabet, s.ksize as int)} \\
        --ka-survival-out ka_survival.csv \\
        --kmer-stats-out spectrum.csv.gz 2>&1 | tee index.log
    mv ka_survival.csv spectrum.csv.gz index.log ${idx}/ 2>/dev/null || true
    """
}

process kmerseekSearch {
    tag "${org}.${s.alphabet}.k${s.ksize}"
    label 'kmerseek'
    publishDir "${params.outdir}/${org}/kmerseek", mode: 'copy', pattern: '*.{zst,log,status}'
    cpus 8
    memory { (s.search_memory_gb as int).GB * task.attempt }
    time   { (s.search_time_h as int).h * task.attempt }

    input:
    tuple val(org), val(s), path(index), path(queries), path(pmap)

    output:
    tuple val(org), val(s), path("${slug(org, s)}.regions.csv.zst"), path("${slug(org, s)}.status"), emit: regions
    path "${slug(org, s)}.log"

    script:
    def sl = slug(org, s)
    def c  = s.mismatch_penalty
    """
    set -uo pipefail
    set +e
    kmerseek search --alphabet ${s.alphabet} --ksize ${s.ksize} \\
        --query ${queries} --target ${index} \\
        --threshold ${params.threshold} --min-shared-kmers ${params.min_shared_kmers} \\
        --max-query-pvalue ${params.max_query_pvalue} --min-region-score ${params.min_region_score} \\
        --extend-mismatch-penalty ${c} --extend-xdrop ${xdrop(c)} --ka-queries 0 \\
        --chain-max-gap ${params.chain_max_gap} --chain-max-shift ${params.chain_max_shift} \\
        2> ${sl}.log \\
      | awk -F, -v max=${params.keep_region_evalue} 'NR==1 { for (i=1;i<=NF;i++) if (\$i=="region_evalue") e=i;
            if (!e) { print "no region_evalue column" > "/dev/stderr"; exit 3 } print; next } (\$e+0) <= max' \\
      | zstd -T2 -q -o ${sl}.regions.csv.zst
    status=(\${PIPESTATUS[@]})
    set -e
    # A search refused because the index has no Karlin-Altschul fit for this penalty is a
    # result, not a crash: the setting is recorded as unfitted and lands nothing.
    if [ "\${status[0]}" -ne 0 ] && grep -q "no Karlin-Altschul fit" ${sl}.log; then
        echo unfitted > ${sl}.status
        printf 'query_name,target_name,target_start,target_end,region_start,region_end,region_evalue,region_ka_bits,region_poisson_score,region_n_shared_kmers\\n' \\
          | zstd -q -f -o ${sl}.regions.csv.zst
        exit 0
    fi
    [ "\${status[0]}" -eq 0 ] && [ "\${status[1]}" -eq 0 ] && [ "\${status[2]}" -eq 0 ] || exit 1
    echo fitted > ${sl}.status
    """
}

// The Makefile measures history before starting the head; tasks read only that snapshot.
def landingAsk(org, setting, column) {
    def tag = "${org}.${setting.alphabet}.k${setting.ksize}"
    def resourceFile = file("${launchDir}/measured/landings.tsv")
    if (!resourceFile.exists()) error "Run make measure-resources before launch"
    def row = resourceFile.readLines().find { it.startsWith(tag + "\t") }
    if (row == null) error "No measured resources for ${tag}"
    row.split('\t')[column] as int
}

process landings {
    tag "${org}.${s.alphabet}.k${s.ksize}"
    label 'python'
    publishDir "${params.outdir}/${org}/landings", mode: 'copy'
    cpus 2
    memory { Math.min(500, landingAsk(org, s, 1) * task.attempt).GB }
    time   { landingAsk(org, s, 2).h * task.attempt }

    input:
    tuple val(org), val(s), path(regions), path(status), path(pmap), path(domains), path(key), path(labellings)

    output:
    path "${slug(org, s)}.summary.tsv", emit: summary
    path "${slug(org, s)}.landings.tsv.gz"

    script:
    def sl = slug(org, s)
    """
    set -o pipefail
    # Read bounded batches from stdin; never inflate the full compressed CSV in memory.
    zstd -dc < ${regions} | landings.py --regions - --domains ${domains} --peptide-map ${pmap} --key ${key} \\
        --labellings ${labellings} --organism ${org} --setting ${s.alphabet}.k${s.ksize} \\
        --status \$(cat ${status}) --out-prefix ${sl}
    """
}

process seedDistance {
    tag "${org}"
    label 'diamond'
    publishDir "${params.outdir}/${org}/distance", mode: 'copy'
    cpus 4
    memory { 16.GB * task.attempt }
    time   { 2.h * task.attempt }

    input:
    tuple val(org), path(proteins), path(seeds)

    output:
    path "${org}.seed_blastp.tsv"

    script:
    """
    diamond makedb --in ${seeds} -d seeds -p ${task.cpus}
    diamond blastp -d seeds -q ${proteins} -o ${org}.seed_blastp.tsv --ultra-sensitive -k 1000 -e 10 \\
        -p ${task.cpus} --outfmt 6 qseqid sseqid pident length evalue bitscore qstart qend sstart send
    """
}

process seedSequences {
    tag "${org}"
    label 'python'
    cpus 1
    memory { 4.GB * task.attempt }
    time   { 1.h * task.attempt }

    input:
    tuple val(org), path(key), path(seed)

    output:
    tuple val(org), path("${org}.seeds.fasta")

    script:
    """
    seed_sequences.py ${seed} ${key} ${org}.seeds.fasta
    """
}

process collectSummaries {
    label 'python'
    publishDir params.outdir, mode: 'copy'
    cpus 1
    memory 4.GB
    time 30.m

    input:
    path summaries

    output:
    path 'summary.tsv'

    script:
    """
    #!/usr/bin/env python3
    import glob, polars as pl
    fs = sorted(glob.glob('*.summary.tsv'))
    pl.concat([pl.read_csv(f, separator='\\t', infer_schema_length=0) for f in fs], how='diagonal') \\
      .sort('organism', 'path', 'setting', 'labelling', 'evalue_cutoff').write_csv('summary.tsv', separator='\\t')
    print(len(fs), 'summary files')
    """
}

// ---------------------------------------------------------------------------------------

def slug(org, s) { "${org}.${s.alphabet}.k${s.ksize}" }

// The give-up margin is 4 x the mismatch penalty, as in the dark set; the same string
// reaches index and search, because the stored fit is looked up by exact penalty and X-drop.
def xdrop(c) {
    def v = (params.xdrop_per_penalty as double) * (c as double)
    def sf = String.format('%.2f', v)
    sf.contains('.') ? sf.replaceAll(/0+$/, '').replaceAll(/\.$/, '') : sf
}

// The dark set's --ka-queries rule: 500 up to 30 bits per k-mer, doubling every 3 bits
// above, capped at 3000, rounded to the nearest 50.
def kaQueries(String alphabet, int k) {
    def m = alphabet =~ /(\d+)$/
    double bits = k * (Math.log(m[0][1] as double) / Math.log(2.0d))
    double base = params.ka_queries_bits_base as double
    if (bits <= base) return params.ka_queries as int
    double n = (params.ka_queries as double) * Math.pow(2.0d, (bits - base) / (params.ka_queries_bits_per_doubling as double))
    (int) (Math.round(Math.min(params.ka_queries_max as double, n) / 50.0d) * 50)
}

workflow {
    orgs = Channel.fromList(params.organisms.tokenize(',')*.trim())
    keyFiles = orgs.map { o ->
        def d = "${params.key_dir}/${o}"
        tuple(o, file("${d}/transcripts.fa", checkIfExists: true), file("${d}/answer_key.tsv", checkIfExists: true),
              file("${d}/search_db.fasta", checkIfExists: true), file("${d}/search_db.pfam.tsv", checkIfExists: true),
              file("${d}/proteins.fa", checkIfExists: true))
    }
    labellings = file(params.labellings, checkIfExists: true)
    only = params.only_settings.toString().tokenize(',')*.trim().findAll { it }
    // Read by hand: the table opens with '#' comment lines, which splitCsv does not skip
    // before taking the header.
    def lines = file(params.settings, checkIfExists: true).readLines().findAll { it.trim() && !it.startsWith('#') }
    def head = lines[0].split('\t') as List
    def rows = lines.drop(1).collect { l -> [head, l.split('\t') as List].transpose().collectEntries { it } }
    def unknown = only - rows.collect { "${it.alphabet}.k${it.ksize}".toString() }
    if (unknown) error "--only_settings names ${unknown.join(', ')}, not in ${params.settings}"
    settings = Channel.fromList(rows.findAll { !only || "${it.alphabet}.k${it.ksize}".toString() in only })

    reads = simulateReads(keyFiles.map { o, tx, key, db, dom, prot -> tuple(o, tx, key) }).reads

    // Standard path
    tri = trinity(reads).fasta
    quant = salmonQuant(tri.join(reads)).quant
    hits = diamondBlastx(tri.join(keyFiles.map { o, tx, key, db, dom, prot -> tuple(o, db) })).hits

    // K-mer path
    kmers = jellyfishCount(reads).kmers
    counts = countMatrix(quant.map { o, f -> tuple(o, 'transcripts', f) }.mix(kmers.map { o, f -> tuple(o, 'kmers', f) })).counts
    de = deseq2(counts.map { o, k, c -> tuple(o, k, c, labellings) }).results
    deTx = de.filter { o, k, f -> k == 'transcripts' }.map { o, k, f -> tuple(o, f) }
    deKm = de.filter { o, k, f -> k == 'kmers' }.map { o, k, f -> tuple(o, f) }

    std = standardCalls(hits.join(keyFiles.map { o, tx, key, db, dom, prot -> tuple(o, dom, key) }).join(deTx)
                          .map { o, h, dom, key, f -> tuple(o, h, dom, key, f) }).summary

    q = deKmerQueries(deKm).queries
    idx = kmerseekIndex(keyFiles.map { o, tx, key, db, dom, prot -> tuple(o, db) }.combine(settings)
                          .map { o, db, s -> tuple(o, s, db) }).index
    found = kmerseekSearch(idx.combine(q, by: 0).map { o, s, i, qs, pm -> tuple(o, s, i, qs, pm) }).regions
    land = landings(found.combine(q, by: 0)
                         .combine(keyFiles.map { o, tx, key, db, dom, prot -> tuple(o, dom, key) }, by: 0)
                         .map { o, s, r, st, qs, pm, dom, key -> tuple(o, s, r, st, pm, dom, key, labellings) }).summary

    // Distance from Pfam seeds
    seeds = seedSequences(keyFiles.map { o, tx, key, db, dom, prot -> tuple(o, key, file(params.pfam_seed, checkIfExists: true)) })
    seedDistance(keyFiles.map { o, tx, key, db, dom, prot -> tuple(o, prot) }.join(seeds))

    collectSummaries(std.mix(land).collect())
}
