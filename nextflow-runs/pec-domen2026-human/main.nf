#!/usr/bin/env nextflow
/*
 * Notebook 361, human exercise series (added 2026-11-03): the reference-free routes of the
 * Botryllus PEC analysis (../pec-domen2026/main.nf) run on the human RNA-seq series that
 * Domen et al. 2026 cite in their data statement: GSE99963 (ref 66), GSE144304 (ref 67) and
 * GSE120862 (ref 63). Each series is its own set of contrasts; no test mixes series.
 *
 * Per run:      ENA download (MD5 checked) -> fastp, single-end
 * Per contrast: SPLASH 2.11.9 (reference -1, other +1) -> compactors of every anchor with
 *               corrected p <= 0.05 -> protein search
 *               DE-kupl 1.3.5 k-mer counts (limma test) -> KaMRaT v1.2 contigs (t-test)
 *                 -> KaMRaT contigs, corrected p <= 0.05 (KaMRaT's own)  -> protein search
 *                 -> KaMRaT contigs, t-test p < 0.05                     -> protein search
 *                 -> KaMRaT contig counts + PyDESeq2, p < 0.05           -> protein search
 * Protein search: six-frame peptides >= 20 aa -> sourmash manysearch against GENCODE v50
 *                 translations -> parasail check (E <= 1e-5, query coverage >= 50 %)
 * Then one table of gene names per contrast, route and direction.
 *
 * Settings are those of the Botryllus pipeline, except single-end reads (no interleaving)
 * and the protein database (GENCODE v50 instead of the Botryllus proteomes and Swiss-Prot).
 * Every result is kept with storeDir under params.out_root, so a relaunch reuses it.
 */
nextflow.enable.dsl = 2

params.pec_root  = null   // $SCRATCH/nb361-pec: tools, env, images (shared with the Botryllus run)
params.out_root  = null   // $SCRATCH/nb361-pec/human
params.code_dir  = null   // .../analysis/pec-domen2026-splash
params.series    = 'GSE99963,GSE144304,GSE120862'
params.contrasts = ''     // comma list to run only these contrasts; empty = every contrast of params.series
params.reuse_dir = ''     // folder of earlier MD5-checked downloads (<run>.fastq.gz + .done), hard-linked

if (!params.pec_root || !params.out_root || !params.code_dir) {
    error "set --pec_root, --out_root and --code_dir"
}

ENV      = "${params.pec_root}/env/bin"
SPLASH   = "${params.pec_root}/splash"
PECTOOLS = "${params.pec_root}/pec-tools/target/release/pec-tools"
SCRIPTS  = "${params.code_dir}/scripts"
DESIGN   = "${params.code_dir}/design/human"
O        = params.out_root
TIME     = "/usr/bin/time -v -o"

// ---------------------------------------------------------------- protein database

// GENCODE v50 translations with stable IDs (GCP######), one sketch per protein at scaled 5 for
// protein k7, Dayhoff k12 and HP k30, and a RocksDB index per setting at scaled 5, 10 and 100,
// as slurm/06_db_sketch.sbatch made the Botryllus databases.
process gencode_db {
    label 'db'
    storeDir "${O}/db"

    output:
    path("gencode"), emit: db

    script:
    """
    export PATH=${ENV}:\$PATH
    mkdir gencode && cd gencode
    python ${SCRIPTS}/stable_ids.py ${params.pec_root}/db/gencode_v50/gencode.v50.pc_translations.fa.gz GCP gencode.faa gencode.id_map.tsv > ids.log
    printf 'name,genome_filename,protein_filename\\n%s,,%s\\n' gencode \$PWD/gencode.faa > manysketch.csv
    ${TIME} manysketch.time sourmash scripts manysketch manysketch.csv --singleton \\
      -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c ${task.cpus} -o gencode.scaled5.zip 2> manysketch.log
    for spec in protein:7 dayhoff:12 hp:30; do
      m=\${spec%:*}; k=\${spec#*:}
      for sc in 5 10 100; do
        ${TIME} gencode.\$m.k\$k.s\$sc.rocksdb.time sourmash scripts index gencode.scaled5.zip -m \$m -k \$k -s \$sc \\
          -c ${task.cpus} -o gencode.\$m.k\$k.s\$sc.rocksdb 2>> index.log
      done
    done
    test -s gencode.faa && test -d gencode.hp.k30.s100.rocksdb
    """
}

// ---------------------------------------------------------------- reads and trimming

process download {
    tag "${series} ${run}"
    label 'download'
    storeDir "${O}/reads/${series}"

    input:
    tuple val(series), val(run), val(ftp), val(md5)

    output:
    tuple val(series), val(run), path("${run}.fastq.gz")

    script:
    """
    out=${run}.fastq.gz
    reuse=${params.reuse_dir}
    if [ -n "\$reuse" ] && [ -e "\$reuse/\$out.done" ]; then
      ln "\$reuse/\$out" \$out.part   # hard link: same file system, no copy
    else
      for attempt in \$(seq 1 20); do
        # --fail: an HTTP error page must never be saved as the FASTQ
        if curl -sS -L --fail -C - --retry 5 --retry-delay 10 --speed-limit 100000 --speed-time 60 \\
             -o \$out.part "https://${ftp}"; then break; fi
        echo "attempt \$attempt stalled or failed; resuming"; sleep 20
      done
    fi
    got=\$(md5sum \$out.part | cut -d' ' -f1)
    [ "\$got" = "${md5}" ] || { echo "MD5 mismatch for ${run}: \$got vs ${md5}"; rm -f \$out.part; exit 1; }
    mv \$out.part \$out
    """
}

// fastp 1.4.0, the Botryllus settings for single-end reads: --trim_poly_g --length_required 50
// (adapters are detected automatically for single-end reads; --detect_adapter_for_pe is paired-only).
process fastp {
    tag "${series} ${run}"
    label 'fastp'
    storeDir "${O}/trimmed/${series}"

    input:
    tuple val(series), val(run), path(raw, stageAs: 'raw.fastq.gz')

    output:
    tuple val(series), val(run), path("${run}.fastq.gz"), path("${run}.fastp.json")

    script:
    """
    export PATH=${ENV}:\$PATH
    ${TIME} ${run}.fastp.time fastp -i ${raw} -o t.tmp.fastq.gz --compression 4 --trim_poly_g --length_required 50 \\
      --thread ${task.cpus} -j ${run}.fastp.json -h ${run}.fastp.html 2> ${run}.fastp.log
    WANT=\$(python -c "import json;print(json.load(open('${run}.fastp.json'))['summary']['after_filtering']['total_reads'])")
    N=\$(pigz -dc t.tmp.fastq.gz | awk 'END{print NR/4}')
    [ "\$N" = "\$WANT" ] || { echo "reads \$N != fastp \$WANT"; exit 3; }
    mv t.tmp.fastq.gz ${run}.fastq.gz
    """
}

// ---------------------------------------------------------------- SPLASH route

process splash {
    tag "${contrast}"
    label 'splash'
    storeDir "${O}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(cjs), path(fq)

    output:
    tuple val(contrast), path("splash_out")

    script:
    """
    export PATH=${ENV}:\$PATH
    for f in SRR*.fastq.gz; do echo "\${f%.fastq.gz} \$PWD/\$f"; done > input.txt
    [ \$(wc -l < input.txt) = \$(wc -l < ${cjs}) ] || { echo "samples and cjs differ"; exit 2; }
    mkdir splash_out && cd splash_out
    ${TIME} splash.time ${SPLASH}/splash --bin_path ${SPLASH} --Cjs_samplesheet ../${cjs} \\
      --pvals_correction_col_name Cjs_0_pval --with_effect_size_cts --n_most_freq_targets 4 \\
      --dump_sample_anchor_target_count_binary --without_compactors \\
      --n_threads_stage_1 4 --n_threads_stage_1_internal \$(( ${task.cpus} / 4 > 0 ? ${task.cpus} / 4 : 1 )) \\
      --n_threads_stage_2 ${task.cpus} --kmc_max_mem_GB ${(task.memory.toGiga() / 3) as int} \\
      --tmp_dir \$PWD/../splash_tmp --logs_dir logs --outname_prefix result ../input.txt \\
      > splash.stdout 2> splash.stderr
    rm -rf ../splash_tmp result_satc
    test -s result.after_correction.all_anchors.tsv && test -s result.after_correction.scores.tsv
    """
}

process compactors {
    tag "${contrast}"
    label 'compactors'
    storeDir "${O}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(splash_out), path(fq)

    output:
    tuple val(contrast), path("compactors")

    script:
    // every anchor with corrected supervised p <= 0.05 (all rows of scores.tsv), no top-N cut
    """
    mkdir compactors && cd compactors
    for f in ../SRR*.fastq.gz; do readlink -f \$f; done > input_list.txt
    echo "significant anchors: \$(( \$(wc -l < ../${splash_out}/result.after_correction.scores.tsv) - 1 ))" > n_input_anchors.txt
    if [ \$(wc -l < ../${splash_out}/result.after_correction.scores.tsv) -le 1 ]; then
      printf 'anchor\\tcompactor\\n' > compactors.tsv   # no significant anchor: nothing to extend
    else
      ${TIME} compactors.time ${SPLASH}/compactors --max_length 186 --num_threads ${task.cpus} --epsilon 0.001 --beta 0.5 \\
        --num_kmers 2 --input_format fastq input_list.txt ../${splash_out}/result.after_correction.scores.tsv \\
        compactors.tsv > compactors.log 2>&1
    fi
    test -s compactors.tsv
    """
}

// ---------------------------------------------------------------- DE-kupl counting + KaMRaT

process dekupl {
    tag "${contrast}"
    label 'dekupl'
    storeDir "${O}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(design), path(fq)

    output:
    tuple val(contrast), path("dekupl_limma")

    script:
    """
    mkdir -p dekupl_limma/fastq
    # single-end: DE-kupl counts <fastq_dir>/<sample>.fastq.gz (its jellyfish_count_single_end rule)
    for f in SRR*.fastq.gz; do ln -s \$(readlink -f \$f) dekupl_limma/fastq/\$f; done
    cd dekupl_limma
    ${ENV}/python ${SCRIPTS}/dekupl_config.py ../${design} \$PWD/fastq \$PWD/../dekupl_tmp config.json limma
    export APPTAINERENV_R_LIBS=${params.pec_root}/Rlib_dekupl
    ${TIME} dekupl.time apptainer run --bind /scratch ${params.pec_root}/images/dekupl-run_1.3.5.sif \\
      --configfile config.json -j ${task.cpus} --resources ram=${(task.memory.toMega() * 0.9) as long} -p > dekupl.log 2>&1
    rm -rf ../dekupl_tmp DEkupl_result/kmer_counts/*.txt.gz DEkupl_result/kmer_counts/*.jf
    test -s DEkupl_result/kmer_counts/raw-counts.tsv.gz
    """
}

process kamrat {
    tag "${contrast}"
    label 'kamrat'
    storeDir "${O}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(design), path(dekupl_dir)

    output:
    tuple val(contrast), path("kamrat_ttest")

    script:
    """
    mkdir kamrat_ttest && cd kamrat_ttest
    K="apptainer exec --bind /scratch ${params.pec_root}/images/kamrat_v1.2.sif kamrat"
    awk -F'\\t' 'NR > 1 { print \$1 "\\t" \$2 }' ../${design} > design.txt
    mkdir idx
    ${TIME} index.time \$K index -intab ../${dekupl_dir}/DEkupl_result/kmer_counts/raw-counts.tsv.gz -outdir idx \\
      -klen 31 -unstrand -nfbase 1000000000 > kamrat.log 2>&1
    \$K score -idxdir idx -scoreby ttest.padj -design design.txt -outfmt bin -outpath score.bin >> kamrat.log 2>&1
    \$K score -idxdir idx -scoreby ttest.padj -design design.txt -outfmt tab -outpath score.tsv >> kamrat.log 2>&1
    ${TIME} merge.time \$K merge -idxdir idx -overlap 30-15 -with score.bin:min -interv pearson:0.20 \\
      -outfmt tab -counts mean:float -outpath contigs_all.tsv >> kamrat.log 2>&1
    awk -F'\\t' 'NR == 1 || \$3 <= 0.05' contigs_all.tsv > contigs.tsv
    awk -F'\\t' 'NR > 1 { n++ } NR > 1 && \$2 <= 0.05 { s++ } END { print "kmers tested " n ", corrected p <= 0.05: " s+0 }' score.tsv > summary.txt
    awk -F'\\t' 'NR > 1 { n++ } NR > 1 && \$3 <= 0.05 { s++ } END { print "contigs " n ", representative corrected p <= 0.05: " s+0 }' contigs_all.tsv >> summary.txt
    gzip contigs_all.tsv
    rm -rf idx score.bin score.tsv
    test -s contigs.tsv
    """
}

// KaMRaT contigs at uncorrected p < 0.05: (a) KaMRaT's own Welch t-test on log2(count + 1),
// recomputed per contig (KaMRaT 1.2 reports only the corrected p); (b) the contigs' counts put
// back on the read scale and fitted with PyDESeq2 (~ pair + condition when paired by subject,
// ~ condition otherwise). No fold-change cutoff, as for Botryllus (2026-11-02).
process kamrat_routes {
    tag "${contrast}"
    label 'kamrat_deseq'
    storeDir "${O}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(design), val(deseq_design), path(kamrat_dir), path(dekupl_dir)

    output:
    tuple val(contrast), path("kamrat_p05"), path("kamrat_deseq2_p05")

    script:
    """
    export PATH=${ENV}:\$PATH
    W=\$PWD
    mkdir -p kamrat_p05 kamrat_deseq2 kamrat_deseq2_p05
    ${TIME} kamrat_p05/select.time python ${SCRIPTS}/select_kamrat_nominal.py \\
      ${kamrat_dir}/contigs_all.tsv.gz ${design} kamrat_p05 0 > kamrat_p05/select.log
    ${TIME} kamrat_deseq2/counts.time python ${SCRIPTS}/kamrat_contig_counts.py \\
      ${kamrat_dir}/contigs_all.tsv.gz ${dekupl_dir}/DEkupl_result/kmer_counts/raw-counts.tsv.gz ${design} kamrat_deseq2 > kamrat_deseq2/counts.log
    (cd ${SCRIPTS} && ${TIME} \$W/kamrat_deseq2/deseq.time python anchor_deseq.py \\
      \$W/kamrat_deseq2/contig_counts.tsv \$W/kamrat_deseq2/samples.txt \$W/${design} \$W/kamrat_deseq2/deseq ${task.cpus} ${deseq_design} \\
      > \$W/kamrat_deseq2/deseq.log 2>&1)
    gzip kamrat_deseq2/contig_counts.tsv
    ${TIME} kamrat_deseq2_p05/select.time python ${SCRIPTS}/select_kamrat_deseq.py \\
      kamrat_deseq2/deseq/deseq_${deseq_design}.parquet kamrat_deseq2_p05 > kamrat_deseq2_p05/select.log
    mv kamrat_deseq2 kamrat_deseq2_p05/
    test -s kamrat_p05/contigs.tsv && test -s kamrat_deseq2_p05/contigs.tsv
    """
}

// ---------------------------------------------------------------- protein search

process search {
    tag "${contrast} ${src}"
    label 'search'
    storeDir "${O}/contrasts/${contrast}"

    input:
    tuple val(contrast), val(src), path(seqdir), path(db)

    output:
    tuple val(contrast), val(src), path("search_${src}")

    script:
    def prep = src == 'compactors'
        ? "ln -s ../${seqdir}/compactors.tsv seqs.tsv; COL=compactor"
        : "ln -s ../${seqdir}/contigs.tsv seqs.tsv; COL=contig"
    """
    export PATH=${ENV}:\$PATH
    mkdir search_${src} && cd search_${src}
    ${prep}
    ${PECTOOLS} translate seqs.tsv pep 20 \$COL 2> translate.log
    if [ -s pep.peptides.faa ]; then
      seqkit split2 -s 500000 -O chunks pep.peptides.faa 2> split.log
      for fa in chunks/*.faa; do
        c=\$(basename \$fa .faa); c=\${c##*.}
        sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\\n%s,,%s\\n' "\$c" "\$PWD/\$fa") \\
          --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c ${task.cpus} -o \$c.scaled5.zip 2>> manysketch.log
        for spec in protein:7 dayhoff:12 hp:30; do
          m=\${spec%:*}; k=\${spec#*:}
          for s in 10 100 5; do
            sourmash scripts manysearch \$c.scaled5.zip ../${db}/gencode.\$m.k\$k.s\$s.rocksdb -m \$m -k \$k -s \$s \\
              -t 0.000001 -N -c ${task.cpus} -o ms.\$c.gencode.\$m.k\$k.s\$s.csv 2>> manysearch.log
          done
        done
      done
      python ${SCRIPTS}/make_pairs.py pairs.tsv hits.parquet ms.*.csv
      python ${SCRIPTS}/pair_align.py pairs.gencode.tsv pep.peptides.faa ../${db}/gencode.faa align.gencode.parquet ${task.cpus}
      rm -f ms.*.csv *.scaled5.zip
    fi
    rm -f seqs.tsv
    touch search.done
    """
}

// ---------------------------------------------------------------- gene names

process human_genes {
    label 'small_mem'
    publishDir "${O}/genes", mode: 'copy'

    input:
    val(searches)
    path(db)

    output:
    path("human_genes*")

    script:
    """
    ${ENV}/python ${SCRIPTS}/human_genes.py ${O}/contrasts ${DESIGN}/contrasts.tsv ${db}/gencode.id_map.tsv human_genes > human_genes.log 2>&1
    """
}

// ---------------------------------------------------------------- workflow

workflow {
    def series = params.series.tokenize(',')
    def want = params.contrasts ? params.contrasts.tokenize(',') : null
    def contrasts = file("${DESIGN}/contrasts.tsv").splitCsv(header: true, sep: '\t')
        .findAll { it.series in series && (want == null || it.contrast in want) }
    if (!contrasts) error "no contrast selected"
    // run -> contrasts it belongs to
    def run2c = [:].withDefault { [] }
    contrasts.each { c ->
        file("${DESIGN}/contrast_${c.contrast}.design.tsv").splitCsv(header: true, sep: '\t')
            .each { run2c[it.sample] << c.contrast }
    }
    def runs = file("${DESIGN}/samples.tsv").splitCsv(header: true, sep: '\t').findAll { it.run in run2c }

    db = gencode_db()

    raw = download(Channel.fromList(runs.collect { tuple(it.series, it.run, it.fastq_ftp, it.fastq_md5) }))
    trimmed = fastp(raw)

    // one record per (contrast, run); groupTuple with the known size emits each contrast as soon
    // as its last run is trimmed. If a run fails (ignored after its retries), its contrasts never
    // reach their size; when the trimmed channel closes, groupTuple drops those incomplete groups
    // (remainder false), so a contrast is never run with a sample missing and nothing waits forever.
    def nrun = contrasts.collectEntries { c -> [(c.contrast): run2c.count { r, cs -> c.contrast in cs }] }
    def cinfo = contrasts.collectEntries { [(it.contrast): it] }
    per_contrast = trimmed
        .flatMap { s, r, fq, js -> run2c[r].collect { c -> tuple(groupKey(c, nrun[c]), fq) } }
        .groupTuple()
        .map { k, fqs ->
            def c = k.toString()
            tuple(c, file("${DESIGN}/contrast_${c}.cjs"), file("${DESIGN}/contrast_${c}.design.tsv"),
                  cinfo[c].deseq_design, fqs)
        }

    sp   = splash(per_contrast.map { c, cjs, dsg, dd, fq -> tuple(c, cjs, fq) })
    comp = compactors(sp.join(per_contrast.map { c, cjs, dsg, dd, fq -> tuple(c, fq) }))
    dk   = dekupl(per_contrast.map { c, cjs, dsg, dd, fq -> tuple(c, dsg, fq) })
    km   = kamrat(dk.join(per_contrast.map { c, cjs, dsg, dd, fq -> tuple(c, dsg) })
                    .map { c, d, dsg -> tuple(c, dsg, d) })
    kr   = kamrat_routes(per_contrast.map { c, cjs, dsg, dd, fq -> tuple(c, dsg, dd) }
                           .join(km).join(dk)
                           .map { c, dsg, dd, k, d -> tuple(c, dsg, dd, k, d) })

    sr = search(comp.map { c, d -> tuple(c, 'compactors', d) }
                  .mix(km.map { c, d -> tuple(c, 'kamrat_ttest', d) })
                  .mix(kr.map { c, a, b -> tuple(c, 'kamrat_p05', a) })
                  .mix(kr.map { c, a, b -> tuple(c, 'kamrat_deseq2_p05', b) })
                  .combine(db))

    human_genes(sr.map { c, src, d -> "${c}/${src}" }.collect(), db)
}
