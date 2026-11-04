#!/usr/bin/env nextflow
/*
 * Notebook 361: Domen et al. 2026 (PNAS) PEC RNA-seq without a genome.
 *
 * Per contrast (from the frozen design in analysis/pec-domen2026-splash/design/):
 *   ENA download (md5 + read count checked) -> fastp (paired + interleaved copy)
 *   -> SPLASH 2.11.9, supervised (PEC +1, control -1)  -> compactors on every anchor with
 *      corrected p <= 0.05  -> six-frame peptides -> sourmash manysearch -> parasail pair check
 *   -> SPLASH anchor counts -> PyDESeq2
 *   -> DE-kupl 1.3.5 (limma test) -> contigs -> the same peptide search
 *      -> KaMRaT v1.2 on DE-kupl's k-mer counts (t-test, merge) -> contigs -> the same search
 *      -> KaMRaT contigs at uncorrected p <= 0.05 and 2-fold -> the same search
 * plus the known-answer control (simulated reads, same stages, scored against the truth).
 *
 * Every result is kept with storeDir under params.pec_root, in the same places the earlier
 * Slurm stages wrote, so results that already exist are found and not recomputed.
 * The stage logic is the same as analysis/pec-domen2026-splash/slurm/*.sbatch.
 */
nextflow.enable.dsl = 2

params.pec_root     = null      // $SCRATCH/nb361-pec
params.code_dir     = null      // .../analysis/pec-domen2026-splash
params.contrasts    = 'young_2h,young_24h,old_24h,young_13_49d,young_132d,old_41_93d'
params.control      = true
params.dekupl_method = 'limma'
params.reuse_dir    = ''        // other run's verified FASTQs, hard-linked when the MD5 matches
// Label-swap control: a manifest TSV (key, contrast, cjs, design; paths relative to code_dir),
// made by scripts/make_label_swaps.py. When set, only the SPLASH and anchor-count routes run,
// on the trimmed reads already in pec_root/trimmed, and every output goes under
// pec_root/label_swap/<contrast>/<flip>/, never under pec_root/contrasts/.
params.label_swap   = ''
// KaMRaT + DESeq2 on swapped labels: a manifest TSV (key, contrast, flip, design, deseq_design; paths
// relative to code_dir), made by scripts/make_kamrat_label_swaps.py. DE-kupl's k-mer counts do not
// depend on the labels (human swaps: same k-mers and counts), so they are reused. KaMRaT's contigs do:
// 20 % of contigs differed between real and swapped human labels, so KaMRaT, DESeq2 and the protein
// search all run again.
params.kamrat_swap  = ''

if (!params.pec_root || !params.code_dir) {
    error "set --pec_root and --code_dir"
}

ENV      = "${params.pec_root}/env/bin"
SPLASH   = "${params.pec_root}/splash"
PECTOOLS = "${params.pec_root}/pec-tools/target/release/pec-tools"
SCRIPTS  = "${params.code_dir}/scripts"
DB       = "${params.pec_root}/db"

// ---------------------------------------------------------------- reads and trimming

process download {
    tag "${run}"
    label 'download'
    storeDir "${params.pec_root}/reads"

    input:
    val run

    output:
    tuple val(run), path("${run}_1.fastq.gz"), path("${run}_2.fastq.gz"), path("${run}_1.json"), path("${run}_2.json"), path("${run}.done")

    script:
    """
    export PATH=${ENV}:\$PATH
    python ${SCRIPTS}/download_run.py ${params.pec_root}/meta/ena_filereport.tsv ${run} . ${params.reuse_dir}
    """
}

process fastp {
    tag "${sample}"
    label 'fastp'
    storeDir "${store}"

    input:
    // staged under names that cannot match the outputs (control inputs are named <sample>_<mate>.fastq.gz)
    tuple val(sample), val(store), path(r1, stageAs: 'in_1.fastq.gz'), path(r2, stageAs: 'in_2.fastq.gz')

    output:
    tuple val(sample), path("${sample}_1.fastq.gz"), path("${sample}_2.fastq.gz"), path("${sample}.il.fastq.gz"), path("${sample}.fastp.json"), path("${sample}.fastp.html"), path("${sample}.done")

    script:
    """
    export PATH=${ENV}:\$PATH
    # fastp picks gzip from the extension, so temporary names end in .fastq.gz
    fastp -i ${r1} -I ${r2} -o t1.tmp.fastq.gz -O t2.tmp.fastq.gz --compression 4 \\
      --detect_adapter_for_pe --trim_poly_g --length_required 50 --thread ${task.cpus} \\
      -j ${sample}.fastp.json -h ${sample}.fastp.html 2> ${sample}.fastp.log
    WANT=\$(python -c "import json;print(json.load(open('${sample}.fastp.json'))['summary']['after_filtering']['total_reads'])")
    N1=\$(pigz -dc t1.tmp.fastq.gz | awk 'END{print NR/4}'); N2=\$(pigz -dc t2.tmp.fastq.gz | awk 'END{print NR/4}')
    [ "\$N1" = "\$N2" ] && [ \$(( N1 + N2 )) = "\$WANT" ] || { echo "R1 \$N1 R2 \$N2 fastp \$WANT"; exit 3; }
    mv t1.tmp.fastq.gz ${sample}_1.fastq.gz; mv t2.tmp.fastq.gz ${sample}_2.fastq.gz
    # interleaved copy for SPLASH: R1 record, then its R2 record
    paste <(pigz -dc ${sample}_1.fastq.gz | paste - - - -) <(pigz -dc ${sample}_2.fastq.gz | paste - - - -) \\
      | tr '\\t' '\\n' | pigz -p ${task.cpus} -1 > ${sample}.il.fastq.gz
    N=\$(pigz -dc ${sample}.il.fastq.gz | awk 'END{print NR/4}')
    [ "\$N" = "\$WANT" ] || { echo "interleaved \$N != fastp \$WANT"; exit 3; }
    echo "\$N" > ${sample}.done
    """
}

process simulate_control {
    label 'small'
    storeDir "${params.pec_root}/control"

    output:
    path("reads")

    script:
    """
    export PATH=${ENV}:\$PATH
    mkdir reads
    ${PECTOOLS} simulate ${params.code_dir}/control/control_transcripts.fa ${params.code_dir}/control/design.tsv reads 1000000 11 2> simulate.log
    pigz -p ${task.cpus} reads/*.fastq
    """
}

// ---------------------------------------------------------------- SPLASH channel

process splash {
    tag "${contrast}"
    label 'splash'
    storeDir "${params.pec_root}/${contrast.contains('/') ? 'label_swap' : 'contrasts'}/${contrast}"

    input:
    tuple val(contrast), path(cjs), path(il)

    output:
    tuple val(contrast), path("splash_out")

    script:
    """
    export PATH=${ENV}:\$PATH
    for f in *.il.fastq.gz; do echo "\${f%.il.fastq.gz} \$PWD/\$f"; done > input.txt
    [ \$(wc -l < input.txt) = \$(wc -l < ${cjs}) ] || { echo "samples and cjs differ"; exit 2; }
    mkdir splash_out && cd splash_out
    ${SPLASH}/splash --bin_path ${SPLASH} --Cjs_samplesheet ../${cjs} \\
      --pvals_correction_col_name Cjs_0_pval --with_effect_size_cts --n_most_freq_targets 4 \\
      --dump_sample_anchor_target_count_binary --without_compactors \\
      --n_threads_stage_1 4 --n_threads_stage_1_internal \$(( ${task.cpus} / 4 > 0 ? ${task.cpus} / 4 : 1 )) \\
      --n_threads_stage_2 ${task.cpus} --kmc_max_mem_GB ${(task.memory.toGiga() / 3) as int} \\
      --tmp_dir \$PWD/../splash_tmp --logs_dir logs --outname_prefix result ../input.txt \\
      > splash.stdout 2> splash.stderr
    rm -rf ../splash_tmp
    test -s result.after_correction.all_anchors.tsv && test -s result.after_correction.scores.tsv
    """
}

process compactors {
    tag "${contrast}"
    label 'compactors'
    storeDir "${params.pec_root}/${contrast.contains('/') ? 'label_swap' : 'contrasts'}/${contrast}"

    input:
    tuple val(contrast), path(splash_out), path(il)

    output:
    tuple val(contrast), path("compactors")

    script:
    // every anchor with corrected supervised p <= 0.05 (all rows of scores.tsv), no top-N cut
    """
    mkdir compactors && cd compactors
    for f in ../*.il.fastq.gz; do readlink -f \$f; done > input_list.txt
    echo "significant anchors: \$(( \$(wc -l < ../${splash_out}/result.after_correction.scores.tsv) - 1 ))" > n_input_anchors.txt
    ${SPLASH}/compactors --max_length 186 --num_threads ${task.cpus} --epsilon 0.001 --beta 0.5 \\
      --num_kmers 2 --input_format fastq input_list.txt ../${splash_out}/result.after_correction.scores.tsv \\
      compactors.tsv > compactors.log 2>&1
    test -s compactors.tsv
    """
}

// Compactors of the anchors that pass the paper's own cutoff in the anchor-count DESeq2 fit
// (uncorrected p <= 0.05, |log2FC| > 1), so they can be matched to the paper's gene lists.
// Primary design per contrast, as frozen in design/ANALYSIS_PLAN.md (2026-10-30):
// ~ pair + condition, except young_132d (no pairs): ~ genotype + condition.
process compactors_nominal {
    tag "${contrast}"
    label 'compactors'
    storeDir "${params.pec_root}/${contrast.contains('/') ? 'label_swap' : 'contrasts'}/${contrast}"

    input:
    tuple val(contrast), path(deseq), path(il)

    output:
    tuple val(contrast), path("compactors_anchor_nominal")

    script:
    def design = contrast.tokenize('/')[0] == 'young_132d' ? 'genotype_condition' : 'pair_condition'
    """
    mkdir compactors_anchor_nominal && cd compactors_anchor_nominal
    for f in ../*.il.fastq.gz; do readlink -f \$f; done > input_list.txt
    ${ENV}/python ${SCRIPTS}/select_nominal_anchors.py ../${deseq}/deseq_${design}.parquet anchors_nominal.tsv > select.log
    ${SPLASH}/compactors --max_length 186 --num_threads ${task.cpus} --epsilon 0.001 --beta 0.5 \\
      --num_kmers 2 --input_format fastq input_list.txt anchors_nominal.tsv compactors.tsv > compactors.log 2>&1
    test -s compactors.tsv
    """
}

process anchor_counts {
    tag "${contrast}"
    label 'anchor_counts'
    storeDir "${params.pec_root}/${contrast.contains('/') ? 'label_swap' : 'contrasts'}/${contrast}"

    input:
    tuple val(contrast), path(splash_out), path(design)

    output:
    tuple val(contrast), path("anchor_counts.tsv"), path("deseq")

    script:
    """
    export PATH=${ENV}:\$PATH
    tail -n +2 ${design} | cut -f1 > samples.txt
    ALL=${splash_out}/result.after_correction.all_anchors.tsv
    for b in ${splash_out}/result_satc/bin*.satc; do
      ${SPLASH}/satc_dump --sample_names ${splash_out}/sample_name_to_id.mapping.txt \$b /dev/stdout \\
        | ${PECTOOLS} anchor-counts samples.txt \$ALL >> anchor_counts.tsv 2>> anchor_counts.log
    done
    N=\$(wc -l < anchor_counts.tsv); W=\$(( \$(wc -l < \$ALL) - 1 ))
    [ "\$N" = "\$W" ] || { echo "anchor count rows \$N != tested anchors \$W"; exit 3; }
    python ${SCRIPTS}/anchor_deseq.py anchor_counts.tsv samples.txt ${design} deseq ${task.cpus} > deseq.log 2>&1
    test -s deseq/deseq_summary.json
    """
}

// ---------------------------------------------------------------- DE-kupl channel

process dekupl {
    tag "${contrast}"
    label 'dekupl'
    storeDir "${params.pec_root}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(design), path(pairs)

    output:
    tuple val(contrast), path("dekupl_${params.dekupl_method}")

    script:
    def W = "dekupl_${params.dekupl_method}"
    """
    mkdir -p ${W}/fastq
    # absolute targets, so the links still resolve after storeDir moves the folder
    for f in *_1.fastq.gz *_2.fastq.gz; do ln -s \$(readlink -f \$f) ${W}/fastq/\$f; done
    cd ${W}
    ${ENV}/python ${SCRIPTS}/dekupl_config.py ../${design} \$PWD/fastq \$PWD/../dekupl_tmp config.json ${params.dekupl_method}
    # the image's limma script loads pryr (never used) but the image lacks it; supplied beside it
    export APPTAINERENV_R_LIBS=${params.pec_root}/Rlib_dekupl
    apptainer run --bind /scratch ${params.pec_root}/images/dekupl-run_1.3.5.sif \\
      --configfile config.json -j ${task.cpus} --resources ram=${(task.memory.toMega() * 0.9) as long} -p > dekupl.log 2>&1
    rm -rf ../dekupl_tmp
    test -s DEkupl_result/control_vs_PEC_kmer_counts/merged-diff-counts.tsv.gz
    """
}

// ---------------------------------------------------------------- KaMRaT channel

// KaMRaT v1.2 (Xue et al. 2024, Bioinformatics 40:btae090) on DE-kupl's k-mer count table
// (k = 31, unstranded, k-mers seen at least 5 times in at least 2 samples). Every k-mer gets a
// t-test on log2(count + 1) between PEC and control, BH-corrected across all k-mers; then all
// k-mers are merged into contigs (overlap 30 to 15, Pearson distance <= 0.20 between neighbours),
// each contig represented by its smallest corrected p. Outputs: k-mers with corrected p <= 0.05,
// contigs whose representative k-mer has corrected p <= 0.05, and every contig.
process kamrat {
    tag "${contrast}"
    label 'kamrat'
    storeDir "${params.pec_root}/${contrast.contains('/') ? 'label_swap' : 'contrasts'}/${contrast}"

    input:
    tuple val(contrast), path(design), path(dekupl_dir)

    output:
    tuple val(contrast), path("kamrat_ttest")

    script:
    """
    mkdir kamrat_ttest && cd kamrat_ttest
    K="apptainer exec --bind /scratch ${params.pec_root}/images/kamrat_v1.2.sif kamrat"
    awk -F'\\t' 'NR > 1 { print \$1 "\\t" \$2 }' ../${design} > design.txt
    mkdir idx  # KaMRaT index aborts (exit 134) if its output folder does not exist
    \$K index -intab ../${dekupl_dir}/DEkupl_result/kmer_counts/raw-counts.tsv.gz -outdir idx \\
      -klen 31 -unstrand -nfbase 1000000000 > kamrat.log 2>&1
    \$K score -idxdir idx -scoreby ttest.padj -design design.txt -outfmt bin -outpath score.bin >> kamrat.log 2>&1
    \$K score -idxdir idx -scoreby ttest.padj -design design.txt -outfmt tab -outpath score.tsv >> kamrat.log 2>&1
    \$K merge -idxdir idx -overlap 30-15 -with score.bin:min -interv pearson:0.20 \\
      -outfmt tab -counts mean:float -outpath contigs_all.tsv >> kamrat.log 2>&1
    awk -F'\\t' 'NR == 1 || \$2 <= 0.05' score.tsv | gzip > kmers_padj05.tsv.gz
    awk -F'\\t' 'NR == 1 || \$3 <= 0.05' contigs_all.tsv > contigs.tsv
    awk -F'\\t' 'NR > 1 { n++ } NR > 1 && \$2 <= 0.05 { s++ } END { print "kmers tested " n ", corrected p <= 0.05: " s }' score.tsv > summary.txt
    awk -F'\\t' 'NR > 1 { n++ } NR > 1 && \$3 <= 0.05 { s++ } END { print "contigs " n ", representative corrected p <= 0.05: " s }' contigs_all.tsv >> summary.txt
    gzip contigs_all.tsv
    rm -rf idx score.bin score.tsv
    test -s contigs.tsv
    """
}

// KaMRaT contigs at the anchor-count route's uncorrected cutoff (added 2026-11-02). KaMRaT 1.2
// reports only the corrected p, which keeps 0 contigs in every real contrast; this recomputes
// KaMRaT's own Welch t-test on log2(count + 1) per contig (on its mean normalised counts) and
// keeps p <= 0.05 and |log2 fold change| >= 1. The control has no paper list, so it is skipped.
process kamrat_nominal {
    tag "${contrast}"
    label 'small_mem'
    storeDir "${params.pec_root}/contrasts/${contrast}"

    input:
    tuple val(contrast), path(design), path(kamrat_dir)

    output:
    tuple val(contrast), path("kamrat_nominal")

    script:
    """
    ${ENV}/python ${SCRIPTS}/select_kamrat_nominal.py ${kamrat_dir}/contigs_all.tsv.gz ${design} kamrat_nominal > kamrat_nominal.log
    mv kamrat_nominal.log kamrat_nominal/
    test -s kamrat_nominal/contigs.tsv
    """
}

// PyDESeq2 on one swap's KaMRaT contigs, then contigs with p < 0.05, as for the real labels
// (slurm/18_kamrat_dekupl_p05.sbatch). No fold-change cutoff.
process kamrat_swap_deseq {
    tag "${key}"
    label 'kamrat_deseq'
    storeDir "${params.pec_root}/label_swap/${key}"

    input:
    tuple val(key), path(design), val(deseq_design), path(kamrat_dir), path(dekupl_dir)

    output:
    tuple val(key), path("kamrat_deseq2_p05")

    script:
    """
    export PATH=${ENV}:\$PATH
    W=\$PWD
    mkdir -p kamrat_deseq2 kamrat_deseq2_p05
    python ${SCRIPTS}/kamrat_contig_counts.py ${kamrat_dir}/contigs_all.tsv.gz \\
      ${dekupl_dir}/DEkupl_result/kmer_counts/raw-counts.tsv.gz ${design} kamrat_deseq2 > kamrat_deseq2/counts.log
    (cd ${SCRIPTS} && /usr/bin/time -v -o \$W/kamrat_deseq2/deseq.time python anchor_deseq.py \\
      \$W/kamrat_deseq2/contig_counts.tsv \$W/kamrat_deseq2/samples.txt \$W/${design} \$W/kamrat_deseq2/deseq ${task.cpus} ${deseq_design} \\
      > \$W/kamrat_deseq2/deseq.log 2>&1)
    rm kamrat_deseq2/contig_counts.tsv
    python ${SCRIPTS}/select_kamrat_deseq.py kamrat_deseq2/deseq/deseq_${deseq_design}.parquet kamrat_deseq2_p05 > kamrat_deseq2_p05/select.log
    mv kamrat_deseq2 kamrat_deseq2_p05/
    test -s kamrat_deseq2_p05/contigs.tsv
    """
}

// ---------------------------------------------------------------- protein search

process search {
    tag "${contrast} ${src}"
    label 'search'
    storeDir "${params.pec_root}/${contrast.contains('/') ? 'label_swap' : 'contrasts'}/${contrast}"

    input:
    tuple val(contrast), val(src), path(seqdir)

    output:
    tuple val(contrast), val(src), path("search_${src}")

    script:
    // compactors, compactors_anchor_nominal: <dir>/compactors.tsv, column compactor; DE-kupl: its merged contigs, column contig
    // KaMRaT: its contigs with representative corrected p <= 0.05, column contig
    // kamrat_nominal: KaMRaT contigs with uncorrected p <= 0.05 and 2-fold, column contig
    def prep = src.startsWith('compactors')
        ? "ln -s ../${seqdir}/compactors.tsv seqs.tsv; COL=compactor"
        : src in ['kamrat_ttest', 'kamrat_nominal', 'kamrat_deseq2_p05']
        ? "ln -s ../${seqdir}/contigs.tsv seqs.tsv; COL=contig"
        : "zcat ../${seqdir}/DEkupl_result/control_vs_PEC_kmer_counts/merged-diff-counts.tsv.gz > dekupl_contigs.tsv; ln -s dekupl_contigs.tsv seqs.tsv; COL=contig"
    """
    export PATH=${ENV}:\$PATH
    mkdir search_${src} && cd search_${src}
    ${prep}
    ${PECTOOLS} translate seqs.tsv pep 20 \$COL 2> translate.log
    if [ -s pep.peptides.faa ]; then
      # Sketch and search in chunks of 500,000 peptides. young_24h's 3.19 M peptides made one
      # 8.6 GB sketch zip that manysearch could not load ("No query signatures loaded"); the
      # 1.9 GB zip of young_2h's 725,806 peptides loaded fine.
      seqkit split2 -s 500000 -O chunks pep.peptides.faa 2> split.log
      for fa in chunks/*.faa; do
        c=\$(basename \$fa .faa); c=\${c##*.}
        sourmash scripts manysketch <(printf 'name,genome_filename,protein_filename\\n%s,,%s\\n' "\$c" "\$PWD/\$fa") \\
          --singleton -p protein,k=7,scaled=5 -p dayhoff,k=12,scaled=5 -p hp,k=30,scaled=5 -c ${task.cpus} -o \$c.scaled5.zip
        for db in proteome swissprot botznik; do
          for spec in protein:7 dayhoff:12 hp:30; do
            m=\${spec%:*}; k=\${spec#*:}
            for s in 10 100 5; do
              sourmash scripts manysearch \$c.scaled5.zip ${DB}/\$db.\$m.k\$k.s\$s.rocksdb -m \$m -k \$k -s \$s \\
                -t 0.000001 -N -c ${task.cpus} -o ms.\$c.\$db.\$m.k\$k.s\$s.csv 2>> manysearch.log
            done
          done
        done
      done
      python ${SCRIPTS}/make_pairs.py pairs.tsv hits.parquet ms.*.csv
      for db in proteome swissprot botznik; do
        python ${SCRIPTS}/pair_align.py pairs.\$db.tsv pep.peptides.faa ${DB}/\$db.faa align.\$db.parquet ${task.cpus}
      done
    fi
    rm -f seqs.tsv
    touch search.done
    """
}

// ---------------------------------------------------------------- known-answer control scoring

process control_truth {
    label 'small_mem'
    // a new folder since KaMRaT was added (2026-11-01): storeDir would reuse the old files without it
    storeDir "${params.pec_root}/control_eval/with_kamrat"

    input:
    tuple path(splash_out), path(anchor_counts), path(deseq), path(dekupl_dir), path(kamrat_dir)

    output:
    path("control_truth_summary.json")
    path("control_genes_found.tsv")

    script:
    """
    mkdir -p cdir && cd cdir && ln -s ../${splash_out} splash_out && ln -s ../${deseq} deseq && ln -s ../${dekupl_dir} ${dekupl_dir} && ln -s ../${kamrat_dir} ${kamrat_dir}
    ${ENV}/python ${SCRIPTS}/control_truth.py ${params.code_dir}/control/control_transcripts.fa \\
      ${params.code_dir}/control/truth_design.tsv . .. > ../control_truth.log 2>&1
    """
}

// ---------------------------------------------------------------- comparison with the paper's lists

// Fraction of each paper gene list reached by each channel, against 1,000 baseMean-matched random
// gene sets (null frozen in design/ANALYSIS_PLAN.md, 2026-10-30). Reads the search outputs in place
// under pec_root, so it waits for every compactor search to finish.
process paper_recovery {
    label 'small_mem'
    publishDir "${params.pec_root}/paper_recovery", mode: 'copy'

    input:
    val(searches)

    output:
    path("paper_recovery.tsv")
    path("paper_recovery.log")

    script:
    def P = params.pec_root
    """
    ${ENV}/python ${SCRIPTS}/paper_recovery.py ${P}/contrasts ${P}/paper/paper_deseq_all.parquet \\
      ${P}/paper/paper_named_lists.tsv ${P}/db/botznik.id_map.tsv paper_recovery.tsv > paper_recovery.log 2>&1
    """
}

// ---------------------------------------------------------------- workflow

workflow {
  if (params.kamrat_swap) {
    // KaMRaT + DESeq2 on swapped labels, from the real contrast's DE-kupl k-mer counts
    ks = Channel.fromList(file(params.kamrat_swap).splitCsv(header: true, sep: '\t')).map { r ->
        if (!(r.key ==~ /^[a-z0-9_]+\/[a-z0-9_]+$/)) throw new IllegalArgumentException("kamrat_swap key ${r.key} must be <contrast>/<flip>")
        tuple(r.key, file("${params.code_dir}/${r.design}", checkIfExists: true), r.deseq_design,
              file("${params.pec_root}/contrasts/${r.contrast}/dekupl_limma", checkIfExists: true))
    }
    km = kamrat(ks.map { k, dsg, dd, dk -> tuple(k, dsg, dk) })
    kd = kamrat_swap_deseq(ks.map { k, dsg, dd, dk -> tuple(k, dsg, dd, dk) }.join(km)
        .map { k, dsg, dd, dk, kdir -> tuple(k, dsg, dd, kdir, dk) })
    search(kd.map { k, d -> tuple(k, 'kamrat_deseq2_p05', d) })
  } else if (params.label_swap) {
    // label-swap control: the same SPLASH and anchor-count stages on swapped designs
    swap = Channel.fromList(file(params.label_swap).splitCsv(header: true, sep: '\t')).map { r ->
        def dsg = file("${params.code_dir}/${r.design}", checkIfExists: true)
        def cjs = file("${params.code_dir}/${r.cjs}", checkIfExists: true)
        if (!(r.key ==~ /^[a-z0-9_]+\/[a-z0-9_]+$/)) throw new IllegalArgumentException("label_swap key ${r.key} must be <contrast>/<flip>")
        def ils = dsg.splitCsv(header: true, sep: '\t').collect {
            file("${params.pec_root}/trimmed/${it.sample}.il.fastq.gz", checkIfExists: true) }
        tuple(r.key, cjs, dsg, ils)
    }
    sw_sp = splash(swap.map { k, cjs, dsg, il -> tuple(k, cjs, il) })
    sw_comp = compactors(sw_sp.join(swap.map { k, cjs, dsg, il -> tuple(k, il) }))
    sw_ac = anchor_counts(sw_sp.join(swap.map { k, cjs, dsg, il -> tuple(k, dsg) }))
    sw_an = compactors_nominal(sw_ac.map { k, a, d -> tuple(k, d) }.join(swap.map { k, cjs, dsg, il -> tuple(k, il) }))
    search(sw_comp.map { k, d -> tuple(k, 'compactors', d) }
             .mix(sw_an.map { k, d -> tuple(k, 'compactors_anchor_nominal', d) }))
  } else {
    def contrasts = params.contrasts.tokenize(',')
    // frozen design: sample, run, treatment, genotype, age_days, contrast, ...
    def rows = file("${params.code_dir}/design/samples.tsv").splitCsv(header: true, sep: '\t')
    def used = rows.findAll { it.contrast in contrasts }
    def run2sample = used.collectEntries { [(it.run): it.sample] }
    def sample2contrast = used.collectEntries { [(it.sample): it.contrast] }

    reads = download(Channel.fromList(used.collect { it.run }))
        .map { run, r1, r2, j1, j2, d -> tuple(run2sample[run], "${params.pec_root}/trimmed", r1, r2) }

    // one fastp call for real and simulated samples (a DSL2 process runs once per workflow)
    if (params.control) {
        sim = simulate_control().flatMap { dir ->
            def fs = file("${dir}/*_{1,2}.fastq.gz").sort { it.name }
            fs.groupBy { it.name.replaceAll(/_[12]\.fastq\.gz$/, '') }
              .collect { s, pair -> tuple(s, "${params.pec_root}/control/trimmed", pair[0], pair[1]) }
        }
        reads = reads.mix(sim)
    }
    trimmed = fastp(reads)

    // group trimmed samples by contrast; the control's samples are ctrl1-6 and pec1-6
    // an unknown sample is an error, not a control sample
    keyed = trimmed.map { s, p1, p2, il, js, ht, d ->
        def c = sample2contrast.get(s) ?: (params.control && s ==~ /^(ctrl|pec)\d+$/ ? 'control' : null)
        if (c == null) throw new IllegalStateException("sample ${s} is not in the frozen design and is not a control sample")
        tuple(c, s, il, [p1, p2])
    }
    per_contrast = keyed.groupTuple().map { c, ss, ils, pairs ->
        def dsg = c == 'control' ? file("${params.code_dir}/control/control.design.tsv")
                                 : file("${params.code_dir}/design/contrast_${c}.design.tsv")
        def cjs = c == 'control' ? file("${params.code_dir}/control/control.cjs")
                                 : file("${params.code_dir}/design/contrast_${c}.cjs")
        tuple(c, cjs, dsg, ils, pairs.flatten())
    }

    sp = splash(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, cjs, il) })
    comp = compactors(sp.join(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, il) }))
    ac = anchor_counts(sp.join(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, dsg) }))
    dk = dekupl(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, dsg, pairs) })

    km = kamrat(dk.join(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, dsg) })
                  .map { c, d, dsg -> tuple(c, dsg, d) })

    kn = kamrat_nominal(km.filter { it[0] != 'control' }
                          .join(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, dsg) })
                          .map { c, k, dsg -> tuple(c, dsg, k) })

    // the control has no paper list, so it gets no paper-cutoff compactors
    an = compactors_nominal(ac.filter { it[0] != 'control' }.map { c, a, d -> tuple(c, d) }
                              .join(per_contrast.map { c, cjs, dsg, il, pairs -> tuple(c, il) }))

    sr = search(comp.map { c, d -> tuple(c, 'compactors', d) }
               .mix(an.map { c, d -> tuple(c, 'compactors_anchor_nominal', d) })
               .mix(dk.map { c, d -> tuple(c, "dekupl_${params.dekupl_method}", d) })
               .mix(km.map { c, d -> tuple(c, 'kamrat_ttest', d) })
               .mix(kn.map { c, d -> tuple(c, 'kamrat_nominal', d) }))

    paper_recovery(sr.filter { c, src, d -> c != 'control' && src.startsWith('compactors') }
                     .map { c, src, d -> "${c}/${src}" }.collect())

    if (params.control) {
        control_truth(sp.join(ac).join(dk).join(km).filter { it[0] == 'control' }
                        .map { c, s, a, d, k, m -> tuple(s, a, d, k, m) })
    }
  }
}
