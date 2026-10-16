/*
 * Steps every annotator tool shares: the clade-excluded reference, the Pfam domains of
 * the proteins the searches hit, placing those domains on the query, and scoring.
 *
 * A tool module (diamond_best_hit.nf, mmseqs_best_hit.nf, and the ones G2 and G3 add)
 * only has to emit tuple(tool, setting, hits.tsv.gz) in the column order of
 * bin/annotator_best_hit.py: query target qstart qend tstart tend bits evalue pident
 * qaln taln. Everything after that is here.
 *
 * Resources are set on each process, and no process carries a label: the config's
 * `python` label would replace them with its own (24 GB, 12 h on sherlock). The python
 * steps name the kmerseek image directly, the image the main run's scoring uses. Memory and time double on each retry; only a kill by the
 * cluster is retried (killedByCluster, the same test main.nf uses): x1, x2, x4.
 */

def killedByCluster(task) {
    def status = task.exitStatus
    status == null || status == Integer.MAX_VALUE || status in 128..143
}

def retryOnKill(task) {
    (killedByCluster(task) && task.attempt <= 2) ? 'retry' : 'finish'
}

process cladeExcludedReference {
    /*
     * Swiss-Prot with the query's clade removed, by the OC lineage lines
     * (bin/build_clade_excluded_reference.py, which exits if the clade matches nothing).
     * Mammalia for human queries: the closest relative left is a bird or a reptile.
     *
     * History: on Swiss-Prot 2026_03 this excluded 68_008 Mammalia entries and kept
     * 507_740, in 47 s at 1.16 GB peak (Sherlock job 48137871, 2026-10-16).
     */
    tag "swissprot minus ${clade}"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 4.GB * 2 ** (task.attempt - 1) }
    time { 1.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/reference", mode: 'copy', pattern: '*.json'

    input:
    path swissprot_dat
    val clade

    output:
    path "reference.fasta", emit: fasta
    path "reference_summary.json", emit: summary

    script:
    """
    build_clade_excluded_reference.py \\
        --swissprot-dat ${swissprot_dat} \\
        --exclude-clade ${clade} \\
        --out-prefix reference \\
        --summary-out reference_summary.json
    """

    stub:
    """
    sed 's/^>sp|\\([^|]*\\)|.*/>\\1/' ${projectDir}/tests/fixtures/qfo/yeast.fasta > reference.fasta
    echo '{"excluded_clade": "${clade}", "stub": true}' > reference_summary.json
    """
}

process hitTargets {
    /* The distinct best-hit proteins of every search, so hmmscan reads thousands of
     * proteins, not the whole reference. */
    tag "best-hit proteins"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 4.GB * 2 ** (task.attempt - 1) }
    time { 1.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2

    input:
    path hits
    path reference

    output:
    path "hit_targets.fasta"

    script:
    """
    annotator_best_hit.py targets --hits ${hits} --reference ${reference} --out hit_targets.fasta
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/annotator/hit_targets.fasta hit_targets.fasta
    """
}

process hmmscanHitTargets {
    /*
     * Pfam domains on the best-hit proteins: hmmscan with Pfam's gathering thresholds
     * (--cut_ga), the threshold Pfam itself uses to place domains, and the same Pfam-A
     * library the hmmscan ceiling reads. Envelope coordinates ($20, $21), as in
     * hmmscanAnnotate. Swiss-Prot's own DR Pfam lines carry no coordinates, and Pfam
     * 38.2's regions file covers only 317_364 of 575_748 Swiss-Prot 2026_03 proteins.
     *
     * History: hmmscanAnnotate on the 998 midi-plus queries took 8 m 36 s on 4 CPUs at
     * 493 MB peak (run-midi-plus trace 2026-09-01, Slurm job 41567260). At most three
     * best-hit proteins per query (one per search) is at most 2_994 proteins, about three
     * times that work: ~26 min on 4 CPUs. Asked: 8 CPUs, 4 GB, 2 h.
     */
    tag "hmmscan best-hit proteins"
    container 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
    cpus 8
    memory { 4.GB * 2 ** (task.attempt - 1) }
    time { 2.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/hit_targets", mode: 'copy'

    input:
    tuple path(targets), path(pfam_hmm), path(pfam_hmm_aux)

    output:
    path "hit_target_domains.tsv.gz"

    script:
    // -o /dev/null for the reason hmmscanAnnotate gives: without it the report shares
    // stdout with the table.
    """
    set -euo pipefail
    hmmscan \\
        --domtblout /dev/stdout \\
        -o /dev/null \\
        --noali \\
        --cut_ga \\
        --cpu ${task.cpus} \\
        ${pfam_hmm} ${targets} \\
    | { grep -v '^#' || true; } \\
    | awk 'NF >= 22 {print \$4 "\\t" \$2 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -nc > hit_target_domains.tsv.gz
    """

    stub:
    """
    cp ${projectDir}/tests/fixtures/annotator/hit_target_domains.tsv.gz hit_target_domains.tsv.gz
    """
}

process placeBestHitDomains {
    tag "${tool} ${setting}"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 4.GB * 2 ** (task.attempt - 1) }
    time { 1.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/regions", mode: 'copy'

    input:
    tuple val(tool), val(setting), path(hits), path(target_domains), path(queries)

    output:
    tuple val(tool), val(setting), path("${tool}.${setting}.regions.tsv.gz"), emit: regions
    path "${tool}.${setting}.place_qc.json", emit: qc
    path "${tool}.${setting}.placed.parquet", emit: detail

    script:
    """
    annotator_best_hit.py place \\
        --hits ${hits} \\
        --target-domains ${target_domains} \\
        --queries ${queries} \\
        --tool ${tool} \\
        --setting ${setting} \\
        --min-overlap ${params.min_overlap} \\
        --out ${tool}.${setting}.regions.tsv.gz \\
        --detail-out ${tool}.${setting}.placed.parquet \\
        --qc-out ${tool}.${setting}.place_qc.json
    """

    stub:
    def fx = "${projectDir}/tests/fixtures/annotator"
    """
    cp ${fx}/${tool}.${setting}.regions.tsv.gz ${tool}.${setting}.regions.tsv.gz
    cp ${fx}/${tool}.${setting}.place_qc.json ${tool}.${setting}.place_qc.json
    touch ${tool}.${setting}.placed.parquet
    """
}

process scoreAnnotatorTool {
    /*
     * evaluate_domain_calls.py --direct-annotation: the calls are already query-side
     * domains, as hmmscan's are, so the script's own transfer step is skipped. File names
     * follow the main run's calls/<truth>.<tool>.<setting>.<species>.calls.parquet, which
     * is what the pile-up builder (PR 146) reads. `species` names what the tool searched:
     * the clade-excluded Swiss-Prot for the best-hit tools, its own models for the
     * reference-database tools.
     */
    tag "${tool} ${setting} ${species}"
    container "${params.kmerseek_image}"
    cpus 1
    memory { 8.GB * 2 ** (task.attempt - 1) }
    time { 1.h * 2 ** (task.attempt - 1) }
    errorStrategy { retryOnKill(task) }
    maxRetries 2
    publishDir "${params.outdir}/calls",   mode: 'copy', pattern: '*.calls.parquet'
    publishDir "${params.outdir}/metrics", mode: 'copy', pattern: '*.metrics.parquet'
    publishDir "${params.outdir}/curves",  mode: 'copy', pattern: '*.curve.parquet'

    input:
    tuple val(tool), val(setting), val(species), path(regions), path(truth), path(covariates)

    output:
    path "pfam.${tool}.${setting}.${species}.calls.parquet",   emit: calls
    path "pfam.${tool}.${setting}.${species}.metrics.parquet", emit: metrics
    path "pfam.${tool}.${setting}.${species}.curve.parquet",   emit: curve

    script:
    def stem = "pfam.${tool}.${setting}.${species}"
    """
    evaluate_domain_calls.py \\
        --regions      ${regions} \\
        --tool         ${tool} \\
        --variant      ${setting} \\
        --species      ${species} \\
        --truth        ${truth} \\
        --direct-annotation \\
        --covariates   ${covariates} \\
        --min-overlap  ${params.min_overlap} \\
        --strict-iou   ${params.strict_iou} \\
        --calls-out    ${stem}.calls.parquet \\
        --metrics-out  ${stem}.metrics.parquet \\
        --curve-out    ${stem}.curve.parquet
    """

    stub:
    def stem = "pfam.${tool}.${setting}.${species}"
    // Each tool group keeps its fixture in its own folder (annotator/, refdb/, ...).
    def fx = "${projectDir}/tests/fixtures"
    """
    for kind in calls metrics curve; do
        cp ${fx}/*/pfam.${tool}.${setting}.${species}.\$kind.parquet ${stem}.\$kind.parquet
    done
    """
}
