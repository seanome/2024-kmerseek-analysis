/*
 * Sequence-based disorder for the dark set.
 *
 * The dark set is every protein phmmer, jackhmmer and mmseqs2 all failed to place. This
 * process asks whether those proteins are more disordered than the ones the arms did
 * place. A known result on this project is that kmerseek's HP-alphabet arms show a real
 * accuracy dip at low pLDDT -- accuracy tracks disorder -- so if the dark set is markedly
 * the more disordered group, that is both an explanation for why sequence search misses
 * it and a warning about what any method will recover there. Reported either way; see the
 * header of bin/label_dark_disorder.py for why the reverse result is the stronger one for
 * the proteome-annotate claim.
 *
 * Disorder is predicted from SEQUENCE, not read off pLDDT. pLDDT below 50 is a confidence
 * measurement that correlates with disorder, and it also drops on a shallow MSA -- which
 * is the same thing that makes phmmer and jackhmmer fail. Reading this axis off pLDDT
 * would risk measuring the very property that defines the dark set. Botryllus also has
 * usable AFDB coverage for only ~1.6% of the query set, so pLDDT is not on the table here.
 *
 * In its own module file rather than main.nf so that concurrent edits to main.nf reduce to
 * an include line plus the two wiring lines in the darkSet workflow.
 */

process darkSetDisorder {
    tag "${species}"
    publishDir "${params.outdir}/${species}", mode: 'copy'

    /*
     * NO container, cpus, memory or time directive in this body, deliberately.
     *
     * A directive here loses SILENTLY to a config selector, and nextflow.config carries a
     * generic `process { cpus = 2; memory = '8 GB' }` in every profile, so anything set
     * here would be overridden without a warning. That is the same gotcha documented in
     * qfo-pfam-region-benchmark, where `container params.metapredict_image` in the process
     * body lost to a withLabel selector and the process quietly ran the wrong image.
     *
     * So the container AND the resources for this process both live in
     * `withName: darkSetDisorder` in nextflow.config, in both profiles. That selector
     * outranks the generic block and every withLabel, so it is the only place these
     * settings actually take effect. There is no `label` on this process for the same
     * reason: a label would bind container = params.kmerseek_image, which lacks metapredict.
     *
     * errorStrategy is likewise left alone. The global one in nextflow.config already
     * handles the scheduler-kill case (exitStatus == Integer.MAX_VALUE) and ends every
     * path at 'finish'; overriding it here would only reintroduce the bug it fixes.
     */

    input:
    tuple val(species), path(query_fasta), path(dark_parquet)

    output:
    tuple val(species),
          path("${species}_disorder.parquet"),
          path("${species}_disorder_summary.json")

    stub:
    """
    touch ${species}_disorder.parquet
    echo '{"species": "${species}"}' > ${species}_disorder_summary.json
    """

    script:
    def thr = params.metapredict_threshold ? "--threshold ${params.metapredict_threshold}" : ""
    """
    set -euo pipefail

    # An empty or missing input is a failed upstream step, not a proteome with no
    # disorder. Caught here, naming the species, so the run stops on the real cause
    # rather than publishing an empty parquet that reads as a result.
    if [ ! -s "${query_fasta}" ]; then
        echo "${species}: query FASTA ${query_fasta} is missing or empty" >&2
        exit 1
    fi
    if [ ! -s "${dark_parquet}" ]; then
        echo "${species}: dark-set parquet ${dark_parquet} is missing or empty" >&2
        exit 1
    fi

    # --accession-mode verbatim is load-bearing. compute_dark_set.py keys the dark parquet
    # on the first whitespace token with no splitting on '|'; the default 'uniprot' mode
    # would turn a QfO header's tr|A0A024R1R8|A0A024R1R8_HUMAN into A0A024R1R8 and the two
    # tables would share no key at all. Botryllus's FUN000001_FUN000001 ids carry no '|'
    # and so agree under either mode, which is exactly what would let this ship broken for
    # every other species. label_dark_disorder.py re-checks the join and fails loudly.
    predict_disorder_metapredict.py \\
        --fasta          ${query_fasta} \\
        ${thr} \\
        --accession-mode verbatim \\
        --out            ${species}_per_protein_disorder.parquet \\
        --summary-out    ${species}_metapredict_run.json

    label_dark_disorder.py \\
        --disorder    ${species}_per_protein_disorder.parquet \\
        --dark        ${dark_parquet} \\
        --query       ${query_fasta} \\
        --species     ${species} \\
        --out         ${species}_disorder.parquet \\
        --summary-out ${species}_disorder_summary.json
    """
}
