/*
 * Protein length, dark vs placed.
 *
 * The dark fraction conflates two populations: proteins whose homologs sequence search
 * genuinely cannot reach, and gene models that are not real proteins. Botryllus is a 2026
 * annotation, and a new gene set carries a tail of fragments and spurious ORF calls that
 * nothing places into Swiss-Prot because there is nothing to place. Junk models are
 * typically short, so length is the cheapest discriminator available before InterProScan
 * and Chainsaw finish -- and it either qualifies the headline number or leaves it standing.
 *
 * Its own module file rather than main.nf on purpose: two other agents are editing main.nf,
 * and this way the conflict surface is one include line and two lines of wiring.
 *
 * No container directive here. The 'python_scoring' label already resolves to
 * params.kmerseek_image in every profile; a container set in the process body would
 * override the profile and pin the image in two places. That image has polars and numpy
 * but NOT scipy -- checked against the image, not assumed -- which is why
 * compare_dark_lengths.py writes out the Mann-Whitney test instead of importing it.
 *
 * No errorStrategy here either. nextflow.config sets one globally that handles the
 * scheduler-kill case -- exitStatus == Integer.MAX_VALUE, which the usual `in 128..143`
 * idiom never catches -- and overriding it per process is how that protection gets lost.
 */

process compareDarkLengths {
    tag "${species}"
    label 'python_scoring'
    memory '16 GB'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(query_fasta), path(dark_parquet)

    output:
    tuple val(species),
          path("${species}_length_comparison.parquet"),
          path("${species}_length_summary.json")

    stub:
    """
    touch ${species}_length_comparison.parquet
    echo '{"species": "${species}"}' > ${species}_length_summary.json
    """

    script:
    """
    set -euo pipefail
    compare_dark_lengths.py \\
        --query ${query_fasta} --dark ${dark_parquet} --species ${species} \\
        --out ${species}_length_comparison.parquet \\
        --summary-out ${species}_length_summary.json
    """
}
