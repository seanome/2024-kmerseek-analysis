/*
 * Pfam domains on the dark set.
 *
 * The dark fraction's first challenge is "are these real proteins". modules/length.nf
 * answers it with a proxy -- junk gene models are short -- and says so. This answers it
 * directly: a dark protein carrying a recognisable Pfam domain is a real protein that
 * phmmer, jackhmmer and mmseqs2 all still failed to place, which is the strongest form of
 * the claim rather than a weakening of it.
 *
 * Not circular. Darkness is defined by three pairwise SEQUENCE searches against reviewed
 * Swiss-Prot with the query's clade removed. Pfam is a library of profile HMMs built from
 * curated alignments -- a more sensitive instrument, against a different database -- so a
 * protein can carry a Pfam domain and still be dark. Those are the interesting ones.
 *
 * The scan runs on the DARK proteins only, not the proteome. The placed proteins are
 * placed already.
 *
 * Its own module file for the reason modules/length.nf gives: main.nf has several editors
 * and this keeps the conflict surface to one include and a few lines of wiring.
 *
 * No container on the python processes and no errorStrategy anywhere here, both for the
 * reasons modules/length.nf spells out: 'python_scoring' already resolves to
 * params.kmerseek_image per profile, and nextflow.config's global errorStrategy is what
 * catches the scheduler kill that `in 128..143` never sees.
 */

process extractDarkFasta {
    tag "${species}"
    label 'python_scoring'
    memory '8 GB'

    input:
    tuple val(species), path(query_fasta), path(dark_parquet)

    output:
    tuple val(species), path("${species}_dark.fasta"), path(dark_parquet)

    stub:
    """
    touch ${species}_dark.fasta
    """

    script:
    """
    set -euo pipefail
    extract_dark_fasta.py \\
        --query ${query_fasta} --dark ${dark_parquet} \\
        --out ${species}_dark.fasta
    """
}

process pfamSearchDark {
    tag "${species}"
    container 'quay.io/biocontainers/hmmer@sha256:7a2b317b8d2fd3650b4924a8482cddeb940d4a0746c6a1501ff03ac1b7439e0c'
    label 'high_cpu'
    time { Math.min(8 * task.attempt, 24) + 'h' }

    input:
    // No hmmpress'd .h3* files here. hmmscan needs them; hmmsearch reads the plain .hmm,
    // so staging them would only add a way for the task to fail on a library that was
    // never pressed.
    tuple val(species), path(dark_fasta), path(dark_parquet), path(pfam_hmm)

    output:
    tuple val(species), path("${species}_dark_pfam.tsv.gz"), path(dark_parquet)

    stub:
    """
    : | gzip -c > ${species}_dark_pfam.tsv.gz
    """

    script:
    /*
     * hmmsearch, not hmmscan. Both answer the same question; hmmscan reads one sequence at
     * a time against the whole library and is the slow direction when there are many
     * sequences, which a dark set is. The column meanings swap with the direction, which is
     * the only reason this awk differs from the one in qfo-pfam-region-benchmark's
     * hmmscanAnnotate: there the HMM is the target ($2 is its accession, $4 the protein),
     * here the HMM is the query ($5 is its accession, $1 the protein).
     *
     * -o /dev/null is load-bearing, not tidiness, and the reason is worth repeating from
     * hmmscanAnnotate because it cost a run there: --domtblout /dev/stdout puts the table
     * on stdout and without -o the human-readable report goes to the SAME stream. --noali
     * drops the alignments but NOT the per-sequence and per-domain tables, whose rows reach
     * 22 fields and so pass an NF filter. A report row whose $20 and $21 happen to look
     * like coordinates then enters the table as a fabricated domain call.
     */
    """
    set -euo pipefail

    if [ ! -s ${dark_fasta} ]; then
        echo "dark FASTA is empty -- every protein was placed; writing no hits"
        : | gzip -c > ${species}_dark_pfam.tsv.gz
        exit 0
    fi

    hmmsearch \\
        --domtblout /dev/stdout \\
        -o /dev/null \\
        --noali \\
        --cut_ga \\
        --cpu ${task.cpus} \\
        ${pfam_hmm} ${dark_fasta} \\
    | grep -v '^#' \\
    | awk 'NF >= 22 {print \$1 "\\t" \$4 "\\t" \$5 "\\t" \$20 "\\t" \$21 "\\t" \$14 "\\t" \$13}' \\
    | gzip -c > ${species}_dark_pfam.tsv.gz
    """
}

process summarizeDarkPfam {
    tag "${species}"
    label 'python_scoring'
    memory '16 GB'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(pfam_hits), path(dark_parquet)

    output:
    tuple val(species),
          path("${species}_pfam.parquet"),
          path("${species}_pfam_summary.json")

    stub:
    """
    touch ${species}_pfam.parquet
    echo '{"species": "${species}"}' > ${species}_pfam_summary.json
    """

    script:
    """
    set -euo pipefail
    summarize_dark_pfam.py \\
        --hits ${pfam_hits} --dark ${dark_parquet} --species ${species} \\
        --i-evalue ${params.pfam_i_evalue} \\
        --out ${species}_pfam.parquet \\
        --summary-out ${species}_pfam_summary.json
    """
}
