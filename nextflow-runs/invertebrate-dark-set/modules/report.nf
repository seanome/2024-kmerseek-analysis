/*
 * The dark-set MultiQC report.
 *
 * In a module file rather than in main.nf deliberately. main.nf is where the length and
 * disorder arms are also landing, and three sets of edits to one file is three merge
 * conflicts in the same hunk. Everything the report needs lives here; main.nf carries an
 * include and the wiring.
 *
 * Every input except the dark summary is OPTIONAL, and optional here means the section is
 * left out with a note naming what is missing -- not that an empty panel is drawn. An
 * empty panel reads as a measured null, which is a different claim from not having looked.
 */

// Optional products, matched by their published suffix rather than by position. Everything
// the run happens to have is staged into one directory and
// build_dark_multiqc_inputs.py picks out what it recognises, so adding a fourth arm later
// means adding a suffix to that script and nothing here.
process buildDarkMultiqcInputs {
    tag "${species}"
    label 'python_scoring'
    memory '8 GB'
    publishDir "${params.outdir}/${species}/multiqc", mode: 'copy'

    input:
    tuple val(species), path(dark_summary)
    // stageAs with a bare `*` so every file keeps its published name. That is not
    // cosmetic: which arm a file came from is carried ONLY in the filename, and a rename
    // to input.1, input.2 would leave the script unable to tell a length parquet from a
    // disorder one.
    path extras, stageAs: 'extras/*'

    output:
    tuple val(species), path("multiqc_in"), emit: sections

    script:
    """
    set -euo pipefail
    # stageAs creates extras/ only when there is something to stage, so a run with no
    # optional arm has no directory at all. mkdir rather than a conditional flag: the
    # script treats an empty directory and a missing one the same way, and this keeps the
    # command line identical between the two cases.
    mkdir -p extras

    build_dark_multiqc_inputs.py \\
        --species       ${species} \\
        --dark-summary  ${dark_summary} \\
        --extra-dir     extras \\
        --outdir        multiqc_in
    """
}

process darkMultiqcReport {
    /*
     * Pinned by digest, the same multiqc 1.35 image the region benchmark reports with, so
     * the two reports cannot be rendered by two different versions and compared.
     */
    tag "${species}"
    container 'quay.io/biocontainers/multiqc@sha256:b65e3fe879df27b92334dda0fd987a6e21bdee09a2848551d4f287099a93b7ac'
    publishDir "${params.outdir}/${species}", mode: 'copy'

    input:
    tuple val(species), path(sections), path(mqc_config)

    output:
    path "${species}_dark_set_multiqc.html", emit: report
    path "${species}_dark_set_multiqc_data", emit: data
    // export_plots in the config writes png, svg and pdf of every panel. Declared as an
    // output so they are published rather than left behind in the work directory: the svg
    // is what goes into a figure, and a work directory is not a place to keep one.
    path "${species}_dark_set_multiqc_plots", emit: plots

    script:
    // Compute nodes on Sherlock have no outbound internet, so the update check has nothing
    // to reach and only costs a timeout. MPLCONFIGDIR keeps matplotlib's cache inside the
    // task directory; under Apptainer $HOME can be read-only and the plot export needs it.
    """
    set -euo pipefail
    export MPLCONFIGDIR=\$PWD/.mplconfig

    multiqc ${sections} \\
        --config ${mqc_config} \\
        --filename ${species}_dark_set_multiqc.html \\
        --outdir . \\
        --no-version-check \\
        --no-ansi \\
        --force
    """
}

/*
 * Both steps, wired.
 *
 * `dark_ch`  : tuple(species, <species>_dark_summary.json)  -- required
 * `extras_ch`: a channel emitting ONE list of optional files, already collected.
 *
 * The two arrive as separate input declarations rather than as one combined tuple, on
 * purpose. `.combine()` CONCATENATES tuples, so combining a (species, path) pair with a
 * channel carrying a collected list spreads that list into one long tuple and the process
 * is handed N positional arguments instead of a list. That exact bug already broke this
 * pipeline once, in kmerseekDarkGain. Two declarations cannot express it.
 */
workflow darkReportFrom {
    take:
    dark_ch
    extras_ch

    main:
    sections = buildDarkMultiqcInputs(dark_ch, extras_ch).sections
    report = darkMultiqcReport(
        sections.map { sp, dir -> tuple(sp, dir, file(params.multiqc_dark_config)) })

    emit:
    sections = sections
    report   = report.report
}
