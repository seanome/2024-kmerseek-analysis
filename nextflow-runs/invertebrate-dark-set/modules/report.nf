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
    // The clade and the reference's own summary (entries kept and removed) are what the
    // overview section quotes: which proteome, against what, with what taken out. The
    // dark summary does not carry them, and the overview should not guess.
    //
    // stageAs with a bare `*` so every file keeps its published name. That is not
    // cosmetic: which arm a file came from is carried ONLY in the filename, and a rename
    // to input.1, input.2 would leave the script unable to tell a length parquet from a
    // disorder one. `extras` is a list, and may be the empty list for a species that ran
    // no optional arm.
    tuple val(species), val(clade), path(dark_summary),
          path(reference_summary, stageAs: 'reference_summary.json'),
          path(extras, stageAs: 'extras/*')

    output:
    tuple val(species), path("multiqc_in"), emit: sections

    stub:
    """
    mkdir -p multiqc_in
    ls extras 2>/dev/null > multiqc_in/staged.txt || true
    """

    script:
    // The search settings, for the overview's "what was done". Read from params here
    // rather than recorded by the searches, so a report-only re-render (-entry darkReport)
    // quotes the CURRENT params: they match the run unless one was overridden on the
    // command line for the run and not for the re-render. The one that changes a number,
    // evalue_call, is written into the dark summary by computeDarkSet and read from there.
    def run_params = groovy.json.JsonOutput.toJson([
        query_chunk_size:     params.query_chunk_size,
        evalue_report:        params.evalue_report,
        jackhmmer_iterations: params.jackhmmer_iterations,
        mmseqs2_sensitivity:  params.mmseqs2_sensitivity,
        min_region_score:     params.min_region_score,
        max_query_pvalue:     params.max_query_pvalue,
        min_shared_kmers:     params.min_shared_kmers,
    ])
    """
    set -euo pipefail
    # stageAs creates extras/ only when there is something to stage, so a run with no
    # optional arm has no directory at all. mkdir rather than a conditional flag: the
    # script treats an empty directory and a missing one the same way, and this keeps the
    # command line identical between the two cases.
    mkdir -p extras
    # Numbers only, so the single quotes are safe.
    echo '${run_params}' > run_params.json

    build_dark_multiqc_inputs.py \\
        --species           ${species} \\
        --clade             ${clade} \\
        --dark-summary      ${dark_summary} \\
        --reference-summary ${reference_summary} \\
        --run-params        run_params.json \\
        --extra-dir         extras \\
        --outdir            multiqc_in
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

    stub:
    """
    touch ${species}_dark_set_multiqc.html
    mkdir -p ${species}_dark_set_multiqc_data ${species}_dark_set_multiqc_plots
    """

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
 * `report_ch`: tuple(species, clade, <species>_dark_summary.json,
 * minus_<clade>/summary.json, [optional files]) -- one per species. The list is the last
 * element of the tuple rather than a separate collected channel: with several species in
 * one run each species' report needs ITS OWN extras, which a single collected list cannot
 * express, and `.join` on the species key is what pairs them. An empty list is a species
 * that ran no optional arm.
 */
workflow darkReportFrom {
    take:
    report_ch

    main:
    sections = buildDarkMultiqcInputs(report_ch).sections
    report = darkMultiqcReport(
        sections.map { sp, dir -> tuple(sp, dir, file(params.multiqc_dark_config)) })

    emit:
    sections = sections
    report   = report.report
}
